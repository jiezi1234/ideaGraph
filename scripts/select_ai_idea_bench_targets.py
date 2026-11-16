from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import Settings
from src.models.ollama_embeddings import OllamaEmbeddingModel
from src.retrieval.base import EmbeddingModel
from src.retrieval.embedding_cache import PaperEmbeddingCache
from src.workflows.ai_idea_bench_pilot import build_pilot_records


LOGGER = logging.getLogger(__name__)

SparseVector = dict[str, float]
DenseVector = list[float]
Vector = SparseVector | DenseVector

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "can",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "method",
    "methods",
    "model",
    "models",
    "of",
    "on",
    "or",
    "paper",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "via",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select related AI Idea Bench target papers with a local TF-IDF vector "
            "cluster/search, and export reference-only corpus manifests."
        )
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=Path(".cache/ai_idea_bench/target_paper_data.json"),
        help="Path to target_paper_data.json.",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing papers_data PDFs. Defaults to dataset_json parent / papers_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exports/ai-idea-bench-selected-10"),
        help="Directory for selected target reports.",
    )
    parser.add_argument("--query", default=None, help="Optional direction query, e.g. '3D scene understanding'.")
    parser.add_argument("--anchor-title", default=None, help="Optional target title used as the similarity anchor.")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--candidate-pool-size", type=int, default=50)
    parser.add_argument("--min-top-references", type=int, default=1)
    parser.add_argument("--require-target-pdf", action="store_true")
    parser.add_argument(
        "--embedding-backend",
        choices=["tfidf", "ollama"],
        default="tfidf",
        help="Use local TF-IDF vectors or the repo Ollama embedding model with PaperEmbeddingCache.",
    )
    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.15,
        help="MMR diversity penalty in [0, 1]. Lower keeps targets tighter around the query.",
    )
    return parser.parse_args()


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def build_target_text(record: dict) -> str:
    summary = record.get("summary") or {}
    split_topic = summary.get("split_topic") or ""
    if isinstance(split_topic, list):
        split_topic_text = " ".join(
            str(item.get("keyword") or item.get("topic") or item) if isinstance(item, dict) else str(item)
            for item in split_topic
        )
    else:
        split_topic_text = str(split_topic)
    parts = [
        str(record.get("target_paper") or ""),
        str(summary.get("topic") or ""),
        str(summary.get("revised_topic") or ""),
        str(summary.get("motivation") or ""),
        split_topic_text,
    ]
    return "\n".join(part for part in parts if part.strip())


def _term_counts(texts: Iterable[str]) -> list[Counter[str]]:
    return [Counter(tokenize(text)) for text in texts]


def build_tfidf_vectors(texts: list[str]) -> tuple[list[dict[str, float]], dict[str, float]]:
    counts = _term_counts(texts)
    document_count = len(counts)
    dfs: Counter[str] = Counter()
    for counter in counts:
        dfs.update(counter.keys())

    idf = {
        token: math.log((1 + document_count) / (1 + df)) + 1.0
        for token, df in dfs.items()
    }
    vectors = [_normalize_vector(_tfidf(counter, idf)) for counter in counts]
    return vectors, idf


def build_query_vector(query: str, idf: dict[str, float]) -> SparseVector:
    counter = Counter(token for token in tokenize(query) if token in idf)
    return _normalize_vector(_tfidf(counter, idf))


def _tfidf(counter: Counter[str], idf: dict[str, float]) -> SparseVector:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {
        token: (count / total) * idf[token]
        for token, count in counter.items()
        if token in idf
    }


def _normalize_vector(vector: SparseVector) -> SparseVector:
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    if norm == 0:
        return {}
    return {token: weight / norm for token, weight in vector.items()}


def cosine_similarity(left: Vector, right: Vector) -> float:
    if isinstance(left, list) and isinstance(right, list):
        return _dense_cosine_similarity(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return _sparse_cosine_similarity(left, right)
    raise TypeError("Cannot compare sparse and dense vectors.")


def _sparse_cosine_similarity(left: SparseVector, right: SparseVector) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(token, 0.0) for token, weight in left.items())


def _dense_cosine_similarity(left: DenseVector, right: DenseVector) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise ValueError("Dense embedding dimensions do not match.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def select_targets(
    *,
    records: list[dict],
    dataset_root: Path,
    pdf_dir: Path,
    count: int,
    query: str | None = None,
    anchor_title: str | None = None,
    candidate_pool_size: int = 50,
    min_top_references: int = 1,
    require_target_pdf: bool = False,
    diversity_weight: float = 0.15,
    embedding_backend: str = "tfidf",
    embedder: EmbeddingModel | None = None,
    cache: PaperEmbeddingCache | None = None,
    worker_count: int = 1,
) -> tuple[list[dict], list[dict]]:
    if count <= 0:
        raise ValueError("count must be positive")
    if query and anchor_title:
        raise ValueError("Use either query or anchor_title, not both.")

    pdf_by_title = _pdf_title_index(pdf_dir)
    eligible_records = [
        record
        for record in records
        if _is_eligible_record(
            record=record,
            dataset_root=dataset_root,
            pdf_by_title=pdf_by_title,
            min_top_references=min_top_references,
            require_target_pdf=require_target_pdf,
        )
    ]
    if len(eligible_records) < count:
        raise ValueError(f"Only {len(eligible_records)} eligible records available for count={count}.")

    texts = [build_target_text(record) for record in eligible_records]
    vectors, idf = _build_vectors(
        records=eligible_records,
        texts=texts,
        embedding_backend=embedding_backend,
        embedder=embedder,
        cache=cache,
        worker_count=worker_count,
    )
    query_vector = _resolve_query_vector(
        query=query,
        anchor_title=anchor_title,
        records=eligible_records,
        vectors=vectors,
        idf=idf,
        embedding_backend=embedding_backend,
        embedder=embedder,
        cache=cache,
    )

    if query_vector is None:
        anchor_position, scored_positions = _auto_dense_cluster(vectors=vectors, candidate_pool_size=candidate_pool_size)
    else:
        anchor_position = None
        scored_positions = [
            (position, cosine_similarity(query_vector, vector))
            for position, vector in enumerate(vectors)
        ]
        scored_positions.sort(key=lambda item: (-item[1], _record_index(eligible_records[item[0]])))

    pool_size = max(candidate_pool_size, count)
    pool_positions = [position for position, _ in scored_positions[:pool_size]]
    selected_positions = _mmr_select(
        pool_positions=pool_positions,
        vectors=vectors,
        base_scores={position: score for position, score in scored_positions},
        count=count,
        diversity_weight=max(0.0, min(1.0, diversity_weight)),
    )
    if anchor_position is not None and anchor_position not in selected_positions:
        selected_positions = [anchor_position, *selected_positions[: count - 1]]

    selected_records = [eligible_records[position] for position in selected_positions]
    selected = _decorate_records(
        records=selected_records,
        dataset_root=dataset_root,
        pdf_by_title=pdf_by_title,
        scores={position: score for position, score in scored_positions},
        positions=selected_positions,
        embedding_backend=embedding_backend,
    )
    candidates = _decorate_records(
        records=[eligible_records[position] for position in pool_positions],
        dataset_root=dataset_root,
        pdf_by_title=pdf_by_title,
        scores={position: score for position, score in scored_positions},
        positions=pool_positions,
        embedding_backend=embedding_backend,
    )
    for rank, item in enumerate(selected, start=1):
        item["selection_rank"] = rank
    return selected, candidates


def _build_vectors(
    *,
    records: list[dict],
    texts: list[str],
    embedding_backend: str,
    embedder: EmbeddingModel | None,
    cache: PaperEmbeddingCache | None,
    worker_count: int,
) -> tuple[list[Vector], dict[str, float]]:
    if embedding_backend == "tfidf":
        vectors, idf = build_tfidf_vectors(texts)
        return vectors, idf
    if embedding_backend != "ollama":
        raise ValueError(f"Unsupported embedding backend: {embedding_backend}")
    if embedder is None or cache is None:
        raise ValueError("Ollama embedding backend requires embedder and cache.")

    def embed_target(item: tuple[dict, str]) -> DenseVector:
        record, text = item
        cache_id = f"ai-idea-bench-target:{record.get('index')}"
        return cached_embedding(cache=cache, embedder=embedder, cache_id=cache_id, text=text)

    # target 选择阶段只把 target 信息用于抽样，不会写入建图库 manifest。
    with ThreadPoolExecutor(max_workers=min(len(texts), max(worker_count, 1)) or 1) as executor:
        vectors = list(executor.map(embed_target, zip(records, texts, strict=True)))
    return vectors, {}


def cached_embedding(
    *,
    cache: PaperEmbeddingCache,
    embedder: EmbeddingModel,
    cache_id: str,
    text: str,
) -> DenseVector:
    embedding = cache.load(
        paper_id=cache_id,
        model_name=embedder.model_name,
        abstract=text,
    )
    if embedding is not None:
        return embedding
    embedding = embedder.embed(text)
    cache.save(
        paper_id=cache_id,
        model_name=embedder.model_name,
        abstract=text,
        embedding=embedding,
    )
    return embedding


def _resolve_query_vector(
    *,
    query: str | None,
    anchor_title: str | None,
    records: list[dict],
    vectors: list[Vector],
    idf: dict[str, float],
    embedding_backend: str,
    embedder: EmbeddingModel | None,
    cache: PaperEmbeddingCache | None,
) -> Vector | None:
    if query:
        if embedding_backend == "ollama":
            if embedder is None or cache is None:
                raise ValueError("Ollama embedding backend requires embedder and cache.")
            query_vector = cached_embedding(
                cache=cache,
                embedder=embedder,
                cache_id=f"ai-idea-bench-query:{query}",
                text=f"query: {query}",
            )
        else:
            query_vector = build_query_vector(query, idf)
        if not query_vector:
            raise ValueError("Query has no overlap with target vocabulary.")
        return query_vector
    if anchor_title:
        anchor_norm = normalize_title(anchor_title)
        for position, record in enumerate(records):
            title_norm = normalize_title(str(record.get("target_paper") or ""))
            if anchor_norm == title_norm or anchor_norm in title_norm or title_norm in anchor_norm:
                return vectors[position]
        raise ValueError(f"Anchor title not found: {anchor_title}")
    return None


def _auto_dense_cluster(
    *,
    vectors: list[Vector],
    candidate_pool_size: int,
) -> tuple[int, list[tuple[int, float]]]:
    if vectors and isinstance(vectors[0], list):
        return _auto_dense_centroid_cluster(vectors=vectors, candidate_pool_size=candidate_pool_size)
    best_anchor = 0
    best_neighbors: list[tuple[int, float]] = []
    best_density = -1.0
    neighbor_count = max(candidate_pool_size, 10)
    for position, vector in enumerate(vectors):
        neighbors = [
            (other_position, cosine_similarity(vector, other_vector))
            for other_position, other_vector in enumerate(vectors)
            if other_position != position
        ]
        neighbors.sort(key=lambda item: item[1], reverse=True)
        top_neighbors = neighbors[:neighbor_count]
        density = sum(score for _, score in top_neighbors[:10]) / max(1, min(10, len(top_neighbors)))
        if density > best_density:
            best_density = density
            best_anchor = position
            best_neighbors = top_neighbors
    return best_anchor, [(best_anchor, 1.0), *best_neighbors]


def _auto_dense_centroid_cluster(
    *,
    vectors: list[Vector],
    candidate_pool_size: int,
) -> tuple[int, list[tuple[int, float]]]:
    dense_vectors = [vector for vector in vectors if isinstance(vector, list)]
    if not dense_vectors:
        return 0, []
    dimensions = len(dense_vectors[0])
    centroid = [0.0] * dimensions
    for vector in dense_vectors:
        if len(vector) != dimensions:
            raise ValueError("Dense embedding dimensions do not match.")
        for index, value in enumerate(vector):
            centroid[index] += value
    centroid = [value / len(dense_vectors) for value in centroid]
    scored_positions = [
        (position, cosine_similarity(centroid, vector))
        for position, vector in enumerate(vectors)
    ]
    scored_positions.sort(key=lambda item: item[1], reverse=True)
    return scored_positions[0][0], scored_positions[: max(candidate_pool_size, 1)]


def _mmr_select(
    *,
    pool_positions: list[int],
    vectors: list[Vector],
    base_scores: dict[int, float],
    count: int,
    diversity_weight: float,
) -> list[int]:
    selected: list[int] = []
    remaining = list(dict.fromkeys(pool_positions))
    while remaining and len(selected) < count:
        best_position = max(
            remaining,
            key=lambda position: (
                (1.0 - diversity_weight) * base_scores.get(position, 0.0)
                - diversity_weight * _max_similarity_to_selected(position, selected, vectors),
                base_scores.get(position, 0.0),
                -position,
            ),
        )
        selected.append(best_position)
        remaining.remove(best_position)
    return selected


def _max_similarity_to_selected(
    position: int,
    selected: list[int],
    vectors: list[Vector],
) -> float:
    if not selected:
        return 0.0
    return max(cosine_similarity(vectors[position], vectors[selected_position]) for selected_position in selected)


def _decorate_records(
    *,
    records: list[dict],
    dataset_root: Path,
    pdf_by_title: dict[str, list[Path]],
    scores: dict[int, float],
    positions: list[int],
    embedding_backend: str,
) -> list[dict]:
    pilot_records = build_pilot_records(records, count=len(records))
    decorated: list[dict] = []
    for position, record, pilot_record in zip(positions, records, pilot_records, strict=True):
        refs = _reference_status(record=record, dataset_root=dataset_root)
        target_pdf = _target_pdf_path(record=record, pdf_by_title=pdf_by_title, dataset_root=dataset_root)
        existing_refs = sum(1 for ref in refs if ref["exists"])
        decorated.append(
            {
                "dataset_index": record.get("index"),
                "target_paper": record.get("target_paper"),
                "selection_score": round(scores.get(position, 0.0), 6),
                "embedding_backend": embedding_backend,
                "target_pdf_path": target_pdf,
                "target_pdf_exists": bool(target_pdf),
                "top_reference_count": len(refs),
                "existing_top_reference_count": existing_refs,
                "all_top_references_exist": bool(refs) and existing_refs == len(refs),
                "top_references": refs,
                "sanitized_problem": pilot_record["sanitized_problem"],
                "problem_extraction_prompt": pilot_record["problem_extraction_prompt"],
            }
        )
    return decorated


def _is_eligible_record(
    *,
    record: dict,
    dataset_root: Path,
    pdf_by_title: dict[str, list[Path]],
    min_top_references: int,
    require_target_pdf: bool,
) -> bool:
    refs = record.get("find_cite", {}).get("top_references", []) or []
    if len(refs) < min_top_references:
        return False
    if require_target_pdf and not _target_pdf_path(record=record, pdf_by_title=pdf_by_title, dataset_root=dataset_root):
        return False
    return True


def _pdf_title_index(pdf_dir: Path) -> dict[str, list[Path]]:
    indexed: dict[str, list[Path]] = {}
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        indexed.setdefault(normalize_title(pdf_path.stem), []).append(pdf_path)
    return indexed


def _target_pdf_path(
    *,
    record: dict,
    pdf_by_title: dict[str, list[Path]],
    dataset_root: Path,
) -> str | None:
    title = str(record.get("target_paper") or "")
    matches = pdf_by_title.get(normalize_title(title), [])
    if not matches:
        return None
    return _display_path(matches[0], dataset_root)


def _reference_status(*, record: dict, dataset_root: Path) -> list[dict]:
    refs = record.get("find_cite", {}).get("top_references", []) or []
    statuses: list[dict] = []
    for ref in refs:
        raw_path = str(ref.get("paper_local_path") or "")
        relative_path = raw_path.removeprefix("./")
        path = dataset_root / relative_path
        statuses.append(
            {
                "title": ref.get("title"),
                "paper_local_path": relative_path,
                "exists": path.exists(),
            }
        )
    return statuses


def _display_path(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root))
    except ValueError:
        return str(path)


def _record_index(record: dict) -> int:
    value = record.get("index")
    return value if isinstance(value, int) else 0


def write_outputs(
    *,
    selected: list[dict],
    candidates: list[dict],
    output_dir: Path,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "selected_targets.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_selected_csv(output_dir / "selected_targets.csv", selected)
    _write_candidate_csv(output_dir / "candidate_pool.csv", candidates)
    _write_problem_drafts(output_dir / "problem_drafts.jsonl", selected)
    _write_reference_availability(output_dir / "reference_availability.json", selected)
    _write_reference_manifest(output_dir / "reference_corpus_manifest.json", selected)


def _write_selected_csv(path: Path, selected: list[dict]) -> None:
    fieldnames = [
        "selection_rank",
        "dataset_index",
        "target_paper",
        "selection_score",
        "target_pdf_exists",
        "target_pdf_path",
        "top_reference_count",
        "existing_top_reference_count",
        "all_top_references_exist",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in selected:
            writer.writerow({field: item.get(field) for field in fieldnames})


def _write_candidate_csv(path: Path, candidates: list[dict]) -> None:
    fieldnames = [
        "dataset_index",
        "target_paper",
        "selection_score",
        "target_pdf_exists",
        "top_reference_count",
        "existing_top_reference_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in candidates:
            writer.writerow({field: item.get(field) for field in fieldnames})


def _write_problem_drafts(path: Path, selected: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for item in selected:
            payload = {
                "dataset_index": item["dataset_index"],
                "target_paper": item["target_paper"],
                "sanitized_problem": item["sanitized_problem"],
                "problem_extraction_prompt": item["problem_extraction_prompt"],
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_reference_availability(path: Path, selected: list[dict]) -> None:
    payload = [
        {
            "dataset_index": item["dataset_index"],
            "target_paper": item["target_paper"],
            "top_references": item["top_references"],
        }
        for item in selected
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_reference_manifest(path: Path, selected: list[dict]) -> None:
    references_by_path: dict[str, dict] = {}
    for item in selected:
        for ref in item["top_references"]:
            if not ref["exists"]:
                continue
            references_by_path.setdefault(
                ref["paper_local_path"],
                {
                    "paper_local_path": ref["paper_local_path"],
                    "titles": [],
                    "used_by_target_indexes": [],
                },
            )
            references_by_path[ref["paper_local_path"]]["titles"].append(ref["title"])
            references_by_path[ref["paper_local_path"]]["used_by_target_indexes"].append(item["dataset_index"])

    # 这里刻意只导出引用论文，target PDF 只能用于评测侧，不能进入建图库。
    pdfs = []
    for paper_local_path, payload in sorted(references_by_path.items()):
        pdfs.append(
            {
                "paper_local_path": paper_local_path,
                "titles": sorted(set(payload["titles"])),
                "used_by_target_indexes": sorted(set(payload["used_by_target_indexes"])),
            }
        )
    path.write_text(
        json.dumps({"pdf_count": len(pdfs), "pdfs": pdfs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()
    dataset_json = args.dataset_json
    dataset_root = dataset_json.parent
    pdf_dir = args.pdf_dir or dataset_root / "papers_data"
    records = json.loads(dataset_json.read_text(encoding="utf-8"))
    settings = Settings() if args.embedding_backend == "ollama" else None
    embedder = (
        OllamaEmbeddingModel(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_embed_model,
            timeout=settings.ollama_timeout,
        )
        if settings is not None
        else None
    )
    cache = PaperEmbeddingCache(settings.paper_embedding_cache_dir) if settings is not None else None
    selected, candidates = select_targets(
        records=records,
        dataset_root=dataset_root,
        pdf_dir=pdf_dir,
        count=args.count,
        query=args.query,
        anchor_title=args.anchor_title,
        candidate_pool_size=args.candidate_pool_size,
        min_top_references=args.min_top_references,
        require_target_pdf=args.require_target_pdf,
        diversity_weight=args.diversity_weight,
        embedding_backend=args.embedding_backend,
        embedder=embedder,
        cache=cache,
        worker_count=settings.embedding_worker_count if settings is not None else 1,
    )
    summary = {
        "dataset_json": str(dataset_json),
        "pdf_dir": str(pdf_dir),
        "record_count": len(records),
        "selected_count": len(selected),
        "candidate_pool_count": len(candidates),
        "query": args.query,
        "anchor_title": args.anchor_title,
        "require_target_pdf": args.require_target_pdf,
        "embedding_backend": args.embedding_backend,
        "embedding_model": embedder.model_name if embedder is not None else None,
        "embedding_cache_dir": str(settings.paper_embedding_cache_dir) if settings is not None else None,
        "selection_method": _selection_method(args.embedding_backend, bool(args.query or args.anchor_title)),
    }
    write_outputs(
        selected=selected,
        candidates=candidates,
        output_dir=args.output_dir,
        summary=summary,
    )
    LOGGER.info("selected_count=%s", len(selected))
    LOGGER.info("output_dir=%s", args.output_dir)
    LOGGER.info("embedding_backend=%s", args.embedding_backend)
    LOGGER.info("reference_pdf_count=%s", len({ref["paper_local_path"] for item in selected for ref in item["top_references"] if ref["exists"]}))
    return 0


def _selection_method(embedding_backend: str, has_query_or_anchor: bool) -> str:
    if embedding_backend == "ollama" and not has_query_or_anchor:
        return "ollama_auto_centroid_mmr"
    suffix = "query_mmr" if has_query_or_anchor else "auto_dense_cluster_mmr"
    return f"{embedding_backend}_{suffix}"


if __name__ == "__main__":
    raise SystemExit(main())
