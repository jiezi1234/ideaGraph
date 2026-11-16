from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure retrieved-paper hit rate against AI Idea Bench target reference papers."
    )
    parser.add_argument("--run-manifest", required=True, type=Path)
    parser.add_argument("--reference-availability", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def normalize_title(title: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return " ".join(normalized.split())


def load_run_manifest(path: str | Path) -> list[dict[str, Any]]:
    runs = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        if "dataset_index" not in payload or "archive_path" not in payload:
            raise ValueError(f"run manifest line {line_number} must contain dataset_index and archive_path")
        runs.append(payload)
    return runs


def load_reference_records(path: str | Path) -> dict[int, dict[str, Any]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("reference availability file must contain a list")
    return {int(record["dataset_index"]): record for record in records}


def extract_retrieved_paper_titles(archive_path: str | Path) -> list[str]:
    archive = json.loads(Path(archive_path).read_text(encoding="utf-8"))
    titles: list[str] = []

    for paper in archive.get("candidate_papers") or []:
        _append_title(titles, paper.get("title") or paper.get("paper_id"))
    selected = archive.get("selected_paper")
    if isinstance(selected, dict):
        _append_title(titles, selected.get("title") or selected.get("paper_id"))
    for chunk in archive.get("retrieved_chunks") or []:
        if isinstance(chunk, dict):
            # 分块向量 baseline 返回的是 chunk，但命中率对比仍以 chunk 绑定的论文为单位。
            _append_title(titles, chunk.get("title") or chunk.get("paper_id"))

    for trace_item in archive.get("tool_trace") or []:
        observation = trace_item.get("observation") if isinstance(trace_item, dict) else None
        _append_observation_paper_titles(titles, observation)

    _append_observation_paper_titles(titles, archive.get("graph_context"))
    return _dedupe_titles(titles)


def evaluate_retrieval_hits(
    *,
    run_manifest: str | Path,
    reference_availability: str | Path,
) -> list[dict[str, Any]]:
    references_by_index = load_reference_records(reference_availability)
    records = []
    for run in load_run_manifest(run_manifest):
        dataset_index = int(run["dataset_index"])
        reference_record = references_by_index.get(dataset_index)
        if reference_record is None:
            raise ValueError(f"dataset_index not found in reference availability: {dataset_index}")

        gold_titles = [
            str(reference.get("title") or "").strip()
            for reference in reference_record.get("top_references") or []
            if str(reference.get("title") or "").strip()
        ]
        retrieved_titles = extract_retrieved_paper_titles(run["archive_path"])
        gold_by_norm = {normalize_title(title): title for title in gold_titles}
        retrieved_norms = {normalize_title(title) for title in retrieved_titles}
        matched_references = [title for norm, title in gold_by_norm.items() if norm in retrieved_norms]
        missed_references = [title for title in gold_titles if normalize_title(title) not in retrieved_norms]
        hit_count = len(matched_references)
        gold_count = len(gold_titles)
        retrieved_count = len(retrieved_titles)
        records.append(
            {
                "dataset_index": dataset_index,
                "target_paper": reference_record.get("target_paper"),
                "variant": run.get("variant"),
                "archive_path": run["archive_path"],
                "gold_reference_count": gold_count,
                "retrieved_paper_count": retrieved_count,
                "hit_count": hit_count,
                "hit_rate": hit_count / gold_count if gold_count else 0.0,
                "retrieval_precision": hit_count / retrieved_count if retrieved_count else 0.0,
                "matched_references": matched_references,
                "missed_references": missed_references,
                "retrieved_papers": retrieved_titles,
            }
        )
    return records


def write_outputs(records: list[dict[str, Any]], output_json: str | Path, output_csv: str | Path | None) -> None:
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output_csv is None:
        return

    fieldnames = [
        "dataset_index",
        "target_paper",
        "variant",
        "archive_path",
        "gold_reference_count",
        "retrieved_paper_count",
        "hit_count",
        "hit_rate",
        "retrieval_precision",
        "matched_references",
        "missed_references",
    ]
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record[key] for key in fieldnames}
            row["matched_references"] = "; ".join(record["matched_references"])
            row["missed_references"] = "; ".join(record["missed_references"])
            writer.writerow(row)


def _append_observation_paper_titles(titles: list[str], observation: Any) -> None:
    if not isinstance(observation, dict):
        return
    for entity in observation.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        # 图工具里 Paper 节点只暴露 id/abstract；语义实体没有 abstract。
        if "abstract" in entity:
            _append_title(titles, entity.get("id"))


def _append_title(titles: list[str], title: Any) -> None:
    text = str(title or "").strip()
    if text:
        titles.append(text)


def _dedupe_titles(titles: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for title in titles:
        normalized = normalize_title(title)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(title)
    return deduped


def main() -> int:
    args = parse_args()
    records = evaluate_retrieval_hits(
        run_manifest=args.run_manifest,
        reference_availability=args.reference_availability,
    )
    write_outputs(records, args.output_json, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
