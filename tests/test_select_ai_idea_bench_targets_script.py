import json

from src.retrieval.embedding_cache import PaperEmbeddingCache

from scripts.select_ai_idea_bench_targets import select_targets
from scripts.select_ai_idea_bench_targets import write_outputs
from scripts.select_ai_idea_bench_targets import _selection_method


def _record(index, title, topic, motivation, refs):
    return {
        "index": index,
        "target_paper": title,
        "summary": {
            "topic": topic,
            "revised_topic": topic,
            "motivation": motivation,
            "split_topic": [{"rank": 1, "keyword": topic}],
            "method": {"targeted_designs_summary": "Do not expose target method."},
        },
        "find_cite": {"top_references": refs},
    }


def _ref(title, filename):
    return {
        "title": title,
        "paper_local_path": f"./papers_data/{filename}",
    }


class FakeEmbedder:
    model_name = "fake-embedding"

    def __init__(self):
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        lower = text.lower()
        if "query: 3d scene language understanding" in lower:
            return [1.0, 0.0]
        if "scene" in lower or "3d" in lower:
            return [0.9, 0.1]
        if "distillation" in lower:
            return [0.0, 1.0]
        return [0.2, 0.8]


def test_select_targets_prefers_query_related_records_and_preserves_reference_status(tmp_path):
    pdf_dir = tmp_path / "papers_data"
    pdf_dir.mkdir()
    (pdf_dir / "gaussian splatting.pdf").write_text("pdf", encoding="utf-8")
    (pdf_dir / "language fields.pdf").write_text("pdf", encoding="utf-8")
    (pdf_dir / "distillation.pdf").write_text("pdf", encoding="utf-8")

    records = [
        _record(
            1,
            "SceneLang Target",
            "3D language scene understanding",
            "Open vocabulary 3D scene methods need faster language field modeling.",
            [_ref("Gaussian Splatting", "gaussian splatting.pdf")],
        ),
        _record(
            2,
            "ViewReason Target",
            "3D scene reasoning",
            "Vision language systems need robust 3D spatial scene grounding.",
            [_ref("Language Fields", "language fields.pdf")],
        ),
        _record(
            3,
            "Distill Target",
            "knowledge distillation",
            "Student models need better logits for teacher transfer.",
            [_ref("Distillation", "distillation.pdf")],
        ),
    ]

    selected, candidates = select_targets(
        records=records,
        dataset_root=tmp_path,
        pdf_dir=pdf_dir,
        count=2,
        query="3D scene language understanding",
        candidate_pool_size=3,
    )

    assert [item["dataset_index"] for item in selected] == [1, 2]
    assert all(item["all_top_references_exist"] for item in selected)
    assert candidates[0]["dataset_index"] == 1


def test_select_targets_can_use_ollama_embedding_backend_with_cache(tmp_path):
    pdf_dir = tmp_path / "papers_data"
    pdf_dir.mkdir()
    (pdf_dir / "language fields.pdf").write_text("pdf", encoding="utf-8")
    (pdf_dir / "distillation.pdf").write_text("pdf", encoding="utf-8")
    records = [
        _record(
            1,
            "SceneLang Target",
            "3D language scene understanding",
            "Open vocabulary scene grounding requires robust spatial language fields.",
            [_ref("Language Fields", "language fields.pdf")],
        ),
        _record(
            2,
            "Distill Target",
            "knowledge distillation",
            "Student models need better logits for teacher transfer.",
            [_ref("Distillation", "distillation.pdf")],
        ),
    ]
    cache = PaperEmbeddingCache(tmp_path / "embedding-cache")
    embedder = FakeEmbedder()

    selected, _ = select_targets(
        records=records,
        dataset_root=tmp_path,
        pdf_dir=pdf_dir,
        count=1,
        query="3D scene language understanding",
        candidate_pool_size=2,
        embedding_backend="ollama",
        embedder=embedder,
        cache=cache,
    )

    assert selected[0]["dataset_index"] == 1
    assert selected[0]["embedding_backend"] == "ollama"
    assert len(embedder.calls) == 3

    embedder_again = FakeEmbedder()
    selected_again, _ = select_targets(
        records=records,
        dataset_root=tmp_path,
        pdf_dir=pdf_dir,
        count=1,
        query="3D scene language understanding",
        candidate_pool_size=2,
        embedding_backend="ollama",
        embedder=embedder_again,
        cache=cache,
    )

    assert selected_again[0]["dataset_index"] == 1
    assert embedder_again.calls == []


def test_ollama_auto_selection_method_uses_centroid_strategy():
    assert _selection_method("ollama", has_query_or_anchor=False) == "ollama_auto_centroid_mmr"


def test_write_outputs_excludes_target_pdfs_from_reference_manifest(tmp_path):
    pdf_dir = tmp_path / "papers_data"
    pdf_dir.mkdir()
    (pdf_dir / "target paper.pdf").write_text("target", encoding="utf-8")
    (pdf_dir / "reference paper.pdf").write_text("reference", encoding="utf-8")
    record = _record(
        1,
        "Target Paper",
        "3D scene reasoning",
        "Existing scene reasoning methods miss geometric constraints.",
        [_ref("Reference Paper", "reference paper.pdf")],
    )

    selected, candidates = select_targets(
        records=[record],
        dataset_root=tmp_path,
        pdf_dir=pdf_dir,
        count=1,
        query="3D scene reasoning",
        candidate_pool_size=1,
    )

    write_outputs(
        selected=selected,
        candidates=candidates,
        output_dir=tmp_path / "out",
        summary={"selected_count": 1},
    )

    targets = json.loads((tmp_path / "out" / "selected_targets.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "out" / "reference_corpus_manifest.json").read_text(encoding="utf-8"))
    drafts = (tmp_path / "out" / "problem_drafts.jsonl").read_text(encoding="utf-8")

    assert targets[0]["target_pdf_exists"] is True
    assert manifest["pdf_count"] == 1
    assert manifest["pdfs"][0]["paper_local_path"] == "papers_data/reference paper.pdf"
    assert "papers_data/target paper.pdf" not in json.dumps(manifest, ensure_ascii=False)
    assert "Target Paper" in drafts
