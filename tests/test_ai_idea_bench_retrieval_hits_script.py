import json

from scripts.evaluate_ai_idea_bench_retrieval_hits import evaluate_retrieval_hits
from scripts.evaluate_ai_idea_bench_retrieval_hits import normalize_title


def test_normalize_title_ignores_case_and_punctuation():
    assert normalize_title("nuScenes: A Multimodal Dataset!") == normalize_title(
        "Nuscenes A Multimodal Dataset"
    )


def test_evaluate_retrieval_hits_matches_archive_papers_to_target_references(tmp_path):
    reference_availability = tmp_path / "reference_availability.json"
    reference_availability.write_text(
        json.dumps(
            [
                {
                    "dataset_index": 1649,
                    "target_paper": "Target A",
                    "top_references": [
                        {"title": "Reference One"},
                        {"title": "Reference Two"},
                        {"title": "Reference Three"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    archive = tmp_path / "run.json"
    archive.write_text(
        json.dumps(
            {
                "candidate_papers": [
                    {"title": "Reference One"},
                    {"title": "Unrelated Candidate"},
                ],
                "selected_paper": {"title": "Reference One"},
                "retrieved_chunks": [
                    {"title": "Reference Three", "paper_id": "Reference Three"},
                ],
                "tool_trace": [
                    {
                        "observation": {
                            "entities": [
                                {"id": "Reference Two", "abstract": "Paper abstract."},
                                {"id": "semantic entity", "description": "Not a paper."},
                            ]
                        }
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    run_manifest = tmp_path / "runs.jsonl"
    run_manifest.write_text(
        json.dumps({"dataset_index": 1649, "archive_path": str(archive), "variant": "ideagraph"})
        + "\n",
        encoding="utf-8",
    )

    records = evaluate_retrieval_hits(
        run_manifest=run_manifest,
        reference_availability=reference_availability,
    )

    assert records == [
        {
            "dataset_index": 1649,
            "target_paper": "Target A",
            "variant": "ideagraph",
            "archive_path": str(archive),
            "gold_reference_count": 3,
            "retrieved_paper_count": 4,
            "hit_count": 3,
            "hit_rate": 1.0,
            "retrieval_precision": 3 / 4,
            "matched_references": ["Reference One", "Reference Two", "Reference Three"],
            "missed_references": [],
            "retrieved_papers": ["Reference One", "Unrelated Candidate", "Reference Three", "Reference Two"],
        }
    ]
