import json

from src.workflows.ai_idea_bench_pilot import AIIdeaBenchPilotWorkflow
from src.workflows.ai_idea_bench_pilot import build_pilot_records


def _record(index=1, target_paper="TargetNet Fast Routing"):
    return {
        "index": index,
        "target_paper": target_paper,
        "summary": {
            "topic": "TargetNet introduces a fast routing approach for dynamic networks.",
            "revised_topic": "The topic of this paper is dynamic network routing.",
            "motivation": (
                "Current dynamic routing methods suffer from slow adaptation and brittle "
                "performance under topology changes. TargetNet addresses these limitations "
                "with a target-specific bridge module."
            ),
            "method": {
                "targeted_designs_summary": "TargetNet uses a target-specific bridge module.",
            },
            "split_topic": [
                {"rank": 1, "keyword": "dynamic routing"},
                {"rank": 2, "keyword": "topology changes"},
            ],
        },
        "find_cite": {
            "top_references": [
                {
                    "title": "Prior Routing Paper",
                    "paper_local_path": "./papers_data/prior routing paper.pdf",
                },
                {
                    "title": "Topology Robustness",
                    "paper_local_path": "./papers_data/topology robustness.pdf",
                },
            ]
        },
    }


def test_build_pilot_records_excludes_target_title_from_problem():
    records = build_pilot_records([_record()], count=1)

    assert len(records) == 1
    pilot = records[0]
    assert pilot["target_excluded"] is True
    assert pilot["target_paper"] == "TargetNet Fast Routing"
    assert pilot["gold_reference_titles"] == ["Prior Routing Paper", "Topology Robustness"]
    assert "TargetNet" not in pilot["sanitized_problem"]["broad_problem"]
    assert "TargetNet" not in pilot["sanitized_problem"]["focused_problem"]
    assert "target-specific bridge module" not in pilot["sanitized_problem"]["broad_problem"]
    assert "target-specific bridge module" not in pilot["sanitized_problem"]["source_motivation"]
    assert "target-specific bridge module" not in pilot["problem_extraction_prompt"]
    assert pilot["sanitized_problem"]["query_keywords"] == ["dynamic routing", "topology changes"]


def test_workflow_writes_targets_and_csv(tmp_path):
    dataset_path = tmp_path / "target_paper_data.json"
    dataset_path.write_text(json.dumps([_record(), _record(index=2, target_paper="Second Target")]), encoding="utf-8")

    workflow = AIIdeaBenchPilotWorkflow()
    summary = workflow.run(dataset_path=dataset_path, count=1, output_dir=tmp_path / "pilot")

    assert summary == {"target_count": 1, "output_dir": str(tmp_path / "pilot")}
    targets = json.loads((tmp_path / "pilot" / "targets.json").read_text(encoding="utf-8"))
    assert targets[0]["dataset_index"] == 1
    assert targets[0]["gold_reference_titles"] == ["Prior Routing Paper", "Topology Robustness"]

    csv_text = (tmp_path / "pilot" / "targets.csv").read_text(encoding="utf-8")
    assert "dataset_index,target_paper,broad_problem,focused_problem,gold_reference_count" in csv_text
    assert "TargetNet Fast Routing" in csv_text
