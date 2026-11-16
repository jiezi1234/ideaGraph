import json
import subprocess

from scripts.run_ai_idea_bench_queries import extract_archive_path
from scripts.run_ai_idea_bench_queries import load_problem_drafts
from scripts.run_ai_idea_bench_queries import run_queries


def test_load_problem_drafts_uses_focused_problem_by_default(tmp_path):
    path = tmp_path / "problem_drafts.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset_index": 1649,
                "target_paper": "Target A",
                "sanitized_problem": {
                    "broad_problem": "Broad question?",
                    "focused_problem": "Focused question?",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    drafts = load_problem_drafts(path)

    assert drafts == [
        {
            "dataset_index": 1649,
            "target_paper": "Target A",
            "query": "Focused question?",
        }
    ]


def test_load_problem_drafts_can_use_top_level_research_problem(tmp_path):
    path = tmp_path / "problem_drafts_v2.jsonl"
    path.write_text(
        json.dumps(
            {
                "dataset_index": 1649,
                "target_paper": "Target A",
                "research_problem": "Research question?",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    drafts = load_problem_drafts(path, query_field="research_problem")

    assert drafts == [
        {
            "dataset_index": 1649,
            "target_paper": "Target A",
            "query": "Research question?",
        }
    ]


def test_extract_archive_path_from_chat_stdout():
    stdout = "Title: Idea\nArchived Result: exports/runs/20260525-run.json\n------------------------------\n"

    assert extract_archive_path(stdout) == "exports/runs/20260525-run.json"


def test_run_queries_skips_existing_runs_and_writes_manifest(tmp_path):
    problem_drafts = tmp_path / "problem_drafts.jsonl"
    problem_drafts.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "dataset_index": 1649,
                        "target_paper": "Target A",
                        "sanitized_problem": {"focused_problem": "Question A?"},
                    }
                ),
                json.dumps(
                    {
                        "dataset_index": 6378,
                        "target_paper": "Target B",
                        "sanitized_problem": {"focused_problem": "Question B?"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runs_output = tmp_path / "runs.jsonl"
    runs_output.write_text(
        json.dumps(
            {
                "dataset_index": 1649,
                "target_paper": "Target A",
                "query": "Question A?",
                "archive_path": "existing.json",
                "variant": "ideagraph_ai_bench",
                "generation_profile": "ai_idea_bench",
                "retrieval_mode": "graph",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    archive_dir = tmp_path / "chat-runs"
    calls = []

    def fake_runner(cmd, *, cwd, env, text, capture_output):
        calls.append({"cmd": cmd, "cwd": cwd, "env": env, "text": text, "capture_output": capture_output})
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=f"Archived Result: {archive_dir / 'new-run.json'}\n",
            stderr="",
        )

    run_queries(
        problem_drafts=problem_drafts,
        archive_output_dir=archive_dir,
        runs_output=runs_output,
        failures_output=tmp_path / "failures.jsonl",
        neo4j_database="aiidea_lidar10",
        variant="ideagraph_ai_bench",
        retrieval_mode="chunk_vector",
        force=False,
        limit=None,
        runner=fake_runner,
    )

    lines = [json.loads(line) for line in runs_output.read_text(encoding="utf-8").splitlines()]
    assert [line["dataset_index"] for line in lines] == [1649, 6378]
    assert lines[1]["archive_path"] == str(archive_dir / "new-run.json")
    assert lines[1]["generation_profile"] == "ai_idea_bench"
    assert lines[1]["retrieval_mode"] == "chunk_vector"
    assert len(calls) == 1
    assert calls[0]["cmd"][:4] == ["python3", "main.py", "chat-idea", "--query"]
    assert "Question B?" in calls[0]["cmd"]
    assert "--generation-profile" in calls[0]["cmd"]
    assert "--retrieval-mode" in calls[0]["cmd"]
    assert "chunk_vector" in calls[0]["cmd"]
    assert calls[0]["env"]["NEO4J_DATABASE"] == "aiidea_lidar10"


def test_run_queries_records_failures_without_stopping_batch(tmp_path):
    problem_drafts = tmp_path / "problem_drafts.jsonl"
    problem_drafts.write_text(
        json.dumps(
            {
                "dataset_index": 1649,
                "target_paper": "Target A",
                "sanitized_problem": {"focused_problem": "Question A?"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def failing_runner(cmd, *, cwd, env, text, capture_output):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    failures_output = tmp_path / "failures.jsonl"
    run_queries(
        problem_drafts=problem_drafts,
        archive_output_dir=tmp_path / "chat-runs",
        runs_output=tmp_path / "runs.jsonl",
        failures_output=failures_output,
        neo4j_database=None,
        variant="ideagraph_ai_bench",
        retrieval_mode="graph",
        force=False,
        limit=None,
        runner=failing_runner,
    )

    failures = [json.loads(line) for line in failures_output.read_text(encoding="utf-8").splitlines()]
    assert failures[0]["dataset_index"] == 1649
    assert failures[0]["returncode"] == 1
    assert failures[0]["stderr"] == "boom"
