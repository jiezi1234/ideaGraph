import json

from scripts.evaluate_ai_idea_bench_i2i import load_run_manifest


def test_load_run_manifest_reads_jsonl_entries(tmp_path):
    manifest = tmp_path / "runs.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"dataset_index": 1649, "archive_path": "run-a.json", "variant": "ideagraph"}),
                json.dumps({"dataset_index": 6378, "archive_path": "run-b.json"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runs = load_run_manifest(manifest)

    assert runs == [
        {"dataset_index": 1649, "archive_path": "run-a.json", "variant": "ideagraph"},
        {"dataset_index": 6378, "archive_path": "run-b.json"},
    ]
