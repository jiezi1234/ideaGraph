from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_RE = re.compile(r"^Archived Result:\s*(?P<path>.+?)\s*$", re.MULTILINE)
RETRIEVAL_MODE_CHOICES = ("graph", "paper_vector", "chunk_vector")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chat-idea over AI Idea Bench problem drafts and write an I2I run manifest."
    )
    parser.add_argument("--problem-drafts", required=True, type=Path)
    parser.add_argument("--archive-output-dir", required=True, type=Path)
    parser.add_argument("--runs-output", required=True, type=Path)
    parser.add_argument("--failures-output", type=Path, default=None)
    parser.add_argument("--neo4j-database", default=None)
    parser.add_argument("--variant", default="ideagraph_ai_bench")
    parser.add_argument("--retrieval-mode", choices=RETRIEVAL_MODE_CHOICES, default="graph")
    parser.add_argument(
        "--query-field",
        choices=["focused_problem", "broad_problem", "research_problem"],
        default="focused_problem",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def load_problem_drafts(path: str | Path, query_field: str = "focused_problem") -> list[dict[str, Any]]:
    drafts = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        problem = payload.get("sanitized_problem") or {}
        # v1 drafts keep generated questions under sanitized_problem; manually
        # reviewed v2 drafts keep the benchmark-facing question at top level.
        query = str(problem.get(query_field) or payload.get(query_field) or "").strip()
        if not query:
            raise ValueError(
                f"problem draft line {line_number} missing sanitized_problem.{query_field} or {query_field}"
            )
        drafts.append(
            {
                "dataset_index": int(payload["dataset_index"]),
                "target_paper": str(payload.get("target_paper") or "").strip(),
                "query": query,
            }
        )
    return drafts


def extract_archive_path(stdout: str) -> str:
    match = ARCHIVE_RE.search(stdout or "")
    if not match:
        raise ValueError("chat-idea output did not contain an Archived Result line")
    return match.group("path").strip()


def load_existing_runs(path: str | Path) -> list[dict[str, Any]]:
    run_path = Path(path)
    if not run_path.exists():
        return []
    runs = []
    for line in run_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            runs.append(json.loads(line))
    return runs


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_chat_command(*, query: str, archive_output_dir: str | Path, retrieval_mode: str = "graph") -> list[str]:
    return [
        "python3",
        "main.py",
        "chat-idea",
        "--query",
        query,
        "--archive-output-dir",
        str(archive_output_dir),
        "--generation-profile",
        "ai_idea_bench",
        "--retrieval-mode",
        retrieval_mode,
    ]


def run_queries(
    *,
    problem_drafts: str | Path,
    archive_output_dir: str | Path,
    runs_output: str | Path,
    failures_output: str | Path | None,
    neo4j_database: str | None,
    variant: str,
    force: bool,
    limit: int | None,
    retrieval_mode: str = "graph",
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    query_field: str = "focused_problem",
    dry_run: bool = False,
    repo_root: str | Path = REPO_ROOT,
) -> list[dict[str, Any]]:
    archive_output_dir = Path(archive_output_dir)
    archive_output_dir.mkdir(parents=True, exist_ok=True)
    runs_output = Path(runs_output)
    failures_output = Path(failures_output) if failures_output is not None else runs_output.with_name("failures.jsonl")
    repo_root = Path(repo_root)

    runs = load_existing_runs(runs_output)
    existing_indexes = {int(run["dataset_index"]) for run in runs}
    executed = 0

    for draft in load_problem_drafts(problem_drafts, query_field=query_field):
        dataset_index = int(draft["dataset_index"])
        if not force and dataset_index in existing_indexes:
            continue
        if limit is not None and executed >= limit:
            break

        command = build_chat_command(
            query=draft["query"],
            archive_output_dir=archive_output_dir,
            retrieval_mode=retrieval_mode,
        )
        if dry_run:
            print(" ".join(command))
            executed += 1
            continue

        env = os.environ.copy()
        if neo4j_database:
            env["NEO4J_DATABASE"] = neo4j_database
        result = runner(command, cwd=str(repo_root), env=env, text=True, capture_output=True)
        if result.returncode != 0:
            append_jsonl(failures_output, _failure_payload(draft, result, command, "chat_idea_failed"))
            executed += 1
            continue

        try:
            archive_path = extract_archive_path(result.stdout)
        except ValueError:
            append_jsonl(failures_output, _failure_payload(draft, result, command, "archive_path_not_found"))
            executed += 1
            continue

        run_record = {
            "dataset_index": dataset_index,
            "target_paper": draft["target_paper"],
            "query": draft["query"],
            "archive_path": archive_path,
            "variant": variant,
            "generation_profile": "ai_idea_bench",
            "retrieval_mode": retrieval_mode,
        }
        # force 重跑时用同一 dataset_index 的最新结果覆盖旧 manifest 记录。
        runs = [run for run in runs if int(run["dataset_index"]) != dataset_index]
        runs.append(run_record)
        write_jsonl(runs_output, runs)
        existing_indexes.add(dataset_index)
        executed += 1

    if not runs_output.exists():
        write_jsonl(runs_output, runs)
    return runs


def _failure_payload(draft: dict[str, Any], result: subprocess.CompletedProcess, command: list[str], reason: str) -> dict:
    return {
        "dataset_index": draft["dataset_index"],
        "target_paper": draft["target_paper"],
        "query": draft["query"],
        "reason": reason,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
    }


def main() -> int:
    args = parse_args()
    run_queries(
        problem_drafts=args.problem_drafts,
        archive_output_dir=args.archive_output_dir,
        runs_output=args.runs_output,
        failures_output=args.failures_output,
        neo4j_database=args.neo4j_database,
        variant=args.variant,
        retrieval_mode=args.retrieval_mode,
        force=args.force,
        limit=args.limit,
        query_field=args.query_field,
        dry_run=args.dry_run,
        repo_root=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
