from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import Settings
from src.models.openai_chat import OpenAIChatModel
from src.workflows.ai_idea_bench_i2i import build_target_canonical_idea
from src.workflows.ai_idea_bench_i2i import evaluate_idea_to_idea
from src.workflows.ai_idea_bench_i2i import evaluate_idea_to_idea_official
from src.workflows.ai_idea_bench_i2i import load_generated_canonical_idea


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated AI Idea Bench ideas with an idea-to-idea similarity judge."
    )
    parser.add_argument("--dataset-json", required=True, type=Path)
    parser.add_argument(
        "--run-manifest",
        required=True,
        type=Path,
        help='JSONL lines with {"dataset_index": int, "archive_path": "...", "variant": "..."}',
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--judge-mode", choices=["official_i2i", "local"], default="official_i2i")
    return parser.parse_args()


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


def load_dataset_by_index(path: str | Path) -> dict[int, dict[str, Any]]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(record["index"]): record for record in records if "index" in record}


def run_evaluation(
    *,
    dataset_json: Path,
    run_manifest: Path,
    output_json: Path,
    output_csv: Path | None,
    model,
    judge_mode: str = "official_i2i",
) -> list[dict[str, Any]]:
    records_by_index = load_dataset_by_index(dataset_json)
    outputs = []
    output_json.parent.mkdir(parents=True, exist_ok=True)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    for run in load_run_manifest(run_manifest):
        dataset_index = int(run["dataset_index"])
        record = records_by_index.get(dataset_index)
        if record is None:
            raise ValueError(f"dataset_index not found in dataset: {dataset_index}")

        generated_idea = load_generated_canonical_idea(run["archive_path"])
        target_idea = build_target_canonical_idea(record)
        if judge_mode == "local":
            evaluation = evaluate_idea_to_idea(
                generated_idea=generated_idea,
                target_idea=target_idea,
                model=model,
            )
        else:
            evaluation = evaluate_idea_to_idea_official(
                generated_idea=generated_idea,
                target_idea=target_idea,
                model=model,
            )
        payload = {
            "dataset_index": dataset_index,
            "target_paper": record.get("target_paper"),
            "variant": run.get("variant"),
            "judge_mode": judge_mode,
            "archive_path": run["archive_path"],
            "generated_idea": generated_idea.model_dump(),
            "target_idea": target_idea.model_dump(),
            "i2i_evaluation": evaluation.model_dump(),
        }
        outputs.append(payload)
        output_json.write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        LOGGER.info("i2i:evaluated dataset_index=%s archive=%s", dataset_index, run["archive_path"])

    if output_csv is not None:
        write_csv(output_csv, outputs, judge_mode=judge_mode)
    return outputs


def write_csv(path: str | Path, records: list[dict[str, Any]], judge_mode: str = "official_i2i") -> None:
    if judge_mode == "local":
        fieldnames = [
            "dataset_index",
            "target_paper",
            "variant",
            "judge_mode",
            "archive_path",
            "motivation_score",
            "experiment_score",
            "overall_score",
            "summary",
        ]
    else:
        fieldnames = [
            "dataset_index",
            "target_paper",
            "variant",
            "judge_mode",
            "archive_path",
            "motivation_rating",
            "experiment_plan_rating",
            "motivation_explanation",
            "experiment_plan_explanation",
        ]
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            evaluation = record["i2i_evaluation"]
            if judge_mode == "local":
                writer.writerow(
                    {
                        "dataset_index": record["dataset_index"],
                        "target_paper": record["target_paper"],
                        "variant": record.get("variant"),
                        "judge_mode": record.get("judge_mode"),
                        "archive_path": record["archive_path"],
                        "motivation_score": evaluation["motivation_similarity"]["score"],
                        "experiment_score": evaluation["experiment_similarity"]["score"],
                        "overall_score": evaluation["overall_score"],
                        "summary": evaluation["summary"],
                    }
                )
            else:
                writer.writerow(
                    {
                        "dataset_index": record["dataset_index"],
                        "target_paper": record["target_paper"],
                        "variant": record.get("variant"),
                        "judge_mode": record.get("judge_mode"),
                        "archive_path": record["archive_path"],
                        "motivation_rating": evaluation["motivation_similarity"]["rating"],
                        "experiment_plan_rating": evaluation["experiment_plan_similarity"]["rating"],
                        "motivation_explanation": evaluation["motivation_similarity"]["explanation"],
                        "experiment_plan_explanation": evaluation["experiment_plan_similarity"]["explanation"],
                    }
                )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    settings = Settings()
    model = OpenAIChatModel.from_settings(settings, model_name=settings.llm_model_generation)
    run_evaluation(
        dataset_json=args.dataset_json,
        run_manifest=args.run_manifest,
        output_json=args.output_json,
        output_csv=args.output_csv,
        model=model,
        judge_mode=args.judge_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
