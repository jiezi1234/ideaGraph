from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.method_idea_format import NormalizedMethodIdea
from scripts.method_idea_format import normalize_method_idea_file
from scripts.method_idea_format import write_normalized_method_idea


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize one method's idea-generation output into method_idea_v1 JSON."
    )
    parser.add_argument("--input", required=True, type=Path, help="Raw output file from IdeaGraph, ResearchAgent, or another adapter.")
    parser.add_argument("--output-json", required=True, type=Path, help="Normalized method_idea_v1 JSON path.")
    parser.add_argument("--method-name", required=True, help="Human-readable method name kept only in metadata.")
    parser.add_argument(
        "--source-format",
        default="auto",
        choices=["auto", "ideagraph_archive", "researchagent_jsonl", "ai_researcher_local_idea_agent"],
        help="Raw source format. Use auto for the current supported outputs.",
    )
    parser.add_argument("--question", default="", help="Shared research question for traceability.")
    parser.add_argument("--record-index", type=int, default=0, help="JSONL record index for multi-record files.")
    return parser.parse_args()


def normalize_to_file(
    *,
    source_path: Path,
    output_path: Path,
    method_name: str,
    source_format: str = "auto",
    question: str = "",
    record_index: int = 0,
) -> NormalizedMethodIdea:
    normalized = normalize_method_idea_file(
        source_path=source_path,
        method_name=method_name,
        source_format=source_format,
        question=question,
        record_index=record_index,
    )
    write_normalized_method_idea(normalized, output_path)
    LOGGER.info(
        "method_idea_normalized method=%s source_format=%s output=%s",
        method_name,
        normalized.source_format,
        output_path,
    )
    return normalized


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    normalize_to_file(
        source_path=args.input,
        output_path=args.output_json,
        method_name=args.method_name,
        source_format=args.source_format,
        question=args.question,
        record_index=args.record_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
