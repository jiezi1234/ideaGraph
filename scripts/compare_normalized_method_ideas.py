from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.method_idea_format import NormalizedMethodIdea
from scripts.method_idea_format import load_normalized_method_idea
from src.config.settings import Settings
from src.models.openai_chat import OpenAIChatModel
from src.models.structured_cache import CachedStructuredJsonModel
from src.prompts import COMPARE_RESEARCH_IDEA_PROMPT
from src.schemas.agent import IdeaComparison


LOGGER = logging.getLogger(__name__)

CRITERIA = [
    "novelty",
    "clarity",
    "feasibility",
    "relevance",
    "expected_effectiveness",
    "evidence_grounding",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blindly compare normalized method_idea_v1 outputs with reversed A/B order."
    )
    parser.add_argument(
        "--idea-json",
        required=True,
        action="append",
        type=Path,
        help="Normalized method_idea_v1 JSON. Pass at least two; three inputs produce all pairwise comparisons.",
    )
    parser.add_argument("--question", default="", help="Shared research question. Defaults to the first normalized file's question.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def run_pairwise_blind_comparison(
    *,
    idea_paths: list[Path],
    question: str,
    model,
) -> dict[str, Any]:
    ideas = [load_normalized_method_idea(path) for path in idea_paths]
    if len(ideas) < 2:
        raise ValueError("At least two normalized idea JSON files are required")
    effective_question = question or next((idea.question for idea in ideas if idea.question), "")

    pairs = []
    for method_a, method_b in itertools.combinations(ideas, 2):
        pairs.append(_compare_pair(method_a=method_a, method_b=method_b, question=effective_question, model=model))

    report = {
        "question": effective_question,
        "idea_files": [str(path) for path in idea_paths],
        "methods": [
            {
                "method_name": idea.method_name,
                "source_path": idea.source_path,
                "source_format": idea.source_format,
            }
            for idea in ideas
        ],
        "pairs": pairs,
        "global_summary": _global_summary(pairs),
    }
    return report


def write_report(report: dict[str, Any], *, output_json: Path, output_md: Path | None = None) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(_format_markdown_report(report), encoding="utf-8")
    LOGGER.info("blind_method_comparison:wrote output_json=%s output_md=%s", output_json, output_md)


def _compare_pair(
    *,
    method_a: NormalizedMethodIdea,
    method_b: NormalizedMethodIdea,
    question: str,
    model,
) -> dict[str, Any]:
    pair_slug = f"{_slug(method_a.method_name)}_vs_{_slug(method_b.method_name)}"
    blind_runs = [
        _judge_order(
            pair_slug=pair_slug,
            order="forward",
            idea_a=method_a,
            idea_b=method_b,
            question=question,
            model=model,
        ),
        _judge_order(
            pair_slug=pair_slug,
            order="reversed",
            idea_a=method_b,
            idea_b=method_a,
            question=question,
            model=model,
        ),
    ]
    return {
        "method_a": method_a.method_name,
        "method_b": method_b.method_name,
        "blind_runs": blind_runs,
        "aggregate": _aggregate_pair(method_a.method_name, method_b.method_name, blind_runs),
    }


def _judge_order(
    *,
    pair_slug: str,
    order: str,
    idea_a: NormalizedMethodIdea,
    idea_b: NormalizedMethodIdea,
    question: str,
    model,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": _blind_comparison_prompt(),
        },
        {
            "role": "user",
            "content": (
                "<research_question>\n"
                f"{question}\n"
                "</research_question>\n\n"
                "<idea_a>\n"
                f"{idea_a.judge_text}\n"
                "</idea_a>\n\n"
                "<idea_b>\n"
                f"{idea_b.judge_text}\n"
                "</idea_b>"
            ),
        },
    ]
    result = model.structured_json(
        messages=messages,
        schema=IdeaComparison,
        name=f"blind_method_comparison_{pair_slug}_{order}",
        metadata={"stage": "blind_method_comparison", "order": order, "pair": pair_slug},
    )
    comparison = result.parsed if isinstance(result.parsed, IdeaComparison) else IdeaComparison.model_validate(result.parsed)
    return {
        "order": order,
        "idea_a_method": idea_a.method_name,
        "idea_b_method": idea_b.method_name,
        "overall_winner_method": _winner_method(comparison.overall_first, idea_a, idea_b),
        "criterion_winners": {
            criterion: _winner_method(getattr(comparison, criterion).first, idea_a, idea_b)
            for criterion in CRITERIA
        },
        "comparison": comparison.model_dump(),
    }


def _aggregate_pair(method_a: str, method_b: str, blind_runs: list[dict[str, Any]]) -> dict[str, Any]:
    overall_counts = Counter(run["overall_winner_method"] for run in blind_runs)
    criterion_counts = {
        criterion: Counter(run["criterion_winners"][criterion] for run in blind_runs)
        for criterion in CRITERIA
    }
    winners = [run["overall_winner_method"] for run in blind_runs]
    overall_winner = _majority_winner(overall_counts, [method_a, method_b])
    order_sensitivity = "consistent" if len(set(winners)) == 1 else "order_sensitive"
    return {
        "overall_winner": overall_winner,
        "order_sensitivity": order_sensitivity,
        "overall_win_counts": _ordered_counts(overall_counts, method_a, method_b),
        "criterion_win_counts": {
            criterion: _ordered_counts(counts, method_a, method_b)
            for criterion, counts in criterion_counts.items()
        },
        "summary": _pair_summary(method_a, method_b, overall_winner, order_sensitivity),
    }


def _global_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    pair_winners = Counter(pair["aggregate"]["overall_winner"] for pair in pairs)
    return {
        "pair_count": len(pairs),
        "pair_winners": dict(pair_winners),
        "order_sensitive_pair_count": sum(
            1 for pair in pairs if pair["aggregate"]["order_sensitivity"] == "order_sensitive"
        ),
    }


def _winner_method(label: str, idea_a: NormalizedMethodIdea, idea_b: NormalizedMethodIdea) -> str:
    if label == "idea_a":
        return idea_a.method_name
    if label == "idea_b":
        return idea_b.method_name
    return "tie"


def _majority_winner(counts: Counter, methods: list[str]) -> str:
    method_counts = [(method, counts.get(method, 0)) for method in methods]
    best_count = max(count for _, count in method_counts)
    best_methods = [method for method, count in method_counts if count == best_count]
    if len(best_methods) == 1 and best_count > counts.get("tie", 0):
        return best_methods[0]
    return "tie"


def _ordered_counts(counts: Counter, method_a: str, method_b: str) -> dict[str, int]:
    return {
        method_a: counts.get(method_a, 0),
        method_b: counts.get(method_b, 0),
        "tie": counts.get("tie", 0),
    }


def _pair_summary(method_a: str, method_b: str, overall_winner: str, order_sensitivity: str) -> str:
    if overall_winner == "tie":
        result = f"{method_a} 与 {method_b} 在双顺序盲评中没有稳定胜者。"
    else:
        result = f"双顺序盲评整体倾向 {overall_winner}。"
    if order_sensitivity == "order_sensitive":
        result += " 两次顺序反转后的总体结论不一致，需要人工复核。"
    else:
        result += " 两次顺序反转后的总体结论一致。"
    return result


def _blind_comparison_prompt() -> str:
    return (
        COMPARE_RESEARCH_IDEA_PROMPT
        + "\n\n<blind_review_rules>\n"
        "- The method names and system identities are intentionally hidden.\n"
        "- Judge only the visible research idea content and the shared research question.\n"
        "- Do not infer which method produced either idea.\n"
        "</blind_review_rules>"
    )


def _format_markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Blind Method Idea Comparison",
        "",
        f"Question: {report.get('question', '')}",
        "",
        "## Global Summary",
        "",
        f"- Pair count: {report.get('global_summary', {}).get('pair_count', 0)}",
        f"- Order-sensitive pairs: {report.get('global_summary', {}).get('order_sensitive_pair_count', 0)}",
        "",
    ]
    for pair in report.get("pairs", []):
        aggregate = pair["aggregate"]
        lines.extend(
            [
                f"## {pair['method_a']} vs {pair['method_b']}",
                "",
                f"- Overall winner: {aggregate['overall_winner']}",
                f"- Order sensitivity: {aggregate['order_sensitivity']}",
                f"- Summary: {aggregate['summary']}",
                "",
                "### Blind Runs",
                "",
            ]
        )
        for run in pair.get("blind_runs", []):
            comparison = run["comparison"]
            lines.extend(
                [
                    f"- Order: {run['order']}",
                    f"  Overall winner: {run['overall_winner_method']}",
                    f"  Overall explanation: {comparison.get('overall_explanation', '')}",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "method"


def _build_model_from_settings(settings: Settings):
    model = OpenAIChatModel.from_settings(settings, model_name=settings.llm_model_generation)
    if not settings.llm_cache_enabled:
        return model
    return CachedStructuredJsonModel(
        model,
        cache_dir=settings.llm_cache_dir,
        cache_identity={
            "model": settings.llm_model_generation,
            "base_url": settings.llm_base_url,
            "think_mode": settings.llm_think_mode,
            "temperature": 1.0,
        },
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    settings = Settings()
    model = _build_model_from_settings(settings)
    report = run_pairwise_blind_comparison(
        idea_paths=args.idea_json,
        question=args.question,
        model=model,
    )
    write_report(report, output_json=args.output_json, output_md=args.output_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
