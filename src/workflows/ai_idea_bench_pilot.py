from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


class AIIdeaBenchPilotWorkflow:
    def __init__(self, log=None) -> None:
        self.log = log or logger

    def run(self, dataset_path: str | Path, count: int, output_dir: str | Path) -> dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        records = _load_dataset_records(Path(dataset_path))
        pilot_records = build_pilot_records(records, count=count)
        _write_targets_json(output_path / "targets.json", pilot_records)
        _write_targets_csv(output_path / "targets.csv", pilot_records)
        _write_problem_prompts_jsonl(output_path / "problem_extraction_prompts.jsonl", pilot_records)
        _write_gold_references_json(output_path / "gold_references.json", pilot_records)
        self._info(
            "ai_idea_bench_pilot:prepared dataset_path=%s count=%s output_dir=%s",
            dataset_path,
            len(pilot_records),
            output_path,
        )
        return {"target_count": len(pilot_records), "output_dir": str(output_path)}

    def _info(self, message: str, *args) -> None:
        if callable(self.log):
            self.log(message % args if args else message)
            return
        self.log.info(message, *args)


def build_pilot_records(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    pilot_records = []
    for record in records:
        if len(pilot_records) >= count:
            break
        target_paper = str(record.get("target_paper") or "").strip()
        if not target_paper:
            continue
        summary = record.get("summary") or {}
        references = _reference_records(record)
        if not references:
            continue
        sanitized_problem = _build_sanitized_problem(target_paper=target_paper, summary=summary)
        pilot_records.append(
            {
                "dataset_index": record.get("index"),
                "target_paper": target_paper,
                "target_excluded": True,
                "sanitized_problem": sanitized_problem,
                "gold_references": references,
                "gold_reference_titles": [reference["title"] for reference in references],
                "problem_extraction_prompt": _build_problem_extraction_prompt(
                    target_paper=target_paper,
                    summary=summary,
                    sanitized_problem=sanitized_problem,
                ),
            }
        )
    return pilot_records


def _load_dataset_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"AI Idea Bench target data must be a list: {path}")
    return payload


def _reference_records(record: dict[str, Any]) -> list[dict[str, str]]:
    references = []
    for reference in (record.get("find_cite") or {}).get("top_references") or []:
        title = str(reference.get("title") or "").strip()
        if not title:
            continue
        references.append(
            {
                "title": title,
                "paper_local_path": str(reference.get("paper_local_path") or "").strip(),
            }
        )
    return references


def _build_sanitized_problem(target_paper: str, summary: dict[str, Any]) -> dict[str, Any]:
    forbidden_terms = _target_forbidden_terms(target_paper)
    challenge = _first_non_target_sentence(str(summary.get("motivation") or ""), forbidden_terms)
    source_topic = _remove_forbidden_terms(str(summary.get("revised_topic") or summary.get("topic") or ""), forbidden_terms)
    source_topic = _clean_topic_prefix(source_topic)
    source_motivation = challenge
    query_keywords = _query_keywords(summary, forbidden_terms)
    domain = source_topic or _fallback_domain(query_keywords)

    if challenge and domain:
        broad_problem = f"How can researchers improve {domain} by addressing this limitation: {challenge}"
    elif challenge:
        broad_problem = f"How can researchers address this research limitation: {challenge}"
    elif domain:
        broad_problem = f"How can researchers make progress on {domain} using evidence from prior work?"
    else:
        broad_problem = "How can researchers use prior literature to identify a well-grounded research opportunity?"

    if domain:
        focused_problem = (
            f"What prior evidence can support a new approach for {domain} while avoiding the target paper's "
            "specific mechanism, experiments, and conclusions?"
        )
    else:
        focused_problem = (
            "What prior evidence can support a new research approach while avoiding the target paper's specific "
            "mechanism, experiments, and conclusions?"
        )

    # 问题草案只能使用允许暴露的高层主题和动机，target 论文名和方法细节留给评估阶段。
    broad_problem = _remove_forbidden_terms(broad_problem, forbidden_terms)
    focused_problem = _remove_forbidden_terms(focused_problem, forbidden_terms)
    return {
        "broad_problem": _normalize_space(broad_problem),
        "focused_problem": _normalize_space(focused_problem),
        "query_keywords": query_keywords,
        "source_topic": _normalize_space(source_topic),
        "source_motivation": _normalize_space(source_motivation),
        "forbidden_terms": forbidden_terms,
        "notes": (
            "Target paper is excluded from the graph. Problems are drafts for retrieval and must not include "
            "target-specific method, experiment, result, or title wording."
        ),
    }


def _target_forbidden_terms(target_paper: str) -> list[str]:
    terms = [target_paper.strip()]
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*", target_paper)
    if tokens:
        first = tokens[0]
        if _looks_like_method_name(first):
            terms.append(first)
    deduped = []
    seen = set()
    for term in terms:
        key = term.lower()
        if term and key not in seen:
            deduped.append(term)
            seen.add(key)
    return deduped


def _looks_like_method_name(token: str) -> bool:
    if len(token) < 4:
        return False
    return bool(re.search(r"[A-Z].*[A-Z]", token)) or token.isupper()


def _first_non_target_sentence(text: str, forbidden_terms: list[str]) -> str:
    for sentence in _sentences(text):
        if not _contains_forbidden_term(sentence, forbidden_terms):
            return _normalize_space(sentence)
    stripped = _remove_forbidden_terms(text, forbidden_terms)
    return _normalize_space(_sentences(stripped)[0] if _sentences(stripped) else stripped)


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text or "") if part.strip()]


def _contains_forbidden_term(text: str, forbidden_terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in forbidden_terms if term)


def _remove_forbidden_terms(text: str, forbidden_terms: list[str]) -> str:
    cleaned = text
    for term in sorted(forbidden_terms, key=len, reverse=True):
        if not term:
            continue
        cleaned = re.sub(re.escape(term), "", cleaned, flags=re.IGNORECASE)
    return _normalize_space(cleaned)


def _clean_topic_prefix(topic: str) -> str:
    cleaned = re.sub(r"^the topic of this paper is\s+", "", topic, flags=re.IGNORECASE).strip(" .")
    return re.sub(r"^improving\s+", "", cleaned, flags=re.IGNORECASE).strip(" .")


def _query_keywords(summary: dict[str, Any], forbidden_terms: list[str]) -> list[str]:
    keywords = []
    for item in summary.get("split_topic") or []:
        if not isinstance(item, dict):
            continue
        keyword = _remove_forbidden_terms(str(item.get("keyword") or ""), forbidden_terms).strip(" .")
        if keyword and keyword.lower() not in {value.lower() for value in keywords}:
            keywords.append(keyword)
        if len(keywords) >= 8:
            break
    return keywords


def _fallback_domain(query_keywords: list[str]) -> str:
    return ", ".join(query_keywords[:3])


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _build_problem_extraction_prompt(
    target_paper: str,
    summary: dict[str, Any],
    sanitized_problem: dict[str, Any],
) -> str:
    forbidden_terms = ", ".join(sanitized_problem["forbidden_terms"])
    keywords = ", ".join(sanitized_problem["query_keywords"])
    return (
        "Refine the following AI Idea Bench target into two retrieval questions for a target-excluded literature "
        "graph pilot.\n\n"
        "Rules:\n"
        "- Do not reveal the target paper title or distinctive target method names.\n"
        "- Do not mention the target paper's concrete mechanism, experiments, datasets, results, or conclusions.\n"
        "- Keep the broad problem high-level and the focused problem suitable for evidence retrieval.\n"
        "- Use only the high-level topic, motivation, and allowed keywords.\n\n"
        f"Target paper for exclusion only: {target_paper}\n"
        f"Forbidden terms: {forbidden_terms}\n"
        f"High-level topic: {sanitized_problem['source_topic']}\n"
        f"Motivation summary: {sanitized_problem['source_motivation']}\n"
        f"Allowed keywords: {keywords}\n\n"
        "Draft broad problem:\n"
        f"{sanitized_problem['broad_problem']}\n\n"
        "Draft focused problem:\n"
        f"{sanitized_problem['focused_problem']}\n"
    )


def _write_targets_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_targets_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "dataset_index",
            "target_paper",
            "broad_problem",
            "focused_problem",
            "gold_reference_count",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "dataset_index": record["dataset_index"],
                    "target_paper": record["target_paper"],
                    "broad_problem": record["sanitized_problem"]["broad_problem"],
                    "focused_problem": record["sanitized_problem"]["focused_problem"],
                    "gold_reference_count": len(record["gold_reference_titles"]),
                }
            )


def _write_problem_prompts_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    {
                        "dataset_index": record["dataset_index"],
                        "target_paper": record["target_paper"],
                        "prompt": record["problem_extraction_prompt"],
                    },
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def _write_gold_references_json(path: Path, records: list[dict[str, Any]]) -> None:
    payload = [
        {
            "dataset_index": record["dataset_index"],
            "target_paper": record["target_paper"],
            "gold_reference_titles": record["gold_reference_titles"],
            "gold_references": record["gold_references"],
        }
        for record in records
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
