from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = "method_idea_v1"


class NormalizedIdea(BaseModel):
    title: str = ""
    summary: str = ""
    problem: str = ""
    method: str = ""
    experiment: str = ""
    original: str = ""
    core_paper_title: str = ""
    idea_lineage: str = ""
    notes: dict[str, Any] = Field(default_factory=dict)


class NormalizedMethodIdea(BaseModel):
    schema_version: Literal["method_idea_v1"] = SCHEMA_VERSION
    method_name: str
    source_path: str
    source_format: str
    question: str = ""
    idea: NormalizedIdea
    judge_text: str
    source_payload: dict[str, Any] = Field(default_factory=dict)


def normalize_method_idea_file(
    *,
    source_path: str | Path,
    method_name: str,
    source_format: str = "auto",
    question: str = "",
    record_index: int = 0,
) -> NormalizedMethodIdea:
    path = Path(source_path)
    payload = _load_payload(path, record_index=record_index)
    detected_format = detect_source_format(payload, path=path) if source_format == "auto" else source_format
    return normalize_method_idea_payload(
        payload,
        method_name=method_name,
        source_path=path,
        source_format=detected_format,
        question=question,
    )


def normalize_method_idea_payload(
    payload: dict[str, Any],
    *,
    method_name: str,
    source_path: str | Path,
    source_format: str,
    question: str = "",
) -> NormalizedMethodIdea:
    if source_format == "ideagraph_archive":
        idea = _normalize_ideagraph_archive(payload)
    elif source_format == "researchagent_jsonl":
        idea = _normalize_researchagent_record(payload)
    elif source_format == "ai_researcher_local_idea_agent":
        idea = _normalize_ai_researcher_local_idea_agent(payload)
    else:
        raise ValueError(f"Unsupported source format: {source_format}")

    judge_text = _format_judge_text(idea)
    return NormalizedMethodIdea(
        method_name=method_name,
        source_path=str(source_path),
        source_format=source_format,
        question=question,
        idea=idea,
        judge_text=judge_text,
        source_payload=payload,
    )


def detect_source_format(payload: dict[str, Any], *, path: str | Path | None = None) -> str:
    if payload.get("schema_version") == SCHEMA_VERSION:
        return "normalized_method_idea"
    if isinstance(payload.get("idea_output"), dict):
        return "ideagraph_archive"
    if isinstance(payload.get("selected_idea"), str) and isinstance(payload.get("ideas"), list):
        return "ai_researcher_local_idea_agent"
    if {"problem", "method", "experiment"}.issubset(payload.keys()):
        return "researchagent_jsonl"
    suffix = Path(path).suffix if path is not None else ""
    raise ValueError(f"Cannot detect method idea source format for suffix={suffix!r}")


def load_normalized_method_idea(path: str | Path) -> NormalizedMethodIdea:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return NormalizedMethodIdea.model_validate(payload)


def write_normalized_method_idea(idea: NormalizedMethodIdea, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(idea.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_payload(path: Path, *, record_index: int) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if record_index < 0 or record_index >= len(lines):
            raise IndexError(f"record_index {record_index} out of range for {path}")
        payload = json.loads(lines[record_index])
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _normalize_ideagraph_archive(payload: dict[str, Any]) -> NormalizedIdea:
    idea_payload = payload.get("idea_output") or payload.get("canonical_idea") or {}
    if not isinstance(idea_payload, dict):
        raise ValueError("IdeaGraph archive must contain object field 'idea_output' or 'canonical_idea'")
    return NormalizedIdea(
        title=_text(idea_payload.get("title") or idea_payload.get("Title")),
        summary=_text(idea_payload.get("summary")),
        problem=_text(idea_payload.get("problem") or idea_payload.get("Motivation") or idea_payload.get("motivation")),
        method=_text(idea_payload.get("method")),
        experiment=_text(idea_payload.get("experiment") or idea_payload.get("Experiment") or idea_payload.get("experiment_plan")),
        core_paper_title=_text(idea_payload.get("core_paper_title")),
        idea_lineage=_text(idea_payload.get("idea_lineage")),
        notes={
            "run_id": payload.get("run_id"),
            "generation_profile": payload.get("generation_profile"),
            "retrieval_mode": payload.get("retrieval_mode"),
        },
    )


def _normalize_researchagent_record(payload: dict[str, Any]) -> NormalizedIdea:
    paper = payload.get("paper") if isinstance(payload.get("paper"), dict) else {}
    return NormalizedIdea(
        title=_text(payload.get("title") or paper.get("title")),
        summary=_text(payload.get("summary")),
        problem=_text(payload.get("problem")),
        method=_text(payload.get("method")),
        experiment=_text(payload.get("experiment")),
        core_paper_title=_text(paper.get("title")),
        notes={
            "reference_count": len(payload.get("references") or []),
            "entity_count": len(payload.get("entities") or []),
        },
    )


def _normalize_ai_researcher_local_idea_agent(payload: dict[str, Any]) -> NormalizedIdea:
    selected_idea = _text(payload.get("selected_idea"))
    if not selected_idea:
        raise ValueError("AI-Researcher local idea output must contain non-empty 'selected_idea'")
    ideas = payload.get("ideas")
    if not isinstance(ideas, list):
        raise ValueError("AI-Researcher local idea output must contain list field 'ideas'")
    selected_idea_index = _selected_ai_researcher_idea_index(selected_idea)
    if selected_idea_index is None:
        raise ValueError("Cannot determine selected AI-Researcher idea index from 'selected_idea'")
    if selected_idea_index < 1 or selected_idea_index > len(ideas):
        raise ValueError(
            f"Selected AI-Researcher idea index {selected_idea_index} out of range for {len(ideas)} ideas"
        )
    original = _text(ideas[selected_idea_index - 1])
    if not original:
        raise ValueError(f"Selected AI-Researcher idea {selected_idea_index} is empty")

    papers = payload.get("papers") if isinstance(payload.get("papers"), list) else []
    first_paper = papers[0] if papers and isinstance(papers[0], dict) else {}
    return NormalizedIdea(
        title=_ai_researcher_original_title(original),
        original=original,
        core_paper_title=_text(first_paper.get("title")),
        notes={
            "category": payload.get("category"),
            "instance_id": payload.get("instance_id"),
            "candidate_idea_count": len(ideas),
            "selected_idea_index": selected_idea_index,
            "paper_count": len(papers),
        },
    )


def _format_judge_text(idea: NormalizedIdea) -> str:
    if idea.original.strip():
        return f"Original:\n{idea.original.strip()}"
    sections = [
        ("Title", idea.title),
        ("Summary", idea.summary),
        ("Problem", idea.problem),
        ("Method", idea.method),
        ("Experiment", idea.experiment),
    ]
    # 盲评只使用两个系统都能映射到的公共 idea 内容，避免把某一系统特有的 trace/feedback 当成质量证据。
    return "\n\n".join(f"{label}:\n{value.strip()}" for label, value in sections if value and value.strip())


def _markdown_sections(text: str) -> list[dict[str, Any]]:
    matches = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, flags=re.MULTILINE))
    sections = []
    for index, match in enumerate(matches):
        level = len(match.group(1))
        end = len(text)
        for next_match in matches[index + 1:]:
            if len(next_match.group(1)) <= level:
                end = next_match.start()
                break
        sections.append(
            {
                "level": level,
                "heading": match.group(2).strip(),
                "content": text[match.end():end].strip(),
            }
        )
    return sections


def _ai_researcher_title(sections: list[dict[str, Any]], selected_idea: str) -> str:
    heading = ""
    for section in sections:
        if section["level"] == 1:
            heading = str(section["heading"])
            break
    if not heading:
        first_line = selected_idea.strip().splitlines()[0] if selected_idea.strip() else ""
        heading = re.sub(r"^#\s*", "", first_line).strip()
    heading = re.sub(r"^\s*Selected\s+Innovative\s+Idea\s+Report\s*:?\s*", "", heading, flags=re.IGNORECASE)
    return heading.strip() or "AI-Researcher selected idea"


def _ai_researcher_original_title(original: str) -> str:
    sections = _markdown_sections(original)
    for section in sections:
        heading = str(section["heading"]).strip()
        if re.search(r"\bidea\s*\d+\b", heading, flags=re.IGNORECASE):
            return re.sub(r"^\s*Idea\s*\d+\s*:?\s*", "", heading, flags=re.IGNORECASE).strip()
        if int(section["level"]) == 1:
            return heading
    first_line = original.strip().splitlines()[0] if original.strip() else ""
    return re.sub(r"^#+\s*", "", first_line).strip()


def _selected_ai_researcher_idea_index(selected_idea: str) -> int | None:
    match = re.search(r"\bIdea\s*([0-9]+)\b", selected_idea, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _extract_markdown_sections(sections: list[dict[str, Any]], keywords: list[str]) -> str:
    chunks = []
    for section in sections:
        if int(section["level"]) > 2:
            continue
        normalized_heading = _normalize_heading(str(section["heading"]))
        if any(keyword in normalized_heading for keyword in keywords):
            content = str(section["content"]).strip()
            if content:
                chunks.append(f"{section['heading']}\n{content}")
    return "\n\n".join(chunks).strip()


def _normalize_heading(value: str) -> str:
    value = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s*", "", value)
    return value.lower()


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_text(item) for item in value if item is not None).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()
