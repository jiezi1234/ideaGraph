from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel

try:
    from langfuse import Langfuse
except ModuleNotFoundError:  # pragma: no cover - dependency is installed in normal runs
    Langfuse = None

logger = logging.getLogger(__name__)

_JSON_SCALAR_TYPES = (str, int, float, bool, type(None))


def build_langfuse_client(settings) -> Langfuse | None:
    public_key = getattr(settings, "langfuse_public_key", "") or ""
    secret_key = getattr(settings, "langfuse_secret_key", "") or ""
    host = getattr(settings, "langfuse_host", "") or "https://cloud.langfuse.com"
    if not public_key.strip() or not secret_key.strip():
        return None
    if Langfuse is None:
        logger.warning("langfuse:package_missing tracing disabled")
        return None
    return Langfuse(
        public_key=public_key.strip(),
        secret_key=secret_key.strip(),
        host=host.strip() or "https://cloud.langfuse.com",
        tracing_enabled=bool(getattr(settings, "langfuse_tracing_enabled", True)),
        sample_rate=float(getattr(settings, "langfuse_sample_rate", 1.0)),
        flush_at=int(getattr(settings, "langfuse_flush_at", 1)),
    )


def flush_langfuse_client(client) -> None:
    if client is None:
        return
    flush = getattr(client, "flush", None)
    if not callable(flush):
        return
    try:
        flush()
    except Exception as exc:  # pragma: no cover - defensive logging only
        logger.warning("langfuse:flush_failed error=%s message=%s", type(exc).__name__, exc)


@contextmanager
def langfuse_observation(
    client,
    *,
    name: str,
    as_type: str = "span",
    input: Any = None,
    metadata: Any = None,
    parent_observation=None,
):
    if client is None:
        yield None
        return
    observation_factory = client
    if parent_observation is not None and hasattr(parent_observation, "start_as_current_observation"):
        observation_factory = parent_observation
    with observation_factory.start_as_current_observation(
        name=name,
        as_type=as_type,
        input=serialize_for_langfuse(input),
        metadata=serialize_for_langfuse(metadata),
    ) as observation:
        yield observation


def serialize_for_langfuse(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return serialize_for_langfuse(value.model_dump())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): serialize_for_langfuse(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [serialize_for_langfuse(item) for item in value]
    if isinstance(value, list):
        return [serialize_for_langfuse(item) for item in value]
    if isinstance(value, set):
        return [serialize_for_langfuse(item) for item in sorted(value, key=str)]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, _JSON_SCALAR_TYPES):
        return value
    return str(value)


def truncate_text(value: Any, limit: int = 240) -> Any:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def summarize_candidate_papers(candidate_papers, limit: int = 5) -> list[dict[str, Any]]:
    summary = []
    for paper in (candidate_papers or [])[:limit]:
        summary.append(
            {
                "paper_id": getattr(paper, "paper_id", None),
                "title": getattr(paper, "title", None),
                "score": getattr(paper, "score", None),
                "cited_by_count": getattr(paper, "cited_by_count", None),
            }
        )
    return summary


def summarize_selected_paper(selected_paper) -> dict[str, Any] | None:
    if selected_paper is None:
        return None
    return {
        "paper_id": getattr(selected_paper, "paper_id", None),
        "title": getattr(selected_paper, "title", None),
        "score": getattr(selected_paper, "score", None),
        "cited_by_count": getattr(selected_paper, "cited_by_count", None),
    }


def summarize_graph_context(graph_context, *, entity_limit: int = 5, relationship_limit: int = 5) -> dict[str, Any] | None:
    if not graph_context:
        return None
    entities = []
    for entity in (graph_context.get("entities") or [])[:entity_limit]:
        entities.append(
            {
                "id": entity.get("id"),
                "entity_type": entity.get("entity_type") or entity.get("node_type"),
                "description": truncate_text(entity.get("description")),
            }
        )
    relationships = []
    for relationship in (graph_context.get("relationships") or [])[:relationship_limit]:
        relationships.append(
            {
                "source_id": relationship.get("source_id"),
                "target_id": relationship.get("target_id"),
                "relationship_type": relationship.get("relationship_type"),
                "description": truncate_text(relationship.get("description")),
            }
        )
    return {
        "page": graph_context.get("page"),
        "page_size": graph_context.get("page_size"),
        "total_entities": graph_context.get("total_entities"),
        "total_relationships": graph_context.get("total_relationships"),
        "entities": entities,
        "relationships": relationships,
    }


def summarize_idea_output(idea_output) -> dict[str, Any] | None:
    if idea_output is None:
        return None
    return {
        "title": getattr(idea_output, "title", None),
        "summary": truncate_text(getattr(idea_output, "summary", None), limit=320),
        "problem": truncate_text(getattr(idea_output, "problem", None), limit=240),
        "method": truncate_text(getattr(idea_output, "method", None), limit=320),
        "experiment": truncate_text(getattr(idea_output, "experiment", None), limit=240),
        "core_paper_id": getattr(idea_output, "core_paper_id", None),
        "core_paper_title": getattr(idea_output, "core_paper_title", None),
        "idea_lineage": truncate_text(getattr(idea_output, "idea_lineage", None), limit=320),
    }


def summarize_idea_evaluation(idea_evaluation) -> dict[str, Any] | None:
    if idea_evaluation is None:
        return None
    return {
        "overall_score": getattr(idea_evaluation, "overall_score", None),
        "summary": truncate_text(getattr(idea_evaluation, "summary", None), limit=320),
        "scores": {
            "novelty": getattr(getattr(idea_evaluation, "novelty", None), "score", None),
            "clarity": getattr(getattr(idea_evaluation, "clarity", None), "score", None),
            "feasibility": getattr(getattr(idea_evaluation, "feasibility", None), "score", None),
            "relevance": getattr(getattr(idea_evaluation, "relevance", None), "score", None),
            "expected_effectiveness": getattr(getattr(idea_evaluation, "expected_effectiveness", None), "score", None),
            "evidence_grounding": getattr(getattr(idea_evaluation, "evidence_grounding", None), "score", None),
        },
        "revision_instruction": truncate_text(getattr(idea_evaluation, "revision_instruction", None), limit=320),
    }


def summarize_tool_trace(tool_trace) -> list[dict[str, Any]]:
    summary = []
    for round_item in tool_trace or []:
        tool_calls = round_item.get("tool_calls") or []
        observation = round_item.get("observation") or {}
        summary.append(
            {
                "round": round_item.get("round"),
                "tool_call_count": len(tool_calls),
                "tool_names": [call.get("tool_name") for call in tool_calls],
                "node_ids": [call.get("arguments", {}).get("node_id") for call in tool_calls],
                "observation": summarize_graph_context(observation),
            }
        )
    return summary


def summarize_idea_result(result: Mapping[str, Any]) -> dict[str, Any]:
    idea_output = result.get("idea_output")
    idea_evaluation = result.get("idea_evaluation")
    selected_paper = result.get("selected_paper")
    return {
        "user_query": truncate_text(result.get("user_query"), limit=240),
        "retrieval_query": truncate_text(result.get("retrieval_query"), limit=240),
        "query_keywords": result.get("query_keywords", []),
        "selected_paper": summarize_selected_paper(selected_paper),
        "graph_context": summarize_graph_context(result.get("graph_context")),
        "tool_trace": summarize_tool_trace(result.get("tool_trace", [])),
        "idea_output": summarize_idea_output(idea_output),
        "idea_evaluation": summarize_idea_evaluation(idea_evaluation),
        "idea_thinking": truncate_text(result.get("idea_thinking"), limit=500),
    }
