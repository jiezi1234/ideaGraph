from __future__ import annotations

import logging

from pydantic import BaseModel

from src.agent.state import AgentStateUpdate, QueryUnderstandingState
from src.prompts import UNDERSTAND_QUERY_PROMPT


logger = logging.getLogger(__name__)

DEFAULT_QUERY_UNDERSTANDING_SYSTEM_PROMPT = UNDERSTAND_QUERY_PROMPT


class QueryUnderstandingOutput(BaseModel):
    keywords: list[str]
    retrieval_query: str


def query_understanding(
    state: QueryUnderstandingState,
    model=None,
    prompt_template: str = DEFAULT_QUERY_UNDERSTANDING_SYSTEM_PROMPT,
) -> AgentStateUpdate:
    fallback_query = state["user_query"].strip()
    if model is None:
        return {"retrieval_query": fallback_query}

    messages = [
        {
            "role": "system",
            "content": prompt_template,
        },
        {
            "role": "user",
            "content": f"User query: {fallback_query}",
        },
    ]
    try:
        structured = model.structured_json(
            messages=messages,
            schema=QueryUnderstandingOutput,
            name="query_understanding",
            metadata={"stage": "query_understanding"},
        )
        raw = structured.parsed.model_dump() if isinstance(structured.parsed, QueryUnderstandingOutput) else structured.parsed
        keywords = _normalize_keywords(raw.get("keywords"))
        retrieval_query = _normalize_retrieval_query(raw.get("retrieval_query"), keywords, fallback_query)
        return {
            "retrieval_query": retrieval_query,
            "query_keywords": keywords,
        }
    except Exception as exc:
        logger.warning(
            "query_understanding:fallback error=%s message=%s",
            type(exc).__name__,
            exc,
        )
        return {
            "retrieval_query": fallback_query,
            "query_keywords": [],
        }


def _normalize_keywords(value) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("keywords must be a list")
    keywords = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        keyword = " ".join(item.strip().split())
        if not keyword:
            continue
        key = keyword.lower()
        if key in seen:
            continue
        seen.add(key)
        keywords.append(keyword)
    if not keywords:
        raise ValueError("keywords must not be empty")
    return keywords


def _normalize_retrieval_query(value, keywords: list[str], fallback_query: str) -> str:
    if isinstance(value, str):
        retrieval_query = " ".join(value.strip().split())
        if retrieval_query:
            return retrieval_query
    if keywords:
        return " ".join(keywords)
    return fallback_query
