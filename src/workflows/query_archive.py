from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def archive_chat_idea_result(
    result: dict,
    settings,
    user_query: str,
    retrieval_mode: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    run_id = uuid.uuid4().hex[:12]
    created_at = datetime.now(UTC).isoformat()
    output_path = Path(output_dir or settings.chat_idea_archive_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id,
        "created_at": created_at,
        "config": {
            "neo4j_database": settings.neo4j_database,
            "llm_model_generation": settings.llm_model_generation,
            "embed_model": settings.embed_model,
            "agent_retrieval_top_k": settings.agent_retrieval_top_k,
        },
    }
    record.update(_to_jsonable(result))
    record["user_query"] = record.get("user_query") or user_query
    if retrieval_mode is not None:
        record["retrieval_mode"] = retrieval_mode

    filename = f"{created_at[:19].replace(':', '').replace('-', '').replace('T', '-')}-{_slug(user_query)}-{run_id}.json"
    archive_path = output_path / filename
    archive_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return archive_path


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.strip()).strip("-").lower()
    return (slug or "query")[:48]
