from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.models.openai_chat import OpenAIStructuredResult


class CachedStructuredJsonModel:
    def __init__(
        self,
        model,
        *,
        cache_dir: str | Path,
        cache_identity: dict[str, Any],
    ) -> None:
        self._model = model
        self._cache_dir = Path(cache_dir)
        self._cache_identity = cache_identity

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    def structured_json(self, *, messages: list[dict[str, Any]], schema, name: str, metadata: dict | None = None):
        cache_path = self._cache_path(messages=messages, schema=schema, name=name)
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            parsed = schema.model_validate(payload["parsed"])
            return OpenAIStructuredResult(
                parsed=parsed,
                content=payload.get("content") or parsed.model_dump_json(),
                reasoning=payload.get("reasoning", ""),
                usage=payload.get("usage") or {},
            )

        result = self._model.structured_json(messages=messages, schema=schema, name=name, metadata=metadata)
        parsed_payload = _to_jsonable(result.parsed)
        content = result.content or json.dumps(parsed_payload, ensure_ascii=False)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "parsed": parsed_payload,
                    "content": content,
                    "reasoning": result.reasoning,
                    "usage": result.usage or {},
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return result

    def chat(self, **kwargs):
        return self._model.chat(**kwargs)

    def chat_with_tools(self, **kwargs):
        return self._model.chat_with_tools(**kwargs)

    def _cache_path(self, *, messages: list[dict[str, Any]], schema, name: str) -> Path:
        # 缓存键必须覆盖会影响结构化输出的模型配置、schema 和完整消息；metadata 只用于 tracing，不参与缓存。
        serialized = json.dumps(
            {
                "api": "structured_json",
                "cache_identity": self._cache_identity,
                "name": name,
                "schema": schema.model_json_schema(),
                "messages": messages,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return self._cache_dir / f"{digest}.json"


def _to_jsonable(value):
    if hasattr(value, "model_dump"):
        return _to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value
