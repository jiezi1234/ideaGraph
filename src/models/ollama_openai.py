from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenAIToolCall:
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str


@dataclass(frozen=True)
class OpenAIChatResult:
    content: str
    tool_calls: list[OpenAIToolCall]
    finish_reason: str
    reasoning: str = ""
    usage: dict[str, int] | None = None


@dataclass(frozen=True)
class OpenAIStructuredResult:
    parsed: BaseModel
    content: str
    reasoning: str = ""
    usage: dict[str, int] | None = None


class OllamaOpenAIChatModel:
    def __init__(
        self,
        *,
        client,
        model_name: str,
        tracing_enabled: bool,
        temperature: float = 1.0,
        think_mode: str | bool | None = None,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._tracing_enabled = tracing_enabled
        self._temperature = temperature
        self._think_mode = think_mode

    @classmethod
    def from_settings(cls, settings):
        tracing_enabled = (
            bool(getattr(settings, "langfuse_tracing_enabled", True))
            and bool((getattr(settings, "langfuse_public_key", "") or "").strip())
            and bool((getattr(settings, "langfuse_secret_key", "") or "").strip())
        )
        client_class = _load_openai_client_class(tracing_enabled=tracing_enabled)
        base_url = f"{getattr(settings, 'ollama_base_url').rstrip('/')}/v1"
        api_key = "ollama"
        timeout = getattr(settings, "ollama_timeout", None)
        client = client_class(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        return cls(
            client=client,
            model_name=getattr(settings, "ollama_model"),
            tracing_enabled=tracing_enabled,
            temperature=1.0,
            think_mode=getattr(settings, "ollama_think_mode", None),
        )

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> OpenAIChatResult:
        return self._complete(messages=messages, name=name, metadata=metadata)

    def chat_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> OpenAIChatResult:
        return self._complete(
            messages=messages,
            tools=tools,
            tool_choice="auto",
            name=name,
            metadata=metadata,
        )

    def structured_json(
        self,
        *,
        messages: list[dict[str, Any]],
        schema: type[BaseModel],
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> OpenAIStructuredResult:
        # 写代码时要加必要的中文注释：内置 Pydantic 校验与 JSON 解析错误的自动重试机制。
        # 当模型输出不合规时，将报错反馈给模型进行自我修正，最大尝试 3 次，大幅提升 Ollama 兼容性。
        current_messages = list(messages)
        max_attempts = 3
        last_error = None
        for attempt in range(max_attempts):
            try:
                result = self._complete(
                    messages=current_messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.__name__,
                            "schema": schema.model_json_schema(),
                        },
                    },
                    name=name,
                    metadata=metadata,
                )
                clean_content = _extract_json_from_content(result.content)
                parsed_payload = json.loads(clean_content)
                parsed = parsed_payload if isinstance(parsed_payload, schema) else schema(**parsed_payload)
                return OpenAIStructuredResult(
                    parsed=parsed,
                    content=clean_content,
                    reasoning=result.reasoning,
                    usage=result.usage,
                )
            except Exception as error:
                last_error = error
                logger.warning(
                    "structured_json:validation_failed name=%s attempt=%d/%d error=%s",
                    name,
                    attempt + 1,
                    max_attempts,
                    str(error),
                )
                if attempt < max_attempts - 1:
                    assistant_content = ""
                    if "result" in locals() and hasattr(result, "content"):
                        assistant_content = result.content
                    current_messages.append({"role": "assistant", "content": assistant_content})
                    current_messages.append({
                        "role": "user",
                        "content": (
                            f"Your previous response failed validation with error: {error}.\n"
                            f"Please output EXACTLY according to the expected JSON schema. JSON Schema definition:\n"
                            f"{json.dumps(schema.model_json_schema(), ensure_ascii=False)}"
                        )
                    })
        raise last_error

    def _complete(
        self,
        *,
        messages: list[dict[str, Any]],
        name: str,
        metadata: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> OpenAIChatResult:
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "temperature": self._temperature,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if response_format is not None:
            kwargs["response_format"] = response_format
        kwargs.update(_reasoning_kwargs_for_think_mode(self._think_mode))
        if self._tracing_enabled:
            kwargs["name"] = name
            kwargs["metadata"] = metadata or {}

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        return OpenAIChatResult(
            content=getattr(message, "content", None) or "",
            tool_calls=_parse_tool_calls(getattr(message, "tool_calls", None)),
            finish_reason=getattr(choice, "finish_reason", "") or "",
            reasoning=_get_message_field(message, "reasoning") or "",
            usage=_parse_usage(getattr(response, "usage", None)),
        )


def _load_openai_client_class(*, tracing_enabled: bool):
    if tracing_enabled:
        from langfuse.openai import OpenAI
    else:
        from openai import OpenAI
    return OpenAI


def _parse_tool_calls(raw_tool_calls) -> list[OpenAIToolCall]:
    parsed = []
    for raw_call in raw_tool_calls or []:
        function = getattr(raw_call, "function", None)
        if function is None and isinstance(raw_call, dict):
            function = raw_call.get("function", {})
        arguments = _get_message_field(function, "arguments") or "{}"
        parsed.append(
            OpenAIToolCall(
                tool_name=_get_message_field(function, "name") or "",
                arguments=json.loads(arguments),
                tool_call_id=_get_message_field(raw_call, "id") or "",
            )
        )
    return parsed


def _parse_usage(usage) -> dict[str, int]:
    if usage is None:
        return {}
    prompt_tokens = _get_message_field(usage, "prompt_tokens") or 0
    completion_tokens = _get_message_field(usage, "completion_tokens") or 0
    total_tokens = _get_message_field(usage, "total_tokens") or int(prompt_tokens) + int(completion_tokens)
    result = {
        "input": int(prompt_tokens or 0),
        "output": int(completion_tokens or 0),
        "total": int(total_tokens or 0),
    }
    return result if any(result.values()) else {}


def _reasoning_kwargs_for_think_mode(think_mode: str | bool | None) -> dict[str, Any]:
    if think_mode in (None, ""):
        return {}
    if isinstance(think_mode, bool):
        return {"reasoning_effort": "medium" if think_mode else "none"}
    normalized = str(think_mode).strip()
    normalized_lower = normalized.lower()
    if normalized_lower in {"true", "yes", "on"}:
        return {"reasoning_effort": "medium"}
    if normalized_lower in {"false", "no", "off"}:
        return {"reasoning_effort": "none"}
    if normalized_lower in {"high", "medium", "low", "none"}:
        return {"reasoning_effort": normalized_lower}
    return {}


def _get_message_field(value, field: str):
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _extract_json_from_content(content: str) -> str:
    """从模型原始输出中提取 JSON 字符串。

    qwen3.5 等思考模式模型可能在 JSON 前输出 <think>...</think> 块，
    或把 JSON 用 Markdown 代码围栏（```json ... ```）包裹。
    本函数依次尝试剥离这两种包装，返回可直接传给 json.loads() 的纯 JSON 字符串。
    """
    text = (content or "").strip()

    # 1. 剥离 <think>...</think> 思考块（可能出现在开头）
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. 提取 Markdown 代码块（```json ... ``` 或 ``` ... ```）
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        return code_block.group(1).strip()

    # 3. 模型偶尔会输出“说明文字 + JSON + 结束语”，用 JSONDecoder 找第一个完整载荷。
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if index > 0:
            logger.debug("_extract_json_from_content: stripped %d leading chars before JSON", index)
        return text[index : index + end].strip()

    return text
