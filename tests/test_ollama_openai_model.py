from __future__ import annotations

import json

from pydantic import BaseModel


class _Object:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self, exclude_none: bool = False):
        result = dict(self.__dict__)
        if exclude_none:
            return {key: value for key, value in result.items() if value is not None}
        return result


class MiniIdea(BaseModel):
    title: str
    summary: str


def test_openai_ollama_model_parses_tool_calls_and_usage():
    from src.models.ollama_openai import OllamaOpenAIChatModel

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Object(
                choices=[
                    _Object(
                        finish_reason="tool_calls",
                        message=_Object(
                            content="",
                            reasoning="I should inspect the graph.",
                            tool_calls=[
                                _Object(
                                    id="call_1",
                                    type="function",
                                    function=_Object(
                                        name="get_node_neighborhood",
                                        arguments='{"node_id":"Graph-guided discovery","page":1}',
                                    ),
                                )
                            ],
                        ),
                    )
                ],
                usage=_Object(prompt_tokens=11, completion_tokens=7, total_tokens=18),
            )

    fake_client = _Object(chat=_Object(completions=FakeCompletions()))
    model = OllamaOpenAIChatModel(
        client=fake_client,
        model_name="qwen3.6:35b",
        tracing_enabled=False,
        temperature=0.0,
    )

    result = model.chat_with_tools(
        messages=[{"role": "user", "content": "Inspect Graph-guided discovery"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_node_neighborhood",
                    "description": "Get graph context.",
                    "parameters": {"type": "object", "properties": {"node_id": {"type": "string"}}},
                },
            }
        ],
        name="agent_generation_round_1",
        metadata={"stage": "agent_generation_round"},
    )

    assert captured["model"] == "qwen3.6:35b"
    assert captured["temperature"] == 0.0
    assert captured["tool_choice"] == "auto"
    assert "name" not in captured
    assert "metadata" not in captured
    assert result.finish_reason == "tool_calls"
    assert result.reasoning == "I should inspect the graph."
    assert result.usage == {"input": 11, "output": 7, "total": 18}
    assert result.tool_calls[0].tool_call_id == "call_1"
    assert result.tool_calls[0].tool_name == "get_node_neighborhood"
    assert result.tool_calls[0].arguments == {"node_id": "Graph-guided discovery", "page": 1}


def test_openai_ollama_model_passes_langfuse_observation_fields_when_enabled():
    from src.models.ollama_openai import OllamaOpenAIChatModel

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Object(
                choices=[_Object(finish_reason="stop", message=_Object(content="ok", tool_calls=None))],
                usage=None,
            )

    fake_client = _Object(chat=_Object(completions=FakeCompletions()))
    model = OllamaOpenAIChatModel(
        client=fake_client,
        model_name="qwen3.6:35b",
        tracing_enabled=True,
        think_mode="low",
    )

    model.chat(
        messages=[{"role": "user", "content": "Return ok"}],
        name="query_understanding",
        metadata={"stage": "query_understanding"},
    )

    assert captured["name"] == "query_understanding"
    assert captured["metadata"] == {"stage": "query_understanding"}
    assert captured["reasoning_effort"] == "low"


def test_openai_ollama_model_structured_json_uses_json_schema_and_pydantic():
    from src.models.ollama_openai import OllamaOpenAIChatModel

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Object(
                choices=[
                    _Object(
                        finish_reason="stop",
                        message=_Object(
                            content='{"title":"Graph idea","summary":"Use graph context."}',
                            reasoning="Structured response is enough.",
                            tool_calls=None,
                        ),
                    )
                ],
                usage=_Object(prompt_tokens=5, completion_tokens=6, total_tokens=11),
            )

    fake_client = _Object(chat=_Object(completions=FakeCompletions()))
    model = OllamaOpenAIChatModel(
        client=fake_client,
        model_name="qwen3.6:35b",
        tracing_enabled=False,
    )

    result = model.structured_json(
        messages=[{"role": "user", "content": "Create an idea."}],
        schema=MiniIdea,
        name="final_idea_generation",
        metadata={"stage": "final_idea_generation"},
    )

    assert captured["response_format"]["type"] == "json_schema"
    assert captured["response_format"]["json_schema"]["name"] == "MiniIdea"
    assert captured["response_format"]["json_schema"]["schema"]["properties"]["title"]["type"] == "string"
    assert result.parsed == MiniIdea(title="Graph idea", summary="Use graph context.")
    assert result.reasoning == "Structured response is enough."
    assert result.usage == {"input": 5, "output": 6, "total": 11}


def test_openai_ollama_model_structured_json_extracts_prefixed_json_payload():
    from src.models.ollama_openai import OllamaOpenAIChatModel

    class FakeCompletions:
        def create(self, **kwargs):
            del kwargs
            return _Object(
                choices=[
                    _Object(
                        finish_reason="stop",
                        message=_Object(
                            content='可以，结果如下：\n{"title":"Graph idea","summary":"Use graph context."}\n以上。',
                            reasoning="",
                            tool_calls=None,
                        ),
                    )
                ],
                usage=None,
            )

    fake_client = _Object(chat=_Object(completions=FakeCompletions()))
    model = OllamaOpenAIChatModel(
        client=fake_client,
        model_name="qwen3.6:35b",
        tracing_enabled=False,
    )

    result = model.structured_json(
        messages=[{"role": "user", "content": "Create an idea."}],
        schema=MiniIdea,
        name="final_idea_generation",
    )

    assert result.parsed == MiniIdea(title="Graph idea", summary="Use graph context.")


def test_cached_structured_json_model_reuses_structured_response(tmp_path):
    from src.models.ollama_openai import OpenAIStructuredResult
    from src.models.structured_cache import CachedStructuredJsonModel

    class FakeStructuredModel:
        def __init__(self):
            self.calls = 0
            self.metadata = None

        def structured_json(self, *, messages, schema, name: str, metadata=None):
            del messages, schema, name
            self.calls += 1
            self.metadata = metadata
            return OpenAIStructuredResult(
                parsed=MiniIdea(title="Cached idea", summary="Reuse structured output."),
                content='{"title":"Cached idea","summary":"Reuse structured output."}',
                reasoning="first call",
                usage={"input": 3, "output": 4, "total": 7},
            )

    inner = FakeStructuredModel()
    model = CachedStructuredJsonModel(
        inner,
        cache_dir=tmp_path,
        cache_identity={"model": "qwen3.6:35b", "think_mode": "medium"},
    )
    messages = [{"role": "user", "content": "Create an idea."}]

    first = model.structured_json(messages=messages, schema=MiniIdea, name="idea", metadata={"stage": "idea"})
    second = model.structured_json(messages=messages, schema=MiniIdea, name="idea", metadata={"stage": "idea"})

    assert inner.calls == 1
    assert inner.metadata == {"stage": "idea"}
    assert first.parsed == MiniIdea(title="Cached idea", summary="Reuse structured output.")
    assert second.parsed == first.parsed
    assert second.reasoning == "first call"
    assert second.usage == {"input": 3, "output": 4, "total": 7}
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    cached_payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert cached_payload["parsed"] == {"title": "Cached idea", "summary": "Reuse structured output."}


def test_bootstrap_wraps_structured_model_when_ollama_cache_enabled(tmp_path, monkeypatch):
    from src.app import bootstrap
    from src.config.settings import Settings
    from src.models.structured_cache import CachedStructuredJsonModel

    class FakeOllamaOpenAIChatModel:
        @classmethod
        def from_settings(cls, settings):
            assert settings.ollama_cache_enabled is True
            return cls()

    monkeypatch.setattr(bootstrap, "OllamaOpenAIChatModel", FakeOllamaOpenAIChatModel)
    settings = Settings(
        OPENALEX_BASE_URL="https://api.openalex.org",
        OLLAMA_BASE_URL="http://127.0.0.1:11435",
        OLLAMA_MODEL="qwen3.6:35b",
        OLLAMA_EMBED_MODEL="qwen3-embedding:4b",
        OLLAMA_TIMEOUT=300,
        OLLAMA_THINK_MODE="medium",
        OLLAMA_CACHE_ENABLED=True,
        OLLAMA_CACHE_DIR=str(tmp_path / "model-cache"),
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="secret",
        NEO4J_DATABASE="idea",
    )

    model = bootstrap._build_structured_model(settings)

    assert isinstance(model, CachedStructuredJsonModel)
    assert model.cache_dir == tmp_path / "model-cache"
