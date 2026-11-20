from types import SimpleNamespace
import json

import httpx

from src.web import trace_reader


def test_trace_reader_filters_sensitive_fields_and_paginates(monkeypatch):
    requests = []
    def handler(request):
        requests.append(request)
        assert request.url.params["traceId"] == "task-trace"
        assert "fromStartTime" in request.url.params
        if "cursor" not in request.url.params:
            return httpx.Response(200, json={"data": [
                {"id": "one", "traceId": "task-trace", "name": "paper_retrieval", "input": "secret", "output": "private"},
                {"id": "other", "traceId": "other-task", "name": "wrong task"}], "meta": {"cursor": "page2"}})
        return httpx.Response(200, json={"data": [{"id": "two", "traceId": "task-trace", "name": "evaluation", "totalTokens": 12}], "meta": {}})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(trace_reader.httpx, "Client", lambda **kwargs: client)
    result = trace_reader.read_observations(SimpleNamespace(langfuse_host="https://example.test", langfuse_public_key="pk", langfuse_secret_key="sk"), "task-trace", "2026-11-16 10:00:00")
    assert len(requests) == 2
    assert [item["id"] for item in result["observations"]] == ["one", "two"]
    assert "secret" not in str(result) and "private" not in str(result)
    assert result["state"] == "ready"
    assert result["truncated"] is False


def test_observation_detail_requests_io_and_filters_task_and_id(monkeypatch):
    def handler(request):
        assert "io" in request.url.params["fields"]
        assert json.loads(request.url.params["filter"])[0]["value"] == "wanted"
        return httpx.Response(200, json={"data": [
            {"id": "wanted", "traceId": "task-trace", "input": '{"messages":["hello"],"api_key":"secret"}', "output": '{"answer":"world"}'},
            {"id": "wrong", "traceId": "task-trace", "output": "wrong-id"},
            {"id": "wanted", "traceId": "other-task", "output": "wrong-task"}], "meta": {}})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(trace_reader.httpx, "Client", lambda **kwargs: client)
    result = trace_reader.read_observations(SimpleNamespace(langfuse_host="https://example.test", langfuse_public_key="pk", langfuse_secret_key="sk"), "task-trace", "2026-11-16 10:00:00", observation_id="wanted")
    assert len(result["observations"]) == 1
    row = result["observations"][0]
    assert row["input"] == {"messages": ["hello"], "api_key": "[redacted]"}
    assert row["output"] == {"answer": "world"}
