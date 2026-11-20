import asyncio

import pytest

from src.observability.langfuse import langfuse_observation
from src.observability.progress import emit_progress, progress_scope, progress_stage, display_payload


def test_stage_lifecycle_without_langfuse_and_failure():
    events = []
    with progress_scope(events.append):
        with langfuse_observation(None, name="paper_retrieval"):
            pass
        with pytest.raises(ValueError):
            with progress_stage("failed_stage"):
                raise ValueError("private error")
    assert [event["status"] for event in events] == ["running", "completed", "running", "failed"]
    assert events[0]["id"] == events[1]["id"]
    assert "private error" not in str(events)
    emit_progress("outside")
    assert len(events) == 4


def test_progress_context_isolated_and_propagated_to_threads():
    async def run():
        async def task(name):
            events = []
            with progress_scope(events.append):
                await asyncio.to_thread(emit_progress, name)
            return events
        return await asyncio.gather(task("first"), task("second"))
    first, second = asyncio.run(run())
    assert [item["name"] for item in first] == ["first"]
    assert [item["name"] for item in second] == ["second"]


def test_sink_failure_does_not_break_workflow():
    def broken_sink(event):
        raise OSError("disk unavailable")
    with progress_scope(broken_sink), progress_stage("work"):
        pass


def test_nested_stages_preserve_parent_and_io():
    events = []
    with progress_scope(events.append):
        with progress_stage("parent", input={"query": "traffic"}) as parent:
            with progress_stage("child", input={"messages": ["question"]}) as child:
                child["output"] = {"answer": "result"}
            parent["output"] = {"done": True}
        emit_progress("next")
    assert events[1]["parent_id"] == events[0]["id"]
    assert events[2]["parent_id"] == events[0]["id"]
    assert events[2]["output"] == {"answer": "result"}
    assert events[3]["parent_id"] is None
    assert events[4]["parent_id"] is None


def test_payload_redacts_credentials_in_json_and_marks_truncation():
    value = display_payload('{"messages": ["<script>alert(1)</script>"], "api_key": "secret", "nested": {"authorization": "Bearer abc"}}')
    assert value["api_key"] == "[redacted]"
    assert value["nested"]["authorization"] == "[redacted]"
    assert value["messages"] == ["<script>alert(1)</script>"]
    assert display_payload("a" * 100, limit=10)["truncated"] is True
