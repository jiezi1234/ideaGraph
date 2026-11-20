"""任务级进度镜像：与 trace 共用阶段名称，不依赖远端入库速度。"""
import logging
import json
from dataclasses import asdict, is_dataclass
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from uuid import uuid4
from pydantic import BaseModel

logger = logging.getLogger(__name__)
_sink = ContextVar("progress_sink", default=None)
_parent = ContextVar("progress_parent", default=None)


def display_payload(value, limit=100_000):
    """保留业务输入输出，屏蔽凭据字段；大文本明确标记截断。"""
    def clean(item):
        if isinstance(item, BaseModel):
            item = item.model_dump()
        elif is_dataclass(item) and not isinstance(item, type):
            item = asdict(item)
        if isinstance(item, dict):
            return {str(key): "[redacted]" if any(part in str(key).lower() for part in
                    ("api_key", "apikey", "secret", "password", "authorization", "access_token"))
                    else clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, str):
            # Langfuse v2 的 io 通常是 JSON 字符串，先解码再进行字段过滤。
            try:
                parsed = json.loads(item)
            except (ValueError, TypeError):
                return item
            return clean(parsed) if isinstance(parsed, (dict, list)) else item
        if item is None or isinstance(item, (int, float, bool)):
            return item
        return str(item)
    result = clean(value)
    encoded = json.dumps(result, ensure_ascii=False)
    return result if len(encoded) <= limit else {"truncated": True, "total_characters": len(encoded), "preview": encoded[:limit]}


def emit_progress(name, status="completed", *, event_id=None, parent_id=None, input=None, output=None):
    sink = _sink.get()
    if sink is not None:
        try:
            sink({"id": event_id or uuid4().hex, "name": name, "status": status,
                  "parent_id": parent_id or _parent.get(),
                  "input": display_payload(input), "output": display_payload(output),
                  "time": datetime.now(timezone.utc).isoformat()})
        except Exception:
            # 展示故障不能中断图谱写入或触发模型重试。
            logger.warning("progress:persist_failed", exc_info=True)


@contextmanager
def progress_scope(sink):
    token = _sink.set(sink)
    parent_token = _parent.set(None)
    try:
        yield
    finally:
        _sink.reset(token)
        _parent.reset(parent_token)


@contextmanager
def progress_stage(name, *, input=None):
    event_id = uuid4().hex
    parent_id = _parent.get()
    data = {"input": input, "output": None}
    emit_progress(name, "running", event_id=event_id, parent_id=parent_id, input=input)
    token = _parent.set(event_id)
    try:
        yield data
    except BaseException:
        _parent.reset(token)
        emit_progress(name, "failed", event_id=event_id, parent_id=parent_id, **data)
        raise
    else:
        _parent.reset(token)
        emit_progress(name, "completed", event_id=event_id, parent_id=parent_id, **data)
