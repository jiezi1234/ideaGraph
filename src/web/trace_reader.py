"""服务端读取 Langfuse；概览与按需加载的输入输出分开，避免轮询大文本。"""
import json
from datetime import datetime, timedelta, timezone

import httpx
from src.observability.progress import display_payload


def read_observations(settings, trace_id, created, *, observation_id=None):
    start = datetime.fromisoformat(created).replace(tzinfo=timezone.utc) - timedelta(minutes=5)
    params = {"traceId": trace_id, "fromStartTime": start.isoformat(),
              "toStartTime": datetime.now(timezone.utc).isoformat(),
              "fields": "core,basic,time,usage", "limit": 1000}
    if observation_id:
        params["fields"] += ",io"
        params["filter"] = json.dumps([{"type": "string", "column": "id", "operator": "=", "value": observation_id}])
    items = {}
    # 有界分页，防止大规模建图阻塞状态接口；详情与实时进度分别加载。
    with httpx.Client(timeout=8, auth=(settings.langfuse_public_key, settings.langfuse_secret_key)) as client:
        for _ in range(5):
            response = client.get(settings.langfuse_host.rstrip("/") + "/api/public/v2/observations", params=params)
            response.raise_for_status()
            body = response.json()
            for row in body.get("data", []):
                if row.get("traceId") != trace_id or (observation_id and row.get("id") != observation_id):
                    continue
                items[row["id"]] = {key: row.get(key) for key in (
                    "id", "name", "type", "parentObservationId", "startTime", "endTime", "level",
                    "inputTokens", "outputTokens", "totalTokens")}
                if observation_id:
                    items[row["id"]].update(input=display_payload(row.get("input")), output=display_payload(row.get("output")))
            cursor = (body.get("meta") or {}).get("cursor")
            if not cursor:
                break
            params["cursor"] = cursor
    return {"observations": sorted(items.values(), key=lambda item: item.get("startTime") or ""),
            "truncated": bool(cursor), "state": "ready" if items else "pending"}
