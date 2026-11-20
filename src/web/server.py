"""单进程 Web 服务：串行运行科研任务，避免小内存服务器并发建索引。"""
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from contextlib import ExitStack
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from src.app.bootstrap import build_idea_chat_workflow, build_mvp_workflow
from src.config.settings import Settings
from src.logging_config import configure_logging
from src.observability.langfuse import build_langfuse_client, flush_langfuse_client
from src.observability.progress import progress_scope, progress_stage, display_payload
from src.web.trace_reader import read_observations
from src.workflows.query_archive import archive_chat_idea_result, _to_jsonable

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent / "static"
DB = Path(os.getenv("WEB_TASK_DB", "exports/web/tasks.sqlite3"))
REPORT_DIR = Path(os.getenv("WEB_REPORT_DIR", "exports/reports"))
pool = ThreadPoolExecutor(max_workers=1)
bearer = HTTPBearer(auto_error=False)
trace_lock = Lock()

# 图谱接口默认只回传这么多节点，避免小内存浏览器一次性渲染整库。
GRAPH_NODE_LIMIT = 800


def connection():
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB)
    db.row_factory = sqlite3.Row
    return db


@asynccontextmanager
async def lifespan(app):
    if len(os.getenv("WEB_ACCESS_TOKEN", "")) < 24:
        raise RuntimeError("WEB_ACCESS_TOKEN must contain at least 24 characters")
    with connection() as db:
        db.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT, status TEXT, result TEXT, error TEXT, created TEXT DEFAULT CURRENT_TIMESTAMP)")
        db.execute("CREATE TABLE IF NOT EXISTS task_events (task_id TEXT, id TEXT, name TEXT, status TEXT, started TEXT, updated TEXT, PRIMARY KEY(task_id,id))")
        # 旧任务数据库原地迁移；未记录过的 io 保持空值，不伪造历史输入输出。
        columns = {row["name"] for row in db.execute("PRAGMA table_info(task_events)")}
        for column in ("parent_id", "io"):
            if column not in columns:
                db.execute(f"ALTER TABLE task_events ADD COLUMN {column} TEXT")
        db.execute("CREATE TABLE IF NOT EXISTS task_traces (task_id TEXT PRIMARY KEY, trace_id TEXT, snapshot TEXT, fetched REAL DEFAULT 0)")
        # 进程重启不能将未完成任务伪装成仍在运行，也不能自动重复计费调用。
        db.execute("UPDATE tasks SET status='failed', error='服务重启，请重新提交任务' WHERE status IN ('queued','running')")
        db.execute("UPDATE task_events SET status='interrupted' WHERE status='running'")
    yield
    pool.shutdown(wait=True)


app = FastAPI(title="IdeaGraph", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=ROOT), name="static")


def authorize(auth: HTTPAuthorizationCredentials | None = Depends(bearer)):
    expected = os.getenv("WEB_ACCESS_TOKEN", "")
    if not expected or not auth or not secrets.compare_digest(auth.credentials, expected):
        raise HTTPException(401, "请输入正确的工作台访问口令")


class TaskInput(BaseModel):
    kind: str = Field(pattern="^(idea|build)$")
    query: str = Field(min_length=3, max_length=6000)
    seed_paper_id: str = Field(default="", max_length=300)
    constraints: str = Field(default="", max_length=2000)
    max_seed_papers: int = Field(default=3, ge=1, le=10)


def _report_markdown(kind: str, payload: dict, result: dict) -> str:
    created = datetime.now().strftime("%Y-%m-%d %H:%M")
    if kind == "build":
        return "\n".join([
            "# 文献建图报告",
            "",
            f"- 生成时间：{created}",
            f"- 检索关键词：{payload.get('query', '')}",
            f"- 种子论文上限：{payload.get('max_seed_papers', '')}",
            "",
            "## 建图结果",
            "",
            "| 指标 | 数量 |",
            "| --- | --- |",
            f"| 论文 | {result.get('paper_count', 0)} |",
            f"| 引用关系 | {result.get('citation_count', 0)} |",
            f"| 实体 | {result.get('entity_count', 0)} |",
            f"| 实体关系 | {result.get('entity_relation_count', 0)} |",
            "",
            "## 下一步",
            "",
            "打开「图谱探索」查看共享文献库，选择有摘要的论文即可继续研究构思。",
            "",
        ])

    idea = result.get("idea_output") or {}
    lines = [
        f"# {idea.get('title') or '研究方案'}",
        "",
        f"- 生成时间：{created}",
        f"- 研究问题：{payload.get('query', '')}",
    ]
    if payload.get("constraints"):
        lines.append(f"- 研究条件：{payload['constraints']}")
    lines.append("")
    if idea.get("summary"):
        lines += ["## 摘要", "", idea["summary"], ""]
    for key, title in (
        ("problem", "研究问题"),
        ("method", "方法设计"),
        ("experiment", "实验计划"),
        ("idea_lineage", "研究思路来源"),
    ):
        if idea.get(key):
            lines += [f"## {title}", "", idea[key], ""]
    if idea.get("core_paper_title"):
        lines += ["## 起点论文", "", idea["core_paper_title"], ""]

    candidates = result.get("candidate_papers") or []
    if candidates:
        lines += ["## 候选论文", ""]
        for item in candidates:
            lines.append(f"- {item.get('title') or item.get('paper_id') or ''}")
        lines.append("")

    evaluation = result.get("idea_evaluation") or {}
    if evaluation:
        lines += ["## 评审意见", ""]
        if evaluation.get("overall_score") is not None:
            lines.append(f"- **综合评分**：{evaluation['overall_score']} / 5")
        if evaluation.get("summary"):
            lines += ["", evaluation["summary"], ""]
        for key, label in (("novelty", "新颖性"), ("feasibility", "可行性"), ("expected_effectiveness", "预期效果")):
            item = evaluation.get(key) or {}
            if item.get("score") is not None:
                lines.append(f"- **{label}**：{item['score']} / 5 —— {item.get('comment', '')}")
        issues = evaluation.get("major_issues") or []
        if issues:
            lines += ["", "**主要问题**", ""]
            lines += [f"- {issue}" for issue in issues]
        if evaluation.get("revision_instruction"):
            lines += ["", f"**改进建议**：{evaluation['revision_instruction']}"]
        lines.append("")

    comparison = result.get("idea_comparison") or {}
    if comparison.get("overall_explanation"):
        lines += ["## 改进对比", "", comparison["overall_explanation"], ""]
    return "\n".join(lines).rstrip() + "\n"


def _save_event(task_id, event):
    with connection() as db:
        db.execute("""INSERT INTO task_events(task_id,id,name,status,started,updated,parent_id,io) VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(task_id,id) DO UPDATE SET status=excluded.status,updated=excluded.updated,io=excluded.io""",
            (task_id, event["id"], event["name"], event["status"], event["time"], event["time"], event.get("parent_id"),
             json.dumps({"input": display_payload(event.get("input")), "output": display_payload(event.get("output"))}, ensure_ascii=False)))


def execute(task_id, task):
    client = None
    with ExitStack() as stack:
        # 根 span 在工作线程内建立，asyncio / to_thread 继承同一任务上下文。
        try:
            settings = Settings()
            if getattr(settings, "langfuse_tracing_enabled", True):
                client = build_langfuse_client(settings)
            if client is not None:
                span = stack.enter_context(client.start_as_current_observation(
                    name="web_" + task.kind, trace_context={"trace_id": task_id},
                    metadata={"task_id": task_id, "kind": task.kind}))
                with connection() as db:
                    db.execute("INSERT OR REPLACE INTO task_traces(task_id,trace_id) VALUES(?,?)", (task_id, span.trace_id))
        except Exception:
            logger.warning("web:trace_setup_failed", exc_info=True)
            client = None

        def sink(event):
            _save_event(task_id, event)
            if client is not None and ":" in event["name"]:
                # 建图里程碑也写入当前 trace，且不改变 producer/consumer 写入顺序。
                with client.start_as_current_observation(name=event["name"], input=event.get("input")) as milestone:
                    milestone.update(output=event.get("output"))

        with progress_scope(sink):
            _execute(task_id, task)
    flush_langfuse_client(client)


def _execute(task_id, task):
    with connection() as db:
        db.execute("UPDATE tasks SET status='running' WHERE id=?", (task_id,))
    try:
        settings = Settings()
        configure_logging(level=settings.log_level, log_file=settings.log_file)
        if task.kind == "build":
            settings = settings.model_copy(update={"citation_seed_fetch_all": False})
            with progress_stage("prepare_workflow"):
                workflow = build_mvp_workflow(settings)
            with progress_stage("build_graph", input=task.model_dump()) as progress:
                result = workflow.run(keyword=task.query, max_seed_papers=task.max_seed_papers, require_references=False)
                progress["output"] = result
        else:
            query = task.query
            if task.constraints.strip():
                query += "\n\n研究资源与限制：\n" + task.constraints
            with progress_stage("prepare_workflow"):
                workflow = build_idea_chat_workflow(settings)
            with progress_stage("research", input={"query": query, "seed_paper_id": task.seed_paper_id}) as progress:
                result = workflow.run(user_query=query, seed_paper_id=task.seed_paper_id.strip() or None)
                progress["output"] = _to_jsonable(result)
            archive_chat_idea_result(result, settings, user_query=query)
        result = dict(_to_jsonable(result))
        result["report_markdown"] = _report_markdown(task.kind, task.model_dump(), result)
        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / f"{task_id}.md").write_text(result["report_markdown"], encoding="utf-8")
        except OSError:
            logger.exception("Failed to persist report for task %s", task_id)
        encoded = json.dumps(result, ensure_ascii=False)
        with connection() as db:
            db.execute("UPDATE tasks SET status='completed', result=? WHERE id=?", (encoded, task_id))
    except Exception:
        logger.exception("Web task failed: %s", task_id)
        # 异常详情只写服务器日志，避免接口地址和凭据泄露给浏览器。
        with connection() as db:
            db.execute("UPDATE tasks SET status='failed', error='任务未完成，请检查服务端日志、模型配置及文献库后重试' WHERE id=?", (task_id,))


@lru_cache(maxsize=64)
def _asset_version(path: Path, mtime_ns: int, size: int) -> str:
    # 时间和长度用于失效本地缓存；浏览器版本来自内容，跨镜像同一文件可复用。
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _index():
    def versioned_url(match: re.Match) -> str:
        relative = match.group("asset")
        asset = ROOT / relative
        stat = asset.stat()
        version = _asset_version(asset, stat.st_mtime_ns, stat.st_size)
        return f'{match.group("attr")}="static/{relative}?v={version}"'

    # HTML 每次导航重新验证；资源使用相对地址，兼容根路径与 /research/。
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    html = re.sub(r'\b(?P<attr>src|href|data-icon-sprite)="static/(?P<asset>[\w./-]+)"', versioned_url, html)
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    return _index()


@app.get("/build")
def build_page():
    return _index()


@app.get("/idea")
def idea_page():
    return _index()


@app.get("/graph")
def graph_page():
    return _index()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/graph", dependencies=[Depends(authorize)])
def graph_data():
    settings = Settings()
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    try:
        with driver.session(database=settings.neo4j_database) as session:
            papers = session.run(
                "MATCH (p:Paper) RETURN p.id AS id, left(trim(coalesce(p.abstract, '')), 480) AS preview "
                "ORDER BY p.id LIMIT $n",
                n=GRAPH_NODE_LIMIT,
            ).data()
            entities = session.run(
                "MATCH (e:Entity) RETURN e.id AS id, e.entity_type AS type, "
                "left(trim(coalesce(e.description, '')), 480) AS preview ORDER BY e.entity_type, e.id LIMIT $n",
                n=GRAPH_NODE_LIMIT,
            ).data()
            edges = session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN (CASE WHEN a:Paper THEN 'p:' ELSE 'e:' END) + a.id AS source,
                       (CASE WHEN b:Paper THEN 'p:' ELSE 'e:' END) + b.id AS target,
                       type(r) AS rel
                LIMIT $n
                """,
                n=GRAPH_NODE_LIMIT * 4,
            ).data()
    finally:
        driver.close()

    # 无摘要论文保留引用结构；前端只把有摘要的论文作为研究起点。
    nodes = [
        {"id": "p:" + row["id"], "label": row["id"], "kind": "paper", "type": "Paper", "preview": row["preview"]}
        for row in papers
    ]
    nodes += [
        {"id": "e:" + row["id"], "label": row["id"], "kind": "entity", "type": row["type"] or "Entity",
         "preview": row["preview"]}
        for row in entities
    ]
    return {
        "nodes": nodes,
        "edges": [{"from": row["source"], "to": row["target"], "label": row["rel"]} for row in edges],
    }


@app.post("/api/tasks", dependencies=[Depends(authorize)], status_code=202)
def submit(task: TaskInput):
    task_id = uuid4().hex
    with connection() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT COUNT(*) FROM tasks WHERE status IN ('queued','running')").fetchone()[0] >= 4:
            raise HTTPException(429, "任务队列已满，请稍后再试")
        db.execute("INSERT INTO tasks(id,payload,status) VALUES(?,?,'queued')", (task_id, task.model_dump_json()))
    pool.submit(execute, task_id, task)
    return {"id": task_id}


@app.get("/api/tasks", dependencies=[Depends(authorize)])
def history():
    with connection() as db:
        rows = db.execute("SELECT id,payload,status,error,created FROM tasks ORDER BY created DESC,rowid DESC LIMIT 50").fetchall()
    return [dict(row) | {"payload": json.loads(row["payload"])} for row in rows]


@app.get("/api/tasks/{task_id}/report.md", dependencies=[Depends(authorize)])
def report(task_id: str):
    path = REPORT_DIR / f"{task_id}.md"
    if not path.exists():
        raise HTTPException(404, "报告不存在")
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="ideagraph-{task_id[:8]}.md"'},
    )


@app.get("/api/tasks/{task_id}", dependencies=[Depends(authorize)])
def detail(task_id: str):
    with connection() as db:
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        events = db.execute("SELECT * FROM (SELECT rowid AS sequence,id,name,status,started,updated,parent_id FROM task_events WHERE task_id=? ORDER BY rowid DESC LIMIT 200) ORDER BY sequence", (task_id,)).fetchall()
        event_count = db.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (task_id,)).fetchone()[0]
    if row is None:
        raise HTTPException(404, "任务不存在")
    return dict(row) | {"events": [dict(event) for event in events], "events_truncated": event_count > len(events), "payload": json.loads(row["payload"]), "result": json.loads(row["result"]) if row["result"] else None}


@app.get("/api/tasks/{task_id}/events/{event_id}", dependencies=[Depends(authorize)])
def event_detail(task_id: str, event_id: str):
    with connection() as db:
        row = db.execute("SELECT io FROM task_events WHERE task_id=? AND id=?", (task_id, event_id)).fetchone()
    if row is None:
        raise HTTPException(404, "过程节点不存在")
    return json.loads(row["io"]) if row["io"] else {"input": None, "output": None}


@app.get("/api/tasks/{task_id}/trace/{observation_id}", dependencies=[Depends(authorize)])
def observation_detail(task_id: str, observation_id: str):
    with connection() as db:
        row = db.execute("SELECT trace_id,created FROM task_traces JOIN tasks ON tasks.id=task_traces.task_id WHERE task_id=?", (task_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "任务未关联 trace")
    try:
        result = read_observations(Settings(), row["trace_id"], row["created"], observation_id=observation_id)
    except Exception:
        logger.warning("web:observation_read_failed task=%s", task_id, exc_info=True)
        raise HTTPException(503, "暂时无法读取调用详情，请稍后重试") from None
    if not result["observations"]:
        raise HTTPException(404, "调用尚未入库或不属于此任务")
    return result["observations"][0]


@app.get("/api/tasks/{task_id}/trace", dependencies=[Depends(authorize)])
def task_trace(task_id: str):
    # 远端轮询共用缓存，多个浏览器不会重复消耗 Langfuse API 配额。
    with trace_lock:
        with connection() as db:
            task = db.execute("SELECT created FROM tasks WHERE id=?", (task_id,)).fetchone()
            trace = db.execute("SELECT * FROM task_traces WHERE task_id=?", (task_id,)).fetchone()
        if task is None:
            raise HTTPException(404, "任务不存在")
        if trace is None:
            return {"state": "disabled", "observations": []}
        snapshot = json.loads(trace["snapshot"]) if trace["snapshot"] else {"observations": []}
        if time.time() - trace["fetched"] < 15:
            return snapshot
        try:
            snapshot = read_observations(Settings(), trace["trace_id"], task["created"])
        except Exception:
            logger.warning("web:trace_read_failed task=%s", task_id, exc_info=True)
            snapshot = dict(snapshot, state="unavailable")
        snapshot["trace_id"] = trace["trace_id"]
        with connection() as db:
            db.execute("UPDATE task_traces SET snapshot=?,fetched=? WHERE task_id=?",
                       (json.dumps(snapshot), time.time(), task_id))
        return snapshot
