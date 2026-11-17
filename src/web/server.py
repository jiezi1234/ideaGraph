"""单进程 Web 服务：串行运行科研任务，避免小内存服务器并发建索引。"""
import json
import logging
import os
import secrets
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

from src.app.bootstrap import build_idea_chat_workflow, build_mvp_workflow
from src.config.settings import Settings
from src.logging_config import configure_logging
from src.workflows.query_archive import archive_chat_idea_result, _to_jsonable

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent / "static"
DB = Path(os.getenv("WEB_TASK_DB", "exports/web/tasks.sqlite3"))
REPORT_DIR = Path(os.getenv("WEB_REPORT_DIR", "exports/reports"))
pool = ThreadPoolExecutor(max_workers=1)
bearer = HTTPBearer(auto_error=False)

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
        # 进程重启不能将未完成任务伪装成仍在运行，也不能自动重复计费调用。
        db.execute("UPDATE tasks SET status='failed', error='服务重启，请重新提交任务' WHERE status IN ('queued','running')")
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
            "在左侧导航打开「图谱浏览」，可以查看本次写入的论文引用图与实体关系网络。",
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


def execute(task_id, task):
    with connection() as db:
        db.execute("UPDATE tasks SET status='running' WHERE id=?", (task_id,))
    try:
        settings = Settings()
        configure_logging(level=settings.log_level, log_file=settings.log_file)
        if task.kind == "build":
            settings = settings.model_copy(update={"citation_seed_fetch_all": False})
            result = build_mvp_workflow(settings).run(keyword=task.query, max_seed_papers=task.max_seed_papers, require_references=False)
        else:
            query = task.query
            if task.constraints.strip():
                query += "\n\n研究资源与限制：\n" + task.constraints
            result = build_idea_chat_workflow(settings).run(user_query=query, seed_paper_id=task.seed_paper_id.strip() or None)
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


def _index():
    return FileResponse(ROOT / "index.html")


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
                "MATCH (p:Paper) RETURN p.id AS id ORDER BY p.id LIMIT $n",
                n=GRAPH_NODE_LIMIT,
            ).data()
            entities = session.run(
                "MATCH (e:Entity) RETURN e.id AS id, e.entity_type AS type ORDER BY e.entity_type, e.id LIMIT $n",
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

    nodes = [{"id": "p:" + row["id"], "label": row["id"], "kind": "paper", "type": "Paper"} for row in papers]
    nodes += [
        {"id": "e:" + row["id"], "label": row["id"], "kind": "entity", "type": row["type"] or "Entity"}
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
    if row is None:
        raise HTTPException(404, "任务不存在")
    return dict(row) | {"payload": json.loads(row["payload"]), "result": json.loads(row["result"]) if row["result"] else None}
