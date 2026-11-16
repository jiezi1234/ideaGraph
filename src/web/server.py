"""单进程 Web 服务：串行运行科研任务，避免小内存服务器并发建索引。"""
import json
import logging
import os
import secrets
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.app.bootstrap import build_idea_chat_workflow, build_mvp_workflow
from src.config.settings import Settings
from src.logging_config import configure_logging
from src.workflows.query_archive import archive_chat_idea_result, _to_jsonable

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent / "static"
DB = Path(os.getenv("WEB_TASK_DB", "exports/web/tasks.sqlite3"))
pool = ThreadPoolExecutor(max_workers=1)
bearer = HTTPBearer(auto_error=False)


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
        encoded = json.dumps(_to_jsonable(result), ensure_ascii=False)
        with connection() as db:
            db.execute("UPDATE tasks SET status='completed', result=? WHERE id=?", (encoded, task_id))
    except Exception:
        logger.exception("Web task failed: %s", task_id)
        # 异常详情只写服务器日志，避免接口地址和凭据泄露给浏览器。
        with connection() as db:
            db.execute("UPDATE tasks SET status='failed', error='任务未完成，请检查服务端日志、模型配置及文献库后重试' WHERE id=?", (task_id,))


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


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


@app.get("/api/tasks/{task_id}", dependencies=[Depends(authorize)])
def detail(task_id: str):
    with connection() as db:
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "任务不存在")
    return dict(row) | {"payload": json.loads(row["payload"]), "result": json.loads(row["result"]) if row["result"] else None}
