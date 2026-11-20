from concurrent.futures import Future
from contextlib import contextmanager
from types import SimpleNamespace
import hashlib
import re
import sqlite3

from fastapi.testclient import TestClient

from src.web import server


def test_workbench_pages_reference_current_asset_contents():
    client = TestClient(server.app)
    home = client.get("/")
    for route in ("/", "/build", "/graph", "/idea"):
        page = client.get(route)
        assert page.headers["cache-control"] == "no-cache"
        assert page.text == home.text
    refs = re.findall(r'(?:src|href|data-icon-sprite)="(static/[^"]+)"', home.text)
    assert {"static/style.css", "static/app.js", "static/icons.svg"} <= {ref.partition("?")[0] for ref in refs}
    for ref in set(refs):
        asset = client.get("/" + ref)
        assert asset.status_code == 200
        assert ref.partition("?v=")[2] == hashlib.sha256(asset.content).hexdigest()[:16]


def test_asset_url_changes_when_stylesheet_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "ROOT", tmp_path)
    (tmp_path / "index.html").write_text('<link href="static/style.css">', encoding="utf-8")
    css = tmp_path / "style.css"
    css.write_text(".brand { color: green }", encoding="utf-8")
    before = server._index().body
    css.write_text(".brand { color: purple }", encoding="utf-8")
    assert server._index().body != before


def test_graph_browser_keeps_citation_only_papers_and_entity_previews(monkeypatch):
    def run(query, **params):
        if "p:Paper" in query:
            assert "trim(coalesce(p.abstract, ''))" in query
            rows = [{"id": "seed", "preview": "Grounded abstract"}, {"id": "citation", "preview": ""}]
        elif "e:Entity" in query:
            assert "trim(coalesce(e.description, ''))" in query
            rows = [{"id": "method", "type": "Method", "preview": "Method description"}]
        else:
            rows = [{"source": "p:seed", "target": "p:citation", "rel": "CITES"}]
        return SimpleNamespace(data=lambda: rows)

    @contextmanager
    def session(**kwargs):
        yield SimpleNamespace(run=run)

    driver = SimpleNamespace(session=session, close=lambda: None)
    monkeypatch.setattr(server, "Settings", lambda: SimpleNamespace(
        neo4j_uri="", neo4j_username="", neo4j_password="", neo4j_database="test"
    ))
    monkeypatch.setattr(server.GraphDatabase, "driver", lambda *args, **kwargs: driver)
    graph = server.graph_data()
    assert [(node["id"], node["preview"]) for node in graph["nodes"]] == [
        ("p:seed", "Grounded abstract"), ("p:citation", ""), ("e:method", "Method description")
    ]
    assert graph["edges"] == [{"from": "p:seed", "to": "p:citation", "label": "CITES"}]


def test_auth_queue_and_history(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DB", tmp_path / "tasks.db")
    monkeypatch.setenv("WEB_ACCESS_TOKEN", "x" * 32)
    monkeypatch.setattr(server.pool, "submit", lambda *args: Future())
    monkeypatch.setattr(server.pool, "shutdown", lambda **kwargs: None)
    headers = {"Authorization": "Bearer " + "x" * 32}
    with TestClient(server.app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/tasks").status_code == 401
        assert client.post("/api/tasks", headers=headers, json={"kind": "bad", "query": "test"}).status_code == 422
        for _ in range(4):
            response = client.post("/api/tasks", headers=headers, json={"kind": "idea", "query": "test problem"})
            assert response.status_code == 202
        assert client.post("/api/tasks", headers=headers, json={"kind": "idea", "query": "test"}).status_code == 429
        row = client.get("/api/tasks/" + response.json()["id"], headers=headers).json()
        assert row["status"] == "queued"
        assert row["result"] is None
        assert row["events"] == []
        task_id = row["id"]
        assert client.get(f"/api/tasks/{task_id}/trace").status_code == 401
        assert client.get(f"/api/tasks/{task_id}/trace", headers=headers).json()["state"] == "disabled"
    with TestClient(server.app) as client:
        assert all(row["status"] == "failed" for row in client.get("/api/tasks", headers=headers).json())


def test_execution_preserves_seed_and_archives(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DB", tmp_path / "tasks.db")
    monkeypatch.setenv("WEB_ACCESS_TOKEN", "x" * 32)
    monkeypatch.setattr(server.pool, "shutdown", lambda **kwargs: None)
    monkeypatch.setattr(server, "Settings", lambda: type("Settings", (), {"log_level": "INFO", "log_file": str(tmp_path / "log")})())
    seen = {}
    class Workflow:
        def run(self, **kwargs):
            seen.update(kwargs)
            return {"idea_output": {"title": "Test"}, "tool_trace": [{"round": 0}]}
    monkeypatch.setattr(server, "build_idea_chat_workflow", lambda settings: Workflow())
    monkeypatch.setattr(server, "archive_chat_idea_result", lambda *args, **kwargs: seen.update(archived=True))
    with TestClient(server.app):
        with server.connection() as db:
            db.execute("INSERT INTO tasks(id,payload,status) VALUES('test','{}','queued')")
        server.execute("test", server.TaskInput(kind="idea", query="Question", seed_paper_id="paper-id", constraints="one GPU"))
        assert seen["seed_paper_id"] == "paper-id"
        assert "one GPU" in seen["user_query"]
        assert seen["archived"]
        assert server.detail("test")["result"]["tool_trace"] == [{"round": 0}]
        events = server.detail("test")["events"]
        assert [event["name"] for event in events] == ["prepare_workflow", "research"]
        assert all(event["status"] == "completed" for event in events)


def test_trace_cache_failure_and_task_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DB", tmp_path / "tasks.db")
    monkeypatch.setattr(server, "Settings", lambda: SimpleNamespace())
    monkeypatch.setenv("WEB_ACCESS_TOKEN", "x" * 32)
    monkeypatch.setattr(server.pool, "shutdown", lambda **kwargs: None)
    calls = []
    def reader(settings, trace_id, created):
        calls.append(trace_id)
        raise RuntimeError("private credentials")
    monkeypatch.setattr(server, "read_observations", reader)
    with TestClient(server.app):
        with server.connection() as db:
            for task_id in ("one", "two"):
                db.execute("INSERT INTO tasks(id,payload,status) VALUES(?,'{}','completed')", (task_id,))
                db.execute("INSERT INTO task_traces(task_id,trace_id) VALUES(?,?)", (task_id, task_id + "-trace"))
        server._save_event("one", {"id": "event", "name": "research", "status": "running", "time": "2026-11-16T10:00:00+00:00"})
        assert server.detail("two")["events"] == []
        assert server.task_trace("one")["state"] == "unavailable"
        assert "private credentials" not in str(server.task_trace("one"))
        assert calls == ["one-trace"]
        server.task_trace("two")
        assert calls == ["one-trace", "two-trace"]


def test_web_build_limits_seed_scan(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DB", tmp_path / "tasks.db")
    monkeypatch.setenv("WEB_ACCESS_TOKEN", "x" * 32)
    monkeypatch.setattr(server.pool, "shutdown", lambda **kwargs: None)
    seen = {}
    class Config:
        log_level = "INFO"
        log_file = str(tmp_path / "log")
        def model_copy(self, update):
            seen.update(update)
            return self
    class Workflow:
        def run(self, **kwargs):
            seen.update(kwargs)
            return {"papers": 1}
    monkeypatch.setattr(server, "Settings", Config)
    monkeypatch.setattr(server, "build_mvp_workflow", lambda settings: Workflow())
    with TestClient(server.app):
        with server.connection() as db:
            db.execute("INSERT INTO tasks(id,payload,status) VALUES('test','{}','queued')")
        server.execute("test", server.TaskInput(kind="build", query="traffic", max_seed_papers=2))
        assert seen["citation_seed_fetch_all"] is False
        assert seen["max_seed_papers"] == 2
        assert server.detail("test")["status"] == "completed"


def test_web_root_trace_and_failed_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DB", tmp_path / "tasks.db")
    monkeypatch.setenv("WEB_ACCESS_TOKEN", "x" * 32)
    monkeypatch.setattr(server.pool, "shutdown", lambda **kwargs: None)
    monkeypatch.setattr(server, "Settings", lambda: SimpleNamespace(log_level="INFO", log_file=str(tmp_path / "log")))
    seen = {}
    class Tracer:
        @contextmanager
        def start_as_current_observation(self, **kwargs):
            seen.update(kwargs)
            yield SimpleNamespace(trace_id="bound-trace")
        def flush(self):
            seen["flushed"] = True
    monkeypatch.setattr(server, "build_langfuse_client", lambda settings: Tracer())
    def fail(settings):
        raise RuntimeError("private backend URL")
    monkeypatch.setattr(server, "build_idea_chat_workflow", fail)
    with TestClient(server.app):
        with server.connection() as db:
            db.execute("INSERT INTO tasks(id,payload,status) VALUES('test','{}','queued')")
        server.execute("test", server.TaskInput(kind="idea", query="Question"))
        result = server.detail("test")
        assert result["status"] == "failed"
        assert result["events"][0]["status"] == "failed"
        assert "private backend URL" not in str(result)
        assert seen["trace_context"] == {"trace_id": "test"}
        assert seen["flushed"]
        with server.connection() as db:
            assert db.execute("SELECT trace_id FROM task_traces WHERE task_id='test'").fetchone()[0] == "bound-trace"


def test_event_io_migration_auth_and_task_isolation(tmp_path, monkeypatch):
    path = tmp_path / "tasks.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE task_events (task_id TEXT,id TEXT,name TEXT,status TEXT,started TEXT,updated TEXT,PRIMARY KEY(task_id,id))")
        db.execute("INSERT INTO task_events VALUES('old','legacy','step','completed','now','now')")
    monkeypatch.setattr(server, "DB", path)
    monkeypatch.setenv("WEB_ACCESS_TOKEN", "x" * 32)
    monkeypatch.setattr(server.pool, "shutdown", lambda **kwargs: None)
    headers = {"Authorization": "Bearer " + "x" * 32}
    with TestClient(server.app) as client:
        assert client.get("/api/tasks/old/events/legacy").status_code == 401
        assert client.get("/api/tasks/old/events/legacy", headers=headers).json() == {"input": None, "output": None}
        with server.connection() as db:
            db.execute("INSERT INTO tasks(id,payload,status) VALUES('one','{}','running')")
        server._save_event("one", {"id": "node", "name": "stage", "status": "running", "time": "now", "parent_id": "root", "input": {"api_key": "secret", "messages": ["question"]}})
        server._save_event("one", {"id": "node", "name": "stage", "status": "completed", "time": "later", "parent_id": "root", "input": {"messages": ["question"]}, "output": {"answer": "answer"}})
        result = client.get("/api/tasks/one/events/node", headers=headers).json()
        assert result == {"input": {"messages": ["question"]}, "output": {"answer": "answer"}}
        assert client.get("/api/tasks/two/events/node", headers=headers).status_code == 404
        summary = server.detail("one")["events"][0]
        assert summary["parent_id"] == "root"
        assert "io" not in summary
        assert summary["started"] == "now"


def test_observation_detail_is_authorized_scoped_and_recoverable(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DB", tmp_path / "tasks.db")
    monkeypatch.setenv("WEB_ACCESS_TOKEN", "x" * 32)
    monkeypatch.setattr(server.pool, "shutdown", lambda **kwargs: None)
    monkeypatch.setattr(server, "Settings", lambda: SimpleNamespace())
    calls = []
    def reader(settings, trace_id, created, **kwargs):
        calls.append((trace_id, kwargs["observation_id"]))
        if kwargs["observation_id"] == "error":
            raise RuntimeError("private url")
        return {"observations": [{"id": "node", "input": "question", "output": "answer"}] if kwargs["observation_id"] == "node" else []}
    monkeypatch.setattr(server, "read_observations", reader)
    headers = {"Authorization": "Bearer " + "x" * 32}
    with TestClient(server.app) as client:
        with server.connection() as db:
            db.execute("INSERT INTO tasks(id,payload,status) VALUES('one','{}','completed')")
            db.execute("INSERT INTO task_traces(task_id,trace_id) VALUES('one','bound-trace')")
        assert client.get("/api/tasks/one/trace/node").status_code == 401
        assert client.get("/api/tasks/two/trace/node", headers=headers).status_code == 404
        assert client.get("/api/tasks/one/trace/node", headers=headers).json()["output"] == "answer"
        assert calls == [("bound-trace", "node")]
        assert client.get("/api/tasks/one/trace/missing", headers=headers).status_code == 404
        response = client.get("/api/tasks/one/trace/error", headers=headers)
        assert response.status_code == 503
        assert "private url" not in response.text
