from concurrent.futures import Future

from fastapi.testclient import TestClient

from src.web import server


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
