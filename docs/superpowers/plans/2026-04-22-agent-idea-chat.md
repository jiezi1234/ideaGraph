# Agent Idea Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph-based single-turn agent that takes a user question, retrieves candidate papers from the local paper graph using abstract embeddings, selects one core paper, and generates one idea from the paper title and abstract.

**Architecture:** Keep graph building and agent querying separate. Implement a new retrieval layer that embeds the user query and scores existing `Paper` nodes from Neo4j in memory. Cache paper embeddings locally by `Paper.id` and embedding model so repeated retrieval does not recompute abstract vectors. Add a short LangGraph pipeline with four nodes: query understanding, paper retrieval, paper selection, and idea generation.

**Tech Stack:** Python 3.12, LangGraph, Ollama, Neo4j, pytest, pydantic

---

## File Structure

Planned files and responsibilities:

- Create: `src/schemas/agent.py`
  - Agent-facing data contracts such as `CandidatePaper`, `IdeaOutput`, and error/result wrappers.
- Create: `src/models/ollama_embeddings.py`
  - Thin Ollama embedding client used only for retrieval.
- Create: `src/retrieval/embedding_cache.py`
  - Local paper embedding cache keyed by `Paper.id` and embedding model.
- Create: `src/retrieval/base.py`
  - Retriever and selector interfaces.
- Create: `src/retrieval/vector_retriever.py`
  - In-memory embedding retrieval over existing `Paper` nodes.
- Create: `src/retrieval/selector.py`
  - Default paper selector that picks the top-scoring candidate.
- Create: `src/agent/state.py`
  - LangGraph state definition.
- Create: `src/agent/nodes/query_understanding.py`
  - Query rewrite node.
- Create: `src/agent/nodes/paper_selection.py`
  - Candidate selection node.
- Create: `src/agent/nodes/idea_generation.py`
  - Idea generation node.
- Create: `src/agent/graph.py`
  - Graph topology and node wiring.
- Create: `src/workflows/idea_chat.py`
  - Application-facing workflow wrapper.
- Create: `src/interfaces/cli/chat_idea.py`
  - CLI subcommand parser additions.
- Modify: `src/storage/graph/neo4j_repository.py`
  - Add read method to fetch papers with non-empty abstracts.
- Modify: `src/models/ollama.py`
  - Optionally add a reusable freeform JSON generation method so the agent nodes do not fake title/abstract payloads.
- Modify: `src/config/settings.py`
  - Add agent-related settings.
- Modify: `src/app/bootstrap.py`
  - Build retriever, selector, and idea chat workflow.
- Modify: `src/interfaces/cli/build_graph.py`
  - Extend the root parser with `chat-idea`.
- Modify: `main.py`
  - Route new command.
- Modify: `pyproject.toml`
  - Add `langgraph` dependency.
- Modify: `README.md`
  - Document agent usage.

Tests:

- Create: `tests/test_embedding_cache.py`
- Create: `tests/test_vector_retriever.py`
- Create: `tests/test_agent_nodes.py`
- Create: `tests/test_idea_chat_workflow.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_main.py`
- Modify: `tests/test_settings_env.py`
- Modify: `tests/test_neo4j_repository.py`

## Task 1: Add Contracts, Config, and Dependencies

**Files:**
- Create: `src/schemas/agent.py`
- Modify: `src/config/settings.py`
- Modify: `pyproject.toml`
- Modify: `tests/test_settings_env.py`

- [ ] **Step 1: Write the failing config and schema tests**

```python
from src.config.settings import Settings
from src.schemas.agent import CandidatePaper, IdeaOutput


def test_settings_reads_agent_fields(monkeypatch):
    monkeypatch.setenv("OPENALEX_BASE_URL", "https://api.openalex.org")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11435")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.6:35b")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "300")
    monkeypatch.setenv("OLLAMA_THINK_MODE", "medium")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "pass")
    monkeypatch.setenv("NEO4J_DATABASE", "testidea")
    monkeypatch.setenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setenv("AGENT_RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("PAPER_EMBEDDING_CACHE_DIR", ".cache/paper_embeddings")

    settings = Settings()

    assert settings.ollama_embed_model == "nomic-embed-text"
    assert settings.agent_retrieval_top_k == 5
    assert settings.paper_embedding_cache_dir == ".cache/paper_embeddings"


def test_agent_schema_objects():
    candidate = CandidatePaper(
        paper_id="p1",
        title="Graph-based idea generation",
        abstract="We study graph-guided idea generation.",
        cited_by_count=10,
        score=0.91,
    )
    idea = IdeaOutput(
        title="Idea title",
        summary="Idea summary",
        problem="Target problem",
        method="Proposed method",
        core_paper_id="p1",
        core_paper_title="Graph-based idea generation",
    )

    assert candidate.paper_id == "p1"
    assert idea.core_paper_id == "p1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_settings_env.py -q`
Expected: FAIL with missing `ollama_embed_model`, `agent_retrieval_top_k`, or missing `src.schemas.agent`

- [ ] **Step 3: Add minimal schema, settings, and dependency changes**

```python
# src/schemas/agent.py
from pydantic import BaseModel, Field


class CandidatePaper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    cited_by_count: int | None = None
    score: float = 0.0


class IdeaOutput(BaseModel):
    title: str
    summary: str
    problem: str
    method: str
    core_paper_id: str
    core_paper_title: str
```

```python
# src/config/settings.py
ollama_embed_model: str = Field(validation_alias="OLLAMA_EMBED_MODEL")
agent_retrieval_top_k: int = Field(default=5, validation_alias="AGENT_RETRIEVAL_TOP_K")
paper_embedding_cache_dir: str = Field(default=".cache/paper_embeddings", validation_alias="PAPER_EMBEDDING_CACHE_DIR")
```

```toml
# pyproject.toml
dependencies = [
    "httpx>=0.27.0",
    "langgraph>=0.2.0",
    "neo4j>=5.20.0",
    "ollama>=0.4.0",
    "pydantic>=2.7.0",
    "pydantic-settings>=2.3.0",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_settings_env.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/config/settings.py src/schemas/agent.py tests/test_settings_env.py
git commit -m "feat: 增加 agent 配置与数据契约"
```

## Task 2: Add Paper Read Access and Embedding Retrieval

**Files:**
- Create: `src/models/ollama_embeddings.py`
- Create: `src/retrieval/embedding_cache.py`
- Create: `src/retrieval/base.py`
- Create: `src/retrieval/vector_retriever.py`
- Modify: `src/storage/graph/neo4j_repository.py`
- Create: `tests/test_embedding_cache.py`
- Create: `tests/test_vector_retriever.py`
- Modify: `tests/test_neo4j_repository.py`

- [ ] **Step 1: Write the failing repository and retriever tests**

```python
from src.retrieval.embedding_cache import PaperEmbeddingCache
from src.retrieval.vector_retriever import VectorPaperRetriever
from src.schemas.agent import CandidatePaper
from src.schemas.document import PaperDocument


class StubEmbedder:
    def embed_text(self, text: str) -> list[float]:
        vectors = {
            "graph idea": [1.0, 0.0],
            "paper one abstract": [0.9, 0.1],
            "paper two abstract": [0.1, 0.9],
        }
        return vectors[text]


class StubRepository:
    def list_papers_with_abstracts(self):
        return [
            PaperDocument(
                document_id="p1",
                source_type="openalex",
                source_ref="p1",
                title="Paper One",
                abstract="paper one abstract",
                year=2024,
                cited_by_count=10,
                ingestion_status="ready",
            ),
            PaperDocument(
                document_id="p2",
                source_type="openalex",
                source_ref="p2",
                title="Paper Two",
                abstract="paper two abstract",
                year=2024,
                cited_by_count=20,
                ingestion_status="ready",
            ),
        ]


def test_embedding_cache_round_trip(tmp_path):
    cache = PaperEmbeddingCache(cache_dir=tmp_path, model_name="qwen3-embedding:4b")

    assert cache.load("p1") is None

    cache.save("p1", "Paper One", "paper one abstract", [0.9, 0.1])

    loaded = cache.load("p1")

    assert loaded is not None
    assert loaded["paper_id"] == "p1"
    assert loaded["embedding"] == [0.9, 0.1]


def test_vector_retriever_returns_top_k_sorted_by_score():
    retriever = VectorPaperRetriever(
        repository=StubRepository(),
        embedder=StubEmbedder(),
        cache=None,
        top_k=1,
    )

    results = retriever.retrieve("graph idea")

    assert len(results) == 1
    assert isinstance(results[0], CandidatePaper)
    assert results[0].paper_id == "p1"
    assert results[0].score > 0
```

```python
def test_repository_lists_only_papers_with_non_empty_abstract():
    records = [
        {"id": "p1", "title": "A", "abstract": "Alpha", "year": 2024, "cited_by_count": 3},
        {"id": "p2", "title": "B", "abstract": "", "year": 2024, "cited_by_count": 4},
    ]
    # fake driver assertion should ensure Cypher filters out empty abstracts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_vector_retriever.py tests/test_neo4j_repository.py -q`
Expected: FAIL with missing cache/retriever/embedder/repository read method

- [ ] **Step 3: Implement repository read path and embedding retriever**

```python
# src/models/ollama_embeddings.py
from ollama import Client


class OllamaEmbeddingModel:
    def __init__(self, base_url: str, model_name: str, timeout: float = 300.0) -> None:
        self._client = Client(host=base_url.rstrip("/"), timeout=timeout)
        self._model_name = model_name

    def embed_text(self, text: str) -> list[float]:
        body = self._client.embed(model=self._model_name, input=text)
        return body["embeddings"][0]
```

```python
# src/retrieval/embedding_cache.py
import hashlib
import json
from pathlib import Path


class PaperEmbeddingCache:
    def __init__(self, cache_dir: str | Path, model_name: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._model_name = model_name

    def load(self, paper_id: str):
        path = self._path_for(paper_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, paper_id: str, title: str, abstract: str, embedding: list[float]) -> None:
        path = self._path_for(paper_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "paper_id": paper_id,
            "model": self._model_name,
            "title": title,
            "abstract": abstract,
            "embedding": embedding,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _path_for(self, paper_id: str) -> Path:
        digest = hashlib.sha256(f"{self._model_name}:{paper_id}".encode("utf-8")).hexdigest()
        return self._cache_dir / self._model_name.replace("/", "_") / f"{digest}.json"
```

```python
# src/retrieval/base.py
class PaperRetriever:
    def retrieve(self, query: str):
        raise NotImplementedError


class PaperSelector:
    def select(self, user_query: str, candidates):
        raise NotImplementedError
```

```python
# src/retrieval/vector_retriever.py
from math import sqrt

from src.schemas.agent import CandidatePaper


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class VectorPaperRetriever:
    def __init__(self, repository, embedder, cache, top_k: int = 5) -> None:
        self._repository = repository
        self._embedder = embedder
        self._cache = cache
        self._top_k = top_k

    def retrieve(self, query: str) -> list[CandidatePaper]:
        query_vector = self._embedder.embed_text(query)
        candidates = []
        for paper in self._repository.list_papers_with_abstracts():
            cached = self._cache.load(paper.document_id) if self._cache else None
            if cached is None:
                paper_vector = self._embedder.embed_text(paper.abstract)
                if self._cache:
                    self._cache.save(paper.document_id, paper.title, paper.abstract, paper_vector)
            else:
                paper_vector = cached["embedding"]
            score = cosine_similarity(query_vector, paper_vector)
            candidates.append(
                CandidatePaper(
                    paper_id=paper.document_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    cited_by_count=paper.cited_by_count,
                    score=score,
                )
            )
        return sorted(candidates, key=lambda item: item.score, reverse=True)[: self._top_k]
```

```python
# src/storage/graph/neo4j_repository.py
def list_papers_with_abstracts(self):
    query = """
    MATCH (p:Paper)
    WHERE p.abstract IS NOT NULL AND trim(p.abstract) <> ""
    RETURN p.id AS id,
           p.title AS title,
           p.abstract AS abstract,
           p.year AS year,
           p.cited_by_count AS cited_by_count
    """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_embedding_cache.py tests/test_vector_retriever.py tests/test_neo4j_repository.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/ollama_embeddings.py src/retrieval/embedding_cache.py src/retrieval/base.py src/retrieval/vector_retriever.py src/storage/graph/neo4j_repository.py tests/test_embedding_cache.py tests/test_vector_retriever.py tests/test_neo4j_repository.py
git commit -m "feat: 增加论文向量缓存与召回能力"
```

## Task 3: Add Selector, Agent State, and LangGraph Nodes

**Files:**
- Create: `src/retrieval/selector.py`
- Create: `src/agent/state.py`
- Create: `src/agent/nodes/query_understanding.py`
- Create: `src/agent/nodes/paper_selection.py`
- Create: `src/agent/nodes/idea_generation.py`
- Modify: `src/models/ollama.py`
- Create: `tests/test_agent_nodes.py`

- [ ] **Step 1: Write the failing node tests**

```python
from src.agent.nodes.paper_selection import select_core_paper
from src.agent.nodes.query_understanding import build_retrieval_query
from src.retrieval.selector import TopScorePaperSelector
from src.schemas.agent import CandidatePaper


def test_query_understanding_node_sets_retrieval_query():
    state = {"user_query": "graph idea generation"}

    new_state = build_retrieval_query(state)

    assert new_state["retrieval_query"] == "graph idea generation"


def test_top_score_selector_picks_first_candidate():
    selector = TopScorePaperSelector()
    selected = selector.select(
        "query",
        [
            CandidatePaper(paper_id="p1", title="A", abstract="aa", score=0.9),
            CandidatePaper(paper_id="p2", title="B", abstract="bb", score=0.5),
        ],
    )

    assert selected.paper_id == "p1"
```

```python
def test_idea_generation_node_returns_structured_output():
    class StubModel:
        def generate_messages_json(self, system_prompt: str, user_prompt: str) -> dict:
            return {
                "title": "Idea title",
                "summary": "Idea summary",
                "problem": "Problem",
                "method": "Method",
            }

    state = {
        "user_query": "graph idea generation",
        "selected_paper": {
            "paper_id": "p1",
            "title": "Paper A",
            "abstract": "Paper abstract",
            "score": 0.9,
        },
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_agent_nodes.py -q`
Expected: FAIL with missing node and selector modules

- [ ] **Step 3: Implement selector, state, node helpers, and reusable model API**

```python
# src/retrieval/selector.py
class TopScorePaperSelector:
    def select(self, user_query: str, candidates):
        if not candidates:
            raise ValueError("no candidate papers available")
        return candidates[0]
```

```python
# src/agent/state.py
from typing import TypedDict


class AgentIdeaState(TypedDict, total=False):
    user_query: str
    retrieval_query: str
    candidate_papers: list
    selected_paper: dict
    idea_output: dict
    errors: list[str]
```

```python
# src/agent/nodes/query_understanding.py
def build_retrieval_query(state):
    return {"retrieval_query": state["user_query"].strip()}
```

```python
# src/agent/nodes/paper_selection.py
def select_core_paper(state, selector):
    selected = selector.select(state["user_query"], state["candidate_papers"])
    return {"selected_paper": selected.model_dump()}
```

```python
# src/models/ollama.py
def generate_messages_json(self, system_prompt: str, user_prompt: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    ...
```

```python
# src/agent/nodes/idea_generation.py
from src.schemas.agent import IdeaOutput


def generate_idea(state, model):
    paper = state["selected_paper"]
    user_prompt = (
        f"User question: {state['user_query']}\n\n"
        f"Core paper title: {paper['title']}\n"
        f"Core paper abstract: {paper['abstract']}\n\n"
        "Return JSON only."
    )
    raw = model.generate_messages_json(SYSTEM_PROMPT, user_prompt)
    idea = IdeaOutput(
        title=raw["title"],
        summary=raw["summary"],
        problem=raw["problem"],
        method=raw["method"],
        core_paper_id=paper["paper_id"],
        core_paper_title=paper["title"],
    )
    return {"idea_output": idea.model_dump()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent_nodes.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/selector.py src/agent/state.py src/agent/nodes/query_understanding.py src/agent/nodes/paper_selection.py src/agent/nodes/idea_generation.py src/models/ollama.py tests/test_agent_nodes.py
git commit -m "feat: 增加 agent 节点与论文选择逻辑"
```

## Task 4: Wire LangGraph and Workflow Entry

**Files:**
- Create: `src/agent/graph.py`
- Create: `src/workflows/idea_chat.py`
- Modify: `src/app/bootstrap.py`
- Create: `tests/test_idea_chat_workflow.py`

- [ ] **Step 1: Write the failing workflow tests**

```python
from src.workflows.idea_chat import IdeaChatWorkflow
from src.schemas.agent import CandidatePaper


class StubRetriever:
    def retrieve(self, query: str):
        return [
            CandidatePaper(
                paper_id="p1",
                title="Paper A",
                abstract="Abstract A",
                cited_by_count=11,
                score=0.95,
            )
        ]


class StubSelector:
    def select(self, user_query: str, candidates):
        return candidates[0]


class StubModel:
    def generate_messages_json(self, system_prompt: str, user_prompt: str) -> dict:
        return {
            "title": "Idea title",
            "summary": "Idea summary",
            "problem": "Problem",
            "method": "Method",
        }


def test_idea_chat_workflow_returns_structured_result():
    workflow = IdeaChatWorkflow(
        retriever=StubRetriever(),
        selector=StubSelector(),
        model=StubModel(),
    )

    result = workflow.run("graph idea generation")

    assert result["idea_output"]["core_paper_id"] == "p1"
    assert result["idea_output"]["title"] == "Idea title"
```

```python
def test_idea_chat_workflow_returns_error_when_no_candidates():
    class EmptyRetriever:
        def retrieve(self, query: str):
            return []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_idea_chat_workflow.py -q`
Expected: FAIL with missing workflow/graph modules

- [ ] **Step 3: Implement LangGraph wiring and workflow wrapper**

```python
# src/agent/graph.py
from langgraph.graph import END, StateGraph

from src.agent.state import AgentIdeaState


def build_agent_graph(retriever, selector, model):
    graph = StateGraph(AgentIdeaState)
    graph.add_node("query_understanding", build_retrieval_query)
    graph.add_node("paper_retrieval", lambda state: {"candidate_papers": [p.model_dump() for p in retriever.retrieve(state["retrieval_query"])]})
    graph.add_node("paper_selection", lambda state: select_core_paper(state, selector))
    graph.add_node("idea_generation", lambda state: generate_idea(state, model))
    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "paper_retrieval")
    graph.add_edge("paper_retrieval", "paper_selection")
    graph.add_edge("paper_selection", "idea_generation")
    graph.add_edge("idea_generation", END)
    return graph.compile()
```

```python
# src/workflows/idea_chat.py
class IdeaChatWorkflow:
    def __init__(self, retriever, selector, model, log=None) -> None:
        self._graph = build_agent_graph(retriever, selector, model)
        self._log = log or logger.info

    def run(self, user_query: str) -> dict:
        result = self._graph.invoke({"user_query": user_query, "errors": []})
        self._log("agent:workflow_done query=%s", user_query)
        return result
```

```python
# src/app/bootstrap.py
def build_idea_chat_workflow(settings: Settings) -> IdeaChatWorkflow:
    repository = Neo4jGraphRepository.from_connection(...)
    embedder = OllamaEmbeddingModel(...)
    cache = PaperEmbeddingCache(cache_dir=settings.paper_embedding_cache_dir, model_name=settings.ollama_embed_model)
    retriever = VectorPaperRetriever(repository=repository, embedder=embedder, cache=cache, top_k=settings.agent_retrieval_top_k)
    selector = TopScorePaperSelector()
    model = OllamaJsonModel(...)
    return IdeaChatWorkflow(retriever=retriever, selector=selector, model=model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_idea_chat_workflow.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/graph.py src/workflows/idea_chat.py src/app/bootstrap.py tests/test_idea_chat_workflow.py
git commit -m "feat: 接入 LangGraph idea chat workflow"
```

## Task 5: Add CLI Entry, Logging, and Documentation

**Files:**
- Create: `src/interfaces/cli/chat_idea.py`
- Modify: `src/interfaces/cli/build_graph.py`
- Modify: `main.py`
- Modify: `README.md`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Write the failing CLI tests**

```python
from src.interfaces.cli.build_graph import build_parser


def test_parser_supports_chat_idea_command():
    parser = build_parser()

    args = parser.parse_args(["chat-idea", "--query", "graph idea generation"])

    assert args.command == "chat-idea"
    assert args.query == "graph idea generation"
```

```python
def test_main_routes_chat_idea(monkeypatch):
    called = {}

    def fake_run_chat_idea(args):
        called["query"] = args.query

    monkeypatch.setattr("main.run_chat_idea", fake_run_chat_idea)
    monkeypatch.setattr("sys.argv", ["main.py", "chat-idea", "--query", "graph idea generation"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_cli.py tests/test_main.py -q`
Expected: FAIL because `chat-idea` is not supported yet

- [ ] **Step 3: Implement CLI command and documentation**

```python
# src/interfaces/cli/build_graph.py
chat = subparsers.add_parser("chat-idea")
chat.add_argument("--query", required=True)
```

```python
# main.py
from src.app.bootstrap import build_idea_chat_workflow


def run_chat_idea(args) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    workflow = build_idea_chat_workflow(settings)
    result = workflow.run(user_query=args.query)
    logger.info("chat_idea:summary %s", result)


if args.command == "chat-idea":
    run_chat_idea(args)
```

```markdown
# README.md
## Agent Idea Chat

```bash
uv run python main.py chat-idea --query "graph-based scientific idea generation for low-resource domains"
```
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cli.py tests/test_main.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/interfaces/cli/build_graph.py main.py README.md tests/test_cli.py tests/test_main.py
git commit -m "feat: 增加对话式 idea 生成命令入口"
```

## Task 6: Final Verification and Environment Template

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `src/config/settings.py`

- [ ] **Step 1: Add missing environment variables to the template**

```env
OLLAMA_EMBED_MODEL=nomic-embed-text
AGENT_RETRIEVAL_TOP_K=5
PAPER_EMBEDDING_CACHE_DIR=.cache/paper_embeddings
```

- [ ] **Step 2: Run targeted tests for the new agent path**

Run: `python3 -m pytest tests/test_settings_env.py tests/test_embedding_cache.py tests/test_vector_retriever.py tests/test_agent_nodes.py tests/test_idea_chat_workflow.py tests/test_cli.py tests/test_main.py -q`
Expected: PASS

- [ ] **Step 3: Run the full test suite**

Run: `python3 -m pytest -q`
Expected: PASS

- [ ] **Step 4: Manual smoke test against a populated database**

Run:

```bash
uv run python main.py chat-idea --query "graph-based scientific idea generation for low-resource domains"
```

Expected:
- logs include `agent:workflow_done`
- output contains one selected core paper
- output contains `title`, `summary`, `problem`, and `method`

- [ ] **Step 5: Commit**

```bash
git add .env.example README.md
git commit -m "chore: 补齐 agent idea chat 配置说明"
```

## Self-Review

Spec coverage check:

- LangGraph 4 节点链路：Task 3, Task 4
- 摘要向量召回：Task 2
- 核心论文选择：Task 3
- idea 生成：Task 3, Task 4
- CLI / workflow 入口：Task 5
- 配置与文档：Task 1, Task 6

Placeholder scan:

- No `TODO`, `TBD`, or deferred implementation placeholders remain in the plan.

Type consistency:

- Retrieval returns `CandidatePaper`
- Selector consumes `CandidatePaper`
- Workflow state stores serialized `CandidatePaper` in `selected_paper`
- `IdeaOutput` always carries `core_paper_id` and `core_paper_title`
