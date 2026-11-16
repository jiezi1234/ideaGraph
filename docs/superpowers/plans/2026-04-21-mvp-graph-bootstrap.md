# MVP Graph Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first end-to-end MVP that searches seed papers from OpenAlex by keyword, filters/selects papers with usable citation metadata, extracts entities/relations from title and abstract, expands one hop into cited papers, and writes the resulting paper graph plus entity graph into a configurable standalone Neo4j database.

**Architecture:** Keep the implementation as a single Python application with clear internal boundaries: OpenAlex ingestion, model-based extraction, graph persistence, and one orchestration workflow. Use document-level processing only; no chunking, no recursive citation crawling beyond one hop, and skip network/API failures instead of blocking the run.

**Tech Stack:** Python 3.12, `httpx`, `pydantic`, `neo4j`, `pytest`, configurable LLM provider wrapper, CLI entrypoint

---

## File Map

### Create

- `src/config/settings.py`
- `src/schemas/document.py`
- `src/schemas/entity.py`
- `src/schemas/relation.py`
- `src/schemas/extraction.py`
- `src/ingestion/openalex_client.py`
- `src/ingestion/openalex_service.py`
- `src/models/base.py`
- `src/models/provider.py`
- `src/prompts/extract_paper_entities.txt`
- `src/extraction/paper_extractor.py`
- `src/knowledge/normalizer.py`
- `src/storage/graph/neo4j_repository.py`
- `src/workflows/mvp_graph_build.py`
- `src/interfaces/cli/build_graph.py`
- `tests/test_openalex_client.py`
- `tests/test_openalex_service.py`
- `tests/test_paper_extractor.py`
- `tests/test_normalizer.py`
- `tests/test_neo4j_repository.py`
- `tests/test_mvp_graph_build.py`

### Modify

- `pyproject.toml`
- `README.md`
- `main.py`

## Task 1: Scaffold Settings, Schemas, and Dependencies

**Files:**
- Create: `src/config/settings.py`
- Create: `src/schemas/document.py`
- Create: `src/schemas/entity.py`
- Create: `src/schemas/relation.py`
- Create: `src/schemas/extraction.py`
- Modify: `pyproject.toml`
- Test: `tests/test_normalizer.py`

- [ ] **Step 1: Write the failing schema/config test**

```python
from src.config.settings import Settings
from src.schemas.document import PaperDocument
from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation


def test_settings_and_core_schemas_round_trip():
    settings = Settings(
        openalex_base_url="https://api.openalex.org",
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="secret",
        neo4j_database="idea_graph_mvp",
        extraction_model="gpt-5.4-mini",
    )

    doc = PaperDocument(
        document_id="doc-1",
        source_type="openalex",
        source_ref="https://openalex.org/W123",
        title="Graph-based idea generation",
        abstract="We study graph-guided idea generation.",
        year=2025,
        citation_openalex_ids=["W999"],
        ingestion_status="ready",
    )

    entity = ExtractedEntity(
        entity_id="graph reasoning",
        name="graph reasoning",
        entity_type="Technology",
        source_refs=["doc-1"],
    )

    relation = ExtractedRelation(
        source_entity_id="graph reasoning",
        target_entity_id="idea generation",
        relation_type="related_to",
        source_refs=["doc-1"],
    )

    assert settings.neo4j_database == "idea_graph_mvp"
    assert doc.citation_openalex_ids == ["W999"]
    assert entity.entity_type == "Technology"
    assert relation.relation_type == "related_to"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalizer.py::test_settings_and_core_schemas_round_trip -v`

Expected: FAIL with import errors because the files do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# src/config/settings.py
from pydantic import BaseModel


class Settings(BaseModel):
    openalex_base_url: str
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str
    extraction_model: str
```

```python
# src/schemas/document.py
from pydantic import BaseModel, Field


class PaperDocument(BaseModel):
    document_id: str
    source_type: str
    source_ref: str
    title: str
    abstract: str
    year: int | None = None
    citation_openalex_ids: list[str] = Field(default_factory=list)
    ingestion_status: str
```

```python
# src/schemas/entity.py
from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    entity_id: str
    name: str
    entity_type: str
    source_refs: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    score: float | None = None
```

```python
# src/schemas/relation.py
from pydantic import BaseModel, Field


class ExtractedRelation(BaseModel):
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    source_refs: list[str] = Field(default_factory=list)
    score: float | None = None
    attributes: dict[str, str | int | float] = Field(default_factory=dict)


class PaperCitationRelation(BaseModel):
    source_paper_id: str
    target_paper_id: str
    relation_type: str = "CITES"
```

```python
# src/schemas/extraction.py
from pydantic import BaseModel, Field

from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation


class ExtractionResult(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
```

```toml
# pyproject.toml
[project]
name = "ideagraph"
version = "0.1.0"
description = "IdeaGraph MVP"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "httpx>=0.27.0",
  "neo4j>=5.20.0",
  "pydantic>=2.7.0",
]

[dependency-groups]
dev = [
  "pytest>=8.2.0",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalizer.py::test_settings_and_core_schemas_round_trip -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/config/settings.py src/schemas/document.py src/schemas/entity.py src/schemas/relation.py src/schemas/extraction.py tests/test_normalizer.py
git commit -m "feat: add core mvp settings and schemas"
```

## Task 2: Add OpenAlex Search Client with Safe Network Failure Handling

**Files:**
- Create: `src/ingestion/openalex_client.py`
- Test: `tests/test_openalex_client.py`

- [ ] **Step 1: Write the failing client tests**

```python
import httpx

from src.ingestion.openalex_client import OpenAlexClient


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_search_works_filters_items_without_reference_list(monkeypatch):
    payload = {
        "results": [
            {"id": "https://openalex.org/W1", "title": "A", "referenced_works": ["https://openalex.org/W2"]},
            {"id": "https://openalex.org/W3", "title": "B", "referenced_works": []},
        ]
    }

    def fake_get(self, url, params):
        return DummyResponse(payload)

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org")

    results = client.search_works("idea graph", must_have_references=True, per_page=10)

    assert [item["id"] for item in results] == ["https://openalex.org/W1"]


def test_search_skips_network_errors(monkeypatch):
    def fake_get(self, url, params):
        raise httpx.ReadTimeout("boom")

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    client = OpenAlexClient(base_url="https://api.openalex.org")

    assert client.search_works("idea graph", must_have_references=True, per_page=10) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_openalex_client.py -v`

Expected: FAIL with import errors because `OpenAlexClient` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ingestion/openalex_client.py
from __future__ import annotations

import httpx


class OpenAlexClient:
    def __init__(self, base_url: str, timeout: float = 20.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def search_works(self, keyword: str, must_have_references: bool, per_page: int) -> list[dict]:
        params = {
            "search": keyword,
            "per-page": per_page,
        }
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(f"{self._base_url}/works", params=params)
                response.raise_for_status()
        except httpx.HTTPError:
            return []

        items = response.json().get("results", [])
        if must_have_references:
            items = [item for item in items if item.get("referenced_works")]
        return items

    def get_work(self, openalex_id: str) -> dict | None:
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(openalex_id, params={})
                response.raise_for_status()
        except httpx.HTTPError:
            return None
        return response.json()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_openalex_client.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/openalex_client.py tests/test_openalex_client.py
git commit -m "feat: add openalex client with safe failure handling"
```

## Task 3: Add OpenAlex Paper Selection and One-Hop Expansion Service

**Files:**
- Create: `src/ingestion/openalex_service.py`
- Test: `tests/test_openalex_service.py`

- [ ] **Step 1: Write the failing service tests**

```python
from src.ingestion.openalex_service import OpenAlexPaperService


class FakeClient:
    def search_works(self, keyword, must_have_references, per_page):
        return [
            {
                "id": "https://openalex.org/W1",
                "title": "Seed Paper",
                "abstract_inverted_index": {"Seed": [0], "paper": [1]},
                "publication_year": 2024,
                "referenced_works": ["https://openalex.org/W2"],
            }
        ]

    def get_work(self, openalex_id):
        return {
            "id": openalex_id,
            "title": "Cited Paper",
            "abstract_inverted_index": {"Cited": [0], "paper": [1]},
            "publication_year": 2023,
            "referenced_works": [],
        }


def test_service_returns_seed_and_one_hop_cited_papers():
    service = OpenAlexPaperService(FakeClient())

    seed_docs, cited_docs, paper_links = service.fetch_seed_and_citations(
        keyword="idea graph",
        max_seed_papers=1,
        require_references=True,
    )

    assert [doc.document_id for doc in seed_docs] == ["https://openalex.org/W1"]
    assert [doc.document_id for doc in cited_docs] == ["https://openalex.org/W2"]
    assert paper_links[0].source_paper_id == "https://openalex.org/W1"
    assert paper_links[0].target_paper_id == "https://openalex.org/W2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_openalex_service.py -v`

Expected: FAIL because `OpenAlexPaperService` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ingestion/openalex_service.py
from __future__ import annotations

from src.schemas.document import PaperDocument
from src.schemas.relation import PaperCitationRelation


def _restore_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    tokens = [""] * (max_pos + 1)
    for token, positions in inverted_index.items():
        for pos in positions:
            tokens[pos] = token
    return " ".join(tokens).strip()


class OpenAlexPaperService:
    def __init__(self, client) -> None:
        self._client = client

    def fetch_seed_and_citations(self, keyword: str, max_seed_papers: int, require_references: bool):
        raw_seeds = self._client.search_works(keyword, must_have_references=require_references, per_page=max_seed_papers)
        seed_docs = [self._to_document(item) for item in raw_seeds[:max_seed_papers]]

        cited_docs: list[PaperDocument] = []
        paper_links: list[PaperCitationRelation] = []
        seen_cited_ids: set[str] = set()

        for seed in raw_seeds[:max_seed_papers]:
            for cited_id in seed.get("referenced_works", []):
                paper_links.append(PaperCitationRelation(source_paper_id=seed["id"], target_paper_id=cited_id))
                if cited_id in seen_cited_ids:
                    continue
                cited_raw = self._client.get_work(cited_id)
                if not cited_raw:
                    continue
                cited_docs.append(self._to_document(cited_raw))
                seen_cited_ids.add(cited_id)

        return seed_docs, cited_docs, paper_links

    def _to_document(self, item: dict) -> PaperDocument:
        return PaperDocument(
            document_id=item["id"],
            source_type="openalex",
            source_ref=item["id"],
            title=item.get("title") or "",
            abstract=_restore_abstract(item.get("abstract_inverted_index")),
            year=item.get("publication_year"),
            citation_openalex_ids=item.get("referenced_works", []),
            ingestion_status="ready",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_openalex_service.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/openalex_service.py tests/test_openalex_service.py
git commit -m "feat: add one-hop openalex paper expansion service"
```

## Task 4: Add Model Abstraction and Single-Call Paper Extraction

**Files:**
- Create: `src/models/base.py`
- Create: `src/models/provider.py`
- Create: `src/prompts/extract_paper_entities.txt`
- Create: `src/extraction/paper_extractor.py`
- Test: `tests/test_paper_extractor.py`

- [ ] **Step 1: Write the failing extractor test**

```python
from src.extraction.paper_extractor import PaperExtractor
from src.schemas.document import PaperDocument


class FakeModel:
    def generate_json(self, prompt: str, payload: dict) -> dict:
        assert payload["title"] == "Seed Paper"
        assert payload["abstract"] == "Graph methods improve idea generation."
        return {
            "entities": [
                {"entity_id": "graph methods", "name": "graph methods", "entity_type": "Technology"},
                {"entity_id": "idea generation", "name": "idea generation", "entity_type": "Problem"},
            ],
            "relations": [
                {
                    "source_entity_id": "graph methods",
                    "target_entity_id": "idea generation",
                    "relation_type": "related_to",
                }
            ],
        }


def test_extractor_uses_one_model_call_per_paper():
    extractor = PaperExtractor(FakeModel(), prompt_template="extract paper entities")
    doc = PaperDocument(
        document_id="doc-1",
        source_type="openalex",
        source_ref="https://openalex.org/W1",
        title="Seed Paper",
        abstract="Graph methods improve idea generation.",
        ingestion_status="ready",
    )

    result = extractor.extract(doc)

    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert result.relations[0].relation_type == "related_to"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_paper_extractor.py -v`

Expected: FAIL because extractor and model abstractions do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# src/models/base.py
from typing import Protocol


class JsonModel(Protocol):
    def generate_json(self, prompt: str, payload: dict) -> dict: ...
```

```python
# src/models/provider.py
class ModelProvider:
    def __init__(self, models: dict[str, object]) -> None:
        self._models = models

    def get(self, name: str):
        return self._models[name]
```

```text
# src/prompts/extract_paper_entities.txt
Extract entities and relations from one paper title and abstract.
Return JSON with keys "entities" and "relations".
Use a single response for the whole paper.
```

```python
# src/extraction/paper_extractor.py
from src.schemas.entity import ExtractedEntity
from src.schemas.extraction import ExtractionResult
from src.schemas.relation import ExtractedRelation


class PaperExtractor:
    def __init__(self, model, prompt_template: str) -> None:
        self._model = model
        self._prompt_template = prompt_template

    def extract(self, document) -> ExtractionResult:
        payload = {
            "title": document.title,
            "abstract": document.abstract,
        }
        raw = self._model.generate_json(self._prompt_template, payload)
        entities = [
            ExtractedEntity(**item, source_refs=[document.document_id])
            for item in raw.get("entities", [])
        ]
        relations = [
            ExtractedRelation(**item, source_refs=[document.document_id])
            for item in raw.get("relations", [])
        ]
        return ExtractionResult(entities=entities, relations=relations)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_paper_extractor.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/base.py src/models/provider.py src/prompts/extract_paper_entities.txt src/extraction/paper_extractor.py tests/test_paper_extractor.py
git commit -m "feat: add single-call paper extraction flow"
```

## Task 5: Add Entity Merge and Relation Cleanup Logic

**Files:**
- Create: `src/knowledge/normalizer.py`
- Test: `tests/test_normalizer.py`

- [ ] **Step 1: Write the failing normalizer tests**

```python
from src.knowledge.normalizer import merge_entities, normalize_relations
from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation


def test_merge_entities_merges_same_name_case_insensitively():
    entities = [
        ExtractedEntity(entity_id="Graph Reasoning", name="Graph Reasoning", entity_type="Technology", source_refs=["d1"]),
        ExtractedEntity(entity_id="graph reasoning", name="graph reasoning", entity_type="Technology", source_refs=["d2"]),
    ]

    merged = merge_entities(entities)

    assert len(merged) == 1
    assert merged[0].entity_id == "graph reasoning"
    assert sorted(merged[0].source_refs) == ["d1", "d2"]


def test_normalize_relations_rewrites_ids_to_merged_entity_ids():
    relations = [
        ExtractedRelation(
            source_entity_id="Graph Reasoning",
            target_entity_id="Idea Generation",
            relation_type="related_to",
            source_refs=["d1"],
        )
    ]
    id_map = {
        "Graph Reasoning": "graph reasoning",
        "Idea Generation": "idea generation",
    }

    normalized = normalize_relations(relations, id_map)

    assert normalized[0].source_entity_id == "graph reasoning"
    assert normalized[0].target_entity_id == "idea generation"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalizer.py -v`

Expected: FAIL because normalizer functions do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# src/knowledge/normalizer.py
from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def merge_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    merged: dict[str, ExtractedEntity] = {}
    for entity in entities:
        normalized_id = _normalize_name(entity.name)
        if normalized_id not in merged:
            merged[normalized_id] = entity.model_copy(
                update={
                    "entity_id": normalized_id,
                    "name": normalized_id,
                    "aliases": [entity.name] if entity.name != normalized_id else [],
                }
            )
            continue
        current = merged[normalized_id]
        current.source_refs = sorted(set(current.source_refs + entity.source_refs))
        current.aliases = sorted(set(current.aliases + [entity.name]))
    return list(merged.values())


def normalize_relations(relations: list[ExtractedRelation], id_map: dict[str, str]) -> list[ExtractedRelation]:
    normalized: list[ExtractedRelation] = []
    for relation in relations:
        normalized.append(
            relation.model_copy(
                update={
                    "source_entity_id": id_map.get(relation.source_entity_id, _normalize_name(relation.source_entity_id)),
                    "target_entity_id": id_map.get(relation.target_entity_id, _normalize_name(relation.target_entity_id)),
                }
            )
        )
    return normalized
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalizer.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/knowledge/normalizer.py tests/test_normalizer.py
git commit -m "feat: add entity merge and relation normalization"
```

## Task 6: Add Neo4j Repository with Configurable Database Target

**Files:**
- Create: `src/storage/graph/neo4j_repository.py`
- Test: `tests/test_neo4j_repository.py`

- [ ] **Step 1: Write the failing repository tests**

```python
from src.schemas.document import PaperDocument
from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation, PaperCitationRelation
from src.storage.graph.neo4j_repository import Neo4jGraphRepository


class FakeDriver:
    def __init__(self):
        self.calls = []

    def execute_query(self, query, parameters_=None, database_=None):
        self.calls.append((query, parameters_, database_))


def test_repository_targets_configured_database():
    driver = FakeDriver()
    repo = Neo4jGraphRepository(driver=driver, database="idea_graph_mvp")
    repo.upsert_papers(
        [PaperDocument(
            document_id="p1",
            source_type="openalex",
            source_ref="https://openalex.org/W1",
            title="Seed Paper",
            abstract="A",
            ingestion_status="ready",
        )]
    )

    assert driver.calls[0][2] == "idea_graph_mvp"


def test_repository_writes_paper_and_entity_edges():
    driver = FakeDriver()
    repo = Neo4jGraphRepository(driver=driver, database="idea_graph_mvp")
    repo.upsert_citations([PaperCitationRelation(source_paper_id="p1", target_paper_id="p2")])
    repo.upsert_entity_relations([
        ExtractedRelation(
            source_entity_id="graph reasoning",
            target_entity_id="idea generation",
            relation_type="related_to",
            source_refs=["p1"],
        )
    ])

    rendered = "\n".join(call[0] for call in driver.calls)
    assert "CITES" in rendered
    assert "RELATED_TO" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_neo4j_repository.py -v`

Expected: FAIL because repository does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# src/storage/graph/neo4j_repository.py
from neo4j import GraphDatabase


class Neo4jGraphRepository:
    def __init__(self, driver, database: str) -> None:
        self._driver = driver
        self._database = database

    @classmethod
    def from_connection(cls, uri: str, username: str, password: str, database: str):
        driver = GraphDatabase.driver(uri, auth=(username, password))
        return cls(driver=driver, database=database)

    def upsert_papers(self, papers) -> None:
        query = """
        UNWIND $papers AS paper
        MERGE (p:Paper {id: paper.document_id})
        SET p.title = paper.title,
            p.abstract = paper.abstract,
            p.source_type = paper.source_type,
            p.source_ref = paper.source_ref,
            p.year = paper.year
        """
        payload = [paper.model_dump() for paper in papers]
        self._driver.execute_query(query, parameters_={"papers": payload}, database_=self._database)

    def upsert_entities(self, entities) -> None:
        query = """
        UNWIND $entities AS entity
        MERGE (e:Entity {id: entity.entity_id})
        SET e.name = entity.name,
            e.entity_type = entity.entity_type,
            e.aliases = entity.aliases
        """
        payload = [entity.model_dump() for entity in entities]
        self._driver.execute_query(query, parameters_={"entities": payload}, database_=self._database)

    def upsert_citations(self, citations) -> None:
        query = """
        UNWIND $citations AS citation
        MATCH (src:Paper {id: citation.source_paper_id})
        MATCH (dst:Paper {id: citation.target_paper_id})
        MERGE (src)-[:CITES]->(dst)
        """
        payload = [citation.model_dump() for citation in citations]
        self._driver.execute_query(query, parameters_={"citations": payload}, database_=self._database)

    def upsert_entity_relations(self, relations) -> None:
        for relation in relations:
            rel_type = relation.relation_type.upper()
            query = f"""
            MATCH (src:Entity {{id: $source_entity_id}})
            MATCH (dst:Entity {{id: $target_entity_id}})
            MERGE (src)-[r:{rel_type}]->(dst)
            SET r.source_refs = $source_refs
            """
            self._driver.execute_query(
                query,
                parameters_=relation.model_dump(),
                database_=self._database,
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_neo4j_repository.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/storage/graph/neo4j_repository.py tests/test_neo4j_repository.py
git commit -m "feat: add neo4j graph repository for mvp graph write"
```

## Task 7: Orchestrate End-to-End Graph Build Workflow

**Files:**
- Create: `src/workflows/mvp_graph_build.py`
- Test: `tests/test_mvp_graph_build.py`

- [ ] **Step 1: Write the failing workflow test**

```python
from src.schemas.document import PaperDocument
from src.schemas.extraction import ExtractionResult
from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation, PaperCitationRelation
from src.workflows.mvp_graph_build import MvpGraphBuildWorkflow


class FakePaperService:
    def fetch_seed_and_citations(self, keyword, max_seed_papers, require_references):
        seed = PaperDocument(
            document_id="p1",
            source_type="openalex",
            source_ref="https://openalex.org/W1",
            title="Seed Paper",
            abstract="Graph methods improve idea generation",
            ingestion_status="ready",
        )
        cited = PaperDocument(
            document_id="p2",
            source_type="openalex",
            source_ref="https://openalex.org/W2",
            title="Cited Paper",
            abstract="Idea generation needs graph planning",
            ingestion_status="ready",
        )
        return [seed], [cited], [PaperCitationRelation(source_paper_id="p1", target_paper_id="p2")]


class FakeExtractor:
    def extract(self, document):
        return ExtractionResult(
            entities=[
                ExtractedEntity(entity_id="graph planning", name="graph planning", entity_type="Technology", source_refs=[document.document_id]),
                ExtractedEntity(entity_id="idea generation", name="idea generation", entity_type="Problem", source_refs=[document.document_id]),
            ],
            relations=[
                ExtractedRelation(
                    source_entity_id="graph planning",
                    target_entity_id="idea generation",
                    relation_type="related_to",
                    source_refs=[document.document_id],
                )
            ],
        )


class FakeRepo:
    def __init__(self):
        self.papers = []
        self.entities = []
        self.citations = []
        self.entity_relations = []

    def upsert_papers(self, papers):
        self.papers.extend(papers)

    def upsert_entities(self, entities):
        self.entities.extend(entities)

    def upsert_citations(self, citations):
        self.citations.extend(citations)

    def upsert_entity_relations(self, relations):
        self.entity_relations.extend(relations)


def test_workflow_builds_graph_for_seed_and_one_hop_cited_papers():
    repo = FakeRepo()
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakePaperService(),
        extractor=FakeExtractor(),
        repository=repo,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["paper_count"] == 2
    assert len(repo.citations) == 1
    assert len(repo.entities) == 2
    assert len(repo.entity_relations) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mvp_graph_build.py -v`

Expected: FAIL because workflow does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# src/workflows/mvp_graph_build.py
from src.knowledge.normalizer import merge_entities, normalize_relations


class MvpGraphBuildWorkflow:
    def __init__(self, paper_service, extractor, repository) -> None:
        self._paper_service = paper_service
        self._extractor = extractor
        self._repository = repository

    def run(self, keyword: str, max_seed_papers: int, require_references: bool) -> dict:
        seed_docs, cited_docs, paper_links = self._paper_service.fetch_seed_and_citations(
            keyword=keyword,
            max_seed_papers=max_seed_papers,
            require_references=require_references,
        )
        all_docs = seed_docs + cited_docs

        all_entities = []
        all_relations = []
        for doc in all_docs:
            extraction = self._extractor.extract(doc)
            all_entities.extend(extraction.entities)
            all_relations.extend(extraction.relations)

        merged_entities = merge_entities(all_entities)
        id_map = {entity.name: entity.entity_id for entity in merged_entities}
        merged_relations = normalize_relations(all_relations, id_map)

        self._repository.upsert_papers(all_docs)
        self._repository.upsert_entities(merged_entities)
        self._repository.upsert_citations(paper_links)
        self._repository.upsert_entity_relations(merged_relations)

        return {
            "paper_count": len(all_docs),
            "entity_count": len(merged_entities),
            "citation_count": len(paper_links),
            "entity_relation_count": len(merged_relations),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mvp_graph_build.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/workflows/mvp_graph_build.py tests/test_mvp_graph_build.py
git commit -m "feat: add mvp graph build workflow"
```

## Task 8: Add CLI Entrypoint and README for MVP Operation

**Files:**
- Create: `src/interfaces/cli/build_graph.py`
- Modify: `main.py`
- Modify: `README.md`
- Test: `tests/test_openalex_service.py`

- [ ] **Step 1: Write the failing CLI smoke test**

```python
from src.interfaces.cli.build_graph import build_parser


def test_build_parser_accepts_keyword_and_seed_limit():
    parser = build_parser()
    args = parser.parse_args(["build-graph", "--keyword", "idea graph", "--max-seed-papers", "3"])

    assert args.command == "build-graph"
    assert args.keyword == "idea graph"
    assert args.max_seed_papers == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_openalex_service.py::test_build_parser_accepts_keyword_and_seed_limit -v`

Expected: FAIL because the CLI parser does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# src/interfaces/cli/build_graph.py
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-graph")
    build.add_argument("--keyword", required=True)
    build.add_argument("--max-seed-papers", type=int, default=3)
    build.add_argument("--require-references", action="store_true", default=False)
    return parser
```

```python
# main.py
from src.interfaces.cli.build_graph import build_parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()


if __name__ == "__main__":
    main()
```

```markdown
# README.md
## MVP Goal

Build a Neo4j graph from OpenAlex seed papers and their one-hop cited papers.

## Run

```bash
python main.py build-graph --keyword "idea graph" --max-seed-papers 3 --require-references
```

## Constraints

- only one-hop citation expansion
- skip network failures instead of aborting
- write into a configurable standalone Neo4j database
- extract entities and relations in one model call per paper
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_openalex_service.py::test_build_parser_accepts_keyword_and_seed_limit -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/interfaces/cli/build_graph.py main.py README.md tests/test_openalex_service.py
git commit -m "feat: add cli entrypoint for mvp graph build"
```

## Task 9: Run End-to-End Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Install dependencies**

Run: `uv sync`

Expected: environment created successfully with runtime and dev dependencies installed

- [ ] **Step 2: Run the unit test suite**

Run: `pytest -v`

Expected: all tests added in this plan pass

- [ ] **Step 3: Run the CLI against a real keyword**

Run: `python main.py build-graph --keyword "knowledge graph idea generation" --max-seed-papers 3 --require-references`

Expected:
- seed papers are fetched from OpenAlex
- one-hop cited papers are fetched when available
- network failures are skipped instead of aborting the run
- the workflow prints a summary with paper, entity, relation, and citation counts

- [ ] **Step 4: Verify Neo4j contents manually**

Run in Neo4j Browser:

```cypher
MATCH (p:Paper) RETURN count(p);
MATCH (p1:Paper)-[r:CITES]->(p2:Paper) RETURN count(r);
MATCH (e1:Entity)-[r]->(e2:Entity) RETURN type(r), count(r);
```

Expected:
- papers exist
- directed `CITES` edges exist
- entity-to-entity edges exist

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document mvp verification flow"
```

## Spec Coverage Check

- OpenAlex keyword search with filterable citation metadata: covered by Task 2 and Task 3
- Seed-paper selection and one-hop citation expansion only: covered by Task 3 and Task 7
- Entity and relation extraction from title/abstract in one model call per paper: covered by Task 4 and Task 7
- Merge same or similar entity items: covered by Task 5
- Build paper-to-paper directed citation edges and entity-to-entity edges: covered by Task 6 and Task 7
- Skip network failures instead of aborting: covered by Task 2 and Task 9
- Use configurable standalone Neo4j database: covered by Task 1 and Task 6
- Confirm graph visibility in Neo4j: covered by Task 9

## Self-Review Notes

- No unresolved placeholders remain.
- The plan intentionally stays document-level and does not introduce chunk/evidence/full-text work.
- The plan keeps filtering/ranking extensible later, but MVP implementation uses a minimal normalization-and-formula path first.
