# Evaluation Benchmark and OpenAlex Seed Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two occasional-use workflows: accepted ICLR paper benchmarking and OpenAlex explicit-seed graph building, without changing the existing `build-graph` and `chat-idea` behavior.

**Architecture:** Add focused ingestion and workflow modules for benchmark data, then reuse the existing model, full-text, evaluation, and graph build components. Refactor `MvpGraphBuildWorkflow` only enough to support an explicit OpenAlex seed producer that expands exactly one reference hop and then runs the same paper writing, full-text enrichment, entity extraction, and graph writing path.

**Tech Stack:** Python 3.12, httpx, pydantic, Ollama JSON model, OpenAlex, OpenReview public API/pages, arXiv fallback, Neo4j, pytest

---

## File Structure

Planned files and responsibilities:

- Create: `src/ingestion/openreview_client.py`
  - Public OpenReview fetcher plus helpers to normalize note content, accepted status, PDF URLs, review scores, and text-query filtering.
- Create: `src/workflows/accepted_paper_benchmark.py`
  - Benchmark orchestration: fetch accepted papers, filter, resolve full text, convert to `IdeaOutput`, evaluate, and export JSON/CSV/HTML.
- Create: `src/prompts/convert_paper_to_idea.txt`
  - Prompt for converting an accepted paper into the existing `IdeaOutput` schema.
- Modify: `src/ingestion/openalex_service.py`
  - Add explicit OpenAlex work lookup for one or more user-specified seed IDs.
- Modify: `src/workflows/mvp_graph_build.py`
  - Factor shared graph-building producer/consumer logic and add a `run_openalex_seed_async()` path that expands exactly one reference hop.
- Modify: `src/app/bootstrap.py`
  - Add builders for the benchmark workflow and the OpenAlex seed graph workflow using existing settings and components.
- Modify: `src/interfaces/cli/build_graph.py`
  - Add `benchmark-accepted-ideas` and `build-openalex-seed-graph` parsers.
- Modify: `main.py`
  - Route the two new commands and print concise summaries.
- Create: `tests/test_openreview_client.py`
  - Unit tests for accepted-paper filtering, RAG query filtering, PDF extraction, and score parsing.
- Create: `tests/test_accepted_paper_benchmark.py`
  - Unit tests for paper-to-idea conversion, evaluator usage, reference stripping, and exports.
- Modify: `tests/test_openalex_service.py`
  - Unit tests for explicit OpenAlex seed lookup.
- Modify: `tests/test_mvp_graph_build.py`
  - Unit tests for one-hop explicit seed graph behavior and full-text-before-extraction ordering.
- Modify: `tests/test_cli.py`
  - CLI parser tests for the two new commands.
- Modify: `tests/test_main.py`
  - Command routing tests for the new workflows.

## Task 1: OpenReview Accepted Paper Ingestion

**Files:**
- Create: `src/ingestion/openreview_client.py`
- Create: `tests/test_openreview_client.py`

- [ ] **Step 1: Write failing OpenReview helper tests**

Add `tests/test_openreview_client.py`:

```python
from src.ingestion.openreview_client import (
    OpenReviewPaper,
    extract_review_scores,
    filter_accepted_papers,
    filter_papers_by_query,
    normalize_openreview_note,
)


def test_filter_accepted_papers_keeps_only_final_accept_statuses():
    papers = [
        OpenReviewPaper(openreview_id="a", title="A", abstract="RAG", status="ICLR 2026 Oral"),
        OpenReviewPaper(openreview_id="b", title="B", abstract="RAG", status="ICLR 2026 Poster"),
        OpenReviewPaper(openreview_id="c", title="C", abstract="RAG", status="ICLR 2026 Conference Desk Rejected Submission"),
        OpenReviewPaper(openreview_id="d", title="D", abstract="RAG", status="Submitted to ICLR 2026"),
        OpenReviewPaper(openreview_id="e", title="E", abstract="RAG", status="ICLR 2026 Conference Withdrawn Submission"),
    ]

    accepted = filter_accepted_papers(papers)

    assert [paper.openreview_id for paper in accepted] == ["a", "b"]


def test_filter_papers_by_query_matches_title_abstract_keywords_and_metadata():
    papers = [
        OpenReviewPaper(openreview_id="a", title="Retrieval augmented generation", abstract="", keywords=[], status="ICLR 2026 Poster"),
        OpenReviewPaper(openreview_id="b", title="Vision model", abstract="No retrieval here", keywords=["rag"], status="ICLR 2026 Poster"),
        OpenReviewPaper(openreview_id="c", title="Optimizer", abstract="Training curves", keywords=[], status="ICLR 2026 Poster"),
    ]

    matched = filter_papers_by_query(papers, "rag retrieval")

    assert [paper.openreview_id for paper in matched] == ["a", "b"]


def test_normalize_openreview_note_reads_content_fields_and_pdf_url():
    note = {
        "id": "note-1",
        "content": {
            "title": {"value": "Paper title"},
            "abstract": {"value": "Paper abstract"},
            "keywords": {"value": ["RAG", "LLM"]},
            "venue": {"value": "ICLR 2026 Poster"},
            "pdf": {"value": "/pdf?id=note-1"},
        },
    }

    paper = normalize_openreview_note(note, base_url="https://openreview.net")

    assert paper.openreview_id == "note-1"
    assert paper.title == "Paper title"
    assert paper.abstract == "Paper abstract"
    assert paper.keywords == ["RAG", "LLM"]
    assert paper.status == "ICLR 2026 Poster"
    assert paper.pdf_url == "https://openreview.net/pdf?id=note-1"


def test_extract_review_scores_reads_numeric_rating_values():
    notes = [
        {"content": {"rating": {"value": "8: accept, good paper"}}},
        {"content": {"recommendation": {"value": "6: marginally above acceptance threshold"}}},
        {"content": {"rating": {"value": "not numeric"}}},
    ]

    scores = extract_review_scores(notes)

    assert scores == [8.0, 6.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_openreview_client.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion.openreview_client'`.

- [ ] **Step 3: Implement OpenReview ingestion helpers**

Create `src/ingestion/openreview_client.py`:

```python
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx


logger = logging.getLogger(__name__)
_ACCEPT_MARKERS = ("oral", "spotlight", "poster")
_REJECT_MARKERS = ("rejected", "withdrawn", "desk rejected", "submitted to")


@dataclass(frozen=True)
class OpenReviewPaper:
    openreview_id: str
    title: str
    abstract: str
    status: str
    keywords: list[str] = field(default_factory=list)
    pdf_url: str | None = None
    metadata: dict = field(default_factory=dict)
    iclr_scores: list[float] = field(default_factory=list)


class OpenReviewClient:
    def __init__(
        self,
        base_url: str = "https://openreview.net",
        timeout: float = 30.0,
        http_client=None,
        log=None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http_client = http_client
        self._log = log or logger.info

    def fetch_venue_papers(self, venue: str, limit: int | None = None) -> list[OpenReviewPaper]:
        notes = self._fetch_notes(venue=venue, limit=limit)
        papers = [normalize_openreview_note(note, base_url=self._base_url) for note in notes]
        self._log("openreview:papers_fetched venue=%s count=%s", venue, len(papers))
        return papers

    def fetch_review_scores(self, forum_id: str) -> list[float]:
        try:
            response = self._get(
                f"{self._base_url}/notes",
                params={"forum": forum_id, "details": "replyCount"},
            )
            notes = response.json().get("notes") or response.json().get("results") or []
        except httpx.HTTPError as exc:
            logger.warning(
                "openreview:reviews_failed forum=%s error=%s message=%s",
                forum_id,
                type(exc).__name__,
                exc,
            )
            return []
        return extract_review_scores(notes)

    def _fetch_notes(self, venue: str, limit: int | None) -> list[dict]:
        notes: list[dict] = []
        offset = 0
        page_size = 100
        while limit is None or len(notes) < limit:
            params = {
                "content.venueid": venue,
                "limit": min(page_size, limit - len(notes)) if limit is not None else page_size,
                "offset": offset,
            }
            try:
                response = self._get(f"{self._base_url}/notes", params=params)
            except httpx.HTTPError as exc:
                logger.warning(
                    "openreview:notes_failed venue=%s offset=%s error=%s message=%s",
                    venue,
                    offset,
                    type(exc).__name__,
                    exc,
                )
                break
            batch = response.json().get("notes") or response.json().get("results") or []
            if not batch:
                break
            notes.extend(batch)
            offset += len(batch)
            if len(batch) < page_size:
                break
        return notes[:limit] if limit is not None else notes

    def _get(self, url: str, params: dict) -> httpx.Response:
        if self._http_client is not None:
            response = self._http_client.get(url, params=params)
        else:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url, params=params)
        response.raise_for_status()
        return response


def normalize_openreview_note(note: dict, base_url: str = "https://openreview.net") -> OpenReviewPaper:
    content = note.get("content") or {}
    title = _content_value(content, "title")
    abstract = _content_value(content, "abstract")
    keywords = _content_list(content, "keywords")
    status = (
        _content_value(content, "venue")
        or _content_value(content, "venueid")
        or _content_value(content, "decision")
        or ""
    )
    pdf = _content_value(content, "pdf") or note.get("pdf")
    pdf_url = urljoin(base_url.rstrip("/") + "/", pdf.lstrip("/")) if isinstance(pdf, str) and pdf else None
    return OpenReviewPaper(
        openreview_id=note.get("id") or note.get("forum") or "",
        title=title,
        abstract=abstract,
        keywords=keywords,
        status=status,
        pdf_url=pdf_url,
        metadata=note,
    )


def filter_accepted_papers(papers: list[OpenReviewPaper]) -> list[OpenReviewPaper]:
    accepted = []
    for paper in papers:
        status = paper.status.lower()
        if any(marker in status for marker in _REJECT_MARKERS):
            continue
        if any(marker in status for marker in _ACCEPT_MARKERS):
            accepted.append(paper)
    return accepted


def filter_papers_by_query(papers: list[OpenReviewPaper], query: str) -> list[OpenReviewPaper]:
    tokens = [token for token in re.split(r"[^a-z0-9]+", query.lower()) if token]
    if not tokens:
        return papers
    matched = []
    for paper in papers:
        haystack = " ".join(
            [
                paper.title,
                paper.abstract,
                " ".join(paper.keywords),
                paper.status,
            ]
        ).lower()
        if any(token in haystack for token in tokens):
            matched.append(paper)
    return matched


def extract_review_scores(notes: list[dict]) -> list[float]:
    scores = []
    for note in notes:
        content = note.get("content") or {}
        for key in ("rating", "recommendation", "score"):
            value = _content_value(content, key)
            if not value:
                continue
            match = re.match(r"\s*(\d+(?:\.\d+)?)", str(value))
            if match:
                scores.append(float(match.group(1)))
                break
    return scores


def _content_value(content: dict, key: str) -> str:
    value = content.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value or "").strip()


def _content_list(content: dict, key: str) -> list[str]:
    value = content.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_openreview_client.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/openreview_client.py tests/test_openreview_client.py
git commit -m "feat: 增加 OpenReview 已接收论文读取"
```

## Task 2: Accepted Paper Benchmark Workflow

**Files:**
- Create: `src/prompts/convert_paper_to_idea.txt`
- Create: `src/workflows/accepted_paper_benchmark.py`
- Create: `tests/test_accepted_paper_benchmark.py`

- [ ] **Step 1: Write failing benchmark workflow tests**

Create `tests/test_accepted_paper_benchmark.py`:

```python
import json
from pathlib import Path

from src.ingestion.openreview_client import OpenReviewPaper
from src.schemas.agent import IdeaOutput
from src.workflows.accepted_paper_benchmark import (
    AcceptedPaperBenchmarkWorkflow,
    convert_paper_to_idea,
    load_review_score_csv,
)


class FakeModel:
    def __init__(self):
        self.messages = []

    def generate_messages_json(self, messages):
        self.messages.append(messages)
        return {
            "title": "图检索增强生成评测",
            "summary": "将论文方法改写为研究想法。",
            "problem": "RAG 系统缺少可靠评测。",
            "method": "结合检索证据和生成质量评分。",
            "core_paper_id": "or-1",
            "core_paper_title": "RAG Paper",
        }


class FakeEvaluatorModel:
    def __init__(self):
        self.messages = []

    def generate_messages_json(self, messages):
        self.messages.append(messages)
        return {
            "overall_score": 4,
            "summary": "基准想法清楚。",
            "novelty": {"score": 4, "comment": "有新意", "revision_suggestion": "补对比"},
            "clarity": {"score": 4, "comment": "清楚", "revision_suggestion": "补流程"},
            "feasibility": {"score": 4, "comment": "可做", "revision_suggestion": "补数据"},
            "relevance": {"score": 5, "comment": "相关", "revision_suggestion": "保持"},
            "expected_effectiveness": {"score": 3, "comment": "待验证", "revision_suggestion": "补实验"},
            "evidence_grounding": {"score": 4, "comment": "有依据", "revision_suggestion": "补引用"},
            "strengths": ["清晰"],
            "major_issues": ["实验少"],
            "revision_instruction": "补实验。",
        }


class FakeOpenReviewClient:
    def fetch_venue_papers(self, venue, limit=None):
        return [
            OpenReviewPaper(
                openreview_id="or-1",
                title="RAG Paper",
                abstract="Retrieval augmented generation",
                keywords=["rag"],
                status="ICLR 2026 Poster",
                pdf_url="https://openreview.net/pdf?id=or-1",
                iclr_scores=[8.0],
            )
        ]

    def fetch_review_scores(self, forum_id):
        assert forum_id == "or-1"
        return [8.0, 7.0]


class FakeFullTextProvider:
    def enrich(self, document):
        return document.model_copy(
            update={
                "full_text_markdown": "# Body\n\nUseful content\n\n# References\n\n[1] cited",
                "text_source": "full_text_md",
                "full_text_markdown_path": "paper.md",
            }
        )


def test_convert_paper_to_idea_prefers_full_text_and_validates_schema():
    model = FakeModel()
    paper = OpenReviewPaper(
        openreview_id="or-1",
        title="RAG Paper",
        abstract="Short abstract",
        status="ICLR 2026 Poster",
    )

    idea = convert_paper_to_idea(
        paper=paper,
        text="Full paper body",
        text_source="full_text_md",
        model=model,
        prompt_template="<role>convert</role>",
    )

    assert isinstance(idea, IdeaOutput)
    assert idea.core_paper_id == "or-1"
    assert "Full paper body" in model.messages[0][1]["content"]


def test_load_review_score_csv_maps_ids_and_titles(tmp_path):
    csv_path = tmp_path / "scores.csv"
    csv_path.write_text("openreview_id,title,iclr_score\nor-1,RAG Paper,7.5\n,RAG Other,6\n", encoding="utf-8")

    scores = load_review_score_csv(csv_path)

    assert scores["or-1"] == 7.5
    assert scores["title:RAG Other"] == 6.0


def test_benchmark_workflow_strips_references_evaluates_and_exports(tmp_path):
    workflow = AcceptedPaperBenchmarkWorkflow(
        openreview_client=FakeOpenReviewClient(),
        full_text_provider=FakeFullTextProvider(),
        conversion_model=FakeModel(),
        evaluation_model=FakeEvaluatorModel(),
        conversion_prompt_template="<role>convert</role>",
        evaluation_prompt_template="<role>evaluate</role>",
    )

    summary = workflow.run(
        venue="ICLR.cc/2026/Conference",
        query="rag",
        max_papers=5,
        output_dir=tmp_path,
        review_score_csv=None,
    )

    assert summary["paper_count"] == 1
    results = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert results[0]["idea_output"]["core_paper_id"] == "or-1"
    assert results[0]["idea_evaluation"]["overall_score"] == 4
    assert results[0]["iclr_score"] == 7.5
    assert "# References" not in results[0]["input_text"]
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "report.html").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_accepted_paper_benchmark.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'src.workflows.accepted_paper_benchmark'`.

- [ ] **Step 3: Add conversion prompt**

Create `src/prompts/convert_paper_to_idea.txt`:

```text
<role>
You convert an accepted research paper into the same research-idea schema used by this system.
</role>

<task>
Read the paper content and describe the paper's core research idea as if it were a generated idea candidate.
Use Chinese for all user-facing fields.
</task>

<schema>
Return JSON with exactly these fields:
- title
- summary
- problem
- method
- core_paper_id
- core_paper_title
</schema>

<rules>
Do not include references, bibliography entries, author lists, venue metadata, or markdown.
Do not invent extra fields.
The core_paper_id must be the provided paper id.
The core_paper_title must be the provided paper title.
</rules>
```

- [ ] **Step 4: Implement benchmark workflow**

Create `src/workflows/accepted_paper_benchmark.py`:

```python
from __future__ import annotations

import csv
import html
import json
import logging
from pathlib import Path

from src.agent.nodes.idea_evaluation import idea_evaluation
from src.extraction.pdf_markdown import strip_references
from src.ingestion.openreview_client import (
    OpenReviewPaper,
    filter_accepted_papers,
    filter_papers_by_query,
)
from src.schemas.agent import IdeaOutput
from src.schemas.document import PaperDocument


logger = logging.getLogger(__name__)


class AcceptedPaperBenchmarkWorkflow:
    def __init__(
        self,
        openreview_client,
        full_text_provider,
        conversion_model,
        evaluation_model,
        conversion_prompt_template: str,
        evaluation_prompt_template: str,
        log=None,
    ) -> None:
        self._openreview_client = openreview_client
        self._full_text_provider = full_text_provider
        self._conversion_model = conversion_model
        self._evaluation_model = evaluation_model
        self._conversion_prompt_template = conversion_prompt_template
        self._evaluation_prompt_template = evaluation_prompt_template
        self._log = log or logger.info

    def run(
        self,
        venue: str,
        query: str,
        max_papers: int,
        output_dir: str | Path,
        review_score_csv: str | Path | None = None,
    ) -> dict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        manual_scores = load_review_score_csv(review_score_csv) if review_score_csv else {}
        papers = self._openreview_client.fetch_venue_papers(venue, limit=None)
        papers = filter_papers_by_query(filter_accepted_papers(papers), query)[:max_papers]
        records = []
        for paper in papers:
            records.append(self._process_paper(paper, manual_scores))
        _write_json(output_path / "results.json", records)
        _write_csv(output_path / "results.csv", records)
        _write_html_report(output_path / "report.html", records)
        summary = {"paper_count": len(records), "output_dir": str(output_path)}
        self._log("benchmark:done papers=%s output_dir=%s", len(records), output_path)
        return summary

    def _process_paper(self, paper: OpenReviewPaper, manual_scores: dict[str, float]) -> dict:
        scores = self._openreview_client.fetch_review_scores(paper.openreview_id)
        text, text_source, markdown_path = self._resolve_text(paper)
        idea = convert_paper_to_idea(
            paper=paper,
            text=text,
            text_source=text_source,
            model=self._conversion_model,
            prompt_template=self._conversion_prompt_template,
        )
        evaluation = idea_evaluation(
            {"idea_output": idea},
            model=self._evaluation_model,
            prompt_template=self._evaluation_prompt_template,
        )["idea_evaluation"]
        iclr_score = _resolve_iclr_score(paper, scores, manual_scores)
        return {
            "openreview_id": paper.openreview_id,
            "title": paper.title,
            "status": paper.status,
            "pdf_url": paper.pdf_url,
            "text_source": text_source,
            "full_text_markdown_path": markdown_path,
            "input_text": text,
            "iclr_scores": scores,
            "iclr_score": iclr_score,
            "idea_output": idea.model_dump(),
            "idea_evaluation": evaluation.model_dump(),
        }

    def _resolve_text(self, paper: OpenReviewPaper) -> tuple[str, str, str | None]:
        document = PaperDocument(
            document_id=paper.title,
            source_type="openreview",
            source_ref=paper.openreview_id,
            abstract=paper.abstract,
            pdf_url=paper.pdf_url,
            ingestion_status="ready",
        )
        if self._full_text_provider is not None:
            enriched = self._full_text_provider.enrich(document)
            content = strip_references((enriched.full_text_markdown or "").strip())
            if content:
                return content, enriched.text_source, enriched.full_text_markdown_path
        fallback = f"Title: {paper.title}\nAbstract: {paper.abstract}".strip()
        return fallback, "title_abstract", None


def convert_paper_to_idea(
    paper: OpenReviewPaper,
    text: str,
    text_source: str,
    model,
    prompt_template: str,
) -> IdeaOutput:
    messages = [
        {"role": "system", "content": prompt_template},
        {
            "role": "user",
            "content": (
                f"Paper id: {paper.openreview_id}\n"
                f"Paper title: {paper.title}\n"
                f"Text source: {text_source}\n\n"
                "<paper_content>\n"
                f"{text}\n"
                "</paper_content>\n\n"
                "Return JSON only."
            ),
        },
    ]
    raw = model.generate_messages_json(messages)
    return IdeaOutput(**raw)


def load_review_score_csv(path: str | Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_score = (row.get("iclr_score") or "").strip()
            if not raw_score:
                continue
            score = float(raw_score)
            openreview_id = (row.get("openreview_id") or "").strip()
            title = (row.get("title") or "").strip()
            if openreview_id:
                scores[openreview_id] = score
            if title:
                scores[f"title:{title}"] = score
    return scores


def _resolve_iclr_score(
    paper: OpenReviewPaper,
    parsed_scores: list[float],
    manual_scores: dict[str, float],
) -> float | None:
    if paper.openreview_id in manual_scores:
        return manual_scores[paper.openreview_id]
    title_key = f"title:{paper.title}"
    if title_key in manual_scores:
        return manual_scores[title_key]
    if parsed_scores:
        return sum(parsed_scores) / len(parsed_scores)
    return None


def _write_json(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, records: list[dict]) -> None:
    fieldnames = ["openreview_id", "title", "status", "text_source", "iclr_score", "overall_score"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "openreview_id": record["openreview_id"],
                    "title": record["title"],
                    "status": record["status"],
                    "text_source": record["text_source"],
                    "iclr_score": record["iclr_score"],
                    "overall_score": record["idea_evaluation"]["overall_score"],
                }
            )


def _write_html_report(path: Path, records: list[dict]) -> None:
    rows = []
    for record in records:
        overall = record["idea_evaluation"]["overall_score"]
        iclr_score = "" if record["iclr_score"] is None else f"{record['iclr_score']:.2f}"
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['title'])}</td>"
            f"<td>{html.escape(record['status'])}</td>"
            f"<td>{overall}</td>"
            f"<td>{html.escape(iclr_score)}</td>"
            "</tr>"
        )
    bars = "\n".join(
        f"<div class='bar' style='width:{record['idea_evaluation']['overall_score'] * 20}%'>"
        f"{html.escape(record['title'])}: {record['idea_evaluation']['overall_score']}/5</div>"
        for record in records
    )
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px}"
        ".bar{background:#4f7cac;color:white;margin:6px 0;padding:4px;min-width:32px}</style>"
        "</head><body><h1>Accepted Paper Benchmark</h1>"
        f"<section>{bars}</section>"
        "<table><tr><th>Title</th><th>Status</th><th>Evaluator</th><th>ICLR</th></tr>"
        + "".join(rows)
        + "</table></body></html>",
        encoding="utf-8",
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
python3 -m pytest tests/test_accepted_paper_benchmark.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/prompts/convert_paper_to_idea.txt src/workflows/accepted_paper_benchmark.py tests/test_accepted_paper_benchmark.py
git commit -m "feat: 增加 ICLR 已接收论文基准流程"
```

## Task 3: Explicit OpenAlex Seed Lookup

**Files:**
- Modify: `src/ingestion/openalex_service.py`
- Modify: `tests/test_openalex_service.py`

- [ ] **Step 1: Write failing OpenAlex seed lookup tests**

Append to `tests/test_openalex_service.py`:

```python
def test_fetch_openalex_seed_documents_skips_missing_ids_and_keeps_raws():
    from src.ingestion.openalex_service import OpenAlexPaperService

    class FakeClient:
        def __init__(self):
            self.requested = []

        def get_work(self, openalex_id):
            self.requested.append(openalex_id)
            if openalex_id == "missing":
                return None
            return {
                "id": f"https://openalex.org/{openalex_id}",
                "title": f"Seed {openalex_id}",
                "abstract_inverted_index": {"Seed": [0], openalex_id: [1]},
                "publication_year": 2025,
                "cited_by_count": 9,
                "referenced_works": [],
            }

    client = FakeClient()
    service = OpenAlexPaperService(client)

    docs, raws, failed = service.fetch_openalex_seed_documents(["W1", "missing", "W2"])

    assert [doc.document_id for doc in docs] == ["Seed W1", "Seed W2"]
    assert [raw["id"] for raw in raws] == ["https://openalex.org/W1", "https://openalex.org/W2"]
    assert failed == ["missing"]
    assert client.requested == ["W1", "missing", "W2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest tests/test_openalex_service.py::test_fetch_openalex_seed_documents_skips_missing_ids_and_keeps_raws -q
```

Expected: FAIL with `AttributeError: 'OpenAlexPaperService' object has no attribute 'fetch_openalex_seed_documents'`.

- [ ] **Step 3: Add explicit seed lookup methods**

Modify `src/ingestion/openalex_service.py`:

```python
    def fetch_openalex_seed_documents(self, openalex_ids: list[str]):
        seed_docs = []
        raw_seeds = []
        failed_ids = []
        seen_ids: set[str] = set()
        for openalex_id in openalex_ids:
            normalized_id = openalex_id.strip()
            if not normalized_id or normalized_id in seen_ids:
                continue
            seen_ids.add(normalized_id)
            raw = self._client.get_work(normalized_id)
            if not raw or not _has_non_empty_abstract(raw):
                failed_ids.append(normalized_id)
                continue
            raw_seeds.append(raw)
            seed_docs.append(self._to_document(raw))
        return seed_docs, raw_seeds, failed_ids

    async def fetch_openalex_seed_documents_async(self, openalex_ids: list[str]):
        return await asyncio.to_thread(self.fetch_openalex_seed_documents, openalex_ids)
```

Place these methods near the other sync/async service methods. Do not change existing keyword search methods.

- [ ] **Step 4: Run OpenAlex service tests**

Run:

```bash
python3 -m pytest tests/test_openalex_service.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/openalex_service.py tests/test_openalex_service.py
git commit -m "feat: 支持 OpenAlex 显式种子论文读取"
```

## Task 4: OpenAlex Seed Graph Workflow Path

**Files:**
- Modify: `src/workflows/mvp_graph_build.py`
- Modify: `tests/test_mvp_graph_build.py`

- [ ] **Step 1: Write failing explicit seed graph tests**

Append to `tests/test_mvp_graph_build.py`:

```python
class FakeExplicitSeedPaperService:
    def __init__(self):
        self.keyword_search_called = False
        self.expansion_called = False

    async def fetch_seed_documents_async(self, *args, **kwargs):
        self.keyword_search_called = True
        raise AssertionError("explicit seed graph must not call keyword search")

    async def fetch_openalex_seed_documents_async(self, openalex_ids):
        assert openalex_ids == ["W1", "W2"]
        return (
            [
                PaperDocument(
                    document_id="Seed A",
                    source_type="openalex",
                    source_ref="https://openalex.org/W1",
                    title="Seed A",
                    abstract="Seed A abstract",
                    citation_openalex_ids=["https://openalex.org/W3"],
                    ingestion_status="ready",
                ),
                PaperDocument(
                    document_id="Seed B",
                    source_type="openalex",
                    source_ref="https://openalex.org/W2",
                    title="Seed B",
                    abstract="Seed B abstract",
                    citation_openalex_ids=[],
                    ingestion_status="ready",
                ),
            ],
            [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Seed A",
                    "abstract_inverted_index": {"Seed": [0], "A": [1]},
                    "referenced_works": ["https://openalex.org/W3"],
                },
                {
                    "id": "https://openalex.org/W2",
                    "title": "Seed B",
                    "abstract_inverted_index": {"Seed": [0], "B": [1]},
                    "referenced_works": [],
                },
            ],
            [],
        )

    async def fetch_cited_documents_async(self, raw_seeds, seed_docs):
        self.expansion_called = True
        assert [raw["id"] for raw in raw_seeds] == ["https://openalex.org/W1", "https://openalex.org/W2"]
        return (
            [
                PaperDocument(
                    document_id="Cited C",
                    source_type="openalex",
                    source_ref="https://openalex.org/W3",
                    title="Cited C",
                    abstract="Cited C abstract",
                    ingestion_status="ready",
                )
            ],
            [PaperCitationRelation(source_paper_id="Seed A", target_paper_id="Cited C")],
        )


class FakeOrderingFullTextProvider:
    def __init__(self):
        self.enriched = []

    def enrich(self, doc):
        self.enriched.append(doc.document_id)
        return doc.model_copy(update={"full_text_markdown": "# Body", "text_source": "full_text_md"})


def test_openalex_seed_graph_uses_explicit_seeds_and_expands_one_hop_only():
    repo = FakeRepo()
    logs = []
    service = FakeExplicitSeedPaperService()
    workflow = MvpGraphBuildWorkflow(
        paper_service=service,
        extractor=FakeExtractor(),
        repository=repo,
        log=logs.append,
        worker_count=1,
        extract_entities_enabled=False,
        citation_graph_strategy="priority_expand",
        citation_max_total_papers=1,
    )

    summary = workflow.run_openalex_seed_graph(openalex_ids=["W1", "W2"])

    assert summary["paper_count"] == 3
    assert summary["citation_count"] == 1
    assert service.keyword_search_called is False
    assert service.expansion_called is True
    assert [paper.document_id for paper in repo.papers] == ["Seed A", "Seed B", "Cited C"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [("Seed A", "Cited C")]
    assert "openalex_seed:failed count=0 ids=[]" in logs


def test_openalex_seed_graph_enriches_full_text_before_extraction():
    repo = FakeRepo()
    full_text_provider = FakeOrderingFullTextProvider()
    extractor = FakeExtractor()
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakeExplicitSeedPaperService(),
        extractor=extractor,
        repository=repo,
        log=lambda message: None,
        worker_count=1,
        extract_entities_enabled=True,
        full_text_provider=full_text_provider,
    )

    summary = workflow.run_openalex_seed_graph(openalex_ids=["W1", "W2"])

    assert summary["paper_count"] == 3
    assert full_text_provider.enriched == ["Seed A", "Seed B", "Cited C"]
    assert len(repo.entities) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_mvp_graph_build.py::test_openalex_seed_graph_uses_explicit_seeds_and_expands_one_hop_only tests/test_mvp_graph_build.py::test_openalex_seed_graph_enriches_full_text_before_extraction -q
```

Expected: FAIL with `AttributeError: 'MvpGraphBuildWorkflow' object has no attribute 'run_openalex_seed_graph'`.

- [ ] **Step 3: Refactor `MvpGraphBuildWorkflow` with an explicit seed path**

Modify `src/workflows/mvp_graph_build.py`:

```python
    def run_openalex_seed_graph(self, openalex_ids: list[str]) -> dict:
        return asyncio.run(self.run_openalex_seed_graph_async(openalex_ids))

    async def run_openalex_seed_graph_async(self, openalex_ids: list[str]) -> dict:
        return await self._run_with_producer(
            producer_factory=lambda context: self._openalex_seed_one_hop_producer(context, openalex_ids)
        )
```

Then factor the current `run_async()` body into `_run_with_producer()`. The extracted context should hold the shared counters, extraction queue, `enqueue_document()`, worker setup, and final summary. Keep the current `run_async(keyword, max_seed_papers, require_references)` signature and existing strategy behavior.

Add this producer inside the class:

```python
    async def _openalex_seed_one_hop_producer(self, context, openalex_ids: list[str]) -> list:
        self._log(f"openalex_seed:start count={len(openalex_ids)} ids={openalex_ids}")
        known_paper_ids: set[str] = set()
        pending_citations = []

        seed_docs, raw_seeds, failed_ids = await self._paper_service.fetch_openalex_seed_documents_async(openalex_ids)
        self._log(f"openalex_seed:failed count={len(failed_ids)} ids={failed_ids}")
        for doc in seed_docs:
            if doc.document_id in known_paper_ids:
                continue
            self._repository.upsert_papers([doc])
            known_paper_ids.add(doc.document_id)
            context.paper_count += 1
            context.log_paper_node_written(doc.document_id)
            await context.enqueue_document(doc)

        cited_docs, paper_links = await self._paper_service.fetch_cited_documents_async(raw_seeds, seed_docs)
        pending_citations.extend(paper_links)
        for doc in cited_docs:
            if doc.document_id not in known_paper_ids:
                self._repository.upsert_papers([doc])
                known_paper_ids.add(doc.document_id)
                context.paper_count += 1
                context.log_paper_node_written(doc.document_id)
                await context.enqueue_document(doc)

        ready = [
            citation
            for citation in pending_citations
            if citation.source_paper_id in known_paper_ids and citation.target_paper_id in known_paper_ids
        ]
        if ready:
            self._repository.upsert_citations(ready)
            context.citation_count += len(ready)
        unresolved_count = len(pending_citations) - len(ready)
        if unresolved_count:
            self._log(f"citation:pending_unresolved count={unresolved_count}")
        self._log(f"search:done papers={context.paper_count} citations={context.citation_count}")
        return []
```

The concrete context can be a small local dataclass inside the module:

```python
@dataclass
class _BuildContext:
    paper_count: int
    citation_count: int
    total_entities: int
    total_entity_relations: int
    enqueue_document: object
    log_paper_node_written: object
```

If mutable integer fields are updated inside nested workers, keep updates on the context object rather than separate `nonlocal` integer variables.

- [ ] **Step 4: Run graph build tests**

Run:

```bash
python3 -m pytest tests/test_mvp_graph_build.py tests/test_full_text_workflow.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/workflows/mvp_graph_build.py tests/test_mvp_graph_build.py
git commit -m "feat: 增加 OpenAlex 种子一跳构图流程"
```

## Task 5: Bootstrap and CLI Routing

**Files:**
- Modify: `src/app/bootstrap.py`
- Modify: `src/interfaces/cli/build_graph.py`
- Modify: `main.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Write failing CLI tests**

Extend `tests/test_cli.py`:

```python
def test_build_parser_accepts_benchmark_accepted_ideas():
    parser = build_parser()
    args = parser.parse_args(
        [
            "benchmark-accepted-ideas",
            "--venue",
            "ICLR.cc/2026/Conference",
            "--query",
            "rag",
            "--max-papers",
            "10",
            "--output-dir",
            "exports/iclr",
            "--review-score-csv",
            "scores.csv",
        ]
    )

    assert args.command == "benchmark-accepted-ideas"
    assert args.venue == "ICLR.cc/2026/Conference"
    assert args.query == "rag"
    assert args.max_papers == 10
    assert args.output_dir == "exports/iclr"
    assert args.review_score_csv == "scores.csv"


def test_build_parser_accepts_repeated_openalex_seed_ids():
    parser = build_parser()
    args = parser.parse_args(
        [
            "build-openalex-seed-graph",
            "--openalex-id",
            "https://openalex.org/W1",
            "--openalex-id",
            "W2",
        ]
    )

    assert args.command == "build-openalex-seed-graph"
    assert args.openalex_ids == ["https://openalex.org/W1", "W2"]
```

Extend `tests/test_main.py` with monkeypatched route tests matching existing style:

```python
def test_main_routes_openalex_seed_graph(monkeypatch):
    import main

    calls = {}

    class FakeWorkflow:
        def run_openalex_seed_graph(self, openalex_ids):
            calls["openalex_ids"] = openalex_ids
            return {"paper_count": 2, "citation_count": 1}

    monkeypatch.setattr(main, "Settings", lambda: object())
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_mvp_workflow", lambda settings: FakeWorkflow())

    main.run_build_openalex_seed_graph(type("Args", (), {"openalex_ids": ["W1", "W2"]})())

    assert calls["openalex_ids"] == ["W1", "W2"]


def test_main_routes_benchmark_accepted_ideas(monkeypatch, tmp_path):
    import main

    calls = {}

    class FakeWorkflow:
        def run(self, venue, query, max_papers, output_dir, review_score_csv):
            calls.update(
                {
                    "venue": venue,
                    "query": query,
                    "max_papers": max_papers,
                    "output_dir": output_dir,
                    "review_score_csv": review_score_csv,
                }
            )
            return {"paper_count": 1, "output_dir": str(output_dir)}

    monkeypatch.setattr(main, "Settings", lambda: object())
    monkeypatch.setattr(main, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(main, "build_accepted_paper_benchmark_workflow", lambda settings: FakeWorkflow())

    main.run_benchmark_accepted_ideas(
        type(
            "Args",
            (),
            {
                "venue": "ICLR.cc/2026/Conference",
                "query": "rag",
                "max_papers": 3,
                "output_dir": tmp_path,
                "review_score_csv": None,
            },
        )()
    )

    assert calls["venue"] == "ICLR.cc/2026/Conference"
    assert calls["query"] == "rag"
    assert calls["max_papers"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python3 -m pytest tests/test_cli.py tests/test_main.py -q
```

Expected: FAIL with missing parser commands and missing main route functions.

- [ ] **Step 3: Add bootstrap builder**

Modify `src/app/bootstrap.py` imports:

```python
from src.ingestion.openreview_client import OpenReviewClient
from src.workflows.accepted_paper_benchmark import AcceptedPaperBenchmarkWorkflow
```

Add prompt loader:

```python
def load_convert_paper_to_idea_prompt_template() -> str:
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / "convert_paper_to_idea.txt"
    return prompt_path.read_text(encoding="utf-8")
```

Add builder:

```python
def build_accepted_paper_benchmark_workflow(settings: Settings) -> AcceptedPaperBenchmarkWorkflow:
    model = OllamaJsonModel(
        base_url=settings.ollama_base_url,
        model_name=settings.ollama_model,
        timeout=settings.ollama_timeout,
        think_mode=settings.ollama_think_mode,
        cache_enabled=settings.ollama_cache_enabled,
        cache_dir=settings.ollama_cache_dir,
    )
    full_text_provider = FullTextProvider(
        output_dir=Path(settings.full_text_output_dir) / settings.neo4j_database / "benchmark",
        arxiv_client=ArxivClient(),
        downloader=PdfDownloader(),
        extractor=OpenDataLoaderPdfMarkdownExtractor(
            hybrid_url=settings.pdf_hybrid_url,
            hybrid_backend=settings.pdf_hybrid_backend,
            hybrid_mode=settings.pdf_hybrid_mode,
        ),
    )
    return AcceptedPaperBenchmarkWorkflow(
        openreview_client=OpenReviewClient(),
        full_text_provider=full_text_provider,
        conversion_model=model,
        evaluation_model=model,
        conversion_prompt_template=load_convert_paper_to_idea_prompt_template(),
        evaluation_prompt_template=load_evaluate_idea_prompt_template(),
    )
```

The benchmark builder intentionally creates a full-text provider regardless of `FULL_TEXT_EXTRACTION_ENABLED`; this command is explicitly for benchmark extraction. PDF failures still fall back to title and abstract inside the workflow.

- [ ] **Step 4: Add CLI parser commands**

Modify `src/interfaces/cli/build_graph.py`:

```python
    benchmark = subparsers.add_parser("benchmark-accepted-ideas")
    benchmark.add_argument("--venue", default="ICLR.cc/2026/Conference")
    benchmark.add_argument("--query", default="rag")
    benchmark.add_argument("--max-papers", type=int, default=50)
    benchmark.add_argument("--output-dir", required=True)
    benchmark.add_argument("--review-score-csv", default=None)

    openalex_seed = subparsers.add_parser("build-openalex-seed-graph")
    openalex_seed.add_argument("--openalex-id", action="append", dest="openalex_ids", required=True)
```

- [ ] **Step 5: Add main routes**

Modify `main.py` imports:

```python
from src.app.bootstrap import build_accepted_paper_benchmark_workflow
```

Add functions:

```python
def run_build_openalex_seed_graph(args) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    workflow = build_mvp_workflow(settings)
    summary = workflow.run_openalex_seed_graph(openalex_ids=args.openalex_ids)
    logger.info("build_openalex_seed_graph:summary %s", summary)
    print(f"OpenAlex seed graph done: {summary}")


def run_benchmark_accepted_ideas(args) -> None:
    settings = Settings()
    configure_logging(level=settings.log_level, log_file=settings.log_file)
    workflow = build_accepted_paper_benchmark_workflow(settings)
    summary = workflow.run(
        venue=args.venue,
        query=args.query,
        max_papers=args.max_papers,
        output_dir=args.output_dir,
        review_score_csv=args.review_score_csv,
    )
    logger.info("benchmark_accepted_ideas:summary %s", summary)
    print(f"Accepted paper benchmark done: {summary}")
```

Extend `main()`:

```python
    elif args.command == "build-openalex-seed-graph":
        run_build_openalex_seed_graph(args)
    elif args.command == "benchmark-accepted-ideas":
        run_benchmark_accepted_ideas(args)
```

- [ ] **Step 6: Run CLI and main tests**

Run:

```bash
python3 -m pytest tests/test_cli.py tests/test_main.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/app/bootstrap.py src/interfaces/cli/build_graph.py main.py tests/test_cli.py tests/test_main.py
git commit -m "feat: 接入基准与 OpenAlex 种子构图命令"
```

## Task 6: Full Validation and Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add README usage section**

Append concise usage examples to `README.md` near the existing CLI examples:

```markdown
### Accepted Paper Benchmark

Build a benchmark from accepted OpenReview venue papers, convert each paper into
the same `IdeaOutput` schema used by `chat-idea`, evaluate it with the existing
judge prompt, and export JSON/CSV/HTML results:

```bash
uv run python main.py benchmark-accepted-ideas \
  --venue ICLR.cc/2026/Conference \
  --query rag \
  --max-papers 50 \
  --output-dir exports/iclr2026-rag-benchmark
```

If public review scores are unavailable or incomplete, provide a manual score
file:

```bash
uv run python main.py benchmark-accepted-ideas \
  --venue ICLR.cc/2026/Conference \
  --query rag \
  --max-papers 50 \
  --output-dir exports/iclr2026-rag-benchmark \
  --review-score-csv scores.csv
```

The score CSV should include `openreview_id,title,iclr_score`.

### OpenAlex Explicit Seed Graph

Build a graph from one or more explicit OpenAlex works. This command expands
exactly one reference hop from the provided seed works, then uses the existing
full-text and entity extraction settings for downstream graph construction:

```bash
uv run python main.py build-openalex-seed-graph \
  --openalex-id https://openalex.org/Wxxxx \
  --openalex-id Wyyyy
```
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_openreview_client.py tests/test_accepted_paper_benchmark.py tests/test_openalex_service.py tests/test_mvp_graph_build.py tests/test_cli.py tests/test_main.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
python3 -m pytest -q
```

Expected: PASS.

- [ ] **Step 4: Inspect final diff**

Run:

```bash
git diff --stat
git diff -- README.md main.py src tests
```

Expected: Changes are limited to benchmark ingestion/workflow, OpenAlex explicit seed support, CLI/bootstrap wiring, README usage, tests, and the conversion prompt.

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: 补充基准与 OpenAlex 种子构图用法"
```

## Self-Review

Spec coverage:

- Accepted ICLR 2026 RAG benchmark is covered by Tasks 1, 2, 5, and 6.
- Full-paper acquisition, arXiv fallback through `FullTextProvider`, and reference stripping are covered by Task 2.
- Existing `IdeaOutput` conversion and evaluator reuse are covered by Task 2.
- JSON/CSV/HTML export is covered by Task 2.
- Manual ICLR score CSV override is covered by Task 2.
- OpenAlex one-or-many explicit seed graph is covered by Tasks 3, 4, and 5.
- Exact one-hop reference expansion and no quantity limit are covered by Task 4.
- Full-text and entity extraction reuse for OpenAlex seed graphs is covered by Task 4.
- Existing `build-graph` and `chat-idea` behavior is preserved by keeping new routes separate and running the existing tests in Task 6.

Type consistency:

- `OpenReviewPaper.openreview_id` is used as `IdeaOutput.core_paper_id` for benchmark records.
- `PaperDocument.document_id` remains title-based in OpenAlex graph construction, matching existing repository lookup behavior.
- `AcceptedPaperBenchmarkWorkflow.run()` returns a summary dictionary like the existing workflows.
- `MvpGraphBuildWorkflow.run_openalex_seed_graph()` mirrors the existing sync wrapper style used by `run()`.
