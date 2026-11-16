from __future__ import annotations

import csv
import html
import json
import logging
from pathlib import Path
from typing import Any

from src.agent.nodes.idea_evaluation import idea_evaluation
from src.extraction.pdf_markdown import strip_references
from src.ingestion.openreview_client import (
    OpenReviewPaper,
    filter_papers_by_decision_scope,
    filter_papers_by_query,
)
from src.schemas.agent import IdeaEvaluation, IdeaOutput
from src.schemas.document import PaperDocument


logger = logging.getLogger(__name__)
MAX_BENCHMARK_INPUT_CHARS = 60000
CRITERION_SCORE_MAX = 5
OVERALL_SCORE_MAX = 10
EVALUATION_CRITERIA = [
    "novelty",
    "clarity",
    "feasibility",
    "relevance",
    "expected_effectiveness",
    "evidence_grounding",
]


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
        self.openreview_client = openreview_client
        self.full_text_provider = full_text_provider
        self.conversion_model = conversion_model
        self.evaluation_model = evaluation_model
        self.conversion_prompt_template = conversion_prompt_template
        self.evaluation_prompt_template = evaluation_prompt_template
        self.log = log or logger

    def run(
        self,
        venue: str,
        query: str,
        max_papers: int,
        output_dir: str | Path,
        decision_scope: str = "accepted",
    ) -> dict:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        existing_records = _load_existing_records(output_path / "results.json", self.log)

        papers = self.openreview_client.fetch_venue_papers(venue, limit=None)
        total_count = len(papers)
        papers = filter_papers_by_decision_scope(papers, decision_scope)
        if len(papers) != total_count:
            self._info(
                "accepted_paper_benchmark:decision_filter decision_scope=%s before=%s after=%s",
                decision_scope,
                total_count,
                len(papers),
            )
        filtered_count = len(papers)
        papers = filter_papers_by_query(papers, query, include_metadata=True)
        self._info(
            "accepted_paper_benchmark:query_filter venue=%s query=%s decision_scope=%s filtered_count=%s matched_count=%s",
            venue,
            query,
            decision_scope,
            filtered_count,
            len(papers),
        )
        papers = papers[:max_papers]
        self._info(
            "accepted_paper_benchmark:processing_limit max_papers=%s selected_count=%s",
            max_papers,
            len(papers),
        )

        records = []
        for paper in papers:
            try:
                records.append(self._process_paper(paper))
            except Exception as exc:
                self._exception(
                    "accepted_paper_benchmark:paper_failed openreview_id=%s title=%s error=%s",
                    paper.openreview_id,
                    paper.title,
                    exc,
                )
                records.append(self._error_record(paper, exc))
            _write_outputs(output_path, _merge_records_by_openreview_id(existing_records, records))
        if not records:
            _write_outputs(output_path, existing_records)

        self._info(
            "accepted_paper_benchmark:completed venue=%s query=%s decision_scope=%s count=%s output_dir=%s",
            venue,
            query,
            decision_scope,
            len(records),
            output_path,
        )
        return {"paper_count": len(records), "output_dir": str(output_path)}

    def _process_paper(self, paper: OpenReviewPaper) -> dict[str, Any]:
        iclr_scores = self.openreview_client.fetch_review_scores(paper.openreview_id)
        input_text, text_source, full_text_markdown_path, input_truncated = self._resolve_input_text(paper)
        idea = convert_paper_to_idea(
            paper=paper,
            text=input_text,
            text_source=text_source,
            model=self.conversion_model,
            prompt_template=self.conversion_prompt_template,
        )
        evaluation = self._evaluate_idea(idea)

        return {
            "openreview_id": paper.openreview_id,
            "title": paper.title,
            "status": paper.status,
            "decision_group": _decision_group(paper.status),
            "pdf_url": paper.pdf_url,
            "text_source": text_source,
            "full_text_markdown_path": full_text_markdown_path,
            "input_text": input_text,
            "input_truncated": input_truncated,
            "iclr_scores": iclr_scores,
            "iclr_score": _select_iclr_score(iclr_scores),
            "idea_output": idea.model_dump(),
            "idea_evaluation": evaluation.model_dump(),
        }

    def _resolve_input_text(self, paper: OpenReviewPaper) -> tuple[str, str, str | None, bool]:
        document = PaperDocument(
            document_id=paper.title or paper.openreview_id,
            source_type="openreview",
            source_ref=paper.openreview_id,
            abstract=paper.abstract,
            pdf_url=paper.pdf_url,
            ingestion_status="accepted_paper_benchmark",
        )
        if self.full_text_provider is not None:
            for candidate in self._full_text_candidates(document):
                enriched = self.full_text_provider.enrich(candidate)
                stripped_text = self._strip_full_text(enriched.full_text_markdown)
                if stripped_text:
                    capped_text, input_truncated = _cap_input_text(stripped_text)
                    return (
                        capped_text,
                        enriched.text_source or "full_text_markdown",
                        enriched.full_text_markdown_path,
                        input_truncated,
                    )

        fallback_text, input_truncated = _cap_input_text(_title_abstract_text(paper))
        return (fallback_text, "title_abstract", None, input_truncated)

    def _full_text_candidates(self, document: PaperDocument) -> list[PaperDocument]:
        candidates = [document]
        if document.pdf_url:
            candidates.append(document.model_copy(update={"pdf_url": None}))
        return candidates

    def _strip_full_text(self, markdown: str | None) -> str:
        if not markdown:
            return ""
        return strip_references(markdown)

    def _error_record(self, paper: OpenReviewPaper, exc: Exception) -> dict[str, Any]:
        try:
            iclr_scores = self.openreview_client.fetch_review_scores(paper.openreview_id)
        except Exception as score_exc:
            self._exception(
                "accepted_paper_benchmark:score_fetch_failed openreview_id=%s error=%s",
                paper.openreview_id,
                score_exc,
            )
            iclr_scores = []

        input_text, text_source, full_text_markdown_path, input_truncated = self._error_input_fields(paper)
        return {
            "openreview_id": paper.openreview_id,
            "title": paper.title,
            "status": paper.status,
            "decision_group": _decision_group(paper.status),
            "pdf_url": paper.pdf_url,
            "text_source": text_source,
            "full_text_markdown_path": full_text_markdown_path,
            "input_text": input_text,
            "input_truncated": input_truncated,
            "iclr_scores": iclr_scores,
            "iclr_score": _select_iclr_score(iclr_scores),
            "idea_output": None,
            "idea_evaluation": None,
            "error": str(exc),
        }

    def _error_input_fields(self, paper: OpenReviewPaper) -> tuple[str, str, str | None, bool]:
        try:
            return self._resolve_input_text(paper)
        except Exception as exc:
            self._exception(
                "accepted_paper_benchmark:error_input_failed openreview_id=%s error=%s",
                paper.openreview_id,
                exc,
            )
            fallback_text, input_truncated = _cap_input_text(_title_abstract_text(paper))
            return (
                fallback_text,
                "title_abstract",
                None,
                input_truncated,
            )

    def _evaluate_idea(self, idea: IdeaOutput) -> IdeaEvaluation:
        result = idea_evaluation(
            {"idea_output": idea},
            model=self.evaluation_model,
            prompt_template=self.evaluation_prompt_template,
        )
        return result["idea_evaluation"]

    def _exception(self, message: str, *args) -> None:
        if callable(self.log):
            self.log(message % args if args else message)
            return
        self.log.exception(message, *args)

    def _info(self, message: str, *args) -> None:
        if callable(self.log):
            self.log(message % args if args else message)
            return
        self.log.info(message, *args)


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
                f"paper_id: {paper.openreview_id}\n"
                f"paper_title: {paper.title}\n"
                f"text_source: {text_source}\n"
                "<paper_content>\n"
                f"{text}\n"
                "</paper_content>"
            ),
        },
    ]
    result = model.structured_json(
        messages=messages,
        schema=IdeaOutput,
        name="benchmark_convert_paper_to_idea",
        metadata={"stage": "benchmark_convert_paper_to_idea"},
    )
    parsed = result.parsed
    return parsed if isinstance(parsed, IdeaOutput) else IdeaOutput(**parsed)


def _select_iclr_score(parsed_scores: list[float]) -> float | None:
    if parsed_scores:
        return sum(parsed_scores) / len(parsed_scores)
    return None


def _title_abstract_text(paper: OpenReviewPaper) -> str:
    return f"# {paper.title}\n\n{paper.abstract}".strip()


def _cap_input_text(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_BENCHMARK_INPUT_CHARS:
        return text, False
    return text[:MAX_BENCHMARK_INPUT_CHARS], True


def _write_results_json(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_results_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "openreview_id",
            "title",
            "status",
            "decision_group",
            "text_source",
            "iclr_score",
            "overall_score",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "openreview_id": record["openreview_id"],
                    "title": record["title"],
                    "status": record["status"],
                    "decision_group": record.get("decision_group") or _decision_group(record.get("status", "")),
                    "text_source": record["text_source"],
                    "iclr_score": record["iclr_score"],
                    "overall_score": _overall_score(record),
                }
            )


def _write_report_html(path: Path, records: list[dict[str, Any]]) -> None:
    report_payload = {
        "criteria": EVALUATION_CRITERIA,
        "criterion_score_max": CRITERION_SCORE_MAX,
        "overall_score_max": OVERALL_SCORE_MAX,
        "records": [_report_record(record) for record in records],
    }
    data_json = _html_json(report_payload)
    content = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OpenReview Paper Benchmark</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --panel: #ffffff;
      --border: #d9e2ec;
      --text: #1f2937;
      --muted: #5f6b7a;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --ok: #0f766e;
      --warn: #b45309;
      --danger: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    }}
    .page {{
      padding: 24px;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 16px;
      margin-bottom: 20px;
    }}
    .header h1 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
    }}
    .header .meta {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      text-align: right;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 16px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    .sidebar {{
      position: sticky;
      top: 16px;
      max-height: calc(100vh - 32px);
      display: flex;
      flex-direction: column;
      min-height: 0;
    }}
    .sidebar-head {{
      padding: 16px;
      border-bottom: 1px solid var(--border);
    }}
    .sidebar-head h2 {{
      margin: 0 0 4px;
      font-size: 16px;
    }}
    .sidebar-head .hint {{
      color: var(--muted);
      font-size: 12px;
    }}
    .filter-row {{
      margin-top: 10px;
      display: grid;
      gap: 6px;
      font-size: 12px;
    }}
    .filter-row label {{
      color: var(--muted);
    }}
    .filter-row select {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      background: white;
      color: var(--text);
      font: inherit;
    }}
    .paper-list {{
      overflow: auto;
      padding: 8px;
    }}
    .paper-item {{
      width: 100%;
      display: block;
      text-align: left;
      border: 1px solid transparent;
      background: transparent;
      border-radius: 10px;
      padding: 12px;
      margin-bottom: 8px;
      cursor: pointer;
      color: inherit;
    }}
    .paper-item:hover {{
      background: #f8fafc;
      border-color: #e2e8f0;
    }}
    .paper-item.active {{
      background: var(--accent-soft);
      border-color: #93c5fd;
    }}
    .paper-title {{
      font-weight: 600;
      font-size: 14px;
      line-height: 1.35;
      margin-bottom: 6px;
    }}
    .paper-sub {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      font-size: 11px;
      color: var(--muted);
    }}
    .pill.accepted {{
      background: #dcfce7;
      color: #166534;
    }}
    .pill.rejected {{
      background: #fee2e2;
      color: #991b1b;
    }}
    .pill.other {{
      background: #f1f5f9;
      color: #475569;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 8px;
      border-radius: 999px;
      background: #eef2ff;
      color: #4338ca;
      font-size: 11px;
    }}
    .pill.gray {{
      background: #edf2f7;
      color: #475569;
    }}
    .content {{
      display: grid;
      gap: 16px;
      min-width: 0;
    }}
    .section {{
      padding: 16px;
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 16px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .stat {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      background: #fafcff;
    }}
    .stat .label {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .stat .value {{
      font-size: 20px;
      font-weight: 700;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .chart-card {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: white;
      padding: 10px;
      min-width: 0;
    }}
    .chart-card.wide {{
      grid-column: 1 / -1;
    }}
    .chart-title {{
      font-size: 13px;
      font-weight: 600;
      margin: 0 0 8px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--text);
    }}
    .chart-note {{
      color: var(--muted);
      font-weight: 400;
      font-size: 12px;
    }}
    .selected-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .selected-head h3 {{
      margin: 0 0 6px;
      font-size: 18px;
    }}
    .selected-head .sub {{
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }}
    .selected-grid {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .field {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      background: #fff;
    }}
    .field .k {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }}
    .field .v {{
      white-space: pre-wrap;
      line-height: 1.55;
      font-size: 13px;
    }}
    .evaluation-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .criterion {{
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #fcfdff;
      padding: 12px;
    }}
    .criterion .name {{
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
      gap: 8px;
    }}
    .criterion .score {{
      color: var(--accent);
      font-weight: 700;
    }}
    .criterion .text {{
      color: var(--text);
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
    }}
    .list-block {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .list-box {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      background: #fff;
    }}
    .list-box h4 {{
      margin: 0 0 8px;
      font-size: 13px;
    }}
    .list-box ul {{
      margin: 0;
      padding-left: 18px;
      color: var(--text);
      font-size: 13px;
      line-height: 1.5;
    }}
    .error-box {{
      margin-top: 12px;
      border: 1px solid #fecaca;
      background: #fff1f2;
      color: var(--danger);
      padding: 12px;
      border-radius: 10px;
      font-size: 13px;
    }}
    .empty {{
      color: var(--muted);
      font-size: 13px;
    }}
    svg {{
      max-width: 100%;
      display: block;
    }}
    @media (max-width: 1200px) {{
      .layout, .selected-grid, .chart-grid, .stats, .detail-grid, .evaluation-grid, .list-block {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: static;
        max-height: none;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="header">
      <div>
        <h1>OpenReview Paper Benchmark</h1>
        <div class="meta">Interactive report for OpenReview papers</div>
      </div>
      <div class="meta" id="page-meta"></div>
    </div>
    <div class="layout">
      <aside class="panel sidebar">
        <div class="sidebar-head">
          <h2>Paper List</h2>
          <div class="hint">Click a paper to inspect its converted idea and score breakdown.</div>
          <div class="filter-row">
            <label for="decision-group-filter">Status Filter</label>
            <select id="decision-group-filter">
              <option value="all">All papers</option>
              <option value="accepted">Accepted</option>
              <option value="rejected">Rejected</option>
              <option value="other">Other</option>
            </select>
          </div>
        </div>
        <div class="paper-list" id="paper-list"></div>
      </aside>
      <main class="content">
        <section class="panel section">
          <h2>Global Analysis</h2>
          <div class="stats" id="summary-stats"></div>
          <div class="chart-grid">
            <div class="chart-card wide">
              <div class="chart-title">
                <span>Average radar</span>
                <span class="chart-note">Mean of each scoring dimension across scored papers</span>
              </div>
              <div id="summary-radar"></div>
            </div>
            <div class="chart-card wide">
              <div class="chart-title">
                <span>Correlation grid</span>
                <span class="chart-note">ICLR score vs each evaluation dimension</span>
              </div>
              <div id="correlation-grid" class="chart-grid"></div>
            </div>
          </div>
        </section>
        <section class="panel section">
          <div class="selected-head">
            <div>
              <h2 style="margin:0 0 6px;">Selected Paper</h2>
              <h3 id="selected-title"></h3>
              <div class="sub" id="selected-meta"></div>
            </div>
            <div id="selected-badges" class="paper-sub"></div>
          </div>
          <div class="selected-grid">
            <div class="chart-card">
              <div class="chart-title">
                <span>Selected radar</span>
                <span class="chart-note">Current paper compared against all dimensions</span>
              </div>
              <div id="selected-radar"></div>
            </div>
            <div class="detail-grid">
              <div class="field">
                <div class="k">Idea Title</div>
                <div class="v" id="idea-title"></div>
              </div>
              <div class="field">
                <div class="k">Core Paper</div>
                <div class="v" id="idea-core"></div>
              </div>
              <div class="field" style="grid-column: 1 / -1;">
                <div class="k">Summary</div>
                <div class="v" id="idea-summary"></div>
              </div>
              <div class="field" style="grid-column: 1 / -1;">
                <div class="k">Problem</div>
                <div class="v" id="idea-problem"></div>
              </div>
              <div class="field" style="grid-column: 1 / -1;">
                <div class="k">Method</div>
                <div class="v" id="idea-method"></div>
              </div>
            </div>
          </div>
          <div class="evaluation-grid" id="evaluation-grid"></div>
          <div class="list-block">
            <div class="list-box">
              <h4>Strengths</h4>
              <ul id="evaluation-strengths"></ul>
            </div>
            <div class="list-box">
              <h4>Major Issues</h4>
              <ul id="evaluation-issues"></ul>
            </div>
            <div class="list-box">
              <h4>Revision Instruction</h4>
              <div class="v" id="evaluation-revision"></div>
            </div>
            <div class="list-box">
              <h4>Raw Idea JSON</h4>
              <pre id="idea-raw" class="v" style="margin:0; white-space:pre-wrap;"></pre>
            </div>
          </div>
          <div class="error-box" id="selected-error" style="display:none;"></div>
        </section>
      </main>
    </div>
  </div>
  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
    const reportData = JSON.parse(document.getElementById('report-data').textContent);
    const CRITERIA = reportData.criteria || ['novelty', 'clarity', 'feasibility', 'relevance', 'expected_effectiveness', 'evidence_grounding'];
    const CRITERION_SCORE_MAX = reportData.criterion_score_max || 5;
    const OVERALL_SCORE_MAX = reportData.overall_score_max || 10;
    const CRITERION_LABELS = {{
      novelty: 'Novelty',
      clarity: 'Clarity',
      feasibility: 'Feasibility',
      relevance: 'Relevance',
      expected_effectiveness: 'Expected Effectiveness',
      evidence_grounding: 'Evidence Grounding',
    }};
    const state = {{
      decisionFilter: 'all',
      selectedOpenreviewId: initialSelectedId(reportData.records),
    }};

    function initialSelectedId(records) {{
      const scored = records.find((record) => hasEvaluation(record));
      return (scored && scored.openreview_id) || (records[0] && records[0].openreview_id) || null;
    }}

    function hasEvaluation(record) {{
      return Boolean(record && record.idea_evaluation);
    }}

    function getDecisionGroup(record) {{
      return (record && record.decision_group) || classifyDecisionGroup(record ? record.status : '');
    }}

    function classifyDecisionGroup(status) {{
      const text = String(status || '').toLowerCase();
      if (['rejected', 'withdrawn', 'desk rejected', 'desk reject', 'not accepted'].some((marker) => text.includes(marker))) {{
        return 'rejected';
      }}
      if (['oral', 'spotlight', 'poster'].some((marker) => text.includes(marker))) {{
        return 'accepted';
      }}
      return 'other';
    }}

    function getVisibleRecords() {{
      if (state.decisionFilter === 'all') {{
        return reportData.records;
      }}
      return reportData.records.filter((record) => getDecisionGroup(record) === state.decisionFilter);
    }}

    function getSelectedRecord(records = getVisibleRecords()) {{
      if (!records.length) {{
        return null;
      }}
      const selected = records.find((record) => record.openreview_id === state.selectedOpenreviewId);
      return selected || records[0] || null;
    }}

    function toNumber(value) {{
      const n = Number(value);
      return Number.isFinite(n) ? n : null;
    }}

    function escapeHtml(text) {{
      return String(text ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function formatText(text) {{
      return escapeHtml(text).replaceAll('\\n', '<br>');
    }}

    function getScore(record, key) {{
      if (!record || !record.idea_evaluation || !record.idea_evaluation[key]) {{
        return null;
      }}
      return toNumber(record.idea_evaluation[key].score);
    }}

    function getCriterionText(record, key, field) {{
      if (!record || !record.idea_evaluation || !record.idea_evaluation[key]) {{
        return '';
      }}
      return record.idea_evaluation[key][field] || '';
    }}

    function average(values) {{
      const valid = values.filter((value) => Number.isFinite(value));
      if (!valid.length) {{
        return null;
      }}
      return valid.reduce((sum, value) => sum + value, 0) / valid.length;
    }}

    function pearson(xs, ys) {{
      const points = [];
      for (let i = 0; i < xs.length; i += 1) {{
        if (Number.isFinite(xs[i]) && Number.isFinite(ys[i])) {{
          points.push([xs[i], ys[i]]);
        }}
      }}
      if (points.length < 2) {{
        return null;
      }}
      const meanX = average(points.map((point) => point[0]));
      const meanY = average(points.map((point) => point[1]));
      let numerator = 0;
      let denomX = 0;
      let denomY = 0;
      for (const [x, y] of points) {{
        const dx = x - meanX;
        const dy = y - meanY;
        numerator += dx * dy;
        denomX += dx * dx;
        denomY += dy * dy;
      }}
      const denom = Math.sqrt(denomX * denomY);
      return denom ? numerator / denom : null;
    }}

    function regressionLine(points) {{
      if (points.length < 2) {{
        return null;
      }}
      const xs = points.map((point) => point.x);
      const ys = points.map((point) => point.y);
      const meanX = average(xs);
      const meanY = average(ys);
      let numerator = 0;
      let denom = 0;
      for (const point of points) {{
        numerator += (point.x - meanX) * (point.y - meanY);
        denom += (point.x - meanX) ** 2;
      }}
      if (!denom) {{
        return null;
      }}
      const slope = numerator / denom;
      const intercept = meanY - slope * meanX;
      return {{ slope, intercept }};
    }}

    function colorForScore(score) {{
      if (!Number.isFinite(score)) {{
        return '#94a3b8';
      }}
      const pct = Math.max(0, Math.min(1, score / 10));
      const hue = 12 + (115 * pct);
      return `hsl(${hue}, 72%, 45%)`;
    }}

    function renderStats() {{
      const records = getVisibleRecords();
      const scored = records.filter((record) => hasEvaluation(record));
      const iclrScores = scored.map((record) => toNumber(record.iclr_score)).filter((value) => Number.isFinite(value));
      const overallScores = scored.map((record) => toNumber(record.overall_score)).filter((value) => Number.isFinite(value));
      const avgIclr = average(iclrScores);
      const avgOverall = average(overallScores);
      const selected = getSelectedRecord();
      const cards = [
        ['Visible papers', records.length],
        ['Scored papers', scored.length],
        ['Accepted', records.filter((record) => getDecisionGroup(record) === 'accepted').length],
        ['Rejected', records.filter((record) => getDecisionGroup(record) === 'rejected').length],
        ['Mean ICLR', avgIclr === null ? 'N/A' : avgIclr.toFixed(2)],
        ['Mean overall', avgOverall === null ? 'N/A' : avgOverall.toFixed(2)],
      ];
      document.getElementById('summary-stats').innerHTML = cards.map(([label, value]) => `
        <div class="stat">
          <div class="label">${escapeHtml(label)}</div>
          <div class="value">${escapeHtml(value)}</div>
        </div>
      `).join('');
      document.getElementById('page-meta').innerHTML = `
        <div>${escapeHtml(reportData.records.length)} papers loaded</div>
        <div>${escapeHtml(records.length)} papers visible</div>
        <div>${escapeHtml(scored.length)} papers with scores</div>
        <div>Selected: ${escapeHtml(selected ? selected.title : 'N/A')}</div>
      `;
    }}

    function averageCriterionScores(records) {{
      return CRITERIA.map((criterion) => {{
        const values = records.map((record) => getScore(record, criterion)).filter((value) => Number.isFinite(value));
        return average(values);
      }});
    }}

    function radarSvg(labels, values, options = {{ size: 340 }}) {{
      const size = options.size || 340;
      const cx = size / 2;
      const cy = size / 2;
      const radius = size * 0.33;
      const maxValue = options.maxValue || CRITERION_SCORE_MAX;
      const points = [];
      const axes = [];
      const gridLevels = [0.25, 0.5, 0.75, 1];
      for (const level of gridLevels) {{
        const ring = [];
        for (let i = 0; i < labels.length; i += 1) {{
          const angle = -Math.PI / 2 + (2 * Math.PI * i / labels.length);
          const x = cx + Math.cos(angle) * radius * level;
          const y = cy + Math.sin(angle) * radius * level;
          ring.push(`${x},${y}`);
        }}
        axes.push(`<polygon points="${ring.join(' ')}" fill="none" stroke="#dbe2ea" stroke-width="1"/>`);
      }}
      for (let i = 0; i < labels.length; i += 1) {{
        const angle = -Math.PI / 2 + (2 * Math.PI * i / labels.length);
        const x = cx + Math.cos(angle) * radius;
        const y = cy + Math.sin(angle) * radius;
        axes.push(`<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="#cbd5e1" stroke-width="1"/>`);
        const labelX = cx + Math.cos(angle) * (radius + 28);
        const labelY = cy + Math.sin(angle) * (radius + 28);
        const anchor = Math.cos(angle) > 0.2 ? 'start' : Math.cos(angle) < -0.2 ? 'end' : 'middle';
        axes.push(`<text x="${labelX}" y="${labelY}" text-anchor="${anchor}" dominant-baseline="middle" font-size="11" fill="#475569">${escapeHtml(labels[i])}</text>`);
        axes.push(`<text x="${cx + Math.cos(angle) * (radius + 10)}" y="${cy + Math.sin(angle) * (radius + 10)}" text-anchor="middle" dominant-baseline="middle" font-size="10" fill="#94a3b8">${maxValue}</text>`);
      }}
      for (let i = 0; i < labels.length; i += 1) {{
        const angle = -Math.PI / 2 + (2 * Math.PI * i / labels.length);
        const value = Number.isFinite(values[i]) ? values[i] : 0;
        const ratio = Math.max(0, Math.min(1, value / maxValue));
        const x = cx + Math.cos(angle) * radius * ratio;
        const y = cy + Math.sin(angle) * radius * ratio;
        points.push(`${x},${y}`);
      }}
      const poly = points.length ? `<polygon points="${points.join(' ')}" fill="rgba(37, 99, 235, 0.2)" stroke="#2563eb" stroke-width="2"/>` : '';
      return `
        <svg viewBox="0 0 ${size} ${size}" width="100%" height="${size}" role="img" aria-label="Radar chart">
          ${axes.join('')}
          ${poly}
          <circle cx="${cx}" cy="${cy}" r="2.5" fill="#2563eb"/>
        </svg>
      `;
    }}

    function scatterSvg(points, title) {{
      const width = 330;
      const height = 230;
      const pad = {{ left: 42, right: 16, top: 16, bottom: 34 }};
      const xMin = 0;
      const xMax = OVERALL_SCORE_MAX;
      const yMin = 1;
      const yMax = CRITERION_SCORE_MAX;
      const plotWidth = width - pad.left - pad.right;
      const plotHeight = height - pad.top - pad.bottom;
      function xScale(value) {{
        return pad.left + ((value - xMin) / (xMax - xMin)) * plotWidth;
      }}
      function yScale(value) {{
        return pad.top + plotHeight - ((value - yMin) / (yMax - yMin)) * plotHeight;
      }}
      const valid = points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
      const line = regressionLine(valid);
      const path = line ? `
        <line x1="${xScale(xMin)}" y1="${yScale(line.slope * xMin + line.intercept)}"
              x2="${xScale(xMax)}" y2="${yScale(line.slope * xMax + line.intercept)}"
              stroke="#2563eb" stroke-width="2" stroke-dasharray="6 4"/>
      ` : '';
      const circles = valid.map((point) => `
        <circle cx="${xScale(point.x)}" cy="${yScale(point.y)}" r="4.2" fill="${colorForScore(point.x)}" fill-opacity="0.85">
          <title>ICLR ${point.x.toFixed(1)} | ${title}: ${point.y.toFixed(1)}</title>
        </circle>
      `).join('');
      const xTicks = [0, 2, 4, 6, 8, 10].filter((tick) => tick <= xMax);
      const yTicks = Array.from({{ length: yMax - yMin + 1 }}, (_, index) => yMin + index);
      const grid = [
        ...xTicks.map((tick) => `
        <line x1="${xScale(tick)}" y1="${yScale(yMin)}" x2="${xScale(tick)}" y2="${yScale(yMax)}" stroke="#edf2f7" stroke-width="1"/>
        <text x="${xScale(tick)}" y="${height - 12}" text-anchor="middle" font-size="10" fill="#94a3b8">${tick}</text>
        `),
        ...yTicks.map((tick) => `
        <line x1="${xScale(xMin)}" y1="${yScale(tick)}" x2="${xScale(xMax)}" y2="${yScale(tick)}" stroke="#edf2f7" stroke-width="1"/>
        <text x="14" y="${yScale(tick) + 3}" font-size="10" fill="#94a3b8">${tick}</text>
        `),
      ].join('');
      return `
        <svg viewBox="0 0 ${width} ${height}" width="100%" height="${height}" role="img" aria-label="${escapeHtml(title)} correlation">
          ${grid}
          <line x1="${xScale(xMin)}" y1="${yScale(yMin)}" x2="${xScale(xMax)}" y2="${yScale(yMin)}" stroke="#94a3b8" stroke-width="1.2"/>
          <line x1="${xScale(xMin)}" y1="${yScale(yMin)}" x2="${xScale(xMin)}" y2="${yScale(yMax)}" stroke="#94a3b8" stroke-width="1.2"/>
          ${path}
          ${circles}
          <text x="${width / 2}" y="${height - 2}" text-anchor="middle" font-size="11" fill="#475569">ICLR score</text>
          <text x="10" y="${height / 2}" text-anchor="middle" font-size="11" fill="#475569" transform="rotate(-90 10 ${height / 2})">${escapeHtml(title)}</text>
        </svg>
      `;
    }}

    function renderSummaryRadar() {{
      const averages = averageCriterionScores(getVisibleRecords());
      document.getElementById('summary-radar').innerHTML = radarSvg(
        CRITERIA.map((criterion) => CRITERION_LABELS[criterion] || criterion),
        averages,
        {{ size: 420, maxValue: CRITERION_SCORE_MAX }}
      );
    }}

    function renderCorrelationGrid() {{
      const scored = getVisibleRecords().filter((record) => hasEvaluation(record) && Number.isFinite(toNumber(record.iclr_score)));
      const grid = document.getElementById('correlation-grid');
      if (!scored.length) {{
        grid.innerHTML = '<div class="empty">No scored papers available yet.</div>';
        return;
      }}
      grid.innerHTML = CRITERIA.map((criterion) => {{
        const points = scored.map((record) => {{
          const iclr = toNumber(record.iclr_score);
          const score = getScore(record, criterion);
          return {{ x: iclr, y: score }};
        }}).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
        const corr = pearson(points.map((point) => point.x), points.map((point) => point.y));
        return `
          <div class="chart-card">
            <div class="chart-title">
              <span>${escapeHtml(CRITERION_LABELS[criterion] || criterion)}</span>
              <span class="chart-note">r=${corr === null ? 'N/A' : corr.toFixed(2)}</span>
            </div>
            ${scatterSvg(points, CRITERION_LABELS[criterion] || criterion)}
          </div>
        `;
      }}).join('');
    }}

    function renderPaperList() {{
      const list = document.getElementById('paper-list');
      const records = getVisibleRecords();
      if (!records.length) {{
        list.innerHTML = '<div class="empty" style="padding:12px;">No papers match the current filter.</div>';
        return;
      }}
      if (!records.some((record) => record.openreview_id === state.selectedOpenreviewId)) {{
        state.selectedOpenreviewId = records[0].openreview_id;
      }}
      list.innerHTML = records.map((record) => `
        <button class="paper-item ${record.openreview_id === state.selectedOpenreviewId ? 'active' : ''}" data-openreview-id="${escapeHtml(record.openreview_id)}">
          <div class="paper-title">${escapeHtml(record.title)}</div>
          <div class="paper-sub">
            <span class="pill ${escapeHtml(getDecisionGroup(record))}">${escapeHtml(getDecisionGroup(record))}</span>
            <span class="pill gray">${escapeHtml(record.status || 'Unknown')}</span>
            <span class="pill gray">ICLR ${formatOptionalScore(record.iclr_score)}</span>
            <span class="pill gray">Overall ${formatOptionalScore(record.overall_score)}</span>
          </div>
        </button>
      `).join('');
      list.querySelectorAll('.paper-item').forEach((button) => {{
        button.addEventListener('click', () => {{
          state.selectedOpenreviewId = button.dataset.openreviewId;
          renderPaperList();
          renderSelectedPaper();
          renderStats();
          renderSummaryRadar();
          renderCorrelationGrid();
        }});
      }});
    }}

    function formatOptionalScore(score) {{
      const value = toNumber(score);
      return value === null ? 'N/A' : value.toFixed(1);
    }}

    function renderSelectedPaper() {{
      const record = getSelectedRecord();
      const title = document.getElementById('selected-title');
      const meta = document.getElementById('selected-meta');
      const badges = document.getElementById('selected-badges');
      const errorBox = document.getElementById('selected-error');
      if (!record) {{
        title.textContent = 'No paper selected';
        meta.textContent = '';
        badges.innerHTML = '';
        errorBox.style.display = 'none';
        return;
      }}
      title.textContent = record.title || '';
      const iclr = toNumber(record.iclr_score);
      const overall = toNumber(record.overall_score);
      const delta = (iclr !== null && overall !== null) ? (overall - iclr) : null;
      meta.innerHTML = [
        `OpenReview ID: ${escapeHtml(record.openreview_id)}`,
        `Status: ${escapeHtml(record.status || 'N/A')}`,
        `Group: ${escapeHtml(getDecisionGroup(record))}`,
        `Text source: ${escapeHtml(record.text_source || 'N/A')}`,
        `PDF: ${record.pdf_url ? `<a href="${escapeHtml(record.pdf_url)}" target="_blank" rel="noreferrer">link</a>` : 'N/A'}`,
      ].join(' · ');
      badges.innerHTML = `
        <span class="pill">ICLR ${formatOptionalScore(record.iclr_score)}</span>
        <span class="pill gray">Overall ${formatOptionalScore(record.overall_score)}</span>
        <span class="pill gray">Δ ${delta === null ? 'N/A' : delta.toFixed(1)}</span>
      `;
      errorBox.style.display = record.error ? 'block' : 'none';
      errorBox.textContent = record.error || '';
      document.getElementById('selected-radar').innerHTML = radarSvg(
        CRITERIA.map((criterion) => CRITERION_LABELS[criterion] || criterion),
        CRITERIA.map((criterion) => getScore(record, criterion)),
        {{ size: 360, maxValue: CRITERION_SCORE_MAX }}
      );
      const idea = record.idea_output || null;
      document.getElementById('idea-title').innerHTML = idea ? formatText(idea.title) : '<span class="empty">No converted idea available.</span>';
      document.getElementById('idea-core').innerHTML = idea ? formatText(`${idea.core_paper_title || ''}\n${idea.core_paper_id || ''}`.trim()) : '<span class="empty">No core paper available.</span>';
      document.getElementById('idea-summary').innerHTML = idea ? formatText(idea.summary || '') : '<span class="empty">No summary available.</span>';
      document.getElementById('idea-problem').innerHTML = idea ? formatText(idea.problem || '') : '<span class="empty">No problem statement available.</span>';
      document.getElementById('idea-method').innerHTML = idea ? formatText(idea.method || '') : '<span class="empty">No method description available.</span>';
      document.getElementById('idea-raw').textContent = idea ? JSON.stringify(idea, null, 2) : '';

      const evaluation = record.idea_evaluation || null;
      if (!evaluation) {{
        document.getElementById('evaluation-grid').innerHTML = '<div class="empty">No evaluation available.</div>';
        document.getElementById('evaluation-strengths').innerHTML = '';
        document.getElementById('evaluation-issues').innerHTML = '';
        document.getElementById('evaluation-revision').innerHTML = '';
        return;
      }}

      document.getElementById('evaluation-grid').innerHTML = CRITERIA.map((criterion) => {{
        const data = evaluation[criterion] || {{}};
        return `
          <div class="criterion">
            <div class="name">
              <span>${escapeHtml(CRITERION_LABELS[criterion] || criterion)}</span>
              <span class="score">${escapeHtml(data.score ?? 'N/A')}</span>
            </div>
            <div class="text"><strong>Comment:</strong> ${formatText(data.comment || '')}</div>
            <div class="text" style="margin-top:8px;"><strong>Revision:</strong> ${formatText(data.revision_suggestion || '')}</div>
          </div>
        `;
      }}).join('');
      document.getElementById('evaluation-strengths').innerHTML = (evaluation.strengths || []).map((item) => `<li>${formatText(item)}</li>`).join('') || '<li class="empty">None</li>';
      document.getElementById('evaluation-issues').innerHTML = (evaluation.major_issues || []).map((item) => `<li>${formatText(item)}</li>`).join('') || '<li class="empty">None</li>';
      document.getElementById('evaluation-revision').innerHTML = formatText(evaluation.revision_instruction || '');
    }}

    function bindDecisionFilter() {{
      const filter = document.getElementById('decision-group-filter');
      filter.value = state.decisionFilter;
      filter.addEventListener('change', () => {{
        state.decisionFilter = filter.value;
        const visible = getVisibleRecords();
        state.selectedOpenreviewId = visible.length ? visible[0].openreview_id : null;
        renderStats();
        renderSummaryRadar();
        renderCorrelationGrid();
        renderPaperList();
        renderSelectedPaper();
      }});
    }}

    bindDecisionFilter();
    renderStats();
    renderSummaryRadar();
    renderCorrelationGrid();
    renderPaperList();
    renderSelectedPaper();
  </script>
</body>
</html>
"""
    content = content.replace("{{", "{").replace("}}", "}").replace("__REPORT_DATA__", data_json)
    path.write_text(content, encoding="utf-8")


def _write_outputs(output_dir: Path, records: list[dict[str, Any]]) -> None:
    _write_results_json(output_dir / "results.json", records)
    _write_results_csv(output_dir / "results.csv", records)
    _write_report_html(output_dir / "report.html", records)


def _load_existing_records(path: Path, log: logging.Logger) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("accepted_paper_benchmark:existing_results_ignored path=%s error=%s", path, exc)
        return []
    if not isinstance(data, list):
        log.warning("accepted_paper_benchmark:existing_results_ignored path=%s error=expected_list", path)
        return []
    records = [record for record in data if isinstance(record, dict)]
    if len(records) != len(data):
        log.warning(
            "accepted_paper_benchmark:existing_results_partially_ignored path=%s ignored_count=%s",
            path,
            len(data) - len(records),
        )
    return records


def _merge_records_by_openreview_id(
    existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(existing_records)
    index_by_id = {
        record["openreview_id"]: index
        for index, record in enumerate(merged)
        if record.get("openreview_id")
    }
    for record in new_records:
        openreview_id = record.get("openreview_id")
        if openreview_id and openreview_id in index_by_id:
            merged[index_by_id[openreview_id]] = record
        else:
            if openreview_id:
                index_by_id[openreview_id] = len(merged)
            merged.append(record)
    return merged


def _format_optional_score(score: float | None) -> str:
    if score is None:
        return ""
    return html.escape(str(score))


def _report_record(record: dict[str, Any]) -> dict[str, Any]:
    idea_output = record.get("idea_output")
    idea_evaluation = record.get("idea_evaluation")
    return {
        "openreview_id": record.get("openreview_id"),
        "title": record.get("title"),
        "status": record.get("status"),
        "decision_group": record.get("decision_group") or _decision_group(record.get("status", "")),
        "pdf_url": record.get("pdf_url"),
        "text_source": record.get("text_source"),
        "full_text_markdown_path": record.get("full_text_markdown_path"),
        "input_truncated": record.get("input_truncated"),
        "iclr_scores": record.get("iclr_scores"),
        "iclr_score": record.get("iclr_score"),
        "overall_score": _overall_score(record),
        "idea_output": idea_output,
        "idea_evaluation": idea_evaluation,
        "error": record.get("error"),
    }


def _html_json(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False)
    return raw.replace("</", "<\\/")


def _overall_score(record: dict[str, Any]) -> Any:
    evaluation = record.get("idea_evaluation")
    if not evaluation:
        return None
    return evaluation.get("overall_score")


def _decision_group(status: str | None) -> str:
    normalized = (status or "").casefold()
    if any(marker in normalized for marker in ("rejected", "withdrawn", "desk rejected", "desk reject", "not accepted")):
        return "rejected"
    if any(marker in normalized for marker in ("oral", "spotlight", "poster")):
        return "accepted"
    return "other"
