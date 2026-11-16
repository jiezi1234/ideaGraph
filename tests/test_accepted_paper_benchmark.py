import json

from src.ingestion.openreview_client import OpenReviewPaper
from src.schemas.agent import IdeaEvaluation, IdeaOutput
from src.schemas.document import PaperDocument
import src.workflows.accepted_paper_benchmark as benchmark_module
from src.workflows.accepted_paper_benchmark import (
    AcceptedPaperBenchmarkWorkflow,
    convert_paper_to_idea,
)


class FakeConversionModel:
    def __init__(self):
        self.calls = []

    def structured_json(self, *, messages, schema, name: str, metadata=None):
        self.calls.append((messages, schema, name, metadata))
        return type(
            "Result",
            (),
            {
                "parsed": schema(
                    title="基于图谱的研究想法",
                    summary="将已接收论文转化为可评估的中文研究想法。",
                    problem="现有方法缺少可复用的基准输入。",
                    method="使用论文正文生成结构化想法。",
                    core_paper_id="paper-1",
                    core_paper_title="Accepted Graph Paper",
                ),
                "usage": {},
            },
        )()


class FailingForPaperConversionModel(FakeConversionModel):
    def structured_json(self, *, messages, schema, name: str, metadata=None):
        user_content = messages[1]["content"]
        if "paper_id: paper-bad" in user_content:
            raise RuntimeError("conversion failed for paper-bad")
        return super().structured_json(messages=messages, schema=schema, name=name, metadata=metadata)


class FakeEvaluatorModel:
    def __init__(self):
        self.calls = []

    def structured_json(self, *, messages, schema, name: str, metadata=None):
        self.calls.append((messages, schema, name, metadata))
        return type(
            "Result",
            (),
            {
                "parsed": schema(
                    overall_score=8,
                    summary="想法清晰，证据充足。",
                    novelty={
                        "score": 8,
                        "comment": "有一定新颖性。",
                        "revision_suggestion": "进一步区分已有方法。",
                    },
                    clarity={
                        "score": 9,
                        "comment": "表达清晰。",
                        "revision_suggestion": "补充实验设定。",
                    },
                    feasibility={
                        "score": 7,
                        "comment": "实现可行。",
                        "revision_suggestion": "说明资源需求。",
                    },
                    relevance={
                        "score": 8,
                        "comment": "与任务相关。",
                        "revision_suggestion": "加强场景描述。",
                    },
                    expected_effectiveness={
                        "score": 7,
                        "comment": "预期有效。",
                        "revision_suggestion": "补充消融。",
                    },
                    evidence_grounding={
                        "score": 8,
                        "comment": "基于生成想法本身。",
                        "revision_suggestion": "补充证据来源。",
                    },
                    strengths=["结构完整"],
                    major_issues=["需要更多实验细节"],
                    revision_instruction="补充实验设计。",
                ),
                "usage": {},
            },
        )()


class FakeOpenReviewClient:
    def __init__(self, papers, scores_by_id):
        self.papers = papers
        self.scores_by_id = scores_by_id
        self.fetch_calls = []
        self.score_calls = []

    def fetch_venue_papers(self, venue, limit=None):
        self.fetch_calls.append((venue, limit))
        return self.papers

    def fetch_review_scores(self, paper_id):
        self.score_calls.append(paper_id)
        return self.scores_by_id.get(paper_id, [])


class FakeFullTextProvider:
    def __init__(self, markdown_by_id):
        self.markdown_by_id = markdown_by_id
        self.calls = []

    def enrich(self, document):
        self.calls.append(document)
        markdown = self.markdown_by_id.get(document.document_id)
        if markdown is None:
            return document
        return document.model_copy(
            update={
                "full_text_markdown": markdown,
                "full_text_markdown_path": f"/tmp/{document.document_id}.md",
                "text_source": "full_text_markdown",
            }
        )


class PdfThenArxivFallbackFullTextProvider:
    def __init__(self, pdf_markdown: str, arxiv_markdown: str):
        self.pdf_markdown = pdf_markdown
        self.arxiv_markdown = arxiv_markdown
        self.calls = []

    def enrich(self, document):
        self.calls.append(document)
        if document.pdf_url:
            return document.model_copy(
                update={
                    "full_text_markdown": self.pdf_markdown,
                    "full_text_markdown_path": f"/tmp/{document.document_id}.pdf.md",
                    "text_source": "full_text_markdown",
                }
            )
        return document.model_copy(
            update={
                "full_text_markdown": self.arxiv_markdown,
                "full_text_markdown_path": f"/tmp/{document.document_id}.arxiv.md",
                "text_source": "full_text_markdown",
            }
        )


def test_convert_paper_to_idea_uses_full_text_message_and_validates_schema():
    paper = OpenReviewPaper(
        openreview_id="paper-1",
        title="Accepted Graph Paper",
        abstract="Abstract should not be the primary content.",
        status="ICLR 2026 Poster",
    )
    model = FakeConversionModel()

    idea = convert_paper_to_idea(
        paper=paper,
        text="Full text body without references.",
        text_source="full_text_markdown",
        model=model,
        prompt_template="Convert prompt",
    )

    assert isinstance(idea, IdeaOutput)
    assert idea.title == "基于图谱的研究想法"
    assert model.calls[0][1] is IdeaOutput
    assert model.calls[0][2] == "benchmark_convert_paper_to_idea"
    assert model.calls[0][3] == {"stage": "benchmark_convert_paper_to_idea"}
    assert model.calls[0][0][0] == {"role": "system", "content": "Convert prompt"}
    user_content = model.calls[0][0][1]["content"]
    assert "paper-1" in user_content
    assert "Accepted Graph Paper" in user_content
    assert "full_text_markdown" in user_content
    assert "Full text body without references." in user_content
    assert "Abstract should not be the primary content." not in user_content


def test_workflow_strips_references_evaluates_idea_only_and_exports_files(tmp_path):
    papers = [
        OpenReviewPaper(
            openreview_id="paper-1",
            title="Accepted Graph Paper",
            abstract="Accepted abstract",
            status="ICLR 2026 Poster",
            pdf_url="https://openreview.net/pdf?id=paper-1",
        ),
        OpenReviewPaper(
            openreview_id="paper-2",
            title="Rejected Graph Paper",
            abstract="Rejected abstract",
            status="ICLR 2026 Rejected",
        ),
    ]
    openreview_client = FakeOpenReviewClient(papers=papers, scores_by_id={"paper-1": [8.0, 6.0]})
    full_text_provider = FakeFullTextProvider(
        {
            "Accepted Graph Paper": (
                "# Accepted Graph Paper\n\n"
                "Main method content for benchmark.\n\n"
                "## References\n\n"
                "[1] Should be stripped."
            )
        }
    )
    conversion_model = FakeConversionModel()
    evaluator_model = FakeEvaluatorModel()
    workflow = AcceptedPaperBenchmarkWorkflow(
        openreview_client=openreview_client,
        full_text_provider=full_text_provider,
        conversion_model=conversion_model,
        evaluation_model=evaluator_model,
        conversion_prompt_template="Convert prompt",
        evaluation_prompt_template="Evaluate prompt",
    )

    result = workflow.run(
        venue="ICLR.cc/2026/Conference",
        query="graph secret query",
        max_papers=5,
        output_dir=tmp_path / "benchmark",
    )

    assert result == {"paper_count": 1, "output_dir": str(tmp_path / "benchmark")}
    assert openreview_client.fetch_calls == [("ICLR.cc/2026/Conference", None)]
    assert openreview_client.score_calls == ["paper-1"]
    assert isinstance(full_text_provider.calls[0], PaperDocument)

    output_dir = tmp_path / "benchmark"
    assert (output_dir / "results.json").exists()
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "report.html").exists()

    records = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert len(records) == 1
    record = records[0]
    assert record["openreview_id"] == "paper-1"
    assert record["pdf_url"] == "https://openreview.net/pdf?id=paper-1"
    assert record["text_source"] == "full_text_markdown"
    assert record["full_text_markdown_path"] == "/tmp/Accepted Graph Paper.md"
    assert record["iclr_scores"] == [8.0, 6.0]
    assert record["iclr_score"] == 7.0
    assert "Main method content for benchmark." in record["input_text"]
    assert "Should be stripped" not in record["input_text"]
    assert record["idea_output"]["title"] == "基于图谱的研究想法"
    assert record["idea_evaluation"]["overall_score"] == 8
    assert evaluator_model.calls[0][1] is IdeaEvaluation
    assert evaluator_model.calls[0][2] == "idea_evaluation"
    assert evaluator_model.calls[0][3] == {"stage": "evaluation"}

    csv_text = (output_dir / "results.csv").read_text(encoding="utf-8")
    assert "openreview_id,title,status,decision_group,text_source,iclr_score,overall_score" in csv_text
    assert "paper-1,Accepted Graph Paper,ICLR 2026 Poster,accepted,full_text_markdown,7.0,8" in csv_text
    html_text = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "Selected radar" in html_text
    assert "Correlation grid" in html_text
    assert "paper-list" in html_text
    assert "report-data" in html_text
    embedded_json = html_text.split('<script id="report-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    report_data = json.loads(embedded_json)
    assert report_data["criteria"] == benchmark_module.EVALUATION_CRITERIA
    assert report_data["criterion_score_max"] == 5
    assert report_data["overall_score_max"] == 10
    assert report_data["records"][0]["idea_output"]["title"] == "基于图谱的研究想法"
    assert report_data["records"][0]["idea_evaluation"]["novelty"]["score"] == 8

    evaluator_messages = evaluator_model.calls[0][0]
    assert evaluator_messages[0] == {"role": "system", "content": "Evaluate prompt"}
    evaluator_user_content = evaluator_messages[1]["content"]
    assert "基于图谱的研究想法" in evaluator_user_content
    assert "secret query" not in evaluator_user_content
    assert "Main method content for benchmark" not in evaluator_user_content
    assert "graph_context" not in evaluator_user_content
    assert "selected_paper" not in evaluator_user_content


def test_workflow_filters_by_decision_scope_and_keeps_status_in_reports(tmp_path):
    papers = [
        OpenReviewPaper(
            openreview_id="paper-accepted",
            title="Accepted Graph Paper",
            abstract="Accepted abstract",
            status="ICLR 2026 Poster",
        ),
        OpenReviewPaper(
            openreview_id="paper-rejected",
            title="Rejected Graph Paper",
            abstract="Rejected abstract",
            status="ICLR 2026 Rejected",
        ),
        OpenReviewPaper(
            openreview_id="paper-other",
            title="Other Graph Paper",
            abstract="Other abstract",
            status="ICLR 2026 Under Review",
        ),
    ]
    openreview_client = FakeOpenReviewClient(
        papers=papers,
        scores_by_id={"paper-rejected": [4.0], "paper-accepted": [8.0], "paper-other": [6.0]},
    )
    workflow = AcceptedPaperBenchmarkWorkflow(
        openreview_client=openreview_client,
        full_text_provider=None,
        conversion_model=FakeConversionModel(),
        evaluation_model=FakeEvaluatorModel(),
        conversion_prompt_template="Convert prompt",
        evaluation_prompt_template="Evaluate prompt",
    )

    result = workflow.run(
        venue="ICLR.cc/2026/Conference",
        query="graph",
        max_papers=50,
        output_dir=tmp_path / "benchmark",
        decision_scope="rejected",
    )

    assert result == {"paper_count": 1, "output_dir": str(tmp_path / "benchmark")}
    records = json.loads((tmp_path / "benchmark" / "results.json").read_text(encoding="utf-8"))
    assert [record["openreview_id"] for record in records] == ["paper-rejected"]
    assert records[0]["decision_group"] == "rejected"
    html_text = (tmp_path / "benchmark" / "report.html").read_text(encoding="utf-8")
    assert "Status Filter" in html_text
    assert "decision-group-filter" in html_text


def test_workflow_persists_after_each_paper(tmp_path, monkeypatch):
    papers = [
        OpenReviewPaper(
            openreview_id="paper-1",
            title="Accepted Graph Paper",
            abstract="Accepted abstract",
            status="ICLR 2026 Poster",
        ),
        OpenReviewPaper(
            openreview_id="paper-2",
            title="Second Accepted Paper",
            abstract="Second abstract",
            status="ICLR 2026 Poster",
        ),
    ]
    openreview_client = FakeOpenReviewClient(papers=papers, scores_by_id={})
    workflow = AcceptedPaperBenchmarkWorkflow(
        openreview_client=openreview_client,
        full_text_provider=None,
        conversion_model=FakeConversionModel(),
        evaluation_model=FakeEvaluatorModel(),
        conversion_prompt_template="Convert prompt",
        evaluation_prompt_template="Evaluate prompt",
    )

    write_counts = []

    def record_write(output_dir, records):
        write_counts.append(len(records))

    monkeypatch.setattr(benchmark_module, "_write_outputs", record_write)

    result = workflow.run(
        venue="ICLR.cc/2026/Conference",
        query="paper",
        max_papers=2,
        output_dir=tmp_path / "benchmark",
    )

    assert result == {"paper_count": 2, "output_dir": str(tmp_path / "benchmark")}
    assert write_counts == [1, 2]


def test_workflow_merges_with_existing_outputs_by_openreview_id(tmp_path):
    output_dir = tmp_path / "benchmark"
    output_dir.mkdir()
    existing_records = [
        {
            "openreview_id": "paper-old",
            "title": "Old Paper",
            "status": "ICLR 2026 Poster",
            "pdf_url": "https://openreview.net/pdf?id=paper-old",
            "text_source": "title_abstract",
            "full_text_markdown_path": None,
            "input_text": "# Old Paper\n\nOld abstract.",
            "input_truncated": False,
            "iclr_scores": [6.0],
            "iclr_score": 6.0,
            "idea_output": {
                "title": "旧结果",
                "summary": "旧结果保持不变。",
                "problem": "旧问题",
                "method": "旧方法",
                "core_paper_id": "paper-old",
                "core_paper_title": "Old Paper",
            },
            "idea_evaluation": None,
        },
        {
            "openreview_id": "paper-repeat",
            "title": "Stale Repeat Paper",
            "status": "ICLR 2026 Poster",
            "pdf_url": None,
            "text_source": "title_abstract",
            "full_text_markdown_path": None,
            "input_text": "# Stale Repeat Paper\n\nStale abstract.",
            "input_truncated": False,
            "iclr_scores": [3.0],
            "iclr_score": 3.0,
            "idea_output": None,
            "idea_evaluation": None,
        },
    ]
    (output_dir / "results.json").write_text(json.dumps(existing_records), encoding="utf-8")
    papers = [
        OpenReviewPaper(
            openreview_id="paper-repeat",
            title="Updated Repeat Paper",
            abstract="Updated abstract.",
            status="ICLR 2026 Poster",
        ),
        OpenReviewPaper(
            openreview_id="paper-new",
            title="New Paper",
            abstract="New abstract.",
            status="ICLR 2026 Poster",
        ),
    ]
    workflow = AcceptedPaperBenchmarkWorkflow(
        openreview_client=FakeOpenReviewClient(papers=papers, scores_by_id={"paper-repeat": [9.0], "paper-new": [7.0]}),
        full_text_provider=None,
        conversion_model=FakeConversionModel(),
        evaluation_model=FakeEvaluatorModel(),
        conversion_prompt_template="Convert prompt",
        evaluation_prompt_template="Evaluate prompt",
    )

    workflow.run(
        venue="ICLR.cc/2026/Conference",
        query="paper",
        max_papers=2,
        output_dir=output_dir,
    )

    records = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert [record["openreview_id"] for record in records] == ["paper-old", "paper-repeat", "paper-new"]
    assert records[0]["title"] == "Old Paper"
    assert records[1]["title"] == "Updated Repeat Paper"
    assert records[1]["iclr_score"] == 9.0
    assert records[2]["title"] == "New Paper"

    csv_text = (output_dir / "results.csv").read_text(encoding="utf-8")
    assert "paper-old,Old Paper,ICLR 2026 Poster,accepted,title_abstract,6.0,\n" in csv_text
    assert "paper-repeat,Updated Repeat Paper,ICLR 2026 Poster,accepted,title_abstract,9.0,8" in csv_text
    assert "paper-new,New Paper,ICLR 2026 Poster,accepted,title_abstract,7.0,8" in csv_text


def test_workflow_falls_back_when_stripped_full_text_is_empty(tmp_path):
    paper = OpenReviewPaper(
        openreview_id="paper-refs",
        title="References Only Paper",
        abstract="Useful abstract fallback.",
        status="ICLR 2026 Poster",
    )
    workflow = AcceptedPaperBenchmarkWorkflow(
        openreview_client=FakeOpenReviewClient(papers=[paper], scores_by_id={}),
        full_text_provider=FakeFullTextProvider({"paper-refs": "## References\n\n[1] Ref only."}),
        conversion_model=FakeConversionModel(),
        evaluation_model=FakeEvaluatorModel(),
        conversion_prompt_template="Convert prompt",
        evaluation_prompt_template="Evaluate prompt",
    )

    result = workflow.run(
        venue="ICLR.cc/2026/Conference",
        query="references",
        max_papers=1,
        output_dir=tmp_path / "benchmark",
    )

    assert result["paper_count"] == 1
    records = json.loads((tmp_path / "benchmark" / "results.json").read_text(encoding="utf-8"))
    assert records[0]["text_source"] == "title_abstract"
    assert records[0]["full_text_markdown_path"] is None
    assert records[0]["input_text"] == "# References Only Paper\n\nUseful abstract fallback."


def test_workflow_tries_arxiv_after_pdf_enrichment_fails(tmp_path):
    paper = OpenReviewPaper(
        openreview_id="paper-arxiv",
        title="OpenReview Title Used For Arxiv Lookup",
        abstract="Useful abstract fallback.",
        status="ICLR 2026 Poster",
        pdf_url="https://openreview.net/pdf?id=paper-arxiv",
    )
    provider = PdfThenArxivFallbackFullTextProvider(
        pdf_markdown="## References\n\n[1] PDF only refs.",
        arxiv_markdown="# Title\n\nArxiv body content.\n\n## References\n\n[1] Arxiv refs.",
    )
    workflow = AcceptedPaperBenchmarkWorkflow(
        openreview_client=FakeOpenReviewClient(papers=[paper], scores_by_id={}),
        full_text_provider=provider,
        conversion_model=FakeConversionModel(),
        evaluation_model=FakeEvaluatorModel(),
        conversion_prompt_template="Convert prompt",
        evaluation_prompt_template="Evaluate prompt",
    )

    workflow.run(
        venue="ICLR.cc/2026/Conference",
        query="openreview",
        max_papers=1,
        output_dir=tmp_path / "benchmark",
    )

    records = json.loads((tmp_path / "benchmark" / "results.json").read_text(encoding="utf-8"))
    assert records[0]["text_source"] == "full_text_markdown"
    assert records[0]["input_text"] == "# Title\n\nArxiv body content."
    assert len(provider.calls) == 2
    assert provider.calls[0].pdf_url == "https://openreview.net/pdf?id=paper-arxiv"
    assert provider.calls[1].pdf_url is None


def test_workflow_records_paper_error_and_still_exports_other_papers(tmp_path):
    papers = [
        OpenReviewPaper(
            openreview_id="paper-bad",
            title="Bad Paper",
            abstract="This paper triggers conversion failure.",
            status="ICLR 2026 Poster",
            pdf_url="https://openreview.net/pdf?id=paper-bad",
        ),
        OpenReviewPaper(
            openreview_id="paper-good",
            title="Good Paper",
            abstract="This paper should still export.",
            status="ICLR 2026 Poster",
            pdf_url="https://openreview.net/pdf?id=paper-good",
        ),
    ]
    workflow = AcceptedPaperBenchmarkWorkflow(
        openreview_client=FakeOpenReviewClient(papers=papers, scores_by_id={"paper-bad": [5.0], "paper-good": [8.0]}),
        full_text_provider=None,
        conversion_model=FailingForPaperConversionModel(),
        evaluation_model=FakeEvaluatorModel(),
        conversion_prompt_template="Convert prompt",
        evaluation_prompt_template="Evaluate prompt",
    )

    result = workflow.run(
        venue="ICLR.cc/2026/Conference",
        query="paper",
        max_papers=2,
        output_dir=tmp_path / "benchmark",
    )

    assert result["paper_count"] == 2
    output_dir = tmp_path / "benchmark"
    records = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert [record["openreview_id"] for record in records] == ["paper-bad", "paper-good"]
    assert records[0]["title"] == "Bad Paper"
    assert records[0]["pdf_url"] == "https://openreview.net/pdf?id=paper-bad"
    assert "conversion failed for paper-bad" in records[0]["error"]
    assert records[0]["idea_evaluation"] is None
    assert records[1]["idea_evaluation"]["overall_score"] == 8

    csv_text = (output_dir / "results.csv").read_text(encoding="utf-8")
    assert "paper-bad,Bad Paper,ICLR 2026 Poster,accepted,title_abstract,5.0,\n" in csv_text
    html_text = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "conversion failed for paper-bad" in html_text
    assert "Good Paper" in html_text


def test_workflow_caps_input_text_and_marks_truncation(tmp_path):
    paper = OpenReviewPaper(
        openreview_id="paper-long",
        title="Long Paper",
        abstract="Short abstract.",
        status="ICLR 2026 Poster",
    )
    long_text = "A" * (benchmark_module.MAX_BENCHMARK_INPUT_CHARS + 10)
    workflow = AcceptedPaperBenchmarkWorkflow(
        openreview_client=FakeOpenReviewClient(papers=[paper], scores_by_id={}),
        full_text_provider=FakeFullTextProvider({"Long Paper": long_text}),
        conversion_model=FakeConversionModel(),
        evaluation_model=FakeEvaluatorModel(),
        conversion_prompt_template="Convert prompt",
        evaluation_prompt_template="Evaluate prompt",
    )

    workflow.run(
        venue="ICLR.cc/2026/Conference",
        query="long",
        max_papers=1,
        output_dir=tmp_path / "benchmark",
    )

    records = json.loads((tmp_path / "benchmark" / "results.json").read_text(encoding="utf-8"))
    assert len(records[0]["input_text"]) == benchmark_module.MAX_BENCHMARK_INPUT_CHARS
    assert records[0]["input_truncated"] is True
    user_content = workflow.conversion_model.calls[0][0][1]["content"]
    assert len(user_content.split("<paper_content>\n", 1)[1].split("\n</paper_content>", 1)[0]) == (
        benchmark_module.MAX_BENCHMARK_INPUT_CHARS
    )


def test_benchmark_bootstrap_uses_structured_model_for_conversion_and_evaluation(monkeypatch, tmp_path):
    from src.app import bootstrap
    from src.config.settings import Settings

    calls = {}

    class FakeStructuredModel:
        @classmethod
        def from_settings(cls, settings):
            calls.setdefault("structured_models", []).append(
                {
                    "base_url": settings.ollama_base_url,
                    "model": settings.ollama_model,
                }
            )
            return cls()

    class FakeFullTextProvider:
        def __init__(self, **kwargs) -> None:
            calls["full_text_provider"] = kwargs

    class FakePdfExtractor:
        def __init__(self, **kwargs) -> None:
            calls["pdf_extractor"] = kwargs

    class FakeWorkflow:
        def __init__(self, **kwargs) -> None:
            calls["workflow"] = kwargs

    monkeypatch.setattr(bootstrap, "OllamaOpenAIChatModel", FakeStructuredModel)
    monkeypatch.setattr(bootstrap, "FullTextProvider", FakeFullTextProvider)
    monkeypatch.setattr(bootstrap, "OpenDataLoaderPdfMarkdownExtractor", FakePdfExtractor)
    monkeypatch.setattr(bootstrap, "AcceptedPaperBenchmarkWorkflow", FakeWorkflow)
    monkeypatch.setattr(bootstrap, "OpenReviewClient", lambda: "openreview")
    monkeypatch.setattr(bootstrap, "ArxivClient", lambda: "arxiv")
    monkeypatch.setattr(bootstrap, "PdfDownloader", lambda **kwargs: "downloader")

    settings = Settings(
        OPENALEX_BASE_URL="https://api.openalex.org",
        OLLAMA_BASE_URL="http://127.0.0.1:11435",
        OLLAMA_MODEL="qwen3.6:35b",
        OLLAMA_EMBED_MODEL="qwen3-embedding:4b",
        OLLAMA_TIMEOUT=300,
        OLLAMA_THINK_MODE="medium",
        OLLAMA_CACHE_ENABLED=True,
        OLLAMA_CACHE_DIR=".cache/model_responses",
        FULL_TEXT_OUTPUT_DIR=str(tmp_path / "data"),
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="secret",
        NEO4J_DATABASE="idea",
    )

    workflow = bootstrap.build_accepted_paper_benchmark_workflow(settings)

    assert isinstance(workflow, FakeWorkflow)
    assert calls["structured_models"] == [
        {"base_url": "http://127.0.0.1:11435", "model": "qwen3.6:35b"},
        {"base_url": "http://127.0.0.1:11435", "model": "qwen3.6:35b"},
    ]
    assert calls["workflow"]["conversion_model"] is not calls["workflow"]["evaluation_model"]
