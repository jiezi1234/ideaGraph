from pathlib import Path

import pytest

import main
from src.extraction.paper_abstract import PaperAbstractExtractionOutput, PaperAbstractExtractor
from src.prompts import EXTRACT_PAPER_ABSTRACT_PROMPT
from src.schemas.extraction import ExtractionResult
from src.schemas.relation import PaperCitationRelation
from src.workflows.mvp_graph_build import MvpGraphBuildWorkflow, _local_paper_output_slug


class FakeAbstractModel:
    def __init__(self):
        self.calls = []

    def structured_json(self, *, messages, schema, name: str, metadata=None):
        self.calls.append((messages, schema, name, metadata))
        return type(
            "Result",
            (),
            {
                "parsed": schema(abstract="This paper studies graph-based retrieval for scientific idea generation."),
                "usage": {"input": 42, "output": 9, "total": 51},
            },
        )()


class SummaryOnlyAbstractModel:
    def structured_json(self, *, messages, schema, name: str, metadata=None):
        return type(
            "Result",
            (),
            {
                "parsed": schema(summary="This paper studies graph-based retrieval for scientific idea generation."),
                "usage": {"input": 42, "output": 9, "total": 51},
            },
        )()


def test_paper_abstract_extractor_uses_full_text_markdown():
    model = FakeAbstractModel()
    extractor = PaperAbstractExtractor(model, prompt_template="extract abstract")

    result = extractor.extract(
        paper_title="Local Folder Paper",
        full_text_markdown="# Local Folder Paper\n\n## Abstract\n\nBody text.",
    )

    messages, schema, name, metadata = model.calls[0]
    assert schema is PaperAbstractExtractionOutput
    assert name == "paper_abstract_extraction"
    assert metadata == {"stage": "paper_abstract_extraction"}
    assert "Title: Local Folder Paper" in messages[1]["content"]
    assert "# Local Folder Paper" in messages[1]["content"]
    assert result.abstract == "This paper studies graph-based retrieval for scientific idea generation."
    assert result.input_token_count == 42
    assert result.output_token_count == 9


def test_paper_abstract_extractor_rejects_summary_only_output():
    extractor = PaperAbstractExtractor(SummaryOnlyAbstractModel(), prompt_template="extract abstract")

    with pytest.raises(ValueError, match="abstract"):
        extractor.extract(
            paper_title="Local Folder Paper",
            full_text_markdown="# Local Folder Paper\n\n## Abstract\n\nBody text.",
        )


def test_paper_abstract_prompt_prefers_existing_abstract():
    prompt = EXTRACT_PAPER_ABSTRACT_PROMPT

    assert "If the paper text contains an abstract or 摘要 section" in prompt
    assert "reproduce its content as faithfully as possible" in prompt


class FakePdfMarkdownExtractor:
    def __init__(self):
        self.calls = []

    def extract(self, pdf_path, output_dir, expected_page_count=None):
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        self.calls.append((pdf_path, output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = output_dir / "paper.md"
        markdown_path.write_text(
            "# Local Folder Paper\n\n## Abstract\n\nBody text.\n",
            encoding="utf-8",
        )
        return type("Result", (), {"markdown_path": markdown_path, "mode": "fast"})()


class FakeLocalAbstractExtractor:
    def __init__(self):
        self.calls = []

    def extract(self, paper_title: str, full_text_markdown: str):
        self.calls.append((paper_title, full_text_markdown))
        return type("Result", (), {"abstract": f"Abstract for {paper_title}"})()


class RecordingExtractor:
    def __init__(self):
        self.docs = []

    async def extract_async(self, document):
        self.docs.append(document)
        return ExtractionResult()


class RecordingRepo:
    def __init__(self):
        self.papers = []
        self.entities = []
        self.paper_entity_relations = []
        self.entity_relations = []
        self.citations = []
        self.events = []
        self.replaced_papers = []

    def upsert_papers(self, papers):
        self.papers.extend(papers)
        self.events.extend(("upsert_paper", paper.document_id) for paper in papers)

    def upsert_entities(self, entities):
        self.entities.extend(entities)
        self.events.append(("upsert_entities", len(entities)))

    def upsert_paper_entity_relations(self, paper_id, entity_ids):
        self.paper_entity_relations.extend((paper_id, entity_id) for entity_id in entity_ids)
        self.events.append(("upsert_paper_entity_relations", paper_id, tuple(entity_ids)))

    def upsert_entity_relations(self, relations):
        self.entity_relations.extend(relations)
        self.events.append(("upsert_entity_relations", len(relations)))

    def upsert_citations(self, citations):
        self.citations.extend(citations)
        self.events.append(("upsert_citations", [(c.source_paper_id, c.target_paper_id) for c in citations]))

    def replace_existing_paper_for_rebuild(self, paper_id):
        self.replaced_papers.append(paper_id)
        self.events.append(("replace_paper", paper_id))
        return []


def test_local_paper_folder_workflow_reuses_entity_extraction_for_multiple_folders(tmp_path):
    paper_dir_a = tmp_path / "Paper A"
    paper_dir_b = tmp_path / "Paper B"
    paper_dir_a.mkdir()
    paper_dir_b.mkdir()
    (paper_dir_a / "paper-a.pdf").write_bytes(b"%PDF-1.4")
    (paper_dir_b / "paper-b.pdf").write_bytes(b"%PDF-1.4")

    repo = RecordingRepo()
    extractor = RecordingExtractor()
    workflow = MvpGraphBuildWorkflow(
        paper_service=None,
        extractor=extractor,
        repository=repo,
        log=lambda message: None,
        worker_count=1,
        extract_entities_enabled=True,
        pdf_markdown_extractor=FakePdfMarkdownExtractor(),
        paper_abstract_extractor=FakeLocalAbstractExtractor(),
        local_paper_output_dir=tmp_path / "converted",
    )

    summary = workflow.run_local_paper_folders([paper_dir_a, paper_dir_b])

    assert summary["paper_count"] == 2
    assert [paper.document_id for paper in repo.papers] == ["Paper A", "Paper B"]
    assert [paper.abstract for paper in repo.papers] == ["Abstract for Paper A", "Abstract for Paper B"]
    assert [doc.document_id for doc in extractor.docs] == ["Paper A", "Paper B"]
    assert extractor.docs[0].full_text_markdown.startswith("# Local Folder Paper")
    assert extractor.docs[0].text_source == "full_text_md"


def test_local_paper_folder_workflow_reuses_existing_markdown_without_pdf_conversion(tmp_path):
    paper_dir = tmp_path / "Paper A"
    paper_dir.mkdir()
    (paper_dir / "paper-a.pdf").write_bytes(b"%PDF-1.4")
    output_root = tmp_path / "converted"
    cached_output_dir = output_root / _local_paper_output_slug(paper_dir.resolve())
    cached_output_dir.mkdir(parents=True)
    cached_markdown = cached_output_dir / "paper.md"
    cached_markdown.write_text("# Cached Local Paper\n\n## Abstract\n\nCached body.\n", encoding="utf-8")

    repo = RecordingRepo()
    extractor = RecordingExtractor()
    pdf_extractor = FakePdfMarkdownExtractor()
    abstract_extractor = FakeLocalAbstractExtractor()
    workflow = MvpGraphBuildWorkflow(
        paper_service=None,
        extractor=extractor,
        repository=repo,
        log=lambda message: None,
        worker_count=1,
        extract_entities_enabled=True,
        pdf_markdown_extractor=pdf_extractor,
        paper_abstract_extractor=abstract_extractor,
        local_paper_output_dir=output_root,
    )

    summary = workflow.run_local_paper_folders([paper_dir])

    assert summary["paper_count"] == 1
    assert pdf_extractor.calls == []
    assert abstract_extractor.calls == [("Paper A", "# Cached Local Paper\n\n## Abstract\n\nCached body.\n")]
    assert extractor.docs[0].full_text_markdown.startswith("# Cached Local Paper")
    assert repo.papers[0].full_text_markdown_path == str(cached_markdown)


def test_local_paper_folder_workflow_rebuilds_existing_paper_and_restores_citations(tmp_path):
    class CitationPreservingRepo(RecordingRepo):
        def replace_existing_paper_for_rebuild(self, paper_id):
            self.replaced_papers.append(paper_id)
            self.events.append(("replace_paper", paper_id))
            return [
                PaperCitationRelation(source_paper_id="other-paper", target_paper_id=paper_id),
                PaperCitationRelation(source_paper_id=paper_id, target_paper_id="referenced-paper"),
            ]

    paper_dir = tmp_path / "Paper A"
    paper_dir.mkdir()
    (paper_dir / "paper-a.pdf").write_bytes(b"%PDF-1.4")

    repo = CitationPreservingRepo()
    workflow = MvpGraphBuildWorkflow(
        paper_service=None,
        extractor=RecordingExtractor(),
        repository=repo,
        log=lambda message: None,
        worker_count=1,
        extract_entities_enabled=True,
        pdf_markdown_extractor=FakePdfMarkdownExtractor(),
        paper_abstract_extractor=FakeLocalAbstractExtractor(),
        local_paper_output_dir=tmp_path / "converted",
    )

    summary = workflow.run_local_paper_folders([paper_dir])

    assert summary["paper_count"] == 1
    assert summary["citation_count"] == 2
    assert repo.replaced_papers == ["Paper A"]
    assert repo.citations == [
        PaperCitationRelation(source_paper_id="other-paper", target_paper_id="Paper A"),
        PaperCitationRelation(source_paper_id="Paper A", target_paper_id="referenced-paper"),
    ]
    assert repo.events[0] == ("replace_paper", "Paper A")
    assert repo.events[1] == ("upsert_paper", "Paper A")
    assert repo.events[-1] == (
        "upsert_citations",
        [("other-paper", "Paper A"), ("Paper A", "referenced-paper")],
    )


def test_main_dispatches_build_local_paper_graph(monkeypatch):
    called = {}

    def fake_run(args):
        called["paper_dirs"] = args.paper_dirs

    monkeypatch.setattr(main, "run_build_local_paper_graph", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "main.py",
            "build-local-paper-graph",
            "--paper-dir",
            "/tmp/a",
            "/tmp/b",
        ],
    )

    main.main()

    assert called["paper_dirs"] == ["/tmp/a", "/tmp/b"]
