from src.workflows.mvp_graph_build import MvpGraphBuildWorkflow
from src.schemas.document import PaperDocument
from src.schemas.extraction import ExtractionResult


class FakePaperService:
    async def fetch_seed_documents_async(self, **kwargs):
        del kwargs
        return [
            PaperDocument(
                document_id="Paper A",
                source_type="openalex",
                source_ref="https://openalex.org/W1",
                abstract="Abstract.",
                ingestion_status="ready",
            )
        ], []

    async def fetch_cited_documents_async(self, raw_seeds, seed_docs):
        del raw_seeds, seed_docs
        return [], []


class FakeFullTextProvider:
    def enrich(self, doc):
        return doc.model_copy(update={"full_text_markdown": "# Paper A\n\nFull text", "text_source": "full_text_md"})


class FakeExtractor:
    def __init__(self):
        self.docs = []

    async def extract_async(self, doc):
        self.docs.append(doc)
        return ExtractionResult(input_token_count=10, output_token_count=2)


class FakeRepository:
    def upsert_papers(self, papers):
        del papers

    def upsert_citations(self, citations):
        del citations

    def upsert_entities(self, entities):
        del entities

    def upsert_paper_entity_relations(self, paper_id, entity_ids):
        del paper_id, entity_ids

    def upsert_entity_relations(self, relations):
        del relations


def test_workflow_enriches_document_with_full_text_before_extraction():
    extractor = FakeExtractor()
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakePaperService(),
        extractor=extractor,
        repository=FakeRepository(),
        worker_count=1,
        extract_entities_enabled=True,
        full_text_provider=FakeFullTextProvider(),
    )

    workflow.run(keyword="idea graph", max_seed_papers=1, require_references=False)

    assert extractor.docs[0].text_source == "full_text_md"
    assert extractor.docs[0].full_text_markdown == "# Paper A\n\nFull text"
