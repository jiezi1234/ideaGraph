from src.schemas.document import PaperDocument
from src.extraction.chunked_paper_extractor import ChunkedPaperExtractor
from src.schemas.entity import ExtractedEntity
from src.schemas.extraction import ExtractionResult
from src.schemas.relation import ExtractedRelation, PaperCitationRelation
from src.workflows.mvp_graph_build import MvpGraphBuildWorkflow


class FakePaperService:
    async def fetch_seed_documents_async(self, keyword, max_seed_papers, require_references):
        seed, cited, _ = self.fetch_seed_and_citations(keyword, max_seed_papers, require_references)
        raw_seeds = [
            {
                "id": seed[0].document_id,
                "title": seed[0].document_id,
                "abstract_inverted_index": {"Seed": [0], "Paper": [1]},
                "referenced_works": [cited[0].document_id],
            }
        ]
        return seed, raw_seeds

    async def fetch_cited_documents_async(self, raw_seeds, seed_docs):
        _, cited, links = self.fetch_seed_and_citations("idea graph", len(seed_docs), True)
        return cited, links

    async def fetch_seed_and_citations_async(self, keyword, max_seed_papers, require_references):
        return self.fetch_seed_and_citations(keyword, max_seed_papers, require_references)

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


class FakePriorityPaperService:
    async def fetch_seed_documents_async(
        self,
        keyword,
        max_seed_papers,
        require_references,
        fetch_all=False,
        min_cited_by_count=None,
        min_publication_year=None,
    ):
        assert fetch_all is True
        assert min_cited_by_count == 50
        assert min_publication_year == 2021
        seed_raw = {
            "id": "p1",
            "title": "Seed A",
            "abstract_inverted_index": {"Seed": [0], "A": [1]},
            "publication_year": 2024,
            "referenced_works": ["p2", "p3"],
            "cited_by_count": 80,
        }
        seed = PaperDocument(
            document_id="p1",
            source_type="openalex",
            source_ref="p1",
            title="Seed A",
            abstract="Seed A",
            year=2024,
            citation_openalex_ids=["p2", "p3"],
            ingestion_status="ready",
        )
        return [seed], [seed_raw]

    async def fetch_referenced_documents_async(self, raw_paper):
        payloads = {
            "p1": (
                [
                    PaperDocument(
                        document_id="p2",
                        source_type="openalex",
                        source_ref="p2",
                        title="B",
                        abstract="B",
                        year=2023,
                        citation_openalex_ids=["p4"],
                        ingestion_status="ready",
                    ),
                    PaperDocument(
                        document_id="p3",
                        source_type="openalex",
                        source_ref="p3",
                        title="C",
                        abstract="C",
                        year=2023,
                        citation_openalex_ids=["p5"],
                        ingestion_status="ready",
                    ),
                ],
                [
                    {
                        "id": "p2",
                        "title": "B",
                        "abstract_inverted_index": {"B": [0]},
                        "publication_year": 2023,
                        "referenced_works": ["p4"],
                        "cited_by_count": 120,
                    },
                    {
                        "id": "p3",
                        "title": "C",
                        "abstract_inverted_index": {"C": [0]},
                        "publication_year": 2023,
                        "referenced_works": ["p5"],
                        "cited_by_count": 20,
                    },
                ],
                [
                    PaperCitationRelation(source_paper_id="p1", target_paper_id="p2"),
                    PaperCitationRelation(source_paper_id="p1", target_paper_id="p3"),
                ],
                [],
            ),
            "p2": (
                [
                    PaperDocument(
                        document_id="p4",
                        source_type="openalex",
                        source_ref="p4",
                        title="D",
                        abstract="D",
                        year=2022,
                        citation_openalex_ids=[],
                        ingestion_status="ready",
                    )
                ],
                [
                    {
                        "id": "p4",
                        "title": "D",
                        "abstract_inverted_index": {"D": [0]},
                        "publication_year": 2022,
                        "referenced_works": [],
                        "cited_by_count": 5,
                    }
                ],
                [PaperCitationRelation(source_paper_id="p2", target_paper_id="p4")],
                [],
            ),
            "p3": (
                [
                    PaperDocument(
                        document_id="p5",
                        source_type="openalex",
                        source_ref="p5",
                        title="E",
                        abstract="E",
                        year=2022,
                        citation_openalex_ids=[],
                        ingestion_status="ready",
                    )
                ],
                [
                    {
                        "id": "p5",
                        "title": "E",
                        "abstract_inverted_index": {"E": [0]},
                        "publication_year": 2022,
                        "referenced_works": [],
                        "cited_by_count": 1,
                    }
                ],
                [PaperCitationRelation(source_paper_id="p3", target_paper_id="p5")],
                [],
            ),
            "p4": ([], [], [], []),
            "p5": ([], [], [], []),
        }
        return payloads[raw_paper["id"]]


class FakePriorityPartialFailurePaperService(FakePriorityPaperService):
    async def fetch_referenced_documents_async(self, raw_paper):
        if raw_paper["id"] == "p1":
            return (
                [
                    PaperDocument(
                        document_id="p2",
                        source_type="openalex",
                        source_ref="p2",
                        title="B",
                        abstract="B",
                        year=2023,
                        citation_openalex_ids=[],
                        ingestion_status="ready",
                    )
                ],
                [
                    {
                        "id": "p2",
                        "title": "B",
                        "abstract_inverted_index": {"B": [0]},
                        "publication_year": 2023,
                        "referenced_works": [],
                        "cited_by_count": 120,
                    }
                ],
                [
                    PaperCitationRelation(source_paper_id="p1", target_paper_id="p2"),
                    PaperCitationRelation(source_paper_id="p1", target_paper_id="p3"),
                ],
                ["p3"],
            )
        return await super().fetch_referenced_documents_async(raw_paper)


class FakePrioritySeedFirstPaperService:
    def __init__(self):
        self.expanded_order = []

    async def fetch_seed_documents_async(
        self,
        keyword,
        max_seed_papers,
        require_references,
        fetch_all=False,
        min_cited_by_count=None,
        min_publication_year=None,
    ):
        assert fetch_all is True
        seed_a_raw = {
            "id": "p1",
            "title": "Seed A",
            "abstract_inverted_index": {"Seed": [0], "A": [1]},
            "publication_year": 2024,
            "referenced_works": ["p3"],
            "cited_by_count": 10,
        }
        seed_b_raw = {
            "id": "p2",
            "title": "Seed B",
            "abstract_inverted_index": {"Seed": [0], "B": [1]},
            "publication_year": 2024,
            "referenced_works": [],
            "cited_by_count": 5,
        }
        seed_a = PaperDocument(
            document_id="p1",
            source_type="openalex",
            source_ref="p1",
            title="Seed A",
            abstract="Seed A",
            year=2024,
            citation_openalex_ids=["p3"],
            cited_by_count=10,
            ingestion_status="ready",
        )
        seed_b = PaperDocument(
            document_id="p2",
            source_type="openalex",
            source_ref="p2",
            title="Seed B",
            abstract="Seed B",
            year=2024,
            citation_openalex_ids=[],
            cited_by_count=5,
            ingestion_status="ready",
        )
        return [seed_a, seed_b], [seed_a_raw, seed_b_raw]

    async def fetch_referenced_documents_async(self, raw_paper):
        self.expanded_order.append(raw_paper["id"])
        payloads = {
            "p1": (
                [
                    PaperDocument(
                        document_id="p3",
                        source_type="openalex",
                        source_ref="p3",
                        title="High Cite Child",
                        abstract="High Cite Child",
                        year=2023,
                        citation_openalex_ids=[],
                        cited_by_count=100,
                        ingestion_status="ready",
                    )
                ],
                [
                    {
                        "id": "p3",
                        "title": "High Cite Child",
                        "abstract_inverted_index": {"High": [0], "Child": [1]},
                        "publication_year": 2023,
                        "referenced_works": [],
                        "cited_by_count": 100,
                    }
                ],
                [PaperCitationRelation(source_paper_id="p1", target_paper_id="p3")],
                [],
            ),
            "p2": ([], [], [], []),
            "p3": ([], [], [], []),
        }
        return payloads[raw_paper["id"]]


class FakePriorityOpenAlexIdPaperService:
    async def fetch_seed_documents_async(
        self,
        keyword,
        max_seed_papers,
        require_references,
        fetch_all=False,
        min_cited_by_count=None,
        min_publication_year=None,
    ):
        assert fetch_all is True
        return [
            PaperDocument(
                document_id="Seed A",
                source_type="openalex",
                source_ref="https://openalex.org/W1",
                title="Seed A",
                abstract="Seed A",
                year=2024,
                citation_openalex_ids=["https://openalex.org/W2"],
                cited_by_count=80,
                ingestion_status="ready",
            )
        ], [
            {
                "id": "https://openalex.org/W1",
                "title": "Seed A",
                "abstract_inverted_index": {"Seed": [0], "A": [1]},
                "publication_year": 2024,
                "referenced_works": ["https://openalex.org/W2"],
                "cited_by_count": 80,
            }
        ]

    async def fetch_referenced_documents_async(self, raw_paper):
        if raw_paper["id"] != "https://openalex.org/W1":
            return [], [], [], []
        return (
            [
                PaperDocument(
                    document_id="Cited B",
                    source_type="openalex",
                    source_ref="https://openalex.org/W2",
                    title="Cited B",
                    abstract="Cited B",
                    year=2023,
                    citation_openalex_ids=[],
                    cited_by_count=20,
                    ingestion_status="ready",
                )
            ],
            [
                {
                    "id": "https://openalex.org/W2",
                    "title": "Cited B",
                    "abstract_inverted_index": {"Cited": [0], "B": [1]},
                    "publication_year": 2023,
                    "referenced_works": [],
                    "cited_by_count": 20,
                }
            ],
            [PaperCitationRelation(source_paper_id="Seed A", target_paper_id="Cited B")],
            [],
        )


class FakeExplicitSeedPaperService:
    def __init__(self, failed_ids=None):
        self.failed_ids = failed_ids or []
        self.openalex_seed_ids = None
        self.keyword_search_called = False
        self.cited_fetch_calls = []
        self.referenced_fetch_called = False

    async def fetch_openalex_seed_documents_async(self, openalex_ids):
        self.openalex_seed_ids = openalex_ids
        seed_a = PaperDocument(
            document_id="seed-a",
            source_type="openalex",
            source_ref="https://openalex.org/W1",
            title="Seed A",
            abstract="Seed A",
            ingestion_status="ready",
        )
        seed_b = PaperDocument(
            document_id="seed-b",
            source_type="openalex",
            source_ref="https://openalex.org/W2",
            title="Seed B",
            abstract="Seed B",
            ingestion_status="ready",
        )
        raw_seed_a = {"id": "https://openalex.org/W1", "referenced_works": ["https://openalex.org/W3"]}
        raw_seed_b = {"id": "https://openalex.org/W2", "referenced_works": ["https://openalex.org/W4"]}
        return [seed_a, seed_b], [raw_seed_a, raw_seed_b], self.failed_ids

    async def fetch_cited_documents_async(self, raw_seeds, seed_docs):
        self.cited_fetch_calls.append((raw_seeds, seed_docs))
        cited_a = PaperDocument(
            document_id="cited-a",
            source_type="openalex",
            source_ref="https://openalex.org/W3",
            title="Cited A",
            abstract="Cited A",
            ingestion_status="ready",
        )
        cited_b = PaperDocument(
            document_id="cited-b",
            source_type="openalex",
            source_ref="https://openalex.org/W4",
            title="Cited B",
            abstract="Cited B",
            ingestion_status="ready",
        )
        return [cited_a, cited_b], [
            PaperCitationRelation(source_paper_id="seed-a", target_paper_id="cited-a"),
            PaperCitationRelation(source_paper_id="seed-b", target_paper_id="cited-b"),
            PaperCitationRelation(source_paper_id="missing-seed", target_paper_id="cited-a"),
        ]

    async def fetch_seed_documents_async(self, *args, **kwargs):
        self.keyword_search_called = True
        raise AssertionError("explicit seed graph must not call keyword search")

    async def fetch_referenced_documents_async(self, raw_paper):
        self.referenced_fetch_called = True
        raise AssertionError("explicit seed graph must not use priority expansion")


class FakeMissingAbstractExplicitSeedPaperService(FakeExplicitSeedPaperService):
    async def fetch_openalex_seed_documents_async(self, openalex_ids):
        seed_docs, raw_seeds, failed_ids = await super().fetch_openalex_seed_documents_async(openalex_ids)
        seed_docs[1] = seed_docs[1].model_copy(update={"abstract": ""})
        return seed_docs, raw_seeds, failed_ids


class FakeContentlessExplicitSeedPaperService(FakeExplicitSeedPaperService):
    async def fetch_openalex_seed_documents_async(self, openalex_ids):
        seed_docs, raw_seeds, failed_ids = await super().fetch_openalex_seed_documents_async(openalex_ids)
        seed_docs[1] = seed_docs[1].model_copy(update={"abstract": ""})
        return seed_docs, raw_seeds, failed_ids


class FakeOrderingFullTextProvider:
    def __init__(self, events):
        self.events = events

    def enrich(self, doc):
        self.events.append(("enrich", doc.document_id))
        return doc.model_copy(
            update={
                "full_text_markdown": f"# {doc.document_id}\n\nFull text",
                "text_source": "full_text_md",
            }
        )


class FakeOrderingExtractor:
    def __init__(self, events):
        self.events = events
        self.docs = []

    async def extract_async(self, document):
        self.events.append(("extract", document.document_id))
        self.docs.append(document)
        return ExtractionResult(entities=[], relations=[])


class FakeKeywordSeedMissingInfoPaperService:
    async def fetch_seed_documents_async(self, keyword, max_seed_papers, require_references):
        seed_with_abstract = PaperDocument(
            document_id="seed-ok",
            source_type="openalex",
            source_ref="https://openalex.org/W1",
            abstract="Seed abstract",
            ingestion_status="ready",
        )
        seed_without_info = PaperDocument(
            document_id="seed-empty",
            source_type="openalex",
            source_ref="https://openalex.org/W2",
            abstract="",
            ingestion_status="ready",
        )
        return [seed_with_abstract, seed_without_info], [
            {"id": "https://openalex.org/W1", "referenced_works": ["https://openalex.org/W3"]},
            {"id": "https://openalex.org/W2", "referenced_works": ["https://openalex.org/W4"]},
        ]

    async def fetch_cited_documents_async(self, raw_seeds, seed_docs):
        self.raw_seed_ids = [raw_seed["id"] for raw_seed in raw_seeds]
        self.seed_doc_ids = [doc.document_id for doc in seed_docs]
        cited_doc = PaperDocument(
            document_id="cited-ok",
            source_type="openalex",
            source_ref="https://openalex.org/W3",
            abstract="Cited abstract",
            ingestion_status="ready",
        )
        return [cited_doc], [PaperCitationRelation(source_paper_id="seed-ok", target_paper_id="cited-ok")]


class FakeReferencedMissingInfoPaperService:
    async def fetch_seed_documents_async(self, keyword, max_seed_papers, require_references):
        seed_doc = PaperDocument(
            document_id="seed-ok",
            source_type="openalex",
            source_ref="https://openalex.org/W1",
            abstract="Seed abstract",
            ingestion_status="ready",
        )
        return [seed_doc], [{"id": "https://openalex.org/W1", "referenced_works": ["https://openalex.org/W2"]}]

    async def fetch_cited_documents_async(self, raw_seeds, seed_docs):
        cited_doc = PaperDocument(
            document_id="cited-empty",
            source_type="openalex",
            source_ref="https://openalex.org/W2",
            abstract="",
            ingestion_status="ready",
        )
        return [cited_doc], [PaperCitationRelation(source_paper_id="seed-ok", target_paper_id="cited-empty")]


class FakePriorityReferencedMissingInfoPaperService:
    def __init__(self):
        self.expanded_order = []

    async def fetch_seed_documents_async(
        self,
        keyword,
        max_seed_papers,
        require_references,
        fetch_all=False,
        min_cited_by_count=None,
        min_publication_year=None,
    ):
        seed_doc = PaperDocument(
            document_id="p1",
            source_type="openalex",
            source_ref="p1",
            abstract="Seed abstract",
            citation_openalex_ids=["p2"],
            ingestion_status="ready",
        )
        raw_seed = {
            "id": "p1",
            "referenced_works": ["p2"],
            "cited_by_count": 50,
        }
        return [seed_doc], [raw_seed]

    async def fetch_referenced_documents_async(self, raw_paper):
        self.expanded_order.append(raw_paper["id"])
        if raw_paper["id"] == "p3":
            return [], [], [], []
        if raw_paper["id"] == "p2":
            cited_doc = PaperDocument(
                document_id="p3",
                source_type="openalex",
                source_ref="p3",
                abstract="Recovered downstream abstract",
                citation_openalex_ids=[],
                ingestion_status="ready",
            )
            raw_cited = {
                "id": "p3",
                "referenced_works": [],
                "cited_by_count": 10,
            }
            return [cited_doc], [raw_cited], [PaperCitationRelation(source_paper_id="p2", target_paper_id="p3")], []
        cited_doc = PaperDocument(
            document_id="p2",
            source_type="openalex",
            source_ref="p2",
            abstract="",
            citation_openalex_ids=["p3"],
            ingestion_status="ready",
        )
        raw_cited = {
            "id": "p2",
            "referenced_works": ["p3"],
            "cited_by_count": 200,
        }
        return [cited_doc], [raw_cited], [PaperCitationRelation(source_paper_id="p1", target_paper_id="p2")], []


class FakeOrderingAbstractExtractor:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def extract(self, paper_title, full_text_markdown):
        self.events.append(("abstract", paper_title))
        self.calls.append((paper_title, full_text_markdown))
        return type("Result", (), {"abstract": f"Recovered abstract for {paper_title}"})()


class FakeExtractor:
    def __init__(self):
        self.called = 0

    async def extract_async(self, document):
        return self.extract(document)

    def extract(self, document):
        self.called += 1
        return ExtractionResult(
            entities=[
                ExtractedEntity(entity_id="graph planning", entity_type="Method", source_refs=[document.document_id]),
                ExtractedEntity(entity_id="idea generation", entity_type="Problem", source_refs=[document.document_id]),
            ],
            relations=[
                ExtractedRelation(
                    source_entity_id="graph planning",
                    target_entity_id="idea generation",
                    relation_type="Used-For",
                    source_refs=[document.document_id],
                )
            ],
        )


class FakeRepo:
    def __init__(self):
        self.operations = []
        self.papers = []
        self.entities = []
        self.paper_entity_relations = []
        self.citations = []
        self.entity_relations = []

    def upsert_papers(self, papers):
        self.operations.append(("papers", len(papers)))
        self.papers.extend(papers)

    def upsert_entities(self, entities):
        self.operations.append(("entities", len(entities)))
        self.entities.extend(entities)

    def upsert_paper_entity_relations(self, paper_id, entity_ids):
        self.operations.append(("paper_entities", len(entity_ids)))
        self.paper_entity_relations.extend((paper_id, entity_id) for entity_id in entity_ids)

    def upsert_citations(self, citations):
        self.operations.append(("citations", len(citations)))
        self.citations.extend(citations)

    def upsert_entity_relations(self, relations):
        self.operations.append(("entity_relations", len(relations)))
        self.entity_relations.extend(relations)


class FakeRelationResolver:
    def __init__(self):
        self.calls = []

    async def resolve_relations(self, relations):
        self.calls.append(list(relations))
        return [
            relation.model_copy(update={"description": "Resolved relation evidence."})
            for relation in relations
        ]


def test_build_mvp_workflow_passes_configured_openalex_works_filter(monkeypatch, tmp_path):
    from src.app import bootstrap
    from src.config.settings import Settings

    calls = {}

    class FakeOpenAlexClient:
        def __init__(
            self,
            base_url: str,
            works_filter: str = "",
            api_keys=None,
            api_key_cooldown_seconds: int = 300,
        ) -> None:
            calls["openalex_client"] = {
                "base_url": base_url,
                "works_filter": works_filter,
                "api_keys": api_keys,
                "api_key_cooldown_seconds": api_key_cooldown_seconds,
            }

    class FakePaperService:
        def __init__(self, client) -> None:
            calls["paper_service"] = client

    class FakeModel:
        @classmethod
        def from_settings(cls, settings, *, model_name):
            calls["model"] = {
                "base_url": settings.llm_base_url,
                "model": model_name,
                "think_mode": settings.llm_think_mode,
                "timeout": settings.llm_timeout,
            }
            return cls()

    class FakeExtractor:
        def __init__(self, model, prompt_template: str) -> None:
            calls["extractor_instance"] = self
            calls["extractor"] = {
                "model": model,
                "prompt_template": prompt_template,
            }

    class FakeRepository:
        @classmethod
        def from_connection(cls, uri: str, username: str, password: str, database: str):
            calls["repository"] = database
            return {"repository": "instance"}

    class FakeArxivClient:
        def __init__(self) -> None:
            calls["arxiv_client"] = True

    class FakeDownloader:
        def __init__(self, **kwargs) -> None:
            calls["downloader"] = kwargs

    class FakePdfExtractor:
        def __init__(self, **kwargs) -> None:
            calls["pdf_extractor"] = kwargs

    class FakeFullTextProvider:
        def __init__(self, output_dir, arxiv_client, downloader, extractor) -> None:
            calls["full_text_provider"] = {
                "output_dir": output_dir,
                "arxiv_client": arxiv_client,
                "downloader": downloader,
                "extractor": extractor,
            }

    class FakeHuggingFaceChunkTokenizer:
        @classmethod
        def from_model_path(cls, model_path: str, *, local_files_only: bool = True):
            calls["chunk_tokenizer"] = {
                "model_path": model_path,
                "local_files_only": local_files_only,
            }
            return cls()

    class FakeWorkflow:
        def __init__(self, **kwargs) -> None:
            calls["workflow"] = kwargs

    monkeypatch.setattr(bootstrap, "OpenAlexClient", FakeOpenAlexClient)
    monkeypatch.setattr(bootstrap, "OpenAlexPaperService", FakePaperService)
    monkeypatch.setattr(bootstrap, "OpenAIChatModel", FakeModel)
    monkeypatch.setattr(bootstrap, "PaperExtractor", FakeExtractor)
    monkeypatch.setattr(bootstrap, "Neo4jGraphRepository", FakeRepository)
    monkeypatch.setattr(bootstrap, "ArxivClient", FakeArxivClient)
    monkeypatch.setattr(bootstrap, "PdfDownloader", FakeDownloader)
    monkeypatch.setattr(bootstrap, "OpenDataLoaderPdfMarkdownExtractor", FakePdfExtractor)
    monkeypatch.setattr(bootstrap, "FullTextProvider", FakeFullTextProvider)
    monkeypatch.setattr(bootstrap, "HuggingFaceChunkTokenizer", FakeHuggingFaceChunkTokenizer)
    monkeypatch.setattr(bootstrap, "MvpGraphBuildWorkflow", FakeWorkflow)
    monkeypatch.setattr(bootstrap, "load_extract_prompt_template", lambda: "prompt")

    settings = Settings(
        OPENALEX_BASE_URL="https://api.openalex.org",
        OPENALEX_WORKS_FILTER="topics.field.id:17",
        OPENALEX_API_KEYS="key-a,key-b",
        OPENALEX_API_KEY_COOLDOWN_SECONDS=60,
        LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1",
        LLM_API_KEY="sk-test",
        LLM_MODEL_EXTRACTION="qwen3.7-flash",
        LLM_MODEL_GENERATION="qwen3.7-plus",
        EMBED_MODEL="text-embedding-v4",
        LLM_TIMEOUT=300,
        LLM_THINK_MODE="medium",
        NEO4J_URI="bolt://localhost:7687",
        NEO4J_USERNAME="neo4j",
        NEO4J_PASSWORD="secret",
        NEO4J_DATABASE="idea",
        FULL_TEXT_EXTRACTION_ENABLED=True,
        FULL_TEXT_OUTPUT_DIR=str(tmp_path / "data"),
        PDF_HYBRID_URL="http://127.0.0.1:15002",
        PDF_HYBRID_BACKEND="docling-fast",
        PDF_HYBRID_MODE="auto",
    )

    workflow = bootstrap.build_mvp_workflow(settings)

    assert isinstance(workflow, FakeWorkflow)
    assert calls["openalex_client"] == {
        "base_url": "https://api.openalex.org",
        "works_filter": "topics.field.id:17",
        "api_keys": ["key-a", "key-b"],
        "api_key_cooldown_seconds": 60,
    }
    assert calls["model"] == {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen3.7-flash",
        "think_mode": "medium",
        "timeout": 300.0,
    }
    assert calls["full_text_provider"]["output_dir"] == tmp_path / "data" / "idea"
    assert calls["downloader"] == {
        "openalex_api_keys": ["key-a", "key-b"],
        "openalex_api_key_cooldown_seconds": 60,
    }
    assert calls["pdf_extractor"] == {
        "hybrid_url": "http://127.0.0.1:15002",
        "hybrid_backend": "docling-fast",
        "hybrid_mode": "auto",
    }
    assert calls["chunk_tokenizer"] == {
        "model_path": settings.chunk_tokenizer_model_path,
        "local_files_only": settings.chunk_tokenizer_local_files_only,
    }
    assert isinstance(calls["workflow"]["extractor"], ChunkedPaperExtractor)
    assert calls["workflow"]["extractor"].base_extractor is calls["extractor_instance"]
    assert calls["workflow"]["extractor"].chunk_min_tokens == settings.chunk_min_tokens
    assert calls["workflow"]["extractor"].chunk_max_tokens == settings.chunk_max_tokens
    assert calls["workflow"]["extractor"].chunk_overlap_tokens == settings.chunk_overlap_tokens
    assert calls["workflow"]["full_text_provider"] is not None


def test_workflow_builds_graph_for_seed_and_one_hop_cited_papers():
    repo = FakeRepo()
    logs = []
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakePaperService(),
        extractor=FakeExtractor(),
        repository=repo,
        log=logs.append,
        worker_count=2,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["paper_count"] == 2
    assert len(repo.citations) == 1
    assert len(repo.entities) == 4
    assert len(repo.paper_entity_relations) == 4
    assert len(repo.entity_relations) == 2
    assert repo.operations == [
        ("papers", 1),
        ("papers", 1),
        ("citations", 1),
        ("entities", 2),
        ("paper_entities", 2),
        ("entity_relations", 1),
        ("entities", 2),
        ("paper_entities", 2),
        ("entity_relations", 1),
    ]
    assert logs[0] == "search:start keyword=idea graph max_seed_papers=1 require_references=True"
    assert "search:done papers=2 citations=1" in logs
    assert any(log.startswith("paper:node_written id=p1 progress=1/1 pending_expand=0 eta_seconds=") for log in logs)
    assert any(log.startswith("paper:node_written id=p2 progress=2/2 pending_expand=0 eta_seconds=") for log in logs)
    assert any(log.startswith("queue:enqueue id=p1 ") and "extract_remaining=" in log for log in logs)
    assert any(log.startswith("queue:enqueue id=p2 ") and "extract_remaining=" in log for log in logs)
    assert any(
        log.startswith("paper:start id=p1 ")
        and "extract_progress_started=" in log
        and "extract_progress_done=" in log
        and "extract_remaining=" in log
        for log in logs
    )
    assert any(
        log.startswith("paper:start id=p2 ")
        and "extract_progress_started=" in log
        and "extract_progress_done=" in log
        and "extract_remaining=" in log
        for log in logs
    )
    assert any(
        log.startswith("paper:extract_done id=p1 entities=2 relations=1 ")
        and "extract_progress_started=" in log
        and "extract_progress_done=" in log
        and "extract_remaining=" in log
        for log in logs
    )
    assert any(
        log.startswith("paper:extract_done id=p2 entities=2 relations=1 ")
        and "extract_progress_started=" in log
        and "extract_progress_done=" in log
        and "extract_remaining=" in log
        for log in logs
    )
    assert "search:done papers=2 citations=1" in logs
    assert any(
        log.startswith("paper:write_done id=p1 entities=2 relations=1 ")
        and "extract_progress_started=" in log
        and "extract_progress_done=" in log
        and "extract_remaining=" in log
        for log in logs
    )
    assert any(
        log.startswith("paper:write_done id=p2 entities=2 relations=1 ")
        and "extract_progress_started=" in log
        and "extract_progress_done=2/2" in log
        and "extract_remaining=0" in log
        for log in logs
    )
    assert "worker:done id=0" in logs
    assert "worker:done id=1" in logs
    assert logs[-1] == "workflow:done papers=2 entities=4 citations=1 entity_relations=2"


def test_workflow_resolves_entity_relations_before_writing():
    repo = FakeRepo()
    relation_resolver = FakeRelationResolver()
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakePaperService(),
        extractor=FakeExtractor(),
        repository=repo,
        worker_count=1,
        relation_resolver=relation_resolver,
    )

    workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert len(relation_resolver.calls) == 2
    assert {relation.description for relation in repo.entity_relations} == {"Resolved relation evidence."}


def test_workflow_can_skip_entity_extraction():
    repo = FakeRepo()
    logs = []
    extractor = FakeExtractor()
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakePaperService(),
        extractor=extractor,
        repository=repo,
        log=logs.append,
        worker_count=2,
        extract_entities_enabled=False,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["paper_count"] == 2
    assert summary["entity_count"] == 0
    assert summary["entity_relation_count"] == 0
    assert summary["citation_count"] == 1
    assert extractor.called == 0
    assert len(repo.citations) == 1
    assert len(repo.entities) == 0
    assert len(repo.paper_entity_relations) == 0
    assert len(repo.entity_relations) == 0
    assert "extract:skipped enabled=false" in logs


def test_keyword_seed_without_abstract_or_full_text_is_skipped_before_graph_write():
    repo = FakeRepo()
    logs = []
    service = FakeKeywordSeedMissingInfoPaperService()
    events = []
    extractor = FakeOrderingExtractor(events)
    workflow = MvpGraphBuildWorkflow(
        paper_service=service,
        extractor=extractor,
        repository=repo,
        log=logs.append,
        worker_count=1,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=2, require_references=True)

    assert summary["paper_count"] == 2
    assert [paper.document_id for paper in repo.papers] == ["seed-ok", "cited-ok"]
    assert service.raw_seed_ids == ["https://openalex.org/W1"]
    assert service.seed_doc_ids == ["seed-ok"]
    assert [doc.document_id for doc in extractor.docs] == ["seed-ok", "cited-ok"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [
        ("seed-ok", "cited-ok")
    ]
    assert "paper:skipped id=seed-empty reason=missing_abstract_and_full_text source=keyword_seed" in logs


def test_citation_paper_without_abstract_or_full_text_keeps_node_but_skips_entity_extraction():
    repo = FakeRepo()
    logs = []
    events = []
    extractor = FakeOrderingExtractor(events)
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakeReferencedMissingInfoPaperService(),
        extractor=extractor,
        repository=repo,
        log=logs.append,
        worker_count=1,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["paper_count"] == 2
    assert summary["entity_count"] == 0
    assert summary["citation_count"] == 1
    assert [paper.document_id for paper in repo.papers] == ["seed-ok", "cited-empty"]
    assert [doc.document_id for doc in extractor.docs] == ["seed-ok"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [
        ("seed-ok", "cited-empty")
    ]
    assert (
        "paper:extract_skipped id=cited-empty reason=missing_abstract_and_full_text source=citation_expansion"
        in logs
    )


def test_priority_strategy_expands_higher_cited_papers_first():
    repo = FakeRepo()
    logs = []
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakePriorityPaperService(),
        extractor=FakeExtractor(),
        repository=repo,
        log=logs.append,
        worker_count=1,
        extract_entities_enabled=False,
        citation_graph_strategy="priority_expand",
        citation_seed_min_cited_by_count=50,
        citation_seed_min_publication_year=2021,
        citation_max_total_papers=10,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["paper_count"] == 5
    assert [paper.document_id for paper in repo.papers] == ["p1", "p2", "p3", "p4", "p5"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [
        ("p1", "p2"),
        ("p1", "p3"),
        ("p2", "p4"),
        ("p3", "p5"),
    ]
    assert any(log.startswith("paper:node_written id=p1 progress=1/2 pending_expand=1 eta_seconds=") for log in logs)
    assert any("paper:node_written id=p2" in log for log in logs)


def test_priority_strategy_skips_partial_expansion_when_total_limit_would_be_exceeded():
    repo = FakeRepo()
    logs = []
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakePriorityPaperService(),
        extractor=FakeExtractor(),
        repository=repo,
        log=logs.append,
        worker_count=1,
        extract_entities_enabled=False,
        citation_graph_strategy="priority_expand",
        citation_seed_min_cited_by_count=50,
        citation_seed_min_publication_year=2021,
        citation_max_total_papers=2,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["paper_count"] == 2
    assert summary["citation_count"] == 1
    assert [paper.document_id for paper in repo.papers] == ["p1", "p2"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [("p1", "p2")]
    assert "expand:limit_reached paper=p1 total_papers=2 max_total_papers=2" in logs


def test_priority_strategy_expands_citation_paper_without_content_when_references_exist():
    repo = FakeRepo()
    logs = []
    service = FakePriorityReferencedMissingInfoPaperService()
    extractor = FakeExtractor()
    workflow = MvpGraphBuildWorkflow(
        paper_service=service,
        extractor=extractor,
        repository=repo,
        log=logs.append,
        worker_count=1,
        citation_graph_strategy="priority_expand",
        citation_max_total_papers=10,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["paper_count"] == 3
    assert summary["citation_count"] == 2
    assert [paper.document_id for paper in repo.papers] == ["p1", "p2", "p3"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [("p1", "p2"), ("p2", "p3")]
    assert service.expanded_order == ["p1", "p2", "p3"]
    assert extractor.called == 2
    assert (
        "paper:extract_skipped id=p2 reason=missing_abstract_and_full_text source=citation_expansion"
        in logs
    )


def test_priority_total_limit_ignores_citation_papers_without_content():
    repo = FakeRepo()
    service = FakePriorityReferencedMissingInfoPaperService()
    workflow = MvpGraphBuildWorkflow(
        paper_service=service,
        extractor=FakeExtractor(),
        repository=repo,
        log=lambda message: None,
        worker_count=1,
        citation_graph_strategy="priority_expand",
        citation_max_total_papers=2,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["paper_count"] == 3
    assert summary["citation_count"] == 2
    assert [paper.document_id for paper in repo.papers] == ["p1", "p2", "p3"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [("p1", "p2"), ("p2", "p3")]


def test_priority_strategy_keeps_successful_references_when_some_reference_fetches_fail():
    repo = FakeRepo()
    logs = []
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakePriorityPartialFailurePaperService(),
        extractor=FakeExtractor(),
        repository=repo,
        log=logs.append,
        worker_count=1,
        extract_entities_enabled=False,
        citation_graph_strategy="priority_expand",
        citation_seed_min_cited_by_count=50,
        citation_seed_min_publication_year=2021,
        citation_max_total_papers=10,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["paper_count"] == 3
    assert summary["citation_count"] == 2
    assert [paper.document_id for paper in repo.papers] == ["p1", "p2", "p4"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [
        ("p1", "p2"),
        ("p2", "p4"),
    ]
    assert "expand:partial_fetch paper=p1 fetched=1 failed=1" in logs


def test_priority_strategy_capacity_check_counts_only_successfully_fetched_new_papers():
    repo = FakeRepo()
    logs = []
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakePriorityPartialFailurePaperService(),
        extractor=FakeExtractor(),
        repository=repo,
        log=logs.append,
        worker_count=1,
        extract_entities_enabled=False,
        citation_graph_strategy="priority_expand",
        citation_seed_min_cited_by_count=50,
        citation_seed_min_publication_year=2021,
        citation_max_total_papers=2,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["paper_count"] == 2
    assert summary["citation_count"] == 1
    assert [paper.document_id for paper in repo.papers] == ["p1", "p2"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [("p1", "p2")]


def test_priority_strategy_processes_all_seed_nodes_before_non_seed_nodes():
    repo = FakeRepo()
    logs = []
    service = FakePrioritySeedFirstPaperService()
    workflow = MvpGraphBuildWorkflow(
        paper_service=service,
        extractor=FakeExtractor(),
        repository=repo,
        log=logs.append,
        worker_count=1,
        extract_entities_enabled=False,
        citation_graph_strategy="priority_expand",
        citation_seed_min_cited_by_count=50,
        citation_seed_min_publication_year=2021,
        citation_max_total_papers=10,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=2, require_references=True)

    assert summary["paper_count"] == 3
    assert service.expanded_order == ["p1", "p2", "p3"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [("p1", "p3")]


def test_priority_strategy_keeps_links_when_raw_openalex_id_differs_from_document_id():
    repo = FakeRepo()
    workflow = MvpGraphBuildWorkflow(
        paper_service=FakePriorityOpenAlexIdPaperService(),
        extractor=FakeExtractor(),
        repository=repo,
        log=lambda message: None,
        worker_count=1,
        extract_entities_enabled=False,
        citation_graph_strategy="priority_expand",
        citation_seed_min_cited_by_count=50,
        citation_seed_min_publication_year=2021,
        citation_max_total_papers=10,
    )

    summary = workflow.run(keyword="idea graph", max_seed_papers=1, require_references=True)

    assert summary["citation_count"] == 1
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [("Seed A", "Cited B")]


def test_openalex_seed_graph_uses_explicit_ids_and_expands_exactly_one_hop():
    repo = FakeRepo()
    service = FakeExplicitSeedPaperService()
    logs = []
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

    summary = workflow.run_openalex_seed_graph(["https://openalex.org/W1", "https://openalex.org/W2"])

    assert service.openalex_seed_ids == ["https://openalex.org/W1", "https://openalex.org/W2"]
    assert service.keyword_search_called is False
    assert service.referenced_fetch_called is False
    assert len(service.cited_fetch_calls) == 1
    assert summary == {
        "paper_count": 4,
        "entity_count": 0,
        "citation_count": 2,
        "entity_relation_count": 0,
    }
    assert [paper.document_id for paper in repo.papers] == ["seed-a", "seed-b", "cited-a", "cited-b"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [
        ("seed-a", "cited-a"),
        ("seed-b", "cited-b"),
    ]
    assert logs[0] == "openalex_seed:start count=2"
    assert "search:done papers=4 citations=2" in logs


def test_openalex_seed_graph_enriches_seed_and_cited_docs_before_extraction():
    repo = FakeRepo()
    service = FakeExplicitSeedPaperService()
    events = []
    extractor = FakeOrderingExtractor(events)
    workflow = MvpGraphBuildWorkflow(
        paper_service=service,
        extractor=extractor,
        repository=repo,
        log=lambda message: None,
        worker_count=1,
        full_text_provider=FakeOrderingFullTextProvider(events),
    )

    summary = workflow.run_openalex_seed_graph(["https://openalex.org/W1", "https://openalex.org/W2"])

    assert summary["paper_count"] == 4
    assert [doc.document_id for doc in extractor.docs] == ["seed-a", "seed-b", "cited-a", "cited-b"]
    for doc in extractor.docs:
        assert doc.text_source == "full_text_md"
        assert doc.full_text_markdown == f"# {doc.document_id}\n\nFull text"
        assert events.index(("enrich", doc.document_id)) < events.index(("extract", doc.document_id))


def test_openalex_seed_graph_extracts_missing_abstract_from_full_text_before_entity_extraction():
    repo = FakeRepo()
    service = FakeMissingAbstractExplicitSeedPaperService()
    events = []
    extractor = FakeOrderingExtractor(events)
    abstract_extractor = FakeOrderingAbstractExtractor(events)
    workflow = MvpGraphBuildWorkflow(
        paper_service=service,
        extractor=extractor,
        repository=repo,
        log=lambda message: None,
        worker_count=1,
        full_text_provider=FakeOrderingFullTextProvider(events),
        paper_abstract_extractor=abstract_extractor,
    )

    summary = workflow.run_openalex_seed_graph(["https://openalex.org/W1", "https://openalex.org/W2"])

    assert summary["paper_count"] == 4
    assert abstract_extractor.calls == [("seed-b", "# seed-b\n\nFull text")]
    assert [doc.abstract for doc in extractor.docs[:2]] == ["Seed A", "Recovered abstract for seed-b"]
    assert events.index(("enrich", "seed-b")) < events.index(("abstract", "seed-b"))
    assert events.index(("abstract", "seed-b")) < events.index(("extract", "seed-b"))


def test_openalex_seed_graph_extracts_missing_abstract_even_when_entity_extraction_is_disabled():
    repo = FakeRepo()
    service = FakeMissingAbstractExplicitSeedPaperService()
    events = []
    workflow = MvpGraphBuildWorkflow(
        paper_service=service,
        extractor=FakeExtractor(),
        repository=repo,
        log=lambda message: None,
        worker_count=1,
        extract_entities_enabled=False,
        full_text_provider=FakeOrderingFullTextProvider(events),
        paper_abstract_extractor=FakeOrderingAbstractExtractor(events),
    )

    summary = workflow.run_openalex_seed_graph(["https://openalex.org/W1", "https://openalex.org/W2"])

    assert summary["entity_count"] == 0
    assert [(paper.document_id, paper.abstract) for paper in repo.papers[:2]] == [
        ("seed-a", "Seed A"),
        ("seed-b", "Recovered abstract for seed-b"),
    ]
    assert events == [("enrich", "seed-b"), ("abstract", "seed-b")]


def test_openalex_seed_graph_keeps_contentless_explicit_seed_for_citation_expansion():
    repo = FakeRepo()
    logs = []
    service = FakeContentlessExplicitSeedPaperService()
    events = []
    extractor = FakeOrderingExtractor(events)
    workflow = MvpGraphBuildWorkflow(
        paper_service=service,
        extractor=extractor,
        repository=repo,
        log=logs.append,
        worker_count=1,
    )

    summary = workflow.run_openalex_seed_graph(["https://openalex.org/W1", "https://openalex.org/W2"])

    assert summary["paper_count"] == 4
    assert summary["citation_count"] == 2
    assert len(service.cited_fetch_calls) == 1
    _, admitted_seed_docs = service.cited_fetch_calls[0]
    assert [doc.document_id for doc in admitted_seed_docs] == ["seed-a", "seed-b"]
    assert [paper.document_id for paper in repo.papers] == ["seed-a", "seed-b", "cited-a", "cited-b"]
    assert [doc.document_id for doc in extractor.docs] == ["seed-a", "cited-a", "cited-b"]
    assert [(rel.source_paper_id, rel.target_paper_id) for rel in repo.citations] == [
        ("seed-a", "cited-a"),
        ("seed-b", "cited-b"),
    ]
    assert (
        "paper:extract_skipped id=seed-b reason=missing_abstract_and_full_text source=openalex_seed"
        in logs
    )


def test_openalex_seed_graph_logs_failed_seed_ids_and_continues():
    repo = FakeRepo()
    logs = []
    service = FakeExplicitSeedPaperService(failed_ids=["https://openalex.org/W404"])
    workflow = MvpGraphBuildWorkflow(
        paper_service=service,
        extractor=FakeExtractor(),
        repository=repo,
        log=logs.append,
        worker_count=1,
        extract_entities_enabled=False,
    )

    summary = workflow.run_openalex_seed_graph(["https://openalex.org/W1", "https://openalex.org/W404"])

    assert summary["paper_count"] == 4
    assert "openalex_seed:failed count=1 ids=https://openalex.org/W404" in logs
    assert [paper.document_id for paper in repo.papers] == ["seed-a", "seed-b", "cited-a", "cited-b"]
