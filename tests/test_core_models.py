import pytest

from pydantic import ValidationError

from src.config.settings import Settings
from src.schemas.document import PaperDocument, RetrievedPaperDocument
from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation


def test_settings_and_core_schemas_round_trip():
    settings = Settings(
        _env_file=None,
        openalex_base_url="https://api.openalex.org",
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        llm_model_extraction="qwen3.7-flash",
        llm_model_generation="qwen3.7-plus",
        embed_model="text-embedding-v4",
        llm_timeout=300,
        llm_think_mode="medium",
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="secret",
        neo4j_database="idea_graph_mvp",
    )

    doc = PaperDocument(
        document_id="Graph-based idea generation",
        source_type="openalex",
        source_ref="https://openalex.org/W123",
        abstract="We study graph-guided idea generation.",
        year=2025,
        citation_openalex_ids=["W999"],
        ingestion_status="ready",
    )

    entity = ExtractedEntity(
        entity_id="graph reasoning",
        entity_type="Concept",
        source_refs=["doc-1"],
    )

    relation = ExtractedRelation(
        source_entity_id="graph reasoning",
        target_entity_id="idea generation",
        relation_type="Used-For",
        source_refs=["doc-1"],
    )

    assert settings.neo4j_database == "idea_graph_mvp"
    assert settings.llm_model_generation == "qwen3.7-plus"
    assert doc.citation_openalex_ids == ["W999"]
    assert entity.entity_type == "Concept"
    assert relation.relation_type == "Used-For"


def test_paper_document_requires_ingestion_fields():
    with pytest.raises(ValidationError):
        PaperDocument(
            document_id="Graph-based idea generation",
            source_type=None,
            source_ref=None,
            abstract="We study graph-guided idea generation.",
            ingestion_status=None,
        )


def test_retrieved_paper_document_allows_sparse_read_fields():
    doc = RetrievedPaperDocument(
        document_id="doc-2",
        title="Sparse read paper",
        abstract="Useful abstract",
        year=None,
        cited_by_count=None,
    )

    assert doc.document_id == "doc-2"
    assert doc.cited_by_count is None
