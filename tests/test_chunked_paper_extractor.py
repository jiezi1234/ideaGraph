import asyncio

from src.extraction.chunked_paper_extractor import ChunkedPaperExtractor
from src.schemas.document import PaperDocument
from src.schemas.entity import ExtractedEntity
from src.schemas.extraction import ExtractionResult
from src.schemas.relation import ExtractedRelation


class CharacterTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)

    def split_text(self, text: str, *, max_tokens: int, overlap_tokens: int) -> list[str]:
        if len(text) <= max_tokens:
            return [text.strip()]
        step = max(max_tokens - overlap_tokens, 1)
        parts = []
        for start in range(0, len(text), step):
            part = text[start : start + max_tokens].strip()
            if part:
                parts.append(part)
            if start + max_tokens >= len(text):
                break
        return parts


class RecordingBaseExtractor:
    def __init__(self) -> None:
        self.calls = []

    def extract(self, document, *, existing_entities=None, existing_relations=None):
        self.calls.append(
            {
                "document": document,
                "existing_entities": existing_entities or [],
                "existing_relations": existing_relations or [],
            }
        )
        if len(self.calls) == 1:
            return ExtractionResult(
                entities=[
                    ExtractedEntity(
                        entity_id="Abstract Method",
                        entity_type="Method",
                        description="Initial method description from abstract.",
                    ),
                    ExtractedEntity(
                        entity_id="Body Problem",
                        entity_type="Problem",
                        description="Initial problem description from abstract.",
                    ),
                ],
                relations=[
                    ExtractedRelation(
                        source_entity_id="Abstract Method",
                        target_entity_id="Body Problem",
                        relation_type="Used_For",
                        description="Initial abstract relation.",
                    )
                ],
                input_token_count=10,
                output_token_count=3,
            )
        return ExtractionResult(
            entities=[
                ExtractedEntity(
                    entity_id="Abstract Method",
                    entity_type="Method",
                    description="Updated method description from body chunk.",
                )
            ],
            relations=[
                ExtractedRelation(
                    source_entity_id="Abstract Method",
                    target_entity_id="Body Problem",
                    relation_type="Used_For",
                    description="Updated relation from body chunk.",
                ),
                ExtractedRelation(
                    source_entity_id="Missing Endpoint",
                    target_entity_id="Body Problem",
                    relation_type="Used_For",
                    description="This relation must be skipped.",
                ),
            ],
            input_token_count=20,
            output_token_count=4,
        )


def test_chunked_extractor_runs_abstract_then_reference_stripped_chunks_and_merges_patches():
    base = RecordingBaseExtractor()
    extractor = ChunkedPaperExtractor(
        base,
        tokenizer=CharacterTokenizer(),
        chunk_min_tokens=256,
        chunk_max_tokens=2048,
        chunk_overlap_tokens=0,
    )
    document = PaperDocument(
        document_id="Readable Paper",
        source_type="openalex",
        source_ref="https://openalex.org/W1",
        abstract="The abstract introduces an abstract method.",
        full_text_markdown=(
            "# Readable Paper\n\n"
            "## Abstract\n\n"
            "The abstract introduces an abstract method.\n\n"
            "## Method\n\n"
            "The body explains a body problem.\n\n"
            "## References\n\n"
            "Reference-only method must not be processed."
        ),
        text_source="full_text_md",
        ingestion_status="ready",
    )

    result = extractor.extract(document)

    abstract_doc = base.calls[0]["document"]
    body_doc = base.calls[1]["document"]
    assert abstract_doc.text_source == "title_abstract"
    assert abstract_doc.full_text_markdown is None
    assert body_doc.text_source == "full_text_chunk"
    assert "The body explains a body problem." in body_doc.full_text_markdown
    assert "Reference-only method" not in body_doc.full_text_markdown
    assert [entity.entity_id for entity in base.calls[1]["existing_entities"]] == [
        "abstract method",
        "body problem",
    ]
    assert [entity.entity_id for entity in result.entities] == ["abstract method", "body problem"]
    assert result.entities[0].description == "Updated method description from body chunk."
    assert [(rel.source_entity_id, rel.target_entity_id) for rel in result.relations] == [
        ("abstract method", "body problem")
    ]
    assert result.relations[0].description == "Updated relation from body chunk."
    assert result.input_token_count == 30
    assert result.output_token_count == 7


def test_chunked_extractor_delegates_abstract_only_documents_to_base_extractor():
    base = RecordingBaseExtractor()
    extractor = ChunkedPaperExtractor(base)
    document = PaperDocument(
        document_id="Abstract Paper",
        source_type="openalex",
        source_ref="https://openalex.org/W2",
        abstract="Only abstract text.",
        ingestion_status="ready",
    )

    result = extractor.extract(document)

    assert len(base.calls) == 1
    assert base.calls[0]["document"] is document
    assert result.entities[0].entity_id == "Abstract Method"


def test_chunked_extractor_async_uses_same_incremental_path():
    base = RecordingBaseExtractor()
    extractor = ChunkedPaperExtractor(base)
    document = PaperDocument(
        document_id="Async Paper",
        source_type="openalex",
        source_ref="https://openalex.org/W3",
        abstract="Only abstract text.",
        ingestion_status="ready",
    )

    result = asyncio.run(extractor.extract_async(document))

    assert len(base.calls) == 1
    assert result.entities[0].entity_id == "Abstract Method"
