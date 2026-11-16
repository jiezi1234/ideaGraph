import logging

import pytest

from src.extraction.paper_extractor import PaperExtractionOutput, PaperExtractor
from src.schemas.document import PaperDocument
from src.schemas.entity import ExtractedEntity
from src.schemas.relation import ExtractedRelation


class StructuredResult:
    def __init__(self, parsed, usage=None):
        self.parsed = parsed
        self.usage = usage or {}


class FakeModel:
    def __init__(self):
        self.calls = []

    def structured_json(self, *, messages, schema, name: str, metadata=None):
        self.calls.append((messages, schema, name, metadata))
        user_content = messages[1]["content"]
        assert "Title: Seed Paper" in user_content
        assert "Abstract: Graph methods improve idea generation." in user_content
        return StructuredResult(
            schema(
                entities=[
                    {"entity_id": "graph methods", "entity_type": "Method", "description": ""},
                    {"entity_id": "idea generation", "entity_type": "Problem", "description": ""},
                ],
                relations=[
                    {
                        "source_entity_id": "graph methods",
                        "target_entity_id": "idea generation",
                        "relation_type": "Used_For",
                        "description": "",
                    }
                ],
            )
        )


def test_extractor_uses_structured_json_schema_per_paper():
    model = FakeModel()
    extractor = PaperExtractor(model, prompt_template="extract paper entities")
    doc = PaperDocument(
        document_id="Seed Paper",
        source_type="openalex",
        source_ref="https://openalex.org/W1",
        abstract="Graph methods improve idea generation.",
        ingestion_status="ready",
    )

    result = extractor.extract(doc)

    assert model.calls[0][1] is PaperExtractionOutput
    assert model.calls[0][2] == "paper_entity_extraction"
    assert model.calls[0][3] == {"stage": "paper_entity_extraction"}
    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert result.relations[0].relation_type == "Used_For"


def test_extraction_schema_describes_canonical_entity_names_and_grounded_descriptions():
    schema = PaperExtractionOutput.model_json_schema()
    entity_props = schema["$defs"]["ExtractedEntityOutput"]["properties"]
    relation_props = schema["$defs"]["ExtractedRelationOutput"]["properties"]

    assert "canonical lowercase noun phrase" in entity_props["entity_id"]["description"]
    assert "not a sentence or claim" in entity_props["entity_id"]["description"]
    assert "natural-language phrase with normal spaces" in entity_props["entity_id"]["description"]
    assert "not snake_case or a slug" in entity_props["entity_id"]["description"]
    assert "bus scheduling" in entity_props["entity_id"]["description"]
    assert "transformer decoder detection head" in entity_props["entity_id"]["description"]
    assert "same natural-language entity_id value" in relation_props["source_entity_id"]["description"]
    assert "same natural-language entity_id value" in relation_props["target_entity_id"]["description"]
    assert "Objective entity definition" in entity_props["description"]["description"]
    assert "factually supported" in entity_props["description"]["description"]
    assert "provided paper text" in relation_props["description"]["description"]


class LegacyShapeModel:
    def structured_json(self, *, messages, schema, name: str, metadata=None):
        return StructuredResult(
            schema(
                entities=[
                    {"text": "Random walks on graphs", "type": "Concept"},
                    {"text": "Sciences", "type": "Domain"},
                ],
                relations=[
                    {
                        "source": "Random walks on graphs",
                        "target": "Sciences",
                        "type": "Used-For",
                    }
                ],
                title="Random walks on graphs: ideas, techniques and results",
            )
        )


def test_extractor_rejects_legacy_entity_shape():
    extractor = PaperExtractor(LegacyShapeModel(), prompt_template="extract paper entities")
    doc = PaperDocument(
        document_id="Random walks on graphs: ideas, techniques and results",
        source_type="openalex",
        source_ref="https://openalex.org/W2",
        abstract="A survey of random walks on graphs.",
        ingestion_status="ready",
    )

    with pytest.raises(ValueError, match="entity_id"):
        extractor.extract(doc)


class LegacyRelationShapeModel:
    def structured_json(self, *, messages, schema, name: str, metadata=None):
        return StructuredResult(
            schema(
                entities=[
                    {"entity_id": "Random walks on graphs", "entity_type": "Concept", "description": ""},
                    {"entity_id": "Sciences", "entity_type": "Domain", "description": ""},
                ],
                relations=[
                    {
                        "source": "Random walks on graphs",
                        "target": "Sciences",
                        "type": "Used-For",
                    }
                ],
            )
        )


def test_extractor_rejects_legacy_relation_shape():
    extractor = PaperExtractor(LegacyRelationShapeModel(), prompt_template="extract paper entities")
    doc = PaperDocument(
        document_id="Random walks on graphs: ideas, techniques and results",
        source_type="openalex",
        source_ref="https://openalex.org/W2",
        abstract="A survey of random walks on graphs.",
        ingestion_status="ready",
    )

    with pytest.raises(ValueError, match="source_entity_id"):
        extractor.extract(doc)


class MultiEntityTypeModel:
    def structured_json(self, *, messages, schema, name: str, metadata=None):
        return StructuredResult(
            schema(
                entities=[
                    {"entity_id": "knowledge graph", "entity_type": "Concept", "description": ""},
                    {"entity_id": "idea generation", "entity_type": "Problem, Domain", "description": ""},
                ],
                relations=[],
            )
        )


def test_extractor_rejects_non_exact_entity_types():
    extractor = PaperExtractor(MultiEntityTypeModel(), prompt_template="extract paper entities")
    doc = PaperDocument(
        document_id="Seed Paper",
        source_type="openalex",
        source_ref="https://openalex.org/W1",
        abstract="Knowledge graphs support idea generation.",
        ingestion_status="ready",
    )

    with pytest.raises(ValueError, match="entity_type"):
        extractor.extract(doc)


class RetryModel:
    def __init__(self):
        self.calls = []

    def structured_json(self, *, messages, schema, name: str, metadata=None):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return StructuredResult(
                schema(
                    entities=[{"text": "bad entity only"}],
                    relations=[],
                )
            )
        return StructuredResult(
            schema(
                entities=[
                    {"entity_id": "graph retrieval", "entity_type": "Method", "description": ""},
                    {"entity_id": "idea generation", "entity_type": "Problem", "description": ""},
                ],
                relations=[
                    {
                        "source_entity_id": "graph retrieval",
                        "target_entity_id": "idea generation",
                        "relation_type": "Used_For",
                        "description": "",
                    }
                ],
            )
        )


def test_extractor_retries_with_validation_feedback(caplog):
    model = RetryModel()
    extractor = PaperExtractor(model, prompt_template="extract paper entities")
    doc = PaperDocument(
        document_id="Retry Paper",
        source_type="openalex",
        source_ref="https://openalex.org/W3",
        abstract="Graph retrieval supports idea generation.",
        ingestion_status="ready",
    )

    with caplog.at_level(logging.WARNING):
        result = extractor.extract(doc)

    assert len(result.entities) == 2
    assert "extract:validation_failed" in caplog.text
    assert len(model.calls) == 2
    assert "Previous output was invalid" in model.calls[1][0]["content"]


class FullTextUsageModel:
    def __init__(self):
        self.messages = []

    def structured_json(self, *, messages, schema, name: str, metadata=None):
        self.messages.append(messages)
        return StructuredResult(
            schema(
                entities=[
                    {
                        "entity_id": "retrieval-augmented generation",
                        "entity_type": "Method",
                        "description": "A method that grounds generation with retrieved papers.",
                    }
                ],
                relations=[
                    {
                        "source_entity_id": "retrieval-augmented generation",
                        "target_entity_id": "scientific idea generation",
                        "relation_type": "Used_For",
                        "description": "The paper uses RAG to support scientific idea generation.",
                    }
                ],
            ),
            usage={"input": 1234, "output": 56, "total": 1290},
        )


def test_extractor_prefers_full_text_markdown_and_records_descriptions_and_token_usage():
    model = FullTextUsageModel()
    extractor = PaperExtractor(model, prompt_template="extract paper entities")
    doc = PaperDocument(
        document_id="Full Text Paper",
        source_type="openalex",
        source_ref="https://openalex.org/W4",
        abstract="Only a short abstract.",
        full_text_markdown="# Full Text Paper\n\n# Introduction\n\nDetailed body.",
        text_source="full_text_md",
        ingestion_status="ready",
    )

    result = extractor.extract(doc)

    user_content = model.messages[0][1]["content"]
    assert "<paper_markdown>" in user_content
    assert "# Full Text Paper" in user_content
    assert "Source mode: full_text_md" in user_content
    assert result.entities[0].description == "A method that grounds generation with retrieved papers."
    assert result.relations[0].description == "The paper uses RAG to support scientific idea generation."
    assert result.input_token_count == 1234
    assert result.output_token_count == 56


def test_extractor_can_pass_existing_entities_and_relations_for_incremental_maintenance():
    model = FullTextUsageModel()
    extractor = PaperExtractor(model, prompt_template="extract paper entities")
    doc = PaperDocument(
        document_id="Chunk Paper",
        source_type="local_md",
        source_ref="/tmp/chunk.md",
        abstract="The abstract describes retrieval-augmented generation.",
        full_text_markdown="## Method\n\nThe method supports scientific idea generation.",
        text_source="full_text_chunk",
        ingestion_status="ready",
    )
    existing_entities = [
        ExtractedEntity(
            entity_id="retrieval-augmented generation",
            entity_type="Method",
            description="A method introduced by the abstract.",
        )
    ]
    existing_relations = [
        ExtractedRelation(
            source_entity_id="retrieval-augmented generation",
            target_entity_id="scientific idea generation",
            relation_type="Used_For",
            description="The abstract says RAG supports scientific idea generation.",
        )
    ]

    extractor.extract(
        doc,
        existing_entities=existing_entities,
        existing_relations=existing_relations,
    )

    user_content = model.messages[0][1]["content"]
    assert "Source mode: full_text_chunk" in user_content
    assert "<current_entities>" in user_content
    assert (
        "[retrieval-augmented generation]----[Method]----[A method introduced by the abstract.]"
        in user_content
    )
    assert "<current_relations>" in user_content
    assert (
        "[retrieval-augmented generation]----[scientific idea generation]----[Used_For]----"
        "[The abstract says RAG supports scientific idea generation.]"
        in user_content
    )
    assert '"entity_id"' not in user_content
    assert "Return only changed entities and relations" in user_content
    assert "Do not return unchanged entries" in user_content


class JsonOnlyModel:
    pass


def test_extractor_requires_structured_json():
    extractor = PaperExtractor(JsonOnlyModel(), prompt_template="extract paper entities")
    doc = PaperDocument(
        document_id="Seed Paper",
        source_type="openalex",
        source_ref="https://openalex.org/W1",
        abstract="Graph methods improve idea generation.",
        ingestion_status="ready",
    )

    with pytest.raises(AttributeError):
        extractor.extract(doc)
