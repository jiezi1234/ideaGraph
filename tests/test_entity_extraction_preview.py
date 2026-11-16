import json

from src.schemas.entity import ExtractedEntity
from src.schemas.extraction import ExtractionResult
from src.schemas.relation import ExtractedRelation
from src.workflows.entity_extraction_preview import (
    EntityExtractionPreviewWorkflow,
    format_preview_json,
    format_preview_table,
)


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


class FakeExtractor:
    def __init__(self):
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
                        entity_id="Full Text Method",
                        entity_type="Method",
                        description="A method extracted from the full paper.",
                    )
                ],
                relations=[],
                input_token_count=12,
                output_token_count=34,
            )
        if len(self.calls) == 2:
            return ExtractionResult(
                entities=[
                    ExtractedEntity(
                        entity_id="Abstract Method",
                        entity_type="Method",
                        description="A method extracted from the abstract.",
                    ),
                    ExtractedEntity(
                        entity_id="Body Problem",
                        entity_type="Problem",
                        description="An initial problem description from the abstract.",
                    )
                ],
                relations=[
                    ExtractedRelation(
                        source_entity_id="Abstract Method",
                        relation_type="Used_For",
                        target_entity_id="Body Problem",
                        description="Initial relation description from the abstract.",
                    )
                ],
                input_token_count=5,
                output_token_count=6,
            )
        return ExtractionResult(
            entities=[
                ExtractedEntity(
                    entity_id="Abstract Method",
                    entity_type="Method",
                    description="An updated method description from the body.",
                ),
            ],
            relations=[
                ExtractedRelation(
                    source_entity_id="Abstract Method",
                    relation_type="Used_For",
                    target_entity_id="Body Problem",
                    description="Updated relation evidence from the body.",
                )
            ],
            input_token_count=7,
            output_token_count=8,
        )


def test_preview_entity_extraction_returns_full_text_and_incremental_shapes(tmp_path):
    markdown_path = tmp_path / "paper.md"
    markdown_path.write_text(
        "# Readable Paper Title\n\n"
        "## Abstract\n\n"
        "The abstract introduces an abstract method.\n\n"
        "## Method\n\n"
        "The body explains a body problem.\n\n"
        "## References\n\n"
        "Reference-only method must not be processed.",
        encoding="utf-8",
    )
    extractor = FakeExtractor()
    workflow = EntityExtractionPreviewWorkflow(extractor)

    preview = workflow.run(markdown_path)

    full_text_doc = extractor.calls[0]["document"]
    abstract_doc = extractor.calls[1]["document"]
    body_doc = extractor.calls[2]["document"]
    assert full_text_doc.document_id == "Readable Paper Title"
    assert full_text_doc.source_type == "local_md"
    assert full_text_doc.text_source == "full_text_md"
    assert "Reference-only method" in full_text_doc.full_text_markdown
    assert abstract_doc.abstract == "The abstract introduces an abstract method."
    assert abstract_doc.full_text_markdown is None
    assert body_doc.text_source == "full_text_chunk"
    assert "The body explains a body problem." in body_doc.full_text_markdown
    assert "Reference-only method" not in body_doc.full_text_markdown
    assert [entity.entity_id for entity in extractor.calls[2]["existing_entities"]] == [
        "abstract method",
        "body problem",
    ]
    assert preview == {
        "full_text": {
            "entities": [
                {
                    "entity_id": "full text method",
                    "entity_type": "Method",
                    "description": "A method extracted from the full paper.",
                }
            ],
            "relations": [],
        },
        "incremental": {
            "entities": [
                {
                    "entity_id": "abstract method",
                    "entity_type": "Method",
                    "description": "An updated method description from the body.",
                },
                {
                    "entity_id": "body problem",
                    "entity_type": "Problem",
                    "description": "An initial problem description from the abstract.",
                },
            ],
            "relations": [
                {
                    "source_entity_id": "abstract method",
                    "relation_type": "Used_For",
                    "target_entity_id": "body problem",
                    "description": "Updated relation evidence from the body.",
                }
            ],
        },
    }


def test_preview_incremental_chunks_use_configured_tokenizer_and_pack_toward_limit(tmp_path):
    markdown_path = tmp_path / "paper.md"
    first_paragraph = "a" * 700
    second_paragraph = "b" * 700
    third_paragraph = "c" * 700
    markdown_path.write_text(
        "# Readable Paper Title\n\n"
        "## Abstract\n\n"
        "The abstract introduces an abstract method.\n\n"
        "## Method\n\n"
        f"{first_paragraph}\n\n{second_paragraph}\n\n{third_paragraph}\n\n"
        "## References\n\n"
        "Reference-only content.",
        encoding="utf-8",
    )
    extractor = FakeExtractor()
    workflow = EntityExtractionPreviewWorkflow(
        extractor,
        tokenizer=CharacterTokenizer(),
        chunk_min_tokens=256,
        chunk_max_tokens=2048,
        chunk_overlap_tokens=0,
    )

    workflow.run(markdown_path)

    chunk_texts = [
        call["document"].full_text_markdown
        for call in extractor.calls
        if call["document"].text_source == "full_text_chunk"
    ]
    assert chunk_texts == [
        f"# Readable Paper Title\n\n## Method\n\n{first_paragraph}\n\n{second_paragraph}",
        third_paragraph,
    ]


def test_preview_json_formatter_outputs_valid_pretty_json():
    preview = {
        "full_text": {
            "entities": [
                {
                    "entity_id": "graph planning",
                    "entity_type": "Method",
                    "description": "A planning method.",
                }
            ],
            "relations": [],
        },
        "incremental": {"entities": [], "relations": []},
    }

    formatted = format_preview_json(preview)

    assert json.loads(formatted) == preview
    assert formatted.endswith("\n")


def test_preview_table_formatter_outputs_box_tables():
    preview = {
        "full_text": {
            "entities": [
                {
                    "entity_id": "graph planning",
                    "entity_type": "Method",
                    "description": "A planning method.",
                }
            ],
            "relations": [],
        },
        "incremental": {
            "entities": [],
            "relations": [
                {
                    "source_entity_id": "graph planning",
                    "relation_type": "Used_For",
                    "target_entity_id": "idea generation",
                    "description": "Graph planning supports idea generation.",
                }
            ],
        },
    }

    formatted = format_preview_table(preview)

    assert formatted.endswith("\n")
    assert "Full Text\nEntities\n" in formatted
    assert "Incremental\nEntities\n" in formatted
    assert "┌" in formatted
    assert "┬" in formatted
    assert "└" in formatted
    assert "│ entity_id      │ entity_type │ description        │" in formatted
    assert "│ graph planning │ Method      │ A planning method. │" in formatted
    assert (
        "│ source_entity_id │ relation_type │ target_entity_id │ description                              │"
        in formatted
    )
    assert (
        "│ graph planning   │ Used_For      │ idea generation  │ Graph planning supports idea generation. │"
        in formatted
    )
