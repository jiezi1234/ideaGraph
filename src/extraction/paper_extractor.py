import asyncio
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.entity import ExtractedEntity
from src.schemas.extraction import ExtractionResult
from src.schemas.relation import ExtractedRelation

logger = logging.getLogger(__name__)
EntityType = Literal["Concept", "Domain", "Problem", "Method", "Scenario", "Innovation"]
RelationType = Literal[
    "Used_For",
    "Contributes_To",
    "Dependent_On",
    "Evaluated_With",
    "Compare_With",
    "Subclass_Of",
    "Part_Of",
    "Synonym_Of",
]


class ExtractedEntityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str = Field(
        description=(
            "canonical lowercase noun phrase for the entity, not a sentence or claim; "
            "use a natural-language phrase with normal spaces, not snake_case or a slug "
            '(for example "bus scheduling" or "transformer decoder detection head").'
        ),
        min_length=1,
    )
    entity_type: EntityType = Field(description="Single best semantic entity type")
    description: str = Field(
        description="Objective entity definition factually supported by the provided input"
    )


class ExtractedRelationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_entity_id: str = Field(
        description="Source endpoint using the same natural-language entity_id value from the extracted entities",
        min_length=1,
    )
    target_entity_id: str = Field(
        description="Target endpoint using the same natural-language entity_id value from the extracted entities",
        min_length=1,
    )
    relation_type: RelationType = Field(description="Input-grounded semantic relation type")
    description: str = Field(description="Concise relation explanation grounded in the provided paper text")


class PaperExtractionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntityOutput] = Field(description="Extracted paper entities")
    relations: list[ExtractedRelationOutput] = Field(description="Extracted entity-to-entity relations")


class PaperExtractor:
    def __init__(self, model, prompt_template: str, max_retries: int = 2) -> None:
        self._model = model
        self._prompt_template = prompt_template
        self._max_retries = max_retries

    def extract(
        self,
        document,
        *,
        existing_entities: list[ExtractedEntity] | None = None,
        existing_relations: list[ExtractedRelation] | None = None,
    ) -> ExtractionResult:
        content = (getattr(document, "full_text_markdown", None) or "").strip()
        source_mode = getattr(document, "text_source", "title_abstract")
        payload = {
            "title": document.document_id,
            "abstract": document.abstract,
            "content": content,
            "source_mode": source_mode if content else "title_abstract",
            "existing_entities": existing_entities or [],
            "existing_relations": existing_relations or [],
        }
        retry_reason = None
        for attempt in range(self._max_retries + 1):
            prompt = self._build_prompt(retry_reason)
            try:
                parsed, usage = self._generate(prompt, payload)
                logger.info("extract:raw_received doc=%s keys=%s attempt=%s", document.document_id, ["entities", "relations"], attempt + 1)
                entities = [
                    ExtractedEntity(
                        entity_id=item.entity_id,
                        entity_type=item.entity_type,
                        description=item.description,
                        source_refs=[document.document_id],
                    )
                    for item in parsed.entities
                ]
                relations = [
                    ExtractedRelation(
                        source_entity_id=item.source_entity_id,
                        target_entity_id=item.target_entity_id,
                        relation_type=item.relation_type,
                        description=item.description,
                        source_refs=[document.document_id],
                    )
                    for item in parsed.relations
                ]
                return ExtractionResult(
                    entities=entities,
                    relations=relations,
                    input_token_count=usage.get("input"),
                    output_token_count=usage.get("output"),
                )
            except Exception as exc:
                retry_reason = str(exc)
                logger.warning(
                    "extract:validation_failed doc=%s attempt=%s reason=%s",
                    document.document_id,
                    attempt + 1,
                    retry_reason,
                )
                if attempt >= self._max_retries:
                    raise
        raise RuntimeError("unreachable")

    def _generate(self, prompt: str, payload: dict) -> tuple[PaperExtractionOutput, dict]:
        result = self._model.structured_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": self._build_user_content(payload)},
            ],
            schema=PaperExtractionOutput,
            name="paper_entity_extraction",
            metadata={"stage": "paper_entity_extraction"},
        )
        parsed = result.parsed
        output = parsed if isinstance(parsed, PaperExtractionOutput) else PaperExtractionOutput.model_validate(parsed)
        return output, result.usage or {}

    async def extract_async(self, document) -> ExtractionResult:
        return await asyncio.to_thread(self.extract, document)

    def _build_user_content(self, payload: dict) -> str:
        content = payload.get("content")
        current_graph = self._build_current_graph_content(payload)
        if content:
            user_content = (
                f"Title: {payload.get('title', '')}\n"
                f"Source mode: {payload.get('source_mode', 'full_text_md')}\n\n"
            )
            if current_graph:
                user_content += f"{current_graph}\n\n"
            return (
                f"{user_content}<paper_markdown>\n"
                f"{content}\n"
                "</paper_markdown>"
            )
        user_content = (
            f"Title: {payload.get('title', '')}\n"
            f"Abstract: {payload.get('abstract', '')}"
        )
        if current_graph:
            user_content += f"\n\n{current_graph}"
        return user_content

    def _build_current_graph_content(self, payload: dict) -> str:
        entities = payload.get("existing_entities") or []
        relations = payload.get("existing_relations") or []
        if not entities and not relations:
            return ""
        return (
            "Current graph maintenance instruction: Return only changed entities and relations grounded in "
            "the new paper text. Do not return unchanged entries. Include an existing entity or relation only "
            "when this chunk adds evidence that should replace its description. Do not rename an existing "
            "entity_id or change an existing entity_type. Relation endpoints must refer to existing entities "
            "or entities returned in this response. The current state below uses compact "
            "[field]----[field] line format; your output must still follow the structured JSON schema.\n\n"
            "<current_entities>\n"
            f"{_format_entity_state_lines(entities)}\n"
            "</current_entities>\n\n"
            "<current_relations>\n"
            f"{_format_relation_state_lines(relations)}\n"
            "</current_relations>"
        )

    def _build_prompt(self, retry_reason: str | None) -> str:
        if not retry_reason:
            return self._prompt_template
        return (
            f"{self._prompt_template}\n\n"
            "Previous output was invalid.\n"
            f"Reason: {retry_reason}\n"
            "Please re-extract and return content matching the provided schema exactly."
        )


def _format_entity_state_lines(entities: list[ExtractedEntity]) -> str:
    lines = ["[entity_id]----[entity_type]----[description]"]
    lines.extend(
        _compact_line(entity.entity_id, entity.entity_type, entity.description)
        for entity in entities
    )
    return "\n".join(lines)


def _format_relation_state_lines(relations: list[ExtractedRelation]) -> str:
    lines = ["[source_entity_id]----[target_entity_id]----[relation_type]----[description]"]
    lines.extend(
        _compact_line(
            relation.source_entity_id,
            relation.target_entity_id,
            relation.relation_type,
            relation.description,
        )
        for relation in relations
    )
    return "\n".join(lines)


def _compact_line(*fields: str) -> str:
    return "----".join(f"[{_compact_field(field)}]" for field in fields)


def _compact_field(value: str) -> str:
    # current state 只给模型阅读；压平控制字符并规避分隔符，输出仍走结构化 JSON。
    normalized = " ".join(str(value or "").split())
    return normalized.replace("----", "--").replace("[", "(").replace("]", ")")
