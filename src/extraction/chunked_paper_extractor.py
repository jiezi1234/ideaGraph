import asyncio
import logging
import re

from src.extraction.pdf_markdown import strip_references
from src.knowledge.normalizer import merge_entities, normalize_relations
from src.retrieval.chroma_retriever import CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS, CHUNK_OVERLAP_TOKENS
from src.retrieval.chroma_retriever import ChunkTokenizer, chunk_markdown_text
from src.schemas.entity import ExtractedEntity
from src.schemas.extraction import ExtractionResult
from src.schemas.relation import ExtractedRelation

logger = logging.getLogger(__name__)


class ChunkedPaperExtractor:
    def __init__(
        self,
        base_extractor,
        *,
        tokenizer: ChunkTokenizer | None = None,
        chunk_min_tokens: int = CHUNK_MIN_TOKENS,
        chunk_max_tokens: int = CHUNK_MAX_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self.base_extractor = base_extractor
        self._tokenizer = tokenizer
        self.chunk_min_tokens = chunk_min_tokens
        self.chunk_max_tokens = chunk_max_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def extract(self, document) -> ExtractionResult:
        full_text_markdown = (getattr(document, "full_text_markdown", None) or "").strip()
        if not full_text_markdown:
            return self.base_extractor.extract(document)
        return self._extract_full_text_incrementally(document, full_text_markdown)

    async def extract_async(self, document) -> ExtractionResult:
        return await asyncio.to_thread(self.extract, document)

    def _extract_full_text_incrementally(self, document, full_text_markdown: str) -> ExtractionResult:
        abstract = (getattr(document, "abstract", "") or "").strip()
        current_entities: list[ExtractedEntity] = []
        current_relations: list[ExtractedRelation] = []
        usage = _UsageAccumulator()

        if abstract:
            # 全文抽取先用摘要建立稳定图状态，正文 chunk 只负责补充或更新局部证据。
            abstract_document = document.model_copy(
                update={
                    "full_text_markdown": None,
                    "text_source": "title_abstract",
                }
            )
            initial = self.base_extractor.extract(abstract_document)
            usage.add(initial)
            current_entities, current_relations = apply_extraction_patch(
                current_entities=current_entities,
                current_relations=current_relations,
                patch=initial,
            )

        body_markdown = remove_markdown_abstract(strip_references(full_text_markdown))
        chunks = chunk_markdown_text(
            body_markdown,
            text_source="full_text_chunk",
            tokenizer=self._tokenizer,
            min_tokens=self.chunk_min_tokens,
            max_tokens=self.chunk_max_tokens,
            overlap_tokens=self.chunk_overlap_tokens,
        )
        logger.info(
            "extract:chunked doc=%s abstract=%s chunks=%s",
            getattr(document, "document_id", ""),
            bool(abstract),
            len(chunks),
        )

        for chunk in chunks:
            chunk_document = document.model_copy(
                update={
                    "abstract": abstract,
                    "full_text_markdown": chunk["text"],
                    "text_source": chunk["text_source"],
                }
            )
            patch = self.base_extractor.extract(
                chunk_document,
                existing_entities=current_entities,
                existing_relations=current_relations,
            )
            usage.add(patch)
            current_entities, current_relations = apply_extraction_patch(
                current_entities=current_entities,
                current_relations=current_relations,
                patch=patch,
            )

        return ExtractionResult(
            entities=current_entities,
            relations=current_relations,
            input_token_count=usage.input_tokens,
            output_token_count=usage.output_tokens,
        )


class _UsageAccumulator:
    def __init__(self) -> None:
        self._input_tokens = 0
        self._output_tokens = 0
        self._has_input = False
        self._has_output = False

    @property
    def input_tokens(self) -> int | None:
        return self._input_tokens if self._has_input else None

    @property
    def output_tokens(self) -> int | None:
        return self._output_tokens if self._has_output else None

    def add(self, extraction: ExtractionResult) -> None:
        if extraction.input_token_count is not None:
            self._input_tokens += extraction.input_token_count
            self._has_input = True
        if extraction.output_token_count is not None:
            self._output_tokens += extraction.output_token_count
            self._has_output = True


def extract_markdown_abstract(markdown: str) -> str:
    heading_match = re.search(r"(?im)^\s*#{1,6}\s*(?:abstract|摘要)\s*$", markdown)
    if heading_match:
        section = markdown[heading_match.end() :]
        next_heading = re.search(r"(?m)^\s*#{1,6}\s+\S", section)
        if next_heading:
            section = section[: next_heading.start()]
        return section.strip()

    inline_match = re.search(r"(?is)\babstract\s*[:.]\s*(.+?)(?:\n\s*\n|$)", markdown)
    if inline_match:
        return inline_match.group(1).strip()
    return ""


def remove_markdown_abstract(markdown: str) -> str:
    heading_match = re.search(r"(?im)^\s*#{1,6}\s*(?:abstract|摘要)\s*$", markdown)
    if not heading_match:
        return markdown.strip()
    section_after_heading = markdown[heading_match.end() :]
    next_heading = re.search(r"(?m)^\s*#{1,6}\s+\S", section_after_heading)
    if not next_heading:
        return markdown[: heading_match.start()].strip()
    return (markdown[: heading_match.start()] + section_after_heading[next_heading.start() :]).strip()


def apply_extraction_patch(
    *,
    current_entities: list[ExtractedEntity],
    current_relations: list[ExtractedRelation],
    patch: ExtractionResult,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    patch_entities = merge_entities(patch.entities)
    entity_id_map = {
        entity.entity_id: _normalize_name(entity.entity_id)
        for entity in [*current_entities, *patch.entities]
    }
    normalized_patch_relations = _dedupe_relations(normalize_relations(patch.relations, entity_id_map))

    entity_by_id = {_normalize_name(entity.entity_id): entity for entity in current_entities}
    for entity in patch_entities:
        entity_id = _normalize_name(entity.entity_id)
        existing = entity_by_id.get(entity_id)
        if existing is None:
            entity_by_id[entity_id] = entity.model_copy(update={"entity_id": entity_id})
            continue
        entity_by_id[entity_id] = existing.model_copy(
            update={
                # chunk patch 命中既有实体时，description 以最新 chunk 为准，不做拼接。
                "description": entity.description,
                "source_refs": sorted(set(existing.source_refs + entity.source_refs)),
                "aliases": sorted(set(existing.aliases + entity.aliases + [entity.entity_id])),
            }
        )

    relation_by_key = {_relation_key(relation): relation for relation in current_relations}
    for relation in normalized_patch_relations:
        # 模型可能返回缺少端点实体的关系；跳过可避免写图阶段 MATCH 失败或悬空。
        if relation.source_entity_id not in entity_by_id or relation.target_entity_id not in entity_by_id:
            continue
        key = _relation_key(relation)
        existing = relation_by_key.get(key)
        if existing is None:
            relation_by_key[key] = relation
            continue
        relation_by_key[key] = existing.model_copy(
            update={
                # chunk patch 命中既有关系时，description 以最新 chunk 为准，不做拼接。
                "description": relation.description,
                "source_refs": sorted(set(existing.source_refs + relation.source_refs)),
            }
        )

    return list(entity_by_id.values()), list(relation_by_key.values())


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _relation_key(relation: ExtractedRelation) -> tuple[str, str, str]:
    return (
        _normalize_name(relation.source_entity_id),
        relation.relation_type,
        _normalize_name(relation.target_entity_id),
    )


def _dedupe_relations(relations: list[ExtractedRelation]) -> list[ExtractedRelation]:
    deduped: dict[tuple[str, str, str], ExtractedRelation] = {}
    for relation in relations:
        key = (
            relation.source_entity_id,
            relation.relation_type,
            relation.target_entity_id,
        )
        if key not in deduped:
            deduped[key] = relation
            continue
        current = deduped[key]
        current.source_refs = sorted(set(current.source_refs + relation.source_refs))
        current.description = _merge_text(current.description, relation.description)
    return list(deduped.values())


def _merge_text(left: str, right: str) -> str:
    values = []
    seen = set()
    for value in [left, right]:
        normalized = " ".join((value or "").split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    return "\n".join(values)
