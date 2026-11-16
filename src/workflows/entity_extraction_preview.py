import json
import unicodedata
from pathlib import Path
from typing import Any

from src.extraction.chunked_paper_extractor import ChunkedPaperExtractor, extract_markdown_abstract
from src.knowledge.normalizer import merge_entities, normalize_relations
from src.retrieval.chroma_retriever import CHUNK_MAX_TOKENS, CHUNK_MIN_TOKENS, CHUNK_OVERLAP_TOKENS
from src.retrieval.chroma_retriever import ChunkTokenizer
from src.schemas.document import PaperDocument
from src.schemas.entity import ExtractedEntity
from src.schemas.extraction import ExtractionResult
from src.schemas.relation import ExtractedRelation


class EntityExtractionPreviewWorkflow:
    def __init__(
        self,
        extractor,
        *,
        tokenizer: ChunkTokenizer | None = None,
        chunk_min_tokens: int = CHUNK_MIN_TOKENS,
        chunk_max_tokens: int = CHUNK_MAX_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self._extractor = extractor
        self._tokenizer = tokenizer
        self._chunk_min_tokens = chunk_min_tokens
        self._chunk_max_tokens = chunk_max_tokens
        self._chunk_overlap_tokens = chunk_overlap_tokens

    def run(self, markdown_path: str | Path) -> dict[str, dict[str, list[dict[str, str]]]]:
        path = Path(markdown_path)
        markdown = path.read_text(encoding="utf-8")
        document_id = _resolve_markdown_title(markdown, path)
        full_text_preview, _, _ = self._extract_full_text_preview(path, markdown, document_id)
        incremental_preview, _, _ = self._extract_incremental_preview(path, markdown, document_id)
        return {
            "full_text": full_text_preview,
            "incremental": incremental_preview,
        }

    def _extract_full_text_preview(
        self,
        path: Path,
        markdown: str,
        document_id: str,
    ) -> tuple[dict[str, list[dict[str, str]]], list[ExtractedEntity], list[ExtractedRelation]]:
        document = PaperDocument(
            document_id=document_id,
            source_type="local_md",
            source_ref=str(path),
            abstract="",
            full_text_markdown_path=str(path),
            full_text_markdown=markdown,
            text_source="full_text_md",
            ingestion_status="ready",
        )
        extraction = self._extractor.extract(document)
        return _preview_payload(extraction)

    def _extract_incremental_preview(
        self,
        path: Path,
        markdown: str,
        document_id: str,
    ) -> tuple[dict[str, list[dict[str, str]]], list[ExtractedEntity], list[ExtractedRelation]]:
        document = PaperDocument(
            document_id=document_id,
            source_type="local_md",
            source_ref=str(path),
            abstract=extract_markdown_abstract(markdown),
            full_text_markdown_path=str(path),
            full_text_markdown=markdown,
            text_source="full_text_md",
            ingestion_status="ready",
        )
        chunked_extractor = ChunkedPaperExtractor(
            self._extractor,
            tokenizer=self._tokenizer,
            chunk_min_tokens=self._chunk_min_tokens,
            chunk_max_tokens=self._chunk_max_tokens,
            chunk_overlap_tokens=self._chunk_overlap_tokens,
        )
        return _preview_payload(chunked_extractor.extract(document))


def format_preview_json(preview: dict[str, Any]) -> str:
    return json.dumps(preview, ensure_ascii=False, indent=2) + "\n"


def format_preview_table(preview: dict[str, Any]) -> str:
    entity_columns = ["entity_id", "entity_type", "description"]
    relation_columns = ["source_entity_id", "relation_type", "target_entity_id", "description"]
    sections = []
    for title, key in [("Full Text", "full_text"), ("Incremental", "incremental")]:
        result = preview.get(key, {})
        entity_table = _format_box_table(entity_columns, result.get("entities", []))
        relation_table = _format_box_table(relation_columns, result.get("relations", []))
        sections.append(f"{title}\nEntities\n{entity_table}\n\nRelations\n{relation_table}")
    return "\n\n".join(sections) + "\n"


def _preview_payload(
    extraction: ExtractionResult,
) -> tuple[dict[str, list[dict[str, str]]], list[ExtractedEntity], list[ExtractedRelation]]:
    # 预览结果要模拟入库前的最终形态，但不能触发任何 Neo4j 写入。
    merged_entities = merge_entities(extraction.entities)
    id_map = {
        entity.entity_id: " ".join(entity.entity_id.strip().lower().split())
        for entity in extraction.entities
    }
    normalized_relations = _dedupe_relations(normalize_relations(extraction.relations, id_map))
    payload = {
        "entities": [_entity_payload(entity) for entity in merged_entities],
        "relations": [_relation_payload(relation) for relation in normalized_relations],
    }
    return payload, merged_entities, normalized_relations


def _resolve_markdown_title(markdown: str, path: Path) -> str:
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("# "):
            continue
        title = stripped[2:].strip().rstrip("#").strip()
        if title:
            return title
    return path.stem


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


def _entity_payload(entity: ExtractedEntity) -> dict[str, str]:
    return {
        "entity_id": entity.entity_id,
        "entity_type": entity.entity_type,
        "description": entity.description,
    }


def _relation_payload(relation: ExtractedRelation) -> dict[str, str]:
    return {
        "source_entity_id": relation.source_entity_id,
        "relation_type": relation.relation_type,
        "target_entity_id": relation.target_entity_id,
        "description": relation.description,
    }


def _format_box_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    normalized_rows = [
        {column: _normalize_table_cell(row.get(column, "")) for column in columns}
        for row in rows
    ]
    widths = {
        column: max([_display_width(column)] + [_display_width(row[column]) for row in normalized_rows])
        for column in columns
    }
    lines = [
        _box_border("┌", "┬", "┐", columns, widths),
        _box_row(columns, columns, widths),
        _box_border("├", "┼", "┤", columns, widths),
    ]
    lines.extend(_box_row(columns, [row[column] for column in columns], widths) for row in normalized_rows)
    lines.append(_box_border("└", "┴", "┘", columns, widths))
    return "\n".join(lines)


def _box_border(left: str, joiner: str, right: str, columns: list[str], widths: dict[str, int]) -> str:
    return left + joiner.join("─" * (widths[column] + 2) for column in columns) + right


def _box_row(columns: list[str], values: list[str], widths: dict[str, int]) -> str:
    cells = [
        f" {_pad_display(value, widths[column])} "
        for column, value in zip(columns, values, strict=True)
    ]
    return "│" + "│".join(cells) + "│"


def _normalize_table_cell(value: Any) -> str:
    return " ".join(str(value).split())


def _pad_display(value: str, width: int) -> str:
    return value + " " * (width - _display_width(value))


def _display_width(value: str) -> int:
    # 盒线表面向终端阅读，按字符显示宽度补齐，避免中文宽字符破坏列对齐。
    return sum(2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1 for char in value)
