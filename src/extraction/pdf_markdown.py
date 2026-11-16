from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import opendataloader_pdf


logger = logging.getLogger(__name__)
_REFERENCES_HEADING_RE = re.compile(
    r"(?im)^\s*#{0,6}\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:references|bibliography|参考文献|引用文献)\s*$"
)


@dataclass(frozen=True)
class PdfMarkdownExtractionResult:
    markdown_path: Path
    mode: str
    page_count: int | None
    pages_with_content: int | None


def strip_references(markdown: str) -> str:
    match = _REFERENCES_HEADING_RE.search(markdown)
    if not match:
        return markdown.strip()
    return markdown[: match.start()].strip()


class OpenDataLoaderPdfMarkdownExtractor:
    def __init__(
        self,
        converter: Callable[..., None] | None = None,
        hybrid_backend: str = "docling-fast",
        hybrid_mode: str = "auto",
        hybrid_url: str | None = None,
    ) -> None:
        self._converter = converter or opendataloader_pdf.convert
        self._hybrid_backend = hybrid_backend
        self._hybrid_mode = hybrid_mode
        self._hybrid_url = hybrid_url

    def extract(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        expected_page_count: int | None = None,
    ) -> PdfMarkdownExtractionResult:
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            hybrid = self._extract_with_mode(
                pdf_path,
                output_dir,
                mode="hybrid",
                expected_page_count=expected_page_count,
            )
        except Exception as exc:
            logger.warning(
                "pdf_extract:hybrid_failed pdf=%s fallback=fast error=%s",
                pdf_path,
                exc,
            )
            fast = self._extract_with_mode(pdf_path, output_dir, mode="fast", expected_page_count=expected_page_count)
            self._strip_file_references(fast.markdown_path)
            return fast

        if self._is_complete(hybrid):
            self._strip_file_references(hybrid.markdown_path)
            return hybrid

        logger.warning(
            "pdf_extract:hybrid_incomplete pdf=%s pages=%s expected_pages=%s fallback=fast",
            pdf_path,
            hybrid.pages_with_content,
            expected_page_count,
        )
        fast = self._extract_with_mode(pdf_path, output_dir, mode="fast", expected_page_count=expected_page_count)
        self._strip_file_references(fast.markdown_path)
        return fast

    def _extract_with_mode(
        self,
        pdf_path: Path,
        output_dir: Path,
        mode: str,
        expected_page_count: int | None,
    ) -> PdfMarkdownExtractionResult:
        kwargs = {
            "input_path": str(pdf_path),
            "output_dir": str(output_dir),
            "format": "markdown,json",
            "image_output": "off",
            "quiet": True,
        }
        if mode == "hybrid":
            kwargs["hybrid"] = self._hybrid_backend
            kwargs["hybrid_mode"] = self._hybrid_mode
            if self._hybrid_url:
                kwargs["hybrid_url"] = self._hybrid_url
            else:
                raise RuntimeError("PDF hybrid backend URL is not configured")

        self._converter(**kwargs)
        markdown_path = self._find_markdown(output_dir)
        page_count, pages_with_content = self._read_json_page_counts(output_dir)
        page_count = page_count or expected_page_count
        if pages_with_content is None:
            pages_with_content = expected_page_count
        return PdfMarkdownExtractionResult(
            markdown_path=markdown_path,
            mode=mode,
            page_count=page_count,
            pages_with_content=pages_with_content,
        )

    def _find_markdown(self, output_dir: Path) -> Path:
        markdown_files = sorted(output_dir.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not markdown_files:
            raise FileNotFoundError(f"no markdown output found in {output_dir}")
        return markdown_files[0]

    def _is_complete(self, result: PdfMarkdownExtractionResult) -> bool:
        if result.page_count is None or result.pages_with_content is None:
            return True
        return result.pages_with_content >= result.page_count

    def _strip_file_references(self, markdown_path: Path) -> None:
        markdown = markdown_path.read_text(encoding="utf-8")
        markdown_path.write_text(strip_references(markdown), encoding="utf-8")

    def _read_json_page_counts(self, output_dir: Path) -> tuple[int | None, int | None]:
        json_files = sorted(output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not json_files:
            return None, None
        try:
            payload = json.loads(json_files[0].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, None
        pages = {
            item.get("page number")
            for item in payload.get("kids", [])
            if item.get("page number") is not None
        }
        return payload.get("number of pages"), len(pages) or None
