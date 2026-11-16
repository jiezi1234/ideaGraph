from __future__ import annotations

import logging
import re
from pathlib import Path

from src.extraction.pdf_markdown import strip_references


logger = logging.getLogger(__name__)


class FullTextProvider:
    def __init__(
        self,
        output_dir: str | Path,
        arxiv_client,
        downloader,
        extractor,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.arxiv_client = arxiv_client
        self.downloader = downloader
        self.extractor = extractor

    def enrich(self, document):
        # 已有 Markdown 说明之前解析成功过，优先复用，避免重复下载和解析。
        cached = self._try_cached_markdown(document)
        if cached is not None:
            return cached

        pdf_urls = _unique_urls([document.pdf_url, *(getattr(document, "pdf_urls", []) or [])])
        if pdf_urls:
            enriched = self._try_pdf_urls(document, pdf_urls)
            if enriched is not None:
                return enriched

        arxiv_url = self.arxiv_client.find_pdf_by_title(document.document_id)
        if arxiv_url:
            enriched = self._try_pdf_urls(document, [arxiv_url], start_index=len(pdf_urls) + 1)
            if enriched is not None:
                return enriched

        if not pdf_urls and not arxiv_url:
            logger.info("full_text:missing_pdf id=%s", document.document_id)
        else:
            logger.warning(
                "full_text:failed_all id=%s attempts=%s",
                document.document_id,
                len(pdf_urls) + int(bool(arxiv_url)),
            )
        return document

    def _try_cached_markdown(self, document):
        paper_dir = self.output_dir / _safe_name(document.document_id)
        markdown_path = _find_cached_markdown(paper_dir)
        if markdown_path is None:
            return None
        markdown = strip_references(markdown_path.read_text(encoding="utf-8"))
        if not markdown.strip():
            return None
        markdown_path.write_text(markdown, encoding="utf-8")
        logger.info("full_text:cache_hit id=%s markdown=%s", document.document_id, markdown_path)
        return document.model_copy(
            update={
                "full_text_markdown_path": str(markdown_path),
                "full_text_markdown": markdown,
                "text_source": "full_text_md",
            }
        )

    def _try_pdf_urls(self, document, pdf_urls: list[str], start_index: int = 1):
        paper_dir = self.output_dir / _safe_name(document.document_id)
        for index, pdf_url in enumerate(pdf_urls, start=start_index):
            pdf_path = paper_dir / f"source_{index}.pdf"
            output_dir = paper_dir / f"attempt_{index}"
            try:
                # 候选源只有完整走过下载、PDF 校验和解析后才算成功。
                self.downloader.download(pdf_url, pdf_path)
                _ensure_pdf_file(pdf_path)
                result = self.extractor.extract(pdf_path=pdf_path, output_dir=output_dir)
                markdown = strip_references(result.markdown_path.read_text(encoding="utf-8"))
                result.markdown_path.write_text(markdown, encoding="utf-8")
            except Exception as exc:
                logger.warning(
                    "full_text:candidate_failed id=%s url=%s error=%s message=%s",
                    document.document_id,
                    pdf_url,
                    type(exc).__name__,
                    exc,
                )
                continue

            logger.info(
                "full_text:ready id=%s mode=%s markdown=%s",
                document.document_id,
                getattr(result, "mode", "unknown"),
                result.markdown_path,
            )
            return document.model_copy(
                update={
                    "pdf_url": pdf_url,
                    "full_text_markdown_path": str(result.markdown_path),
                    "full_text_markdown": markdown,
                    "text_source": "full_text_md",
                }
            )
        return None


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())[:120]
    return normalized or "paper"


def _unique_urls(candidates: list[str | None]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        url = candidate.strip()
        if not url or url in seen:
            continue
        urls.append(url)
        seen.add(url)
    return urls


def _find_cached_markdown(paper_dir: Path) -> Path | None:
    if not paper_dir.exists():
        return None
    markdown_files = [path for path in paper_dir.rglob("*.md") if path.is_file()]
    if not markdown_files:
        return None
    return max(markdown_files, key=lambda path: path.stat().st_mtime)


def _ensure_pdf_file(path: Path) -> None:
    if path.read_bytes()[:4] != b"%PDF":
        raise ValueError("downloaded file is not a PDF")
