from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from src.schemas.document import PaperDocument
from src.schemas.relation import PaperCitationRelation


logger = logging.getLogger(__name__)


def _restore_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    max_pos = max(pos for positions in inverted_index.values() for pos in positions)
    tokens = [""] * (max_pos + 1)
    for token, positions in inverted_index.items():
        for pos in positions:
            tokens[pos] = token
    return " ".join(tokens).strip()


class OpenAlexPaperService:
    def __init__(self, client, max_concurrent_requests: int = 5) -> None:
        self._client = client
        self._max_concurrent_requests = max_concurrent_requests

    def fetch_seed_and_citations(self, keyword: str, max_seed_papers: int, require_references: bool):
        raw_seeds = []
        page = 1
        while len(raw_seeds) < max_seed_papers:
            batch = self._client.search_works(
                keyword,
                must_have_references=require_references,
                per_page=20,
                page=page,
            )
            if not batch:
                break
            raw_seeds.extend(batch)
            page += 1
        raw_seeds = raw_seeds[:max_seed_papers]
        seed_docs = [self._to_document(item) for item in raw_seeds]
        cited_docs, paper_links = self._fetch_cited_documents_with_pool(raw_seeds, seed_docs)
        return seed_docs, cited_docs, paper_links

    def fetch_seed_documents(
        self,
        keyword: str,
        max_seed_papers: int,
        require_references: bool,
        fetch_all: bool = False,
        min_cited_by_count: int | None = None,
        min_publication_year: int | None = None,
    ):
        if fetch_all:
            if hasattr(self._client, "search_all_works"):
                raw_seeds = self._client.search_all_works(
                    keyword,
                    must_have_references=require_references,
                    per_page=20,
                    min_cited_by_count=min_cited_by_count,
                    min_publication_year=min_publication_year,
                )
            else:
                raw_seeds = []
                page = 1
                while True:
                    batch = self._client.search_works(
                        keyword,
                        must_have_references=require_references,
                        per_page=20,
                        page=page,
                        min_cited_by_count=min_cited_by_count,
                        min_publication_year=min_publication_year,
                    )
                    if not batch:
                        break
                    raw_seeds.extend(batch)
                    page += 1
        else:
            raw_seeds = []
            page = 1
            while len(raw_seeds) < max_seed_papers:
                batch = self._client.search_works(
                    keyword,
                    must_have_references=require_references,
                    per_page=20,
                    page=page,
                    min_cited_by_count=min_cited_by_count,
                    min_publication_year=min_publication_year,
                )
                if not batch:
                    break
                raw_seeds.extend(batch)
                page += 1
            raw_seeds = raw_seeds[:max_seed_papers]
        seed_docs = [self._to_document(item) for item in raw_seeds]
        return seed_docs, raw_seeds

    def fetch_openalex_seed_documents(self, openalex_ids: list[str]):
        normalized_ids: list[str] = []
        seen_ids: set[str] = set()
        for openalex_id in openalex_ids:
            normalized_id = openalex_id.strip()
            if not normalized_id or normalized_id in seen_ids:
                continue
            seen_ids.add(normalized_id)
            normalized_ids.append(normalized_id)

        seed_docs: list[PaperDocument] = []
        raw_seeds: list[dict] = []
        failed_ids: list[str] = []
        for normalized_id in normalized_ids:
            raw_seed = self._safe_get_work(normalized_id)
            if not raw_seed:
                failed_ids.append(normalized_id)
                continue
            seed_docs.append(self._to_document(raw_seed))
            raw_seeds.append(raw_seed)
        return seed_docs, raw_seeds, failed_ids

    def _safe_get_work(self, openalex_id: str) -> dict | None:
        try:
            return self._client.get_work(openalex_id)
        except Exception:
            logger.warning("openalex:get_work_skipped id=%s", openalex_id, exc_info=True)
            return None

    def _fetch_cited_documents_with_pool(
        self,
        raw_seeds: list[dict],
        seed_docs: list[PaperDocument],
    ):

        cited_docs: list[PaperDocument] = []
        seen_cited_ids: set[str] = set()
        cited_ids: list[str] = []
        raw_id_to_document_id = {raw["id"]: doc.document_id for raw, doc in zip(raw_seeds, seed_docs)}
        direct_openalex_links: list[tuple[str, str]] = []

        for seed in raw_seeds:
            for cited_id in seed.get("referenced_works", []):
                direct_openalex_links.append((seed["id"], cited_id))
                if cited_id in seen_cited_ids:
                    continue
                seen_cited_ids.add(cited_id)
                cited_ids.append(cited_id)

        with ThreadPoolExecutor(max_workers=self._max_concurrent_requests) as executor:
            for cited_raw in executor.map(self._safe_get_work, cited_ids):
                if not cited_raw:
                    continue
                cited_doc = self._to_document(cited_raw)
                cited_docs.append(cited_doc)
                raw_id_to_document_id[cited_raw["id"]] = cited_doc.document_id

        paper_links: list[PaperCitationRelation] = []
        existing_links: set[tuple[str, str]] = set()
        for source_openalex_id, target_openalex_id in direct_openalex_links:
            source_document_id = raw_id_to_document_id.get(source_openalex_id)
            target_document_id = raw_id_to_document_id.get(target_openalex_id)
            if not source_document_id or not target_document_id:
                continue
            pair = (source_document_id, target_document_id)
            if pair in existing_links:
                continue
            paper_links.append(
                PaperCitationRelation(
                    source_paper_id=source_document_id,
                    target_paper_id=target_document_id,
                )
            )
            existing_links.add(pair)

        for cited_doc in cited_docs:
            for ref_id in cited_doc.citation_openalex_ids:
                target_document_id = raw_id_to_document_id.get(ref_id)
                if target_document_id is None:
                    continue
                pair = (cited_doc.document_id, target_document_id)
                if pair in existing_links:
                    continue
                paper_links.append(
                    PaperCitationRelation(
                        source_paper_id=cited_doc.document_id,
                        target_paper_id=target_document_id,
                    )
                )
                existing_links.add(pair)

        return cited_docs, paper_links

    async def fetch_seed_documents_async(
        self,
        keyword: str,
        max_seed_papers: int,
        require_references: bool,
        fetch_all: bool = False,
        min_cited_by_count: int | None = None,
        min_publication_year: int | None = None,
    ):
        return await asyncio.to_thread(
            self.fetch_seed_documents,
            keyword,
            max_seed_papers,
            require_references,
            fetch_all,
            min_cited_by_count,
            min_publication_year,
        )

    async def fetch_openalex_seed_documents_async(self, openalex_ids: list[str]):
        return await asyncio.to_thread(self.fetch_openalex_seed_documents, openalex_ids)

    async def fetch_cited_documents_async(self, raw_seeds: list[dict], seed_docs: list[PaperDocument]):
        return await asyncio.to_thread(self._fetch_cited_documents_with_pool, raw_seeds, seed_docs)

    def fetch_referenced_documents(self, raw_paper: dict):
        referenced_ids = []
        seen_ids: set[str] = set()
        for cited_id in raw_paper.get("referenced_works", []):
            if cited_id in seen_ids:
                continue
            seen_ids.add(cited_id)
            referenced_ids.append(cited_id)

        raw_cited_docs: list[dict] = []
        failed_ids: list[str] = []
        with ThreadPoolExecutor(max_workers=self._max_concurrent_requests) as executor:
            for cited_id, cited_raw in zip(referenced_ids, executor.map(self._safe_get_work, referenced_ids)):
                if not cited_raw:
                    failed_ids.append(cited_id)
                    continue
                raw_cited_docs.append(cited_raw)

        cited_docs = [self._to_document(item) for item in raw_cited_docs]
        raw_id_to_document_id = {item["id"]: doc.document_id for item, doc in zip(raw_cited_docs, cited_docs)}
        source_document_id = self._title_for_raw(raw_paper)
        paper_links = [
            PaperCitationRelation(
                source_paper_id=source_document_id,
                target_paper_id=target_document_id,
            )
            for target_id in referenced_ids
            if (target_document_id := raw_id_to_document_id.get(target_id)) is not None
        ]
        return cited_docs, raw_cited_docs, paper_links, failed_ids

    async def fetch_referenced_documents_async(self, raw_paper: dict):
        return await asyncio.to_thread(self.fetch_referenced_documents, raw_paper)

    async def fetch_seed_and_citations_async(self, keyword: str, max_seed_papers: int, require_references: bool):
        return self.fetch_seed_and_citations(keyword, max_seed_papers, require_references)

    def _to_document(self, item: dict) -> PaperDocument:
        pdf_urls = _extract_pdf_urls(item)
        return PaperDocument(
            document_id=self._title_for_raw(item),
            source_type="openalex",
            source_ref=item["id"],
            abstract=_restore_abstract(item.get("abstract_inverted_index")),
            year=item.get("publication_year"),
            cited_by_count=item.get("cited_by_count"),
            pdf_url=pdf_urls[0] if pdf_urls else None,
            pdf_urls=pdf_urls,
            citation_openalex_ids=item.get("referenced_works", []),
            ingestion_status="ready",
        )

    def _title_for_raw(self, item: dict) -> str:
        title = " ".join((item.get("title") or "").split())
        return title or item["id"]


def _extract_pdf_url(item: dict) -> str | None:
    pdf_urls = _extract_pdf_urls(item)
    return pdf_urls[0] if pdf_urls else None


def _extract_pdf_urls(item: dict) -> list[str]:
    # 先试 OpenAlex 记录到的外部位置；官方 content PDF 需要 API key，作为末级 OpenAlex 兜底。
    locations = item.get("locations") or []
    external_candidates = [
        (item.get("best_oa_location") or {}).get("pdf_url"),
        (item.get("primary_location") or {}).get("pdf_url"),
    ]
    for location in locations:
        external_candidates.append((location or {}).get("pdf_url"))

    landing_candidates = [
        (item.get("best_oa_location") or {}).get("landing_page_url"),
        (item.get("primary_location") or {}).get("landing_page_url"),
    ]
    for location in locations:
        landing_candidates.append((location or {}).get("landing_page_url"))

    oa_url = (item.get("open_access") or {}).get("oa_url")
    if _looks_like_pdf_url(oa_url):
        external_candidates.append(oa_url)
    external_candidates.extend(candidate for candidate in landing_candidates if _looks_like_pdf_url(candidate))

    official_candidates = [(item.get("content_urls") or {}).get("pdf")]
    return _unique_urls([*external_candidates, *official_candidates])


def _looks_like_pdf_url(url: str | None) -> bool:
    return isinstance(url, str) and url.lower().split("?", 1)[0].endswith(".pdf")


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
