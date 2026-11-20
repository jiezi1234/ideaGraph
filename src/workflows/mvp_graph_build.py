import asyncio
import hashlib
import heapq
import logging
import re
import time
from pathlib import Path

from src.knowledge.normalizer import merge_entities, normalize_relations
from src.schemas.document import PaperDocument
from src.observability.progress import emit_progress

logger = logging.getLogger(__name__)


class MvpGraphBuildWorkflow:
    def __init__(
        self,
        paper_service,
        extractor,
        repository,
        log=None,
        worker_count: int = 2,
        extract_entities_enabled: bool = True,
        citation_graph_strategy: str = "one_hop",
        citation_seed_min_cited_by_count: int | None = None,
        citation_seed_min_publication_year: int = 2021,
        citation_max_total_papers: int = 5000,
        full_text_provider=None,
        pdf_markdown_extractor=None,
        paper_abstract_extractor=None,
        local_paper_output_dir: str | Path | None = None,
        resolver=None,
        relation_resolver=None,
        citation_seed_fetch_all: bool = True,
    ) -> None:
        self._paper_service = paper_service
        self._extractor = extractor
        self._repository = repository
        self._log_output = log or logger.info
        self._worker_count = worker_count
        self._extract_entities_enabled = extract_entities_enabled
        self._citation_graph_strategy = citation_graph_strategy
        self._citation_seed_min_cited_by_count = citation_seed_min_cited_by_count
        self._citation_seed_min_publication_year = citation_seed_min_publication_year
        self._citation_max_total_papers = citation_max_total_papers
        self._citation_seed_fetch_all = citation_seed_fetch_all
        self._full_text_provider = full_text_provider
        self._pdf_markdown_extractor = pdf_markdown_extractor
        self._paper_abstract_extractor = paper_abstract_extractor
        self._local_paper_output_dir = Path(local_paper_output_dir) if local_paper_output_dir else None
        self._resolver = resolver
        self._relation_resolver = relation_resolver

    def _log(self, message):
        self._log_output(message)
        # 仅镜像固定业务事件名，异常详情及凭据不进入浏览器。
        name = message.split(" ", 1)[0]
        if name in {"search:start", "search:done", "paper:node_written", "paper:start",
                    "paper:extract_done", "paper:write_done", "paper:extract_error",
                    "paper:extract_skipped", "extract:skipped", "citation:pending_unresolved"}:
            # 仅提取论文标识和计数，绝不将异常原文当作节点输出。
            fields = dict(re.findall(r"(?:^| )(id|papers|citations|entities|relations|extract_progress_done|extract_remaining)=([^ ]+)", message))
            emit_progress(name, "failed" if name == "paper:extract_error" else "completed", output=fields or None)

    def run(self, keyword: str, max_seed_papers: int, require_references: bool) -> dict:
        return asyncio.run(self.run_async(keyword, max_seed_papers, require_references))

    def run_openalex_seed_graph(self, openalex_ids: list[str]) -> dict:
        return asyncio.run(self.run_openalex_seed_graph_async(openalex_ids))

    def run_local_paper_folders(self, paper_dirs: list[str | Path]) -> dict:
        return asyncio.run(self.run_local_paper_folders_async(paper_dirs))

    async def run_async(self, keyword: str, max_seed_papers: int, require_references: bool) -> dict:
        return await self._run_graph_async(
            keyword=keyword,
            max_seed_papers=max_seed_papers,
            require_references=require_references,
        )

    async def run_openalex_seed_graph_async(self, openalex_ids: list[str]) -> dict:
        return await self._run_graph_async(openalex_ids=openalex_ids)

    async def run_local_paper_folders_async(self, paper_dirs: list[str | Path]) -> dict:
        if self._pdf_markdown_extractor is None:
            raise RuntimeError("local paper folder workflow requires a PDF markdown extractor")
        if self._paper_abstract_extractor is None:
            raise RuntimeError("local paper folder workflow requires an abstract extractor")
        if self._local_paper_output_dir is None:
            raise RuntimeError("local paper folder workflow requires a local output directory")

        self._log(f"local_paper:start count={len(paper_dirs)}")
        documents = []
        seen_document_ids: set[str] = set()
        skipped_count = 0
        for paper_dir in paper_dirs:
            try:
                document = await asyncio.to_thread(self._build_local_paper_document, Path(paper_dir))
            except Exception as exc:
                skipped_count += 1
                self._log(
                    f"local_paper:skipped dir={paper_dir} error={type(exc).__name__} message={exc}"
                )
                continue
            if document.document_id in seen_document_ids:
                skipped_count += 1
                self._log(f"local_paper:duplicate id={document.document_id} dir={paper_dir}")
                continue
            seen_document_ids.add(document.document_id)
            documents.append(document)

        summary = await self._run_documents_async(documents)
        if skipped_count:
            self._log(f"local_paper:skipped_count={skipped_count}")
        self._log(
            "local_paper:done "
            f"papers={summary['paper_count']} "
            f"entities={summary['entity_count']} "
            f"citations={summary['citation_count']} "
            f"entity_relations={summary['entity_relation_count']}"
        )
        return summary

    async def _run_graph_async(
        self,
        keyword: str | None = None,
        max_seed_papers: int | None = None,
        require_references: bool | None = None,
        openalex_ids: list[str] | None = None,
    ) -> dict:
        if openalex_ids is None:
            self._log(
                f"search:start keyword={keyword} max_seed_papers={max_seed_papers} "
                f"require_references={require_references}"
            )
        else:
            self._log(f"openalex_seed:start count={len(openalex_ids)}")
        total_entities = 0
        total_entity_relations = 0
        paper_count = 0
        citation_count = 0
        queue: asyncio.Queue = asyncio.Queue()
        started_at = time.monotonic()
        extract_enqueued_count = 0
        extract_started_count = 0
        extract_done_count = 0

        def extract_progress_suffix() -> str:
            remaining = max(extract_enqueued_count - extract_done_count, 0)
            return (
                f"extract_progress_started={extract_started_count}/{extract_enqueued_count} "
                f"extract_progress_done={extract_done_count}/{extract_enqueued_count} "
                f"extract_remaining={remaining}"
            )

        def log_paper_node_written(doc_id: str, pending_expand: int = 0) -> None:
            total_known = paper_count + pending_expand
            average_seconds_per_paper = (time.monotonic() - started_at) / max(paper_count, 1)
            eta_seconds = int(average_seconds_per_paper * pending_expand)
            self._log(
                f"paper:node_written id={doc_id} "
                f"progress={paper_count}/{total_known} "
                f"pending_expand={pending_expand} "
                f"eta_seconds={eta_seconds}"
            )

        async def enqueue_document(
            doc,
            *,
            should_extract: bool = True,
            skip_reason: str | None = None,
            source: str | None = None,
        ) -> None:
            nonlocal extract_enqueued_count
            if not should_extract:
                # 引用扩展拿到的空内容论文只保留论文/引用结构，不进入语义抽取队列。
                if self._extract_entities_enabled:
                    self._log(
                        f"paper:extract_skipped id={doc.document_id} "
                        f"reason={skip_reason} source={source}"
                    )
                return
            if not self._extract_entities_enabled:
                return
            extract_enqueued_count += 1
            self._log(f"queue:enqueue id={doc.document_id} {extract_progress_suffix()}")
            await queue.put(doc)

        async def worker(worker_id: int) -> None:
            nonlocal total_entities, total_entity_relations, extract_started_count, extract_done_count
            while True:
                doc = await queue.get()
                if doc is None:
                    queue.task_done()
                    self._log(f"worker:done id={worker_id}")
                    break

                try:
                    extract_started_count += 1
                    self._log(f"paper:start id={doc.document_id} {extract_progress_suffix()}")
                    extraction = await self._extractor.extract_async(doc)
                    self._log(
                        f"paper:extract_done id={doc.document_id} "
                        f"entities={len(extraction.entities)} relations={len(extraction.relations)} "
                        f"input_tokens={extraction.input_token_count} output_tokens={extraction.output_token_count} "
                        f"{extract_progress_suffix()}"
                    )

                    # 在线实体对齐与共指消解判定
                    if self._resolver is not None:
                        merged_entities, id_map = await self._resolver.resolve_entities(extraction.entities)
                    else:
                        merged_entities = merge_entities(extraction.entities)
                        id_map = {}
                        for entity in extraction.entities:
                            normalized_name = " ".join(entity.entity_id.strip().lower().split())
                            id_map[entity.entity_id] = normalized_name
                    merged_relations = normalize_relations(extraction.relations, id_map)
                    if self._relation_resolver is not None:
                        merged_relations = await self._relation_resolver.resolve_relations(merged_relations)

                    self._repository.upsert_entities(merged_entities)
                    if self._resolver is not None:
                        # 只有实体写库成功后才进入 resolver 缓存，避免未持久化节点参与后续消歧。
                        self._resolver.remember_entities(merged_entities)
                    self._repository.upsert_paper_entity_relations(
                        doc.document_id,
                        [entity.entity_id for entity in merged_entities],
                    )
                    self._repository.upsert_entity_relations(merged_relations)
                    total_entities += len(merged_entities)
                    total_entity_relations += len(merged_relations)
                    extract_done_count += 1
                    self._log(
                        f"paper:write_done id={doc.document_id} "
                        f"entities={len(merged_entities)} relations={len(merged_relations)} "
                        f"{extract_progress_suffix()}"
                    )
                except Exception as exc:
                    extract_done_count += 1
                    self._log(
                        f"paper:extract_error id={doc.document_id} "
                        f"error={type(exc).__name__} message={exc} "
                        f"{extract_progress_suffix()}"
                    )
                finally:
                    queue.task_done()

        async def one_hop_producer() -> list:
            nonlocal paper_count, citation_count
            known_paper_ids: set[str] = set()
            pending_citations = []

            def flush_pending_citations(final: bool = False) -> None:
                nonlocal citation_count, pending_citations
                ready = []
                remaining = []
                for citation in pending_citations:
                    if (
                        citation.source_paper_id in known_paper_ids
                        and citation.target_paper_id in known_paper_ids
                    ):
                        ready.append(citation)
                    else:
                        remaining.append(citation)
                if ready:
                    self._repository.upsert_citations(ready)
                    citation_count += len(ready)
                pending_citations = remaining
                if final and pending_citations:
                    self._log(f"citation:pending_unresolved count={len(pending_citations)}")

            seed_docs, raw_seeds = await self._paper_service.fetch_seed_documents_async(
                keyword=keyword,
                max_seed_papers=max_seed_papers,
                require_references=require_references,
            )
            admitted_raw_seeds = []
            admitted_seed_docs = []
            for index, doc in enumerate(seed_docs):
                # 搜索入口论文必须至少有摘要或全文，否则没有足够信息作为建图起点。
                doc = await self._prepare_seed_document_for_graph(doc, source="keyword_seed")
                if doc is None:
                    continue
                self._repository.upsert_papers([doc])
                known_paper_ids.add(doc.document_id)
                paper_count += 1
                if raw_seed := self._raw_at(raw_seeds, index):
                    admitted_raw_seeds.append(raw_seed)
                    admitted_seed_docs.append(doc)
                log_paper_node_written(doc.document_id)
                await enqueue_document(doc)

            if admitted_raw_seeds:
                cited_docs, paper_links = await self._paper_service.fetch_cited_documents_async(
                    admitted_raw_seeds,
                    admitted_seed_docs,
                )
            else:
                cited_docs, paper_links = [], []
            pending_citations.extend(paper_links)
            for doc in cited_docs:
                # 引用论文即使缺少可抽取内容，也先写 Paper/CITES，便于后续手工导入覆盖。
                doc, should_extract = await self._prepare_citation_document_for_graph(doc)
                self._repository.upsert_papers([doc])
                known_paper_ids.add(doc.document_id)
                paper_count += 1
                log_paper_node_written(doc.document_id)
                flush_pending_citations()
                await enqueue_document(
                    doc,
                    should_extract=should_extract,
                    skip_reason="missing_abstract_and_full_text",
                    source="citation_expansion",
                )

            flush_pending_citations(final=True)
            self._log(f"search:done papers={paper_count} citations={citation_count}")
            return pending_citations

        async def openalex_seed_producer() -> list:
            nonlocal paper_count, citation_count
            known_paper_ids: set[str] = set()
            pending_citations = []

            def flush_pending_citations(final: bool = False) -> None:
                nonlocal citation_count, pending_citations
                ready = []
                remaining = []
                for citation in pending_citations:
                    if (
                        citation.source_paper_id in known_paper_ids
                        and citation.target_paper_id in known_paper_ids
                    ):
                        ready.append(citation)
                    else:
                        remaining.append(citation)
                if ready:
                    self._repository.upsert_citations(ready)
                    citation_count += len(ready)
                pending_citations = remaining
                if final and pending_citations:
                    self._log(f"citation:pending_unresolved count={len(pending_citations)}")

            seed_docs, raw_seeds, failed_ids = await self._paper_service.fetch_openalex_seed_documents_async(
                openalex_ids or []
            )
            if failed_ids:
                self._log(f"openalex_seed:failed count={len(failed_ids)} ids={','.join(failed_ids)}")

            admitted_raw_seeds = []
            admitted_seed_docs = []
            for index, doc in enumerate(seed_docs):
                # 显式指定的 OpenAlex 论文代表用户确认的结构入口；即使缺少正文，也保留节点用于补引用。
                doc, should_extract = await self._prepare_openalex_seed_document_for_graph(doc)
                self._repository.upsert_papers([doc])
                known_paper_ids.add(doc.document_id)
                paper_count += 1
                if raw_seed := self._raw_at(raw_seeds, index):
                    admitted_raw_seeds.append(raw_seed)
                    admitted_seed_docs.append(doc)
                log_paper_node_written(doc.document_id)
                await enqueue_document(
                    doc,
                    should_extract=should_extract,
                    skip_reason="missing_abstract_and_full_text",
                    source="openalex_seed",
                )

            if admitted_raw_seeds:
                cited_docs, paper_links = await self._paper_service.fetch_cited_documents_async(
                    admitted_raw_seeds,
                    admitted_seed_docs,
                )
            else:
                cited_docs, paper_links = [], []
            pending_citations.extend(paper_links)
            for doc in cited_docs:
                doc, should_extract = await self._prepare_citation_document_for_graph(doc)
                self._repository.upsert_papers([doc])
                known_paper_ids.add(doc.document_id)
                paper_count += 1
                log_paper_node_written(doc.document_id)
                flush_pending_citations()
                await enqueue_document(
                    doc,
                    should_extract=should_extract,
                    skip_reason="missing_abstract_and_full_text",
                    source="citation_expansion",
                )

            flush_pending_citations(final=True)
            self._log(f"search:done papers={paper_count} citations={citation_count}")
            return pending_citations

        async def priority_expand_producer() -> list:
            nonlocal paper_count, citation_count
            limit_counted_papers = 0
            known_paper_ids: set[str] = set()
            discovered_raws: set[str] = set()
            expanded_paper_ids: set[str] = set()
            raw_by_id: dict[str, dict] = {}
            priority_queue: list[tuple[int, int, int, str]] = []
            queue_index = 0

            def push_raw_for_expansion(raw_paper: dict, *, is_seed: bool) -> None:
                nonlocal queue_index
                # priority_expand 依赖 OpenAlex raw 引用关系；是否能抽实体不影响继续扩引用。
                paper_id = raw_paper["id"]
                if paper_id in discovered_raws:
                    return
                discovered_raws.add(paper_id)
                raw_by_id[paper_id] = raw_paper
                cited_by_count = raw_paper.get("cited_by_count") or 0
                heapq.heappush(
                    priority_queue,
                    (-int(is_seed), -cited_by_count, queue_index, paper_id),
                )
                queue_index += 1

            seed_docs, raw_seeds = await self._paper_service.fetch_seed_documents_async(
                keyword=keyword,
                max_seed_papers=max_seed_papers,
                require_references=require_references,
                # 网页交互可限制种子数量，离线批量建库默认仍扫描全部候选。
                fetch_all=self._citation_seed_fetch_all,
                min_cited_by_count=self._citation_seed_min_cited_by_count,
                min_publication_year=self._citation_seed_min_publication_year,
            )

            for index, doc in enumerate(seed_docs):
                if limit_counted_papers >= self._citation_max_total_papers:
                    break
                if doc.document_id in known_paper_ids:
                    continue
                # priority 的初始搜索结果仍按 seed 规则准入，避免空入口污染图。
                doc = await self._prepare_seed_document_for_graph(doc, source="keyword_seed")
                if doc is None:
                    continue
                self._repository.upsert_papers([doc])
                known_paper_ids.add(doc.document_id)
                paper_count += 1
                # 总量限制只约束有摘要或全文、后续可提供语义证据的论文。
                limit_counted_papers += 1
                if raw_seed := self._raw_at(raw_seeds, index):
                    push_raw_for_expansion(raw_seed, is_seed=True)
                log_paper_node_written(doc.document_id, pending_expand=len(priority_queue))
                await enqueue_document(doc)

            while priority_queue and limit_counted_papers < self._citation_max_total_papers:
                _, _, _, paper_id = heapq.heappop(priority_queue)
                if paper_id in expanded_paper_ids:
                    continue
                raw_paper = raw_by_id[paper_id]
                cited_docs, raw_cited_docs, paper_links, failed_ids = await self._paper_service.fetch_referenced_documents_async(
                    raw_paper
                )
                if failed_ids:
                    self._log(
                        f"expand:partial_fetch paper={paper_id} fetched={len(cited_docs)} failed={len(failed_ids)}"
                    )
                links_by_target = {
                    link.target_paper_id: link
                    for link in paper_links
                }
                limit_reached = False
                for doc, raw_cited in zip(cited_docs, raw_cited_docs):
                    if doc.document_id not in known_paper_ids:
                        # 引用节点可能只有 raw 引用关系，没有可抽取正文；这类节点保留结构但不消耗配额。
                        doc, should_extract = await self._prepare_citation_document_for_graph(doc)
                        counts_against_limit = should_extract
                        if counts_against_limit and limit_counted_papers >= self._citation_max_total_papers:
                            self._log(
                                f"expand:limit_reached paper={paper_id} "
                                f"total_papers={limit_counted_papers} "
                                f"max_total_papers={self._citation_max_total_papers}"
                            )
                            limit_reached = True
                            break
                        self._repository.upsert_papers([doc])
                        known_paper_ids.add(doc.document_id)
                        paper_count += 1
                        if counts_against_limit:
                            limit_counted_papers += 1
                        log_paper_node_written(doc.document_id, pending_expand=len(priority_queue))
                        await enqueue_document(
                            doc,
                            should_extract=should_extract,
                            skip_reason="missing_abstract_and_full_text",
                            source="citation_expansion",
                        )

                    link = links_by_target.get(doc.document_id)
                    if link and link.source_paper_id in known_paper_ids and link.target_paper_id in known_paper_ids:
                        self._repository.upsert_citations([link])
                        citation_count += 1

                    push_raw_for_expansion(raw_cited, is_seed=False)

                if limit_reached:
                    expanded_paper_ids.add(paper_id)
                    break

                expanded_paper_ids.add(paper_id)

            self._log(f"search:done papers={paper_count} citations={citation_count}")
            return []

        if self._extract_entities_enabled:
            workers = [asyncio.create_task(worker(idx)) for idx in range(self._worker_count)]
        
        if openalex_ids is not None:
            pending_citations = await openalex_seed_producer()
        elif self._citation_graph_strategy == "priority_expand":
            pending_citations = await priority_expand_producer()
        else:
            pending_citations = await one_hop_producer()
            
        if self._extract_entities_enabled:
            await queue.join()
            for _ in range(self._worker_count):
                await queue.put(None)
            await asyncio.gather(*workers)
        else:
            self._log("extract:skipped enabled=false")

        if pending_citations:
            self._log(f"citation:skipped_unresolved count={len(pending_citations)}")

        summary = {
            "paper_count": paper_count,
            "entity_count": total_entities,
            "citation_count": citation_count,
            "entity_relation_count": total_entity_relations,
        }
        self._log(
            "workflow:done "
            f"papers={summary['paper_count']} "
            f"entities={summary['entity_count']} "
            f"citations={summary['citation_count']} "
            f"entity_relations={summary['entity_relation_count']}"
        )
        return summary

    async def _run_documents_async(self, documents: list[PaperDocument]) -> dict:
        total_entities = 0
        total_entity_relations = 0
        paper_count = 0
        citation_count = 0
        queue: asyncio.Queue = asyncio.Queue()
        started_at = time.monotonic()
        extract_enqueued_count = 0
        extract_started_count = 0
        extract_done_count = 0

        def extract_progress_suffix() -> str:
            remaining = max(extract_enqueued_count - extract_done_count, 0)
            return (
                f"extract_progress_started={extract_started_count}/{extract_enqueued_count} "
                f"extract_progress_done={extract_done_count}/{extract_enqueued_count} "
                f"extract_remaining={remaining}"
            )

        def log_paper_node_written(doc_id: str) -> None:
            average_seconds_per_paper = (time.monotonic() - started_at) / max(paper_count, 1)
            eta_seconds = 0
            self._log(
                f"paper:node_written id={doc_id} "
                f"progress={paper_count}/{paper_count} "
                f"pending_expand=0 "
                f"eta_seconds={eta_seconds}"
            )

        async def enqueue_document(doc) -> None:
            nonlocal extract_enqueued_count
            if not self._extract_entities_enabled:
                return
            extract_enqueued_count += 1
            self._log(f"queue:enqueue id={doc.document_id} {extract_progress_suffix()}")
            await queue.put(doc)

        async def worker(worker_id: int) -> None:
            nonlocal total_entities, total_entity_relations, extract_started_count, extract_done_count
            while True:
                doc = await queue.get()
                if doc is None:
                    queue.task_done()
                    self._log(f"worker:done id={worker_id}")
                    break

                try:
                    extract_started_count += 1
                    self._log(f"paper:start id={doc.document_id} {extract_progress_suffix()}")
                    extraction = await self._extractor.extract_async(doc)
                    self._log(
                        f"paper:extract_done id={doc.document_id} "
                        f"entities={len(extraction.entities)} relations={len(extraction.relations)} "
                        f"input_tokens={extraction.input_token_count} output_tokens={extraction.output_token_count} "
                        f"{extract_progress_suffix()}"
                    )

                    # 在线实体对齐与共指消解判定
                    if self._resolver is not None:
                        merged_entities, id_map = await self._resolver.resolve_entities(extraction.entities)
                    else:
                        merged_entities = merge_entities(extraction.entities)
                        id_map = {}
                        for entity in extraction.entities:
                            normalized_name = " ".join(entity.entity_id.strip().lower().split())
                            id_map[entity.entity_id] = normalized_name
                    merged_relations = normalize_relations(extraction.relations, id_map)
                    if self._relation_resolver is not None:
                        merged_relations = await self._relation_resolver.resolve_relations(merged_relations)

                    self._repository.upsert_entities(merged_entities)
                    if self._resolver is not None:
                        # 只有实体写库成功后才进入 resolver 缓存，避免未持久化节点参与后续消歧。
                        self._resolver.remember_entities(merged_entities)
                    self._repository.upsert_paper_entity_relations(
                        doc.document_id,
                        [entity.entity_id for entity in merged_entities],
                    )
                    self._repository.upsert_entity_relations(merged_relations)
                    total_entities += len(merged_entities)
                    total_entity_relations += len(merged_relations)
                    extract_done_count += 1
                    self._log(
                        f"paper:write_done id={doc.document_id} "
                        f"entities={len(merged_entities)} relations={len(merged_relations)} "
                        f"{extract_progress_suffix()}"
                    )
                except Exception as exc:
                    extract_done_count += 1
                    self._log(
                        f"paper:extract_error id={doc.document_id} "
                        f"error={type(exc).__name__} message={exc} "
                        f"{extract_progress_suffix()}"
                    )
                finally:
                    queue.task_done()

        workers = []
        if self._extract_entities_enabled:
            workers = [asyncio.create_task(worker(idx)) for idx in range(self._worker_count)]

        preserved_citations = []
        for doc in documents:
            doc = await self._prepare_document_for_processing(doc)
            restored = self._repository.replace_existing_paper_for_rebuild(doc.document_id)
            if restored:
                self._log(f"paper:rebuild_preserved_citations id={doc.document_id} count={len(restored)}")
                preserved_citations.extend(restored)
            self._repository.upsert_papers([doc])
            paper_count += 1
            log_paper_node_written(doc.document_id)
            await enqueue_document(doc)

        if self._extract_entities_enabled:
            await queue.join()
            for _ in range(self._worker_count):
                await queue.put(None)
            await asyncio.gather(*workers)
        else:
            self._log("extract:skipped enabled=false")

        if preserved_citations:
            self._repository.upsert_citations(preserved_citations)
            citation_count += len(preserved_citations)
            self._log(f"citation:restored count={len(preserved_citations)}")

        summary = {
            "paper_count": paper_count,
            "entity_count": total_entities,
            "citation_count": citation_count,
            "entity_relation_count": total_entity_relations,
        }
        self._log(
            "workflow:done "
            f"papers={summary['paper_count']} "
            f"entities={summary['entity_count']} "
            f"citations={summary['citation_count']} "
            f"entity_relations={summary['entity_relation_count']}"
        )
        return summary

    async def _prepare_document_for_processing(self, doc: PaperDocument) -> PaperDocument:
        needs_full_text = (
            self._full_text_provider is not None
            and not doc.full_text_markdown
            and (
                self._extract_entities_enabled
                or (not doc.abstract.strip() and self._paper_abstract_extractor is not None)
            )
        )
        if needs_full_text:
            doc = await asyncio.to_thread(self._full_text_provider.enrich, doc)
        if not doc.abstract.strip():
            doc = await asyncio.to_thread(self._fill_missing_abstract, doc)
        return doc

    async def _prepare_seed_document_for_graph(self, doc: PaperDocument, *, source: str) -> PaperDocument | None:
        doc = await self._prepare_document_for_processing(doc)
        if self._has_processing_content(doc):
            return doc
        self._log(
            f"paper:skipped id={doc.document_id} "
            f"reason=missing_abstract_and_full_text source={source}"
        )
        return None

    async def _prepare_openalex_seed_document_for_graph(self, doc: PaperDocument) -> tuple[PaperDocument, bool]:
        doc = await self._prepare_document_for_processing(doc)
        # 显式 OpenAlex seed 需要保留 Paper/CITES 结构，布尔值只控制是否进入语义抽取。
        return doc, self._has_processing_content(doc)

    async def _prepare_citation_document_for_graph(self, doc: PaperDocument) -> tuple[PaperDocument, bool]:
        doc = await self._prepare_document_for_processing(doc)
        # 引用论文总是返回给调用方写 Paper；布尔值仅表示是否有内容可做实体抽取。
        return doc, self._has_processing_content(doc)

    def _has_processing_content(self, doc: PaperDocument) -> bool:
        # 这里判断的是文本信息量，不代表 citation raw 是否还能继续扩展。
        return bool(doc.abstract.strip() or (doc.full_text_markdown or "").strip())

    def _raw_at(self, raw_papers: list[dict], index: int) -> dict | None:
        # 写 Paper 以文档对象为准；raw 只用于后续引用扩展，缺失时不能阻断抽取。
        if index >= len(raw_papers):
            return None
        return raw_papers[index]

    def _build_local_paper_document(self, paper_dir: Path) -> PaperDocument:
        paper_dir = paper_dir.expanduser().resolve()
        if not paper_dir.exists() or not paper_dir.is_dir():
            raise FileNotFoundError(f"paper folder not found: {paper_dir}")
        pdf_path = self._find_local_pdf(paper_dir)
        output_dir = self._local_paper_output_dir / _local_paper_output_slug(paper_dir)
        cached_markdown_path = self._find_existing_local_markdown(output_dir)
        if cached_markdown_path is not None:
            # 本地 PDF 转 Markdown 成本高；已有转换结果时只复用文本，后续建图仍可重建。
            self._log(f"local_paper:markdown_cache_hit id={paper_dir.name} path={cached_markdown_path}")
            markdown_path = cached_markdown_path
        else:
            extraction_result = self._pdf_markdown_extractor.extract(pdf_path=pdf_path, output_dir=output_dir)
            markdown_path = extraction_result.markdown_path
        markdown = markdown_path.read_text(encoding="utf-8")
        abstract_result = self._paper_abstract_extractor.extract(
            paper_title=paper_dir.name,
            full_text_markdown=markdown,
        )
        return PaperDocument(
            document_id=paper_dir.name,
            source_type="local_folder",
            source_ref=str(paper_dir),
            abstract=abstract_result.abstract,
            pdf_url=None,
            full_text_markdown_path=str(markdown_path),
            full_text_markdown=markdown,
            text_source="full_text_md",
            citation_openalex_ids=[],
            ingestion_status="ready",
        )

    def _find_existing_local_markdown(self, output_dir: Path) -> Path | None:
        markdown_files = sorted(output_dir.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
        for markdown_path in markdown_files:
            try:
                if markdown_path.read_text(encoding="utf-8").strip():
                    return markdown_path
            except OSError:
                continue
        return None

    def _fill_missing_abstract(self, doc: PaperDocument) -> PaperDocument:
        if doc.abstract.strip():
            return doc
        if self._paper_abstract_extractor is None:
            self._log(f"paper:abstract_skipped id={doc.document_id} reason=extractor_missing")
            return doc
        if not doc.full_text_markdown:
            self._log(f"paper:abstract_skipped id={doc.document_id} reason=full_text_missing")
            return doc
        try:
            abstract_result = self._paper_abstract_extractor.extract(
                paper_title=doc.document_id,
                full_text_markdown=doc.full_text_markdown,
            )
        except Exception as exc:
            self._log(
                f"paper:abstract_error id={doc.document_id} "
                f"error={type(exc).__name__} message={exc}"
            )
            return doc
        self._log(f"paper:abstract_extracted id={doc.document_id}")
        return doc.model_copy(update={"abstract": abstract_result.abstract})

    def _find_local_pdf(self, paper_dir: Path) -> Path:
        direct_pdfs = sorted(path for path in paper_dir.glob("*.pdf") if path.is_file())
        if len(direct_pdfs) == 1:
            return direct_pdfs[0]
        if len(direct_pdfs) > 1:
            raise ValueError(f"multiple pdf files found in {paper_dir}: {', '.join(str(path) for path in direct_pdfs)}")

        recursive_pdfs = sorted(path for path in paper_dir.rglob("*.pdf") if path.is_file())
        if not recursive_pdfs:
            raise FileNotFoundError(f"no pdf file found in {paper_dir}")
        if len(recursive_pdfs) == 1:
            return recursive_pdfs[0]
        raise ValueError(
            f"multiple pdf files found in {paper_dir}: {', '.join(str(path) for path in recursive_pdfs)}"
        )


def _local_paper_output_slug(paper_dir: Path) -> str:
    normalized = paper_dir.name.strip().replace(" ", "_")
    digest = hashlib.sha1(str(paper_dir).encode("utf-8")).hexdigest()[:8]
    return f"{normalized}-{digest}"
