from inspect import signature
from pathlib import Path

from src.config.settings import Settings
from src.extraction.chunked_paper_extractor import ChunkedPaperExtractor
from src.extraction.paper_extractor import PaperExtractor
from src.observability.langfuse import build_langfuse_client
from src.extraction.paper_abstract import PaperAbstractExtractor
from src.extraction.pdf_markdown import OpenDataLoaderPdfMarkdownExtractor
from src.ingestion.arxiv_client import ArxivClient
from src.ingestion.full_text import FullTextProvider
from src.ingestion.openalex_client import OpenAlexClient
from src.ingestion.openalex_service import OpenAlexPaperService
from src.ingestion.openreview_client import OpenReviewClient
from src.ingestion.pdf_downloader import PdfDownloader
from src.models.ollama_embeddings import OllamaEmbeddingModel
from src.models.ollama_openai import OllamaOpenAIChatModel
from src.models.structured_cache import CachedStructuredJsonModel
from src.agent.generation_profiles import resolve_generation_profile
from src.prompts import CHAT_IDEA_AGENT_PROMPT
from src.prompts import CONVERT_PAPER_TO_IDEA_PROMPT
from src.prompts import EVALUATE_RESEARCH_IDEA_PROMPT
from src.prompts import EXTRACT_PAPER_ABSTRACT_PROMPT
from src.prompts import EXTRACT_PAPER_ENTITIES_PROMPT
from src.prompts import UNDERSTAND_QUERY_PROMPT
from src.retrieval.chroma_retriever import (
    ChromaChunkRetriever,
    ChromaEntityRetriever,
    ChromaPaperRetriever,
    ChromaVectorIndex,
    HuggingFaceChunkTokenizer,
    build_chroma_client,
)
from src.retrieval.selector import TopScorePaperSelector
from src.storage.graph.neo4j_repository import Neo4jGraphRepository
from src.tools.graph.context_provider import VectorFallbackGraphContextProvider
from src.workflows.accepted_paper_benchmark import AcceptedPaperBenchmarkWorkflow
from src.workflows.entity_extraction_preview import EntityExtractionPreviewWorkflow
from src.workflows.idea_chat import IdeaChatWorkflow
from src.workflows.mvp_graph_build import MvpGraphBuildWorkflow


def load_extract_prompt_template() -> str:
    return EXTRACT_PAPER_ENTITIES_PROMPT


def load_evaluate_idea_prompt_template() -> str:
    return EVALUATE_RESEARCH_IDEA_PROMPT


def load_understand_query_prompt_template() -> str:
    return UNDERSTAND_QUERY_PROMPT


def load_chat_idea_agent_prompt_template() -> str:
    return CHAT_IDEA_AGENT_PROMPT


def load_convert_paper_to_idea_prompt_template() -> str:
    return CONVERT_PAPER_TO_IDEA_PROMPT


def load_extract_paper_abstract_prompt_template() -> str:
    return EXTRACT_PAPER_ABSTRACT_PROMPT


def build_mvp_workflow(settings: Settings) -> MvpGraphBuildWorkflow:
    openalex_api_keys = [key.strip() for key in settings.openalex_api_keys.split(",") if key.strip()]
    openalex_client = OpenAlexClient(
        base_url=settings.openalex_base_url,
        works_filter=settings.openalex_works_filter,
        api_keys=openalex_api_keys,
        api_key_cooldown_seconds=settings.openalex_api_key_cooldown_seconds,
    )
    paper_service = OpenAlexPaperService(openalex_client)
    model = _build_structured_model(settings)
    extractor = PaperExtractor(model, prompt_template=load_extract_prompt_template())
    if settings.extract_entities_enabled and settings.full_text_extraction_enabled:
        extractor = _build_chunked_paper_extractor(extractor, settings)
    abstract_extractor = PaperAbstractExtractor(model, prompt_template=load_extract_paper_abstract_prompt_template())
    full_text_provider = None
    if settings.full_text_extraction_enabled:
        full_text_provider = FullTextProvider(
            output_dir=Path(settings.full_text_output_dir) / settings.neo4j_database,
            arxiv_client=ArxivClient(),
            downloader=PdfDownloader(
                openalex_api_keys=openalex_api_keys,
                openalex_api_key_cooldown_seconds=settings.openalex_api_key_cooldown_seconds,
            ),
            extractor=OpenDataLoaderPdfMarkdownExtractor(
                hybrid_url=settings.pdf_hybrid_url,
                hybrid_backend=settings.pdf_hybrid_backend,
                hybrid_mode=settings.pdf_hybrid_mode,
            ),
        )
    repository = Neo4jGraphRepository.from_connection(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    resolver = None
    relation_resolver = None
    if settings.entity_resolution_enabled and settings.extract_entities_enabled:
        from src.models.ollama_embeddings import OllamaEmbeddingModel
        from src.retrieval.embedding_cache import PaperEmbeddingCache
        from src.knowledge.entity_resolver import EntityResolutionService
        from src.knowledge.relation_resolver import RelationResolutionService
        embedder = OllamaEmbeddingModel(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_embed_model,
            timeout=settings.ollama_timeout,
        )
        embedding_cache = PaperEmbeddingCache(settings.paper_embedding_cache_dir)
        resolver = EntityResolutionService(
            model=model,
            repository=repository,
            embedder=embedder,
            embedding_cache=embedding_cache,
        )
        relation_resolver = RelationResolutionService(
            model=model,
            repository=repository,
        )
    return MvpGraphBuildWorkflow(
        paper_service=paper_service,
        extractor=extractor,
        repository=repository,
        worker_count=settings.extraction_worker_count,
        extract_entities_enabled=settings.extract_entities_enabled,
        citation_graph_strategy=settings.citation_graph_strategy,
        citation_seed_min_cited_by_count=settings.citation_seed_min_cited_by_count,
        citation_seed_min_publication_year=settings.citation_seed_min_publication_year,
        citation_max_total_papers=settings.citation_max_total_papers,
        full_text_provider=full_text_provider,
        paper_abstract_extractor=abstract_extractor,
        resolver=resolver,
        relation_resolver=relation_resolver,
    )


def build_idea_chat_workflow(
    settings: Settings,
    generation_profile: str = "ideagraph",
    retrieval_mode: str = "graph",
) -> IdeaChatWorkflow:
    profile = resolve_generation_profile(generation_profile)
    repository = Neo4jGraphRepository.from_connection(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    embedder = OllamaEmbeddingModel(
        base_url=settings.ollama_base_url,
        model_name=settings.ollama_embed_model,
        timeout=settings.ollama_timeout,
    )
    # chunk 大小必须贴近生成上下文模型的 tokenizer，避免长中文段落被空白切词低估。
    chunk_tokenizer = HuggingFaceChunkTokenizer.from_model_path(
        settings.chunk_tokenizer_model_path,
        local_files_only=settings.chunk_tokenizer_local_files_only,
    )
    vector_index = ChromaVectorIndex(
        client=build_chroma_client(settings.neo4j_database),
        database_name=settings.neo4j_database,
        embedder=embedder,
        tokenizer=chunk_tokenizer,
        chunk_min_tokens=settings.chunk_min_tokens,
        chunk_max_tokens=settings.chunk_max_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    retriever = ChromaPaperRetriever(
        repository=repository,
        index=vector_index,
    )
    entity_retriever = ChromaEntityRetriever(
        repository=repository,
        index=vector_index,
    )
    chunk_retriever = ChromaChunkRetriever(
        repository=repository,
        index=vector_index,
    )
    graph_context_provider = VectorFallbackGraphContextProvider(
        graph_context_provider=repository,
        entity_retriever=entity_retriever,
    )
    selector = TopScorePaperSelector()
    langfuse_client = build_langfuse_client(settings)
    model = _build_structured_model(settings)
    evaluation_model = _build_structured_model(settings)
    workflow_kwargs = {
        "retriever": retriever,
        "selector": selector,
        "model": model,
        "retrieval_top_k": settings.agent_retrieval_top_k,
        "graph_context_provider": graph_context_provider,
        "initial_entity_retriever": entity_retriever,
        "context_chunk_retriever": chunk_retriever,
        "seed_paper_provider": repository,
        "query_understanding_prompt_template": load_understand_query_prompt_template(),
        "agent_prompt_template": profile.prompt_template,
        "evaluation_prompt_template": load_evaluate_idea_prompt_template()
        if profile.internal_evaluation_enabled
        else None,
    }
    if "evaluation_model" in signature(IdeaChatWorkflow).parameters:
        workflow_kwargs["evaluation_model"] = evaluation_model
    if "generation_profile" in signature(IdeaChatWorkflow).parameters:
        workflow_kwargs["generation_profile"] = profile.name
    if "retrieval_mode" in signature(IdeaChatWorkflow).parameters:
        workflow_kwargs["retrieval_mode"] = retrieval_mode
    if langfuse_client is not None and "langfuse_client" in signature(IdeaChatWorkflow).parameters:
        workflow_kwargs["langfuse_client"] = langfuse_client
    return IdeaChatWorkflow(**workflow_kwargs)


def _build_structured_model(settings: Settings):
    model = OllamaOpenAIChatModel.from_settings(settings)
    if not settings.ollama_cache_enabled:
        return model
    # 结构化输出缓存键包含模型配置和 prompt/schema；Langfuse metadata 不影响模型输出，不参与 hash。
    return CachedStructuredJsonModel(
        model,
        cache_dir=settings.ollama_cache_dir,
        cache_identity={
            "model": settings.ollama_model,
            "think_mode": settings.ollama_think_mode,
            "temperature": 1.0,
        },
    )


def build_accepted_paper_benchmark_workflow(settings: Settings) -> AcceptedPaperBenchmarkWorkflow:
    conversion_model = _build_structured_model(settings)
    evaluation_model = _build_structured_model(settings)
    openalex_api_keys = [key.strip() for key in settings.openalex_api_keys.split(",") if key.strip()]
    full_text_provider = FullTextProvider(
        output_dir=Path(settings.full_text_output_dir) / settings.neo4j_database / "benchmark",
        arxiv_client=ArxivClient(),
        downloader=PdfDownloader(
            openalex_api_keys=openalex_api_keys,
            openalex_api_key_cooldown_seconds=settings.openalex_api_key_cooldown_seconds,
        ),
        extractor=OpenDataLoaderPdfMarkdownExtractor(
            hybrid_url=settings.pdf_hybrid_url,
            hybrid_backend=settings.pdf_hybrid_backend,
            hybrid_mode=settings.pdf_hybrid_mode,
        ),
    )
    return AcceptedPaperBenchmarkWorkflow(
        openreview_client=OpenReviewClient(),
        full_text_provider=full_text_provider,
        conversion_model=conversion_model,
        evaluation_model=evaluation_model,
        conversion_prompt_template=load_convert_paper_to_idea_prompt_template(),
        evaluation_prompt_template=load_evaluate_idea_prompt_template(),
    )


def build_local_paper_folder_workflow(settings: Settings) -> MvpGraphBuildWorkflow:
    model = _build_structured_model(settings)
    extractor = PaperExtractor(model, prompt_template=load_extract_prompt_template())
    if settings.extract_entities_enabled:
        extractor = _build_chunked_paper_extractor(extractor, settings)
    abstract_extractor = PaperAbstractExtractor(model, prompt_template=load_extract_paper_abstract_prompt_template())
    pdf_markdown_extractor = OpenDataLoaderPdfMarkdownExtractor(
        hybrid_url=settings.pdf_hybrid_url,
        hybrid_backend=settings.pdf_hybrid_backend,
        hybrid_mode=settings.pdf_hybrid_mode,
    )
    repository = Neo4jGraphRepository.from_connection(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    resolver = None
    relation_resolver = None
    if settings.entity_resolution_enabled and settings.extract_entities_enabled:
        from src.models.ollama_embeddings import OllamaEmbeddingModel
        from src.retrieval.embedding_cache import PaperEmbeddingCache
        from src.knowledge.entity_resolver import EntityResolutionService
        from src.knowledge.relation_resolver import RelationResolutionService
        embedder = OllamaEmbeddingModel(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_embed_model,
            timeout=settings.ollama_timeout,
        )
        embedding_cache = PaperEmbeddingCache(settings.paper_embedding_cache_dir)
        resolver = EntityResolutionService(
            model=model,
            repository=repository,
            embedder=embedder,
            embedding_cache=embedding_cache,
        )
        relation_resolver = RelationResolutionService(
            model=model,
            repository=repository,
        )
    return MvpGraphBuildWorkflow(
        paper_service=None,
        extractor=extractor,
        repository=repository,
        worker_count=settings.extraction_worker_count,
        extract_entities_enabled=settings.extract_entities_enabled,
        pdf_markdown_extractor=pdf_markdown_extractor,
        paper_abstract_extractor=abstract_extractor,
        local_paper_output_dir=Path(settings.full_text_output_dir) / settings.neo4j_database / "local_papers",
        resolver=resolver,
        relation_resolver=relation_resolver,
    )


def build_entity_extraction_preview_workflow(settings: Settings) -> EntityExtractionPreviewWorkflow:
    model = _build_structured_model(settings)
    extractor = PaperExtractor(model, prompt_template=load_extract_prompt_template())
    return EntityExtractionPreviewWorkflow(
        extractor,
        tokenizer=_build_chunk_tokenizer(settings),
        chunk_min_tokens=settings.chunk_min_tokens,
        chunk_max_tokens=settings.chunk_max_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )


def _build_chunked_paper_extractor(extractor, settings: Settings) -> ChunkedPaperExtractor:
    return ChunkedPaperExtractor(
        extractor,
        tokenizer=_build_chunk_tokenizer(settings),
        chunk_min_tokens=settings.chunk_min_tokens,
        chunk_max_tokens=settings.chunk_max_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )


def _build_chunk_tokenizer(settings: Settings) -> HuggingFaceChunkTokenizer:
    # 全文实体抽取、preview 和 chunk-vector 共用同一 tokenizer/边界，避免三套切块口径漂移。
    return HuggingFaceChunkTokenizer.from_model_path(
        settings.chunk_tokenizer_model_path,
        local_files_only=settings.chunk_tokenizer_local_files_only,
    )
