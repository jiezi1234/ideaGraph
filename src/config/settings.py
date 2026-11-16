from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    openalex_base_url: str = Field(validation_alias="OPENALEX_BASE_URL")
    openalex_works_filter: str = Field(default="", validation_alias="OPENALEX_WORKS_FILTER")
    openalex_api_keys: str = Field(default="", validation_alias="OPENALEX_API_KEYS")
    openalex_api_key_cooldown_seconds: int = Field(default=300, validation_alias="OPENALEX_API_KEY_COOLDOWN_SECONDS")
    llm_base_url: str = Field(validation_alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    llm_timeout: float = Field(validation_alias="LLM_TIMEOUT")
    llm_think_mode: str = Field(default="", validation_alias="LLM_THINK_MODE")
    # 抽取与生成共用同一端点，但模型档位不同：抽取量大用便宜模型，生成追求质量。
    llm_model_extraction: str = Field(validation_alias="LLM_MODEL_EXTRACTION")
    llm_model_generation: str = Field(validation_alias="LLM_MODEL_GENERATION")
    llm_cache_enabled: bool = Field(default=False, validation_alias="LLM_CACHE_ENABLED")
    llm_cache_dir: str = Field(default=".cache/model_responses", validation_alias="LLM_CACHE_DIR")
    embed_base_url: str = Field(default="", validation_alias="EMBED_BASE_URL")
    embed_api_key: str = Field(default="", validation_alias="EMBED_API_KEY")
    embed_model: str = Field(validation_alias="EMBED_MODEL")
    langfuse_public_key: str = Field(default="", validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST")
    langfuse_tracing_enabled: bool = Field(default=True, validation_alias="LANGFUSE_TRACING_ENABLED")
    langfuse_sample_rate: float = Field(default=1.0, ge=0.0, le=1.0, validation_alias="LANGFUSE_SAMPLE_RATE")
    langfuse_flush_at: int = Field(default=1, gt=0, validation_alias="LANGFUSE_FLUSH_AT")
    agent_retrieval_top_k: int = Field(default=3, gt=0, validation_alias="AGENT_RETRIEVAL_TOP_K")
    chat_idea_archive_dir: str = Field(
        default="exports/chat_idea_runs",
        validation_alias="CHAT_IDEA_ARCHIVE_DIR",
    )
    paper_embedding_cache_dir: str = Field(
        default=".cache/paper_embeddings",
        validation_alias="PAPER_EMBEDDING_CACHE_DIR",
    )
    embedding_worker_count: int = Field(default=4, gt=0, validation_alias="EMBEDDING_WORKER_COUNT")
    chunk_tokenizer_model_path: str = Field(
        default="/home/zb/models/Qwen3.6-35B-A3B-GPTQ-Int4",
        validation_alias="CHUNK_TOKENIZER_MODEL_PATH",
    )
    chunk_tokenizer_local_files_only: bool = Field(
        default=True,
        validation_alias="CHUNK_TOKENIZER_LOCAL_FILES_ONLY",
    )
    chunk_min_tokens: int = Field(default=256, gt=0, validation_alias="CHUNK_MIN_TOKENS")
    chunk_max_tokens: int = Field(default=2048, gt=0, validation_alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(default=80, ge=0, validation_alias="CHUNK_OVERLAP_TOKENS")
    extraction_worker_count: int = Field(default=1, validation_alias="EXTRACTION_WORKER_COUNT")
    extract_entities_enabled: bool = Field(default=True, validation_alias="EXTRACT_ENTITIES_ENABLED")
    entity_resolution_enabled: bool = Field(default=True, validation_alias="ENTITY_RESOLUTION_ENABLED")
    citation_graph_strategy: str = Field(default="one_hop", validation_alias="CITATION_GRAPH_STRATEGY")
    citation_seed_min_cited_by_count: int | None = Field(default=None, validation_alias="CITATION_SEED_MIN_CITED_BY_COUNT")
    citation_seed_min_publication_year: int = Field(default=2021, validation_alias="CITATION_SEED_MIN_PUBLICATION_YEAR")
    citation_max_total_papers: int = Field(default=5000, validation_alias="CITATION_MAX_TOTAL_PAPERS")
    full_text_extraction_enabled: bool = Field(default=False, validation_alias="FULL_TEXT_EXTRACTION_ENABLED")
    full_text_output_dir: str = Field(default="data", validation_alias="FULL_TEXT_OUTPUT_DIR")
    pdf_hybrid_url: str = Field(default="http://127.0.0.1:15002", validation_alias="PDF_HYBRID_URL")
    pdf_hybrid_backend: str = Field(default="docling-fast", validation_alias="PDF_HYBRID_BACKEND")
    pdf_hybrid_mode: str = Field(default="auto", validation_alias="PDF_HYBRID_MODE")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_file: str = Field(default="logs/ideagraph.log", validation_alias="LOG_FILE")
    graph_multihop_depth: int = Field(default=2, ge=1, validation_alias="GRAPH_MULTIHOP_DEPTH")
    graph_react_width: int = Field(default=3, ge=1, validation_alias="GRAPH_REACT_WIDTH")
    neo4j_uri: str = Field(validation_alias="NEO4J_URI")
    neo4j_username: str = Field(validation_alias="NEO4J_USERNAME")
    neo4j_password: str = Field(validation_alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(validation_alias="NEO4J_DATABASE")

    @model_validator(mode="after")
    def validate_chunk_token_bounds(self) -> "Settings":
        if self.chunk_min_tokens > self.chunk_max_tokens:
            raise ValueError("CHUNK_MIN_TOKENS must be less than or equal to CHUNK_MAX_TOKENS")
        if self.chunk_overlap_tokens >= self.chunk_max_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS must be less than CHUNK_MAX_TOKENS")
        return self

    @property
    def resolved_embed_base_url(self) -> str:
        # 同厂商（如百炼）只需配 LLM_*，EMBED_BASE_URL 留空即复用。
        return self.embed_base_url.strip() or self.llm_base_url

    @property
    def resolved_embed_api_key(self) -> str:
        return self.embed_api_key.strip() or self.llm_api_key
