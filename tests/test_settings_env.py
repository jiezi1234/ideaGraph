import pytest

from pydantic import ValidationError

from src.config.settings import Settings
from src.schemas.agent import (
    AIIdeaBenchIdeaOutput,
    AIIdeaBenchIdeaToIdeaEvaluation,
    CandidatePaper,
    IdeaEvaluation,
    IdeaOutput,
)


def test_settings_can_load_from_environment(monkeypatch):
    monkeypatch.setenv("OPENALEX_BASE_URL", "https://api.openalex.org")
    monkeypatch.setenv("OPENALEX_WORKS_FILTER", "topics.field.id:17")
    monkeypatch.setenv("OPENALEX_API_KEYS", "key-a,key-b")
    monkeypatch.setenv("OPENALEX_API_KEY_COOLDOWN_SECONDS", "60")
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL_EXTRACTION", "qwen3.7-flash")
    monkeypatch.setenv("LLM_MODEL_GENERATION", "qwen3.7-plus")
    monkeypatch.setenv("EMBED_MODEL", "text-embedding-v4")
    monkeypatch.setenv("NEO4J_URI", "bolt://10.160.4.176:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j102")
    monkeypatch.setenv("NEO4J_DATABASE", "testidea")
    monkeypatch.setenv("LLM_TIMEOUT", "300")
    monkeypatch.setenv("LLM_THINK_MODE", "medium")
    monkeypatch.setenv("LLM_CACHE_ENABLED", "true")
    monkeypatch.setenv("LLM_CACHE_DIR", ".cache/model_responses")
    monkeypatch.setenv("AGENT_RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("CHAT_IDEA_ARCHIVE_DIR", "exports/test-chat-runs")
    monkeypatch.setenv("PAPER_EMBEDDING_CACHE_DIR", ".cache/paper_embeddings")
    monkeypatch.setenv("EMBEDDING_WORKER_COUNT", "6")
    monkeypatch.setenv("CHUNK_TOKENIZER_MODEL_PATH", "/home/zb/models/Qwen3.6-35B-A3B-GPTQ-Int4")
    monkeypatch.setenv("CHUNK_TOKENIZER_LOCAL_FILES_ONLY", "true")
    monkeypatch.setenv("CHUNK_MIN_TOKENS", "256")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "2048")
    monkeypatch.setenv("CHUNK_OVERLAP_TOKENS", "80")
    monkeypatch.setenv("EXTRACTION_WORKER_COUNT", "4")
    monkeypatch.setenv("EXTRACT_ENTITIES_ENABLED", "false")
    monkeypatch.setenv("CITATION_GRAPH_STRATEGY", "priority_expand")
    monkeypatch.setenv("CITATION_SEED_MIN_CITED_BY_COUNT", "60")
    monkeypatch.setenv("CITATION_SEED_MIN_PUBLICATION_YEAR", "2021")
    monkeypatch.setenv("CITATION_MAX_TOTAL_PAPERS", "1234")
    monkeypatch.setenv("FULL_TEXT_EXTRACTION_ENABLED", "true")
    monkeypatch.setenv("FULL_TEXT_OUTPUT_DIR", "data")
    monkeypatch.setenv("PDF_HYBRID_URL", "http://127.0.0.1:15002")
    monkeypatch.setenv("PDF_HYBRID_BACKEND", "docling-fast")
    monkeypatch.setenv("PDF_HYBRID_MODE", "auto")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("LOG_FILE", "logs/test.log")

    settings = Settings()

    assert settings.openalex_base_url == "https://api.openalex.org"
    assert settings.openalex_works_filter == "topics.field.id:17"
    assert settings.openalex_api_keys == "key-a,key-b"
    assert settings.openalex_api_key_cooldown_seconds == 60
    assert settings.llm_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings.llm_api_key == "sk-test"
    assert settings.llm_model_extraction == "qwen3.7-flash"
    assert settings.llm_model_generation == "qwen3.7-plus"
    assert settings.embed_model == "text-embedding-v4"
    assert settings.llm_timeout == 300
    assert settings.llm_think_mode == "medium"
    assert settings.neo4j_uri == "bolt://10.160.4.176:7687"
    assert settings.neo4j_database == "testidea"
    assert settings.llm_cache_enabled is True
    assert settings.llm_cache_dir == ".cache/model_responses"
    assert settings.agent_retrieval_top_k == 5
    assert settings.chat_idea_archive_dir == "exports/test-chat-runs"
    assert settings.paper_embedding_cache_dir == ".cache/paper_embeddings"
    assert settings.embedding_worker_count == 6
    assert settings.chunk_tokenizer_model_path == "/home/zb/models/Qwen3.6-35B-A3B-GPTQ-Int4"
    assert settings.chunk_tokenizer_local_files_only is True
    assert settings.chunk_min_tokens == 256
    assert settings.chunk_max_tokens == 2048
    assert settings.chunk_overlap_tokens == 80
    assert settings.extraction_worker_count == 4
    assert settings.extract_entities_enabled is False
    assert settings.citation_graph_strategy == "priority_expand"
    assert settings.citation_seed_min_cited_by_count == 60
    assert settings.citation_seed_min_publication_year == 2021
    assert settings.citation_max_total_papers == 1234
    assert settings.full_text_extraction_enabled is True
    assert settings.full_text_output_dir == "data"
    assert settings.pdf_hybrid_url == "http://127.0.0.1:15002"
    assert settings.pdf_hybrid_backend == "docling-fast"
    assert settings.pdf_hybrid_mode == "auto"
    assert settings.log_level == "DEBUG"
    assert settings.log_file == "logs/test.log"


def test_settings_embed_credentials_fall_back_to_llm(monkeypatch):
    monkeypatch.setenv("OPENALEX_BASE_URL", "https://api.openalex.org")
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-llm")
    monkeypatch.setenv("LLM_MODEL_EXTRACTION", "qwen3.7-flash")
    monkeypatch.setenv("LLM_MODEL_GENERATION", "qwen3.7-plus")
    monkeypatch.setenv("EMBED_MODEL", "text-embedding-v4")
    monkeypatch.setenv("LLM_TIMEOUT", "300")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    monkeypatch.setenv("NEO4J_DATABASE", "idea")

    settings = Settings(_env_file=None)

    assert settings.resolved_embed_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings.resolved_embed_api_key == "sk-llm"


def test_settings_leaves_citation_seed_threshold_unset_by_default(monkeypatch):
    monkeypatch.delenv("CITATION_SEED_MIN_CITED_BY_COUNT", raising=False)
    monkeypatch.setenv("OPENALEX_BASE_URL", "https://api.openalex.org")
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("LLM_MODEL_EXTRACTION", "qwen3.6:35b")
    monkeypatch.setenv("LLM_MODEL_GENERATION", "qwen3.6:35b")
    monkeypatch.setenv("EMBED_MODEL", "text-embedding-v4")
    monkeypatch.setenv("LLM_TIMEOUT", "300")
    monkeypatch.setenv("LLM_THINK_MODE", "medium")
    monkeypatch.setenv("NEO4J_URI", "bolt://10.160.4.176:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j102")
    monkeypatch.setenv("NEO4J_DATABASE", "testidea")

    settings = Settings(_env_file=None)

    assert settings.citation_seed_min_cited_by_count is None


def test_candidate_paper_schema_has_minimal_fields():
    fields = CandidatePaper.model_fields

    assert "paper_id" in fields
    assert "title" in fields
    assert "abstract" in fields
    assert "cited_by_count" in fields
    assert "score" in fields


def test_candidate_paper_requires_abstract():
    with pytest.raises(ValidationError):
        CandidatePaper(
            paper_id="paper-1",
            title="Graph-based idea generation",
            cited_by_count=12,
            score=0.8,
        )


def test_candidate_paper_accepts_none_cited_by_count():
    paper = CandidatePaper(
        paper_id="paper-1",
        title="Graph-based idea generation",
        abstract="We study graph-guided idea generation.",
        cited_by_count=None,
    )

    assert paper.cited_by_count is None
    assert paper.abstract == "We study graph-guided idea generation."
    assert paper.score == 0.0


def test_candidate_paper_score_defaults_to_zero():
    paper = CandidatePaper(
        paper_id="paper-2",
        title="Graph-guided retrieval",
        abstract="We study retrieval for idea generation.",
    )

    assert paper.score == 0.0


def test_idea_output_schema_has_minimal_fields():
    fields = IdeaOutput.model_fields

    assert "title" in fields
    assert "summary" in fields
    assert "problem" in fields
    assert "method" in fields
    assert "experiment" in fields
    assert "core_paper_id" in fields
    assert "core_paper_title" in fields
    assert "idea_lineage" in fields


def test_ai_idea_bench_idea_output_schema_matches_official_fields():
    fields = AIIdeaBenchIdeaOutput.model_fields

    assert "Name" in fields
    assert "Title" in fields
    assert "Motivation" in fields
    assert "Experiment" in fields
    assert "Interestingness" in fields
    assert "Feasibility" in fields
    assert "Novelty" in fields


def test_ai_idea_bench_i2i_evaluation_schema_has_similarity_fields():
    fields = AIIdeaBenchIdeaToIdeaEvaluation.model_fields

    assert "motivation_similarity" in fields
    assert "experiment_similarity" in fields
    assert "overall_score" in fields


def test_idea_evaluation_schema_has_judge_fields():
    fields = IdeaEvaluation.model_fields

    assert "overall_score" in fields
    assert "novelty" in fields
    assert "feasibility" in fields
    assert "expected_effectiveness" in fields
    assert "major_issues" in fields
    assert "revision_instruction" in fields


@pytest.mark.parametrize("payload", [
    {
        "summary": "summary",
        "problem": "problem",
        "method": "method",
        "core_paper_id": "paper-1",
        "core_paper_title": "Core paper",
    },
    {
        "title": "Idea title",
        "problem": "problem",
        "method": "method",
        "core_paper_id": "paper-1",
        "core_paper_title": "Core paper",
    },
])
def test_idea_output_requires_all_fields(payload):
    with pytest.raises(ValidationError):
        IdeaOutput(**payload)


def test_settings_requires_embed_model_env(monkeypatch):
    monkeypatch.setenv("OPENALEX_BASE_URL", "https://api.openalex.org")
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("LLM_MODEL_EXTRACTION", "qwen3.6:35b")
    monkeypatch.setenv("LLM_MODEL_GENERATION", "qwen3.6:35b")
    import os
    if "EMBED_MODEL" in os.environ:
        del os.environ["EMBED_MODEL"]
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    monkeypatch.setenv("NEO4J_URI", "bolt://10.160.4.176:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j102")
    monkeypatch.setenv("NEO4J_DATABASE", "testidea")
    monkeypatch.setenv("LLM_TIMEOUT", "300")
    monkeypatch.setenv("LLM_THINK_MODE", "medium")

    # also unset it from the loaded environment file
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("top_k", ["0", "-1"])
def test_agent_retrieval_top_k_must_be_positive(monkeypatch, top_k):
    monkeypatch.setenv("OPENALEX_BASE_URL", "https://api.openalex.org")
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("LLM_MODEL_EXTRACTION", "qwen3.6:35b")
    monkeypatch.setenv("LLM_MODEL_GENERATION", "qwen3.6:35b")
    monkeypatch.setenv("EMBED_MODEL", "text-embedding-v4")
    monkeypatch.setenv("NEO4J_URI", "bolt://10.160.4.176:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j102")
    monkeypatch.setenv("NEO4J_DATABASE", "testidea")
    monkeypatch.setenv("LLM_TIMEOUT", "300")
    monkeypatch.setenv("LLM_THINK_MODE", "medium")
    monkeypatch.setenv("AGENT_RETRIEVAL_TOP_K", top_k)

    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize("worker_count", ["0", "-1"])
def test_embedding_worker_count_must_be_positive(monkeypatch, worker_count):
    monkeypatch.setenv("OPENALEX_BASE_URL", "https://api.openalex.org")
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("LLM_MODEL_EXTRACTION", "qwen3.6:35b")
    monkeypatch.setenv("LLM_MODEL_GENERATION", "qwen3.6:35b")
    monkeypatch.setenv("EMBED_MODEL", "text-embedding-v4")
    monkeypatch.setenv("NEO4J_URI", "bolt://10.160.4.176:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j102")
    monkeypatch.setenv("NEO4J_DATABASE", "testidea")
    monkeypatch.setenv("LLM_TIMEOUT", "300")
    monkeypatch.setenv("LLM_THINK_MODE", "medium")
    monkeypatch.setenv("EMBEDDING_WORKER_COUNT", worker_count)

    with pytest.raises(ValidationError):
        Settings()


def test_chunk_min_tokens_cannot_exceed_chunk_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENALEX_BASE_URL", "https://api.openalex.org")
    monkeypatch.setenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("LLM_MODEL_EXTRACTION", "qwen3.6:35b")
    monkeypatch.setenv("LLM_MODEL_GENERATION", "qwen3.6:35b")
    monkeypatch.setenv("EMBED_MODEL", "text-embedding-v4")
    monkeypatch.setenv("NEO4J_URI", "bolt://10.160.4.176:7687")
    monkeypatch.setenv("NEO4J_USERNAME", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "neo4j102")
    monkeypatch.setenv("NEO4J_DATABASE", "testidea")
    monkeypatch.setenv("LLM_TIMEOUT", "300")
    monkeypatch.setenv("LLM_THINK_MODE", "medium")
    monkeypatch.setenv("CHUNK_MIN_TOKENS", "2049")
    monkeypatch.setenv("CHUNK_MAX_TOKENS", "2048")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
