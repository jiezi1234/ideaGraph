from src.retrieval.embedding_cache import PaperEmbeddingCache


def test_paper_embedding_cache_can_save_and_load_embedding(tmp_path):
    cache = PaperEmbeddingCache(cache_dir=tmp_path)

    cache.save(
        paper_id="paper-1",
        model_name="qwen3-embedding:4b",
        abstract="abstract-v1",
        embedding=[0.1, 0.2, 0.3],
    )

    loaded = cache.load(
        paper_id="paper-1",
        model_name="qwen3-embedding:4b",
        abstract="abstract-v1",
    )

    assert loaded == [0.1, 0.2, 0.3]


def test_paper_embedding_cache_scopes_entries_by_model_name(tmp_path):
    cache = PaperEmbeddingCache(cache_dir=tmp_path)
    cache.save("paper-1", "model-a", "abstract-v1", [1.0])

    loaded = cache.load("paper-1", "model-b", "abstract-v1")

    assert loaded is None


def test_paper_embedding_cache_invalidates_when_abstract_changes(tmp_path):
    cache = PaperEmbeddingCache(cache_dir=tmp_path)
    cache.save("paper-1", "model-a", "abstract-v1", [1.0])

    loaded = cache.load("paper-1", "model-a", "abstract-v2")

    assert loaded is None


def test_paper_embedding_cache_sanitizes_model_name_in_cache_path(tmp_path):
    cache = PaperEmbeddingCache(cache_dir=tmp_path)

    cache.save("paper-1", "model/name:with spaces", "abstract-v1", [1.0])

    model_dirs = [path.name for path in tmp_path.iterdir() if path.is_dir()]
    assert model_dirs == ["model_name_with_spaces"]
