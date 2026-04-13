"""Tests for TextSearch with mocked BGE-M3 embeddings."""

from deep_semantic_search.text_searcher import TextSearch


def test_find_similar(mock_bge_m3_model, saved_text_embeddings, sample_text_corpus):
    from deep_semantic_search.text_embedder import TextEmbedder

    embedder = TextEmbedder(metadata_dir=saved_text_embeddings)
    search = TextSearch(embedder)
    results = search.find_similar("machine learning", top_n=2)

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all("text" in r and "score" in r for r in results)


def test_find_similar_returns_correct_keys(mock_bge_m3_model, saved_text_embeddings, sample_text_corpus):
    from deep_semantic_search.text_embedder import TextEmbedder

    embedder = TextEmbedder(metadata_dir=saved_text_embeddings)
    search = TextSearch(embedder)
    results = search.find_similar("test query", top_n=1)

    if results:
        result = results[0]
        assert "index" in result
        assert "text" in result
        assert "path" in result
        assert "score" in result


def test_find_similar_dense_only(mock_bge_m3_model, saved_text_embeddings, sample_text_corpus):
    from deep_semantic_search.text_embedder import TextEmbedder

    embedder = TextEmbedder(metadata_dir=saved_text_embeddings)
    search = TextSearch(embedder)
    results = search.find_similar("test query", top_n=2, hybrid=False)

    assert isinstance(results, list)
    assert len(results) <= 2


def test_find_duplicates(mock_bge_m3_model, saved_text_embeddings, sample_text_corpus):
    from deep_semantic_search.text_embedder import TextEmbedder

    embedder = TextEmbedder(metadata_dir=saved_text_embeddings)
    search = TextSearch(embedder)
    duplicates = search.find_duplicates(threshold=0.5)

    assert isinstance(duplicates, list)
    for path1, path2, sim in duplicates:
        assert isinstance(path1, str)
        assert isinstance(path2, str)
        assert sim >= 0.5

    sims = [s for _, _, s in duplicates]
    assert sims == sorted(sims, reverse=True)
