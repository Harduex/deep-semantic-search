"""Tests for TextSearch with mocked embeddings."""


import torch

from deep_semantic_search.text_searcher import TextSearch


def test_find_similar(mock_sentence_transformers, saved_text_embeddings, sample_text_corpus):
    from deep_semantic_search.text_embedder import TextEmbedder

    mock_sentence_transformers.SentenceTransformer.return_value.encode.return_value = torch.randn(384)
    mock_sentence_transformers.util.pytorch_cos_sim.return_value = torch.randn(1, len(sample_text_corpus))

    embedder = TextEmbedder(metadata_dir=saved_text_embeddings)
    search = TextSearch(embedder)
    results = search.find_similar("machine learning", top_n=2)

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all("text" in r and "score" in r for r in results)


def test_find_similar_returns_correct_keys(mock_sentence_transformers, saved_text_embeddings, sample_text_corpus):
    from deep_semantic_search.text_embedder import TextEmbedder

    mock_sentence_transformers.SentenceTransformer.return_value.encode.return_value = torch.randn(384)
    mock_sentence_transformers.util.pytorch_cos_sim.return_value = torch.randn(1, len(sample_text_corpus))

    embedder = TextEmbedder(metadata_dir=saved_text_embeddings)
    search = TextSearch(embedder)
    results = search.find_similar("test query", top_n=1)

    if results:
        result = results[0]
        assert "index" in result
        assert "text" in result
        assert "path" in result
        assert "score" in result
