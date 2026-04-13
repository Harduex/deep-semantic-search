"""Tests for TextEmbedder with mocked SentenceTransformer."""


import pytest
import torch

from deep_semantic_search.exceptions import EmbeddingError
from deep_semantic_search.text_embedder import TextEmbedder


def test_embed_saves_files(mock_sentence_transformers, tmp_metadata_dir, sample_text_corpus):
    mock_sentence_transformers.SentenceTransformer.return_value.encode.return_value = torch.randn(3, 384)

    embedder = TextEmbedder(metadata_dir=tmp_metadata_dir)
    embedder.embed(sample_text_corpus)

    assert (tmp_metadata_dir / "corpus_embeddings_data.pickle").exists()
    assert (tmp_metadata_dir / "corpus_list_data.pickle").exists()


def test_embed_skips_if_exists(mock_sentence_transformers, tmp_metadata_dir, sample_text_corpus):
    mock_sentence_transformers.SentenceTransformer.return_value.encode.return_value = torch.randn(3, 384)

    embedder = TextEmbedder(metadata_dir=tmp_metadata_dir)
    embedder.embed(sample_text_corpus)

    # Second call should skip
    mock_sentence_transformers.SentenceTransformer.return_value.encode.reset_mock()
    embedder.embed(sample_text_corpus, reindex=False)
    mock_sentence_transformers.SentenceTransformer.return_value.encode.assert_not_called()


def test_load_embedding_without_data(mock_sentence_transformers, tmp_metadata_dir):
    embedder = TextEmbedder(metadata_dir=tmp_metadata_dir)
    with pytest.raises(EmbeddingError, match="No embedding data"):
        embedder.load_embedding()


def test_load_embedding_returns_data(mock_sentence_transformers, saved_text_embeddings, sample_text_corpus):
    embedder = TextEmbedder(metadata_dir=saved_text_embeddings)
    embeddings, corpus = embedder.load_embedding()

    assert embeddings is not None
    assert len(corpus) == len(sample_text_corpus)
