"""Tests for TextEmbedder with mocked BGE-M3 model."""

import pytest

from deep_semantic_search.exceptions import EmbeddingError
from deep_semantic_search.text_embedder import TextEmbedder


def test_embed_saves_files(mock_bge_m3_model, tmp_metadata_dir, sample_text_corpus):
    embedder = TextEmbedder(metadata_dir=tmp_metadata_dir)
    embedder.embed(sample_text_corpus)

    assert (tmp_metadata_dir / "corpus_embeddings.npy").exists()
    assert (tmp_metadata_dir / "corpus_metadata.json").exists()
    assert (tmp_metadata_dir / "text_sparse_vectors.json").exists()
    assert (tmp_metadata_dir / "text_embeddings.usearch").exists()


def test_embed_skips_if_exists(mock_bge_m3_model, tmp_metadata_dir, sample_text_corpus):
    embedder = TextEmbedder(metadata_dir=tmp_metadata_dir)
    embedder.embed(sample_text_corpus)

    # Second call should skip
    mock_bge_m3_model["model"].encode.reset_mock()
    embedder.embed(sample_text_corpus, reindex=False)
    mock_bge_m3_model["model"].encode.assert_not_called()


def test_load_embedding_without_data(mock_bge_m3_model, tmp_metadata_dir):
    embedder = TextEmbedder(metadata_dir=tmp_metadata_dir)
    with pytest.raises(EmbeddingError, match="No embedding data"):
        embedder.load_embedding()


def test_load_embedding_returns_3_tuple(mock_bge_m3_model, saved_text_embeddings, sample_text_corpus):
    embedder = TextEmbedder(metadata_dir=saved_text_embeddings)
    dense, sparse, corpus = embedder.load_embedding()

    assert dense is not None
    assert dense.shape[0] == len(sample_text_corpus)
    assert dense.shape[1] == 1024
    assert sparse is not None
    assert len(sparse) == len(sample_text_corpus)
    assert len(corpus) == len(sample_text_corpus)


def test_index_path_property(mock_bge_m3_model, tmp_metadata_dir):
    embedder = TextEmbedder(metadata_dir=tmp_metadata_dir)
    assert embedder.index_path == tmp_metadata_dir / "text_embeddings.usearch"
    assert embedder.metadata_dir == tmp_metadata_dir
