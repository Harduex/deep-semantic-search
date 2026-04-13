"""Tests for ImageClusterer."""

import pytest

from deep_semantic_search.exceptions import ClusteringError
from deep_semantic_search.image_clusterer import ImageClusterer


def _build_clusterer(sample_images, tmp_metadata_dir, mock_siglip_model):
    """Helper: build an indexed ImageIndexer + ImageClusterer pair."""
    from deep_semantic_search.image_indexer import ImageIndexer

    indexer = ImageIndexer(
        image_list=sample_images,
        metadata_dir=tmp_metadata_dir / "siglip",
    )
    indexer.run_index()
    return ImageClusterer(indexer)


def test_cluster_without_index(tmp_metadata_dir, mock_siglip_model):
    from deep_semantic_search.image_indexer import ImageIndexer

    indexer = ImageIndexer(image_list=[], metadata_dir=tmp_metadata_dir / "siglip")
    clusterer = ImageClusterer(indexer)

    with pytest.raises(ClusteringError, match="No indexed images"):
        clusterer.cluster(n_clusters=2)


def test_get_cluster_images_without_clustering(tmp_metadata_dir, mock_siglip_model):
    from deep_semantic_search.image_indexer import ImageIndexer

    indexer = ImageIndexer(image_list=[], metadata_dir=tmp_metadata_dir / "siglip")
    clusterer = ImageClusterer(indexer)

    with pytest.raises(ClusteringError, match="No clustering data"):
        clusterer.get_cluster_images(0)


def test_custom_llm_fn():
    """Test that custom llm_fn is accepted."""
    from unittest.mock import MagicMock

    from deep_semantic_search.image_indexer import ImageIndexer

    mock_indexer = MagicMock(spec=ImageIndexer)
    def custom_fn(texts):
        return ["custom_topic"]
    clusterer = ImageClusterer(mock_indexer, llm_fn=custom_fn)
    assert clusterer._llm_fn is custom_fn


def test_cluster_kmeans(sample_images, tmp_metadata_dir, mock_siglip_model):
    """KMeans path: explicit n_clusters."""
    clusterer = _build_clusterer(sample_images, tmp_metadata_dir, mock_siglip_model)
    result = clusterer.cluster(n_clusters=2)

    assert "cluster" in result.columns
    assert "topic" in result.columns
    assert len(result) == len(sample_images)


def test_cluster_hdbscan(sample_images, tmp_metadata_dir, mock_siglip_model):
    """HDBSCAN path: n_clusters=None, auto-detect."""
    clusterer = _build_clusterer(sample_images, tmp_metadata_dir, mock_siglip_model)
    result = clusterer.cluster(n_clusters=None, min_cluster_size=2)

    assert "cluster" in result.columns
    assert "topic" in result.columns
    # HDBSCAN noise points get topic "noise"
    noise_rows = result[result["cluster"] == -1]
    if not noise_rows.empty:
        assert (noise_rows["topic"] == "noise").all()
