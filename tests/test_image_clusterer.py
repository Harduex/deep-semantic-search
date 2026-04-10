"""Tests for ImageClusterer."""

import pytest

from deep_semantic_search.exceptions import ClusteringError
from deep_semantic_search.image_clusterer import ImageClusterer


def test_cluster_without_index(tmp_metadata_dir, mock_clip_model):
    from deep_semantic_search.image_indexer import ImageIndexer

    indexer = ImageIndexer(image_list=[], metadata_dir=tmp_metadata_dir / "clip")
    clusterer = ImageClusterer(indexer)

    with pytest.raises(ClusteringError, match="No indexed images"):
        clusterer.cluster(n_clusters=2)


def test_get_cluster_images_without_clustering(tmp_metadata_dir, mock_clip_model):
    from deep_semantic_search.image_indexer import ImageIndexer

    indexer = ImageIndexer(image_list=[], metadata_dir=tmp_metadata_dir / "clip")
    clusterer = ImageClusterer(indexer)

    with pytest.raises(ClusteringError, match="No clustering data"):
        clusterer.get_cluster_images(0)


def test_custom_llm_fn():
    """Test that custom llm_fn is accepted."""
    from deep_semantic_search.image_indexer import ImageIndexer
    from unittest.mock import MagicMock

    mock_indexer = MagicMock(spec=ImageIndexer)
    custom_fn = lambda texts: ["custom_topic"]
    clusterer = ImageClusterer(mock_indexer, llm_fn=custom_fn)
    assert clusterer._llm_fn is custom_fn
