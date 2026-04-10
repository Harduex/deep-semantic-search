"""Tests for ImageIndexer with mocked CLIP model."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from deep_semantic_search.image_indexer import ImageIndexer
from deep_semantic_search.exceptions import ModelLoadError, IndexNotFoundError


def test_indexer_creates_metadata_dir(sample_images, tmp_metadata_dir, mock_clip_model):
    indexer = ImageIndexer(
        image_list=sample_images,
        metadata_dir=tmp_metadata_dir / "clip",
    )
    assert (tmp_metadata_dir / "clip").exists()


def test_indexer_model_load_failure(sample_images, tmp_metadata_dir, mock_transformers):
    mock_transformers.CLIPProcessor.from_pretrained.side_effect = RuntimeError("download failed")
    with pytest.raises(ModelLoadError, match="download failed"):
        ImageIndexer(image_list=sample_images, metadata_dir=tmp_metadata_dir)


def test_run_index(sample_images, tmp_metadata_dir, mock_clip_model):
    indexer = ImageIndexer(
        image_list=sample_images,
        metadata_dir=tmp_metadata_dir / "clip",
    )
    indexer.run_index()

    assert (tmp_metadata_dir / "clip" / "image_data_features.pkl").exists()
    assert (tmp_metadata_dir / "clip" / "image_features_vectors.idx").exists()
    assert not indexer.image_data.empty


def test_add_images_without_index(sample_images, tmp_metadata_dir, mock_clip_model):
    indexer = ImageIndexer(
        image_list=sample_images[:2],
        metadata_dir=tmp_metadata_dir / "clip",
    )
    with pytest.raises(IndexNotFoundError):
        indexer.add_images(sample_images[2:])


def test_image_count_limits(sample_images, tmp_metadata_dir, mock_clip_model):
    indexer = ImageIndexer(
        image_list=sample_images,
        image_count=2,
        metadata_dir=tmp_metadata_dir / "clip",
    )
    assert len(indexer.image_list) == 2
