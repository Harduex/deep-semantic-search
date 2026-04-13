"""Tests for ImageIndexer with mocked SigLIP model."""

import json

import numpy as np
import pytest

from deep_semantic_search.exceptions import IndexNotFoundError, ModelLoadError
from deep_semantic_search.image_indexer import ImageIndexer


def test_indexer_creates_metadata_dir(sample_images, tmp_metadata_dir, mock_siglip_model):
    ImageIndexer(
        image_list=sample_images,
        metadata_dir=tmp_metadata_dir / "siglip",
    )
    assert (tmp_metadata_dir / "siglip").exists()


def test_indexer_model_load_failure(sample_images, tmp_metadata_dir, mock_transformers):
    mock_transformers.SiglipProcessor.from_pretrained.side_effect = RuntimeError("download failed")
    indexer = ImageIndexer(image_list=sample_images, metadata_dir=tmp_metadata_dir)
    with pytest.raises(ModelLoadError, match="download failed"):
        _ = indexer.model


def test_run_index(sample_images, tmp_metadata_dir, mock_siglip_model):
    indexer = ImageIndexer(
        image_list=sample_images,
        metadata_dir=tmp_metadata_dir / "siglip",
    )
    indexer.run_index()

    assert (tmp_metadata_dir / "siglip" / "image_features.npy").exists()
    assert (tmp_metadata_dir / "siglip" / "image_paths.json").exists()
    assert (tmp_metadata_dir / "siglip" / "image_features.usearch").exists()
    assert not indexer.image_data.empty


def test_add_images_without_index(sample_images, tmp_metadata_dir, mock_siglip_model):
    indexer = ImageIndexer(
        image_list=sample_images[:2],
        metadata_dir=tmp_metadata_dir / "siglip",
    )
    with pytest.raises(IndexNotFoundError):
        indexer.add_images(sample_images[2:])


def test_image_count_limits(sample_images, tmp_metadata_dir, mock_siglip_model):
    indexer = ImageIndexer(
        image_list=sample_images,
        image_count=2,
        metadata_dir=tmp_metadata_dir / "siglip",
    )
    assert len(indexer.image_list) == 2


def test_index_path_property(sample_images, tmp_metadata_dir, mock_siglip_model):
    indexer = ImageIndexer(
        image_list=sample_images,
        metadata_dir=tmp_metadata_dir / "siglip",
    )
    assert indexer.index_path == tmp_metadata_dir / "siglip" / "image_features.usearch"
    assert indexer.metadata_dir == tmp_metadata_dir / "siglip"


def test_faiss_migration(sample_images, tmp_metadata_dir, mock_siglip_model):
    """Verify that legacy .npy features are migrated to a USearch index."""
    from deep_semantic_search.config import SIGLIP_EMBEDDING_DIM

    meta_dir = tmp_metadata_dir / "siglip"
    meta_dir.mkdir(parents=True, exist_ok=True)

    n = len(sample_images)
    features = np.random.randn(n, SIGLIP_EMBEDDING_DIM).astype(np.float32)
    np.save(meta_dir / "image_features.npy", features)
    with open(meta_dir / "image_paths.json", "w") as f:
        json.dump(sample_images, f)

    # Create a dummy legacy FAISS index file
    (meta_dir / "image_features_vectors.idx").write_bytes(b"fake")

    indexer = ImageIndexer(
        image_list=sample_images,
        metadata_dir=meta_dir,
    )
    indexer.run_index()

    assert (meta_dir / "image_features.usearch").exists()
    assert not indexer.image_data.empty


def test_add_images_extends_index(sample_images, tmp_metadata_dir, mock_siglip_model):
    indexer = ImageIndexer(
        image_list=sample_images[:3],
        metadata_dir=tmp_metadata_dir / "siglip",
    )
    indexer.run_index()
    initial_count = len(indexer.image_data)

    indexer.add_images(sample_images[3:])
    assert len(indexer.image_data) == initial_count + len(sample_images[3:])
