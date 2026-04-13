"""Shared test fixtures with mocked ML models."""

import json
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def tmp_metadata_dir(tmp_path):
    """Return a temporary metadata directory."""
    d = tmp_path / "metadata"
    d.mkdir()
    return d


@pytest.fixture
def sample_image(tmp_path):
    """Create a minimal valid PNG image file."""
    from PIL import Image

    img = Image.new("RGB", (224, 224), color="red")
    path = tmp_path / "test_image.png"
    img.save(path)
    return str(path)


@pytest.fixture
def sample_images(tmp_path):
    """Create multiple sample image files."""
    from PIL import Image

    paths = []
    for i in range(5):
        img = Image.new("RGB", (100, 100), color=(i * 50, 0, 0))
        path = tmp_path / f"img_{i}.jpg"
        img.save(path)
        paths.append(str(path))
    return paths


@pytest.fixture
def sample_text_corpus():
    """Return a sample text corpus dictionary."""
    return {
        "doc1.txt": "The quick brown fox jumps over the lazy dog.",
        "doc2.txt": "Machine learning is a subset of artificial intelligence.",
        "doc3.txt": "Deep learning uses neural networks with many layers.",
    }


@pytest.fixture
def mock_transformers():
    """Provide a mocked ``transformers`` module via sys.modules."""
    mock_mod = MagicMock()
    original = sys.modules.get("transformers")
    sys.modules["transformers"] = mock_mod
    yield mock_mod
    if original is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = original


@pytest.fixture
def mock_sentence_transformers():
    """Provide a mocked ``sentence_transformers`` module via sys.modules."""
    mock_mod = MagicMock()
    original = sys.modules.get("sentence_transformers")
    sys.modules["sentence_transformers"] = mock_mod
    yield mock_mod
    if original is None:
        sys.modules.pop("sentence_transformers", None)
    else:
        sys.modules["sentence_transformers"] = original


@pytest.fixture
def mock_clip_model(mock_transformers):
    """Mock CLIP model and processor via a faked ``transformers`` module."""
    fake_feature = np.random.rand(512).astype(np.float32)
    fake_feature = fake_feature / np.linalg.norm(fake_feature)

    mock_proc_class = MagicMock()
    mock_proc_class.from_pretrained.return_value = MagicMock()

    mock_model_instance = MagicMock()
    mock_model_instance.to.return_value = mock_model_instance

    feature_tensor = MagicMock(spec=[])
    feature_tensor.detach = MagicMock()
    feature_tensor.detach.return_value.cpu.return_value.numpy.return_value.flatten.return_value = fake_feature
    feature_tensor.detach.return_value.cpu.return_value.numpy.return_value = fake_feature
    mock_model_instance.get_image_features.return_value = feature_tensor
    mock_model_instance.get_text_features.return_value = feature_tensor

    mock_model_class = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model_instance

    mock_transformers.CLIPProcessor = mock_proc_class
    mock_transformers.CLIPModel = mock_model_class

    yield {
        "processor": mock_proc_class,
        "model": mock_model_class,
        "feature": fake_feature,
    }


@pytest.fixture
def saved_text_embeddings(tmp_metadata_dir, sample_text_corpus):
    """Create pre-saved text embeddings for testing TextSearch."""
    import torch

    n_docs = len(sample_text_corpus)
    embeddings = torch.randn(n_docs, 384)

    np.save(tmp_metadata_dir / "corpus_embeddings.npy", embeddings.numpy())
    with open(tmp_metadata_dir / "corpus_metadata.json", "w", encoding="utf-8") as f:
        json.dump(sample_text_corpus, f)

    return tmp_metadata_dir
