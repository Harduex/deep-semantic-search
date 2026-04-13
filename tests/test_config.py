"""Tests for config module."""

import warnings
from pathlib import Path

from deep_semantic_search.config import (
    BGE_M3_MODEL_DEFAULT,
    DEFAULT_METADATA_DIR,
    DEFAULT_RERANK_MODEL,
    FLORENCE_MODEL_DEFAULT,
    IMAGE_EXTENSIONS,
    IMAGE_USEARCH_INDEX_FILE,
    SIGLIP_EMBEDDING_DIM,
    SIGLIP_IMAGE_SIZE,
    SIGLIP_MODEL_DEFAULT,
    TEXT_USEARCH_INDEX_FILE,
)


def test_default_metadata_dir_is_in_home():
    assert DEFAULT_METADATA_DIR.parent == Path.home()
    assert ".deep-semantic-search" in str(DEFAULT_METADATA_DIR)


def test_image_extensions():
    assert ".png" in IMAGE_EXTENSIONS
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".jpeg" in IMAGE_EXTENSIONS


def test_model_defaults_are_strings():
    assert isinstance(SIGLIP_MODEL_DEFAULT, str)
    assert isinstance(FLORENCE_MODEL_DEFAULT, str)
    assert isinstance(BGE_M3_MODEL_DEFAULT, str)
    assert isinstance(DEFAULT_RERANK_MODEL, str)
    assert "siglip" in SIGLIP_MODEL_DEFAULT
    assert "Florence" in FLORENCE_MODEL_DEFAULT
    assert "bge-m3" in BGE_M3_MODEL_DEFAULT


def test_siglip_constants():
    assert SIGLIP_EMBEDDING_DIM == 1152
    assert SIGLIP_IMAGE_SIZE == 384


def test_usearch_index_files():
    assert IMAGE_USEARCH_INDEX_FILE.endswith(".usearch")
    assert TEXT_USEARCH_INDEX_FILE.endswith(".usearch")


def test_deprecated_constant_warning():
    import deep_semantic_search.config as cfg

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.__getattr__("CLIP_MODEL_DEFAULT")
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "SIGLIP_MODEL_DEFAULT" in str(w[0].message)
        assert val == SIGLIP_MODEL_DEFAULT

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.__getattr__("BLIP_MODEL_DEFAULT")
        assert len(w) == 1
        assert val == FLORENCE_MODEL_DEFAULT

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = cfg.__getattr__("TEXT_MODEL_DEFAULT")
        assert len(w) == 1
        assert val == BGE_M3_MODEL_DEFAULT
