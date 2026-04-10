"""Tests for config module."""

from pathlib import Path

from deep_semantic_search.config import (
    BLIP_MODEL_DEFAULT,
    CLIP_MODEL_DEFAULT,
    DEFAULT_METADATA_DIR,
    IMAGE_EXTENSIONS,
    TEXT_MODEL_DEFAULT,
)


def test_default_metadata_dir_is_in_home():
    assert DEFAULT_METADATA_DIR.parent == Path.home()
    assert ".deep-semantic-search" in str(DEFAULT_METADATA_DIR)


def test_image_extensions():
    assert ".png" in IMAGE_EXTENSIONS
    assert ".jpg" in IMAGE_EXTENSIONS
    assert ".jpeg" in IMAGE_EXTENSIONS


def test_model_defaults_are_strings():
    assert isinstance(CLIP_MODEL_DEFAULT, str)
    assert isinstance(BLIP_MODEL_DEFAULT, str)
    assert isinstance(TEXT_MODEL_DEFAULT, str)
    assert "/" in CLIP_MODEL_DEFAULT
