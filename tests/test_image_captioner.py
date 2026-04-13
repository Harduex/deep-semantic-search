"""Tests for ImageCaptioner with mocked Florence-2 model."""

from unittest.mock import MagicMock

import pytest

from deep_semantic_search.exceptions import ModelLoadError


def test_captioner_lazy_loading(mock_transformers):
    """Model should not be loaded at construction time."""
    from deep_semantic_search.image_captioner import ImageCaptioner

    captioner = ImageCaptioner()
    assert not captioner._model_loaded
    mock_transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()


def test_captioner_model_load_failure(mock_transformers):
    mock_transformers.AutoProcessor.from_pretrained.side_effect = RuntimeError("model not found")

    from deep_semantic_search.image_captioner import ImageCaptioner

    captioner = ImageCaptioner()
    with pytest.raises(ModelLoadError, match="model not found"):
        _ = captioner.model


def test_caption_returns_dataframe(sample_images, mock_transformers):
    mock_processor = MagicMock()
    mock_processor_result = MagicMock()
    mock_processor_result.to.return_value = {"input_ids": MagicMock(), "pixel_values": MagicMock()}
    mock_processor.return_value = mock_processor_result
    mock_processor.batch_decode.return_value = ["a test caption"]
    mock_processor.post_process_generation.return_value = {"<DETAILED_CAPTION>": "a test caption"}
    mock_transformers.AutoProcessor.from_pretrained.return_value = mock_processor

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.generate.return_value = MagicMock()
    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model

    from deep_semantic_search.image_captioner import ImageCaptioner

    captioner = ImageCaptioner()
    result = captioner.caption(sample_images[:2])

    assert "image_path" in result.columns
    assert "caption" in result.columns
    assert len(result) == 2


def test_captioner_custom_task(mock_transformers):
    from deep_semantic_search.image_captioner import ImageCaptioner

    captioner = ImageCaptioner(task="<CAPTION>")
    assert captioner.task == "<CAPTION>"
