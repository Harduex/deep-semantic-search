"""Tests for ImageCaptioner with mocked BLIP model."""

from unittest.mock import MagicMock

import pytest

from deep_semantic_search.exceptions import ModelLoadError


def test_captioner_model_load_failure(mock_transformers):
    mock_transformers.BlipProcessor.from_pretrained.side_effect = RuntimeError("model not found")
    with pytest.raises(ModelLoadError, match="model not found"):
        from deep_semantic_search.image_captioner import ImageCaptioner
        ImageCaptioner()


def test_caption_returns_dataframe(sample_images, mock_transformers):
    mock_processor = MagicMock()
    mock_processor_result = MagicMock()
    mock_processor_result.to.return_value = mock_processor_result
    mock_processor.return_value = mock_processor_result
    mock_processor.decode.return_value = "a test caption"
    mock_transformers.BlipProcessor.from_pretrained.return_value = mock_processor

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.generate.return_value = [MagicMock()]
    mock_transformers.BlipForConditionalGeneration.from_pretrained.return_value = mock_model

    from deep_semantic_search.image_captioner import ImageCaptioner
    captioner = ImageCaptioner()

    result = captioner.caption(sample_images[:2])

    assert "image_path" in result.columns
    assert "caption" in result.columns
    assert len(result) == 2
