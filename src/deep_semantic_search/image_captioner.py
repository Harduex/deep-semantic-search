"""Image captioning using the BLIP model."""

from __future__ import annotations

import logging
import math

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from .config import BLIP_MODEL_DEFAULT, get_device
from .exceptions import ModelLoadError

logger = logging.getLogger("deep_semantic_search")


class ImageCaptioner:
    """Generate captions for images using BLIP.

    The model is loaded once during initialization and reused for all caption calls.

    Parameters
    ----------
    model_name : str
        HuggingFace BLIP model identifier.
    """

    def __init__(self, model_name: str = BLIP_MODEL_DEFAULT):
        self.model_name = model_name
        self.device = get_device()

        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor

            logger.info("Loading BLIP model: %s", model_name)
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
            logger.info("BLIP model loaded successfully.")
        except Exception as exc:
            raise ModelLoadError(f"Failed to load BLIP model '{model_name}': {exc}") from exc

    def caption(self, image_paths: list[str], starting_text: str = "This is a") -> pd.DataFrame:
        """Generate captions for a list of images.

        Parameters
        ----------
        image_paths : list[str]
            Paths to images to caption.
        starting_text : str
            Prefix text for conditional generation.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``image_path`` and ``caption``.
        """
        df = pd.DataFrame(image_paths, columns=["image_path"])
        captions: list[str] = []

        for img_path in tqdm(image_paths, desc="Captioning images"):
            raw_image = Image.open(img_path)
            inputs = self.processor(raw_image, text=starting_text, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)

        df["caption"] = captions
        return df

    def plot_captioned_images(self, images_df: pd.DataFrame, caption_col: str | None = None) -> None:
        """Plot images with optional captions.

        Parameters
        ----------
        images_df : pd.DataFrame
            DataFrame with an ``image_path`` column.
        caption_col : str | None
            Column name containing captions to display as titles.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "'matplotlib' is required for plotting. "
                "Install it with: pip install deep-semantic-search[viz]"
            ) from None

        count = len(images_df)
        grid_size = math.ceil(math.sqrt(count))
        fig = plt.figure(figsize=(20, 15))
        for i in range(count):
            fig.add_subplot(grid_size, grid_size, i + 1)
            plt.axis("off")
            img = Image.open(images_df["image_path"].iloc[i])
            img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
            plt.imshow(img_resized)
            if caption_col and caption_col in images_df.columns:
                plt.title(images_df[caption_col].iloc[i])
        fig.tight_layout()
        plt.show()
