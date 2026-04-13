"""Image captioning using Florence-2."""

from __future__ import annotations

import logging
import math

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from .config import FLORENCE_MODEL_DEFAULT, SIGLIP_IMAGE_SIZE, get_device
from .exceptions import ModelLoadError

logger = logging.getLogger("deep_semantic_search")


class ImageCaptioner:
    """Generate captions for images using Florence-2.

    The model is lazily loaded on first use.

    Parameters
    ----------
    model_name : str
        HuggingFace Florence-2 model identifier.
    task : str
        Florence-2 task prompt (e.g. ``"<CAPTION>"``, ``"<DETAILED_CAPTION>"``).
    """

    def __init__(
        self,
        model_name: str = FLORENCE_MODEL_DEFAULT,
        task: str = "<DETAILED_CAPTION>",
    ):
        self.model_name = model_name
        self.task = task
        self.device = get_device()
        self._model = None
        self._processor = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """Load Florence-2 model and processor."""
        if self._model_loaded:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            logger.info("Loading Florence-2 model: %s", self.model_name)
            self._processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to(self.device)
            logger.info("Florence-2 model loaded successfully.")
            self._model_loaded = True
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load Florence-2 model '{self.model_name}': {exc}"
            ) from exc

    @property
    def model(self):
        """Lazy-loaded Florence-2 model."""
        if not self._model_loaded:
            self._load_model()
        return self._model

    @property
    def processor(self):
        """Lazy-loaded Florence-2 processor."""
        if not self._model_loaded:
            self._load_model()
        return self._processor

    def caption(self, image_paths: list[str]) -> pd.DataFrame:
        """Generate captions for a list of images.

        Parameters
        ----------
        image_paths : list[str]
            Paths to images to caption.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``image_path`` and ``caption``.
        """
        df = pd.DataFrame(image_paths, columns=["image_path"])
        captions: list[str] = []

        for img_path in tqdm(image_paths, desc="Captioning images"):
            raw_image = Image.open(img_path).convert("RGB")
            inputs = self.processor(
                text=self.task, images=raw_image, return_tensors="pt"
            ).to(self.device)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=False
            )[0]
            parsed = self.processor.post_process_generation(
                generated_text, task=self.task, image_size=raw_image.size
            )
            caption = parsed.get(self.task, generated_text)
            captions.append(caption)

        df["caption"] = captions
        return df

    def plot_captioned_images(
        self, images_df: pd.DataFrame, caption_col: str | None = None
    ) -> None:
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

        img_size = (SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE)
        count = len(images_df)
        grid_size = math.ceil(math.sqrt(count))
        fig = plt.figure(figsize=(20, 15))
        for i in range(count):
            fig.add_subplot(grid_size, grid_size, i + 1)
            plt.axis("off")
            img = Image.open(images_df["image_path"].iloc[i])
            img_resized = ImageOps.fit(img, img_size, Image.LANCZOS)
            plt.imshow(img_resized)
            if caption_col and caption_col in images_df.columns:
                plt.title(images_df[caption_col].iloc[i])
        fig.tight_layout()
        plt.show()
