"""Image indexing with CLIP embeddings and FAISS."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .config import (
    CLIP_MODEL_DEFAULT,
    DEFAULT_METADATA_DIR,
    IMAGE_DATA_FEATURES_FILE,
    IMAGE_DATA_PATHS_FILE,
    IMAGE_FEATURES_VECTORS_FILE,
    get_device,
)
from .exceptions import IndexNotFoundError, ModelLoadError

logger = logging.getLogger("deep_semantic_search")


class ImageIndexer:
    """Indexes images using CLIP embeddings and FAISS for similarity search.

    Parameters
    ----------
    image_list : list[str]
        Paths to images to index.
    model_name : str
        HuggingFace CLIP model identifier.
    metadata_dir : str | Path | None
        Directory for storing index/metadata files.
        Defaults to ``~/.deep-semantic-search/clip_metadata/{model_name_safe}``.
    image_count : int | None
        Limit the number of images to index. None means all.
    """

    def __init__(
        self,
        image_list: list[str],
        model_name: str = CLIP_MODEL_DEFAULT,
        metadata_dir: str | Path | None = None,
        image_count: int | None = None,
    ):
        self.device = get_device()
        self.model_name = model_name

        if image_count is not None:
            self.image_list = image_list[:image_count]
        else:
            self.image_list = list(image_list)

        # Resolve metadata directory
        model_name_safe = model_name.replace("/", os.sep) if os.sep != "/" else model_name
        if metadata_dir is not None:
            self._metadata_dir = Path(metadata_dir)
        else:
            self._metadata_dir = DEFAULT_METADATA_DIR / "clip_metadata" / model_name_safe

        self._metadata_dir.mkdir(parents=True, exist_ok=True)

        self._features_file = self._metadata_dir / IMAGE_DATA_FEATURES_FILE
        self._paths_file = self._metadata_dir / IMAGE_DATA_PATHS_FILE
        self._index_file = self._metadata_dir / IMAGE_FEATURES_VECTORS_FILE

        # Legacy pickle paths for migration
        self._legacy_features_pkl = self._metadata_dir / "image_data_features.pkl"

        # Lazy model loading
        self._model = None
        self._processor = None
        self._model_loaded = False

        self.image_data: pd.DataFrame = pd.DataFrame()

    def _load_model(self) -> None:
        """Load CLIP model and processor."""
        if self._model_loaded:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info("Loading CLIP model: %s", self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            logger.info("Model loaded successfully: %s", self.model_name)
            self._model_loaded = True
        except Exception as exc:
            raise ModelLoadError(f"Failed to load CLIP model '{self.model_name}': {exc}") from exc

    @property
    def model(self):
        """Lazy-loaded CLIP model."""
        if not self._model_loaded:
            self._load_model()
        return self._model

    @property
    def processor(self):
        """Lazy-loaded CLIP processor."""
        if not self._model_loaded:
            self._load_model()
        return self._processor

    def _extract(self, img: Image.Image) -> np.ndarray:
        """Extract normalized CLIP feature vector from a single image."""
        img = img.resize((224, 224)).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            feature = self.model.get_image_features(**inputs)
        if hasattr(feature, "pooler_output"):
            feature = feature.pooler_output
        feature = feature.detach().cpu().numpy().flatten()
        return feature / np.linalg.norm(feature)

    def _get_features(self, image_paths: list[str]) -> list[np.ndarray | None]:
        """Extract features from multiple images."""
        features: list[np.ndarray | None] = []
        for img_path in tqdm(image_paths, desc="Extracting features"):
            try:
                feature = self._extract(Image.open(img_path))
                features.append(feature)
            except Exception as exc:
                logger.warning("Failed to extract features from %s: %s", img_path, exc)
                features.append(None)
        return features

    def _migrate_legacy_pickle(self) -> bool:
        """Migrate legacy .pkl format to numpy/json if needed."""
        if self._legacy_features_pkl.exists() and not self._features_file.exists():
            try:
                legacy_data = pd.read_pickle(self._legacy_features_pkl)
                paths = legacy_data["images_paths"].to_list()
                features = np.vstack(legacy_data["features"].values).astype(np.float32)
                np.save(self._features_file, features)
                with open(self._paths_file, "w", encoding="utf-8") as f:
                    json.dump(paths, f)
                logger.info("Migrated legacy pickle to numpy/json format.")
                return True
            except Exception as exc:
                logger.warning("Failed to migrate legacy pickle: %s", exc)
        return False

    def _build_features(self) -> pd.DataFrame:
        """Extract features and save metadata."""
        image_data = pd.DataFrame()
        image_data["images_paths"] = self.image_list
        image_data["features"] = self._get_features(self.image_list)
        image_data = image_data.dropna().reset_index(drop=True)

        # Save as numpy/json
        paths = image_data["images_paths"].to_list()
        features = np.vstack(image_data["features"].values).astype(np.float32)
        np.save(self._features_file, features)
        with open(self._paths_file, "w", encoding="utf-8") as f:
            json.dump(paths, f)
        logger.info("Image metadata saved: %s", self._features_file)
        return image_data

    def _load_metadata(self) -> pd.DataFrame:
        """Load metadata from numpy/json files."""
        features = np.load(self._features_file)
        with open(self._paths_file, "r", encoding="utf-8") as f:
            paths = json.load(f)
        image_data = pd.DataFrame()
        image_data["images_paths"] = paths
        image_data["features"] = list(features)
        return image_data

    def _build_index(self, image_data: pd.DataFrame) -> None:
        """Build FAISS index from feature vectors."""
        d = len(image_data["features"].iloc[0])
        index = faiss.IndexFlatL2(d)
        features_matrix = np.vstack(image_data["features"].values).astype(np.float32)
        index.add(features_matrix)
        faiss.write_index(index, str(self._index_file))
        logger.info("FAISS index saved: %s", self._index_file)

    def run_index(self, reindex: bool = False) -> None:
        """Build or load the image index.

        Parameters
        ----------
        reindex : bool
            If True, rebuild even if index already exists.
        """
        existing_files = list(self._metadata_dir.iterdir())
        if not existing_files or reindex:
            data = self._build_features()
            self._build_index(data)
        else:
            # Try migration from legacy format
            self._migrate_legacy_pickle()
            logger.info("Metadata already present at %s, skipping indexing.", self._metadata_dir)

        self.image_data = self._load_metadata()

    def add_images(self, new_image_paths: list[str]) -> None:
        """Add new images to an existing index.

        Parameters
        ----------
        new_image_paths : list[str]
            Paths to new images to add.
        """
        if not self._features_file.exists() or not self._index_file.exists():
            # Try migration first
            if not self._migrate_legacy_pickle():
                raise IndexNotFoundError("No existing index found. Run run_index() first.")

        self.image_data = self._load_metadata()
        index = faiss.read_index(str(self._index_file))

        for path in tqdm(new_image_paths, desc="Adding images"):
            try:
                feature = self._extract(Image.open(path))
            except Exception as exc:
                logger.warning("Failed to extract features from %s: %s", path, exc)
                continue

            new_row = pd.DataFrame({"images_paths": [path], "features": [feature]})
            self.image_data = pd.concat([self.image_data, new_row], axis=0, ignore_index=True)
            index.add(np.array([feature], dtype=np.float32))

        # Save updated metadata
        paths = self.image_data["images_paths"].to_list()
        features = np.vstack(self.image_data["features"].values).astype(np.float32)
        np.save(self._features_file, features)
        with open(self._paths_file, "w", encoding="utf-8") as f:
            json.dump(paths, f)
        faiss.write_index(index, str(self._index_file))
        logger.info("Added %d images to index.", len(new_image_paths))

    def get_metadata(self) -> pd.DataFrame:
        """Return the image metadata DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``images_paths`` and ``features``.
        """
        if self.image_data.empty and self._features_file.exists():
            self.image_data = self._load_metadata()
        return self.image_data
