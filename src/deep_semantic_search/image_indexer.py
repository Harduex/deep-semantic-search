"""Image indexing with SigLIP embeddings and USearch."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .config import (
    DEFAULT_METADATA_DIR,
    IMAGE_DATA_FEATURES_FILE,
    IMAGE_DATA_PATHS_FILE,
    IMAGE_USEARCH_INDEX_FILE,
    SIGLIP_IMAGE_SIZE,
    SIGLIP_MODEL_DEFAULT,
    get_device,
)
from .exceptions import IndexNotFoundError, ModelLoadError

logger = logging.getLogger("deep_semantic_search")


class ImageIndexer:
    """Indexes images using SigLIP embeddings and USearch for similarity search.

    Parameters
    ----------
    image_list : list[str]
        Paths to images to index.
    model_name : str
        HuggingFace SigLIP model identifier.
    metadata_dir : str | Path | None
        Directory for storing index/metadata files.
        Defaults to ``~/.deep-semantic-search/siglip_metadata/{model_name_safe}``.
    image_count : int | None
        Limit the number of images to index. None means all.
    """

    def __init__(
        self,
        image_list: list[str],
        model_name: str = SIGLIP_MODEL_DEFAULT,
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
            self._metadata_dir = DEFAULT_METADATA_DIR / "siglip_metadata" / model_name_safe

        self._metadata_dir.mkdir(parents=True, exist_ok=True)

        self._features_file = self._metadata_dir / IMAGE_DATA_FEATURES_FILE
        self._paths_file = self._metadata_dir / IMAGE_DATA_PATHS_FILE
        self._index_file = self._metadata_dir / IMAGE_USEARCH_INDEX_FILE

        # Legacy paths for migration
        self._legacy_features_pkl = self._metadata_dir / "image_data_features.pkl"
        self._legacy_faiss_idx = self._metadata_dir / "image_features_vectors.idx"

        # Lazy model loading
        self._model = None
        self._processor = None
        self._model_loaded = False

        self.image_data: pd.DataFrame = pd.DataFrame()

    @property
    def index_path(self) -> Path:
        """Path to the USearch index file."""
        return self._index_file

    @property
    def metadata_dir(self) -> Path:
        """Path to the metadata directory."""
        return self._metadata_dir

    def _load_model(self) -> None:
        """Load SigLIP model and processor."""
        if self._model_loaded:
            return
        try:
            from transformers import SiglipModel, SiglipProcessor

            logger.info("Loading SigLIP model: %s", self.model_name)
            self._processor = SiglipProcessor.from_pretrained(self.model_name)
            self._model = SiglipModel.from_pretrained(self.model_name).to(self.device)
            logger.info("Model loaded successfully: %s", self.model_name)
            self._model_loaded = True
        except Exception as exc:
            raise ModelLoadError(f"Failed to load SigLIP model '{self.model_name}': {exc}") from exc

    @property
    def model(self):
        """Lazy-loaded SigLIP model."""
        if not self._model_loaded:
            self._load_model()
        return self._model

    @property
    def processor(self):
        """Lazy-loaded SigLIP processor."""
        if not self._model_loaded:
            self._load_model()
        return self._processor

    def _extract(self, img: Image.Image) -> np.ndarray:
        """Extract normalized SigLIP feature vector from a single image."""
        img = img.resize((SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE)).convert("RGB")
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

    def _migrate_faiss_index(self) -> bool:
        """Migrate legacy FAISS .idx index to USearch format."""
        if self._legacy_faiss_idx.exists() and self._features_file.exists() and not self._index_file.exists():
            try:
                features = np.load(self._features_file).astype(np.float32)
                self._build_usearch_index(features)
                logger.info("Migrated FAISS index to USearch format.")
                return True
            except Exception as exc:
                logger.warning("Failed to migrate FAISS index: %s", exc)
        return False

    def _build_usearch_index(self, features: np.ndarray) -> None:
        """Build and save a USearch index from feature vectors."""
        from usearch.index import Index

        d = features.shape[1]
        index = Index(ndim=d, metric="cos", dtype="f32")
        keys = np.arange(len(features), dtype=np.uint64)
        index.add(keys, features)
        index.save(str(self._index_file))
        logger.info("USearch index saved: %s", self._index_file)

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
        """Build USearch index from feature vectors."""
        features_matrix = np.vstack(image_data["features"].values).astype(np.float32)
        self._build_usearch_index(features_matrix)

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
            # Try migrations from legacy formats
            self._migrate_legacy_pickle()
            self._migrate_faiss_index()
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
            if not self._migrate_legacy_pickle():
                raise IndexNotFoundError("No existing index found. Run run_index() first.")
            self._migrate_faiss_index()

        self.image_data = self._load_metadata()

        from usearch.index import Index

        index = Index.restore(str(self._index_file))
        next_key = int(index.size)

        for path in tqdm(new_image_paths, desc="Adding images"):
            try:
                feature = self._extract(Image.open(path))
            except Exception as exc:
                logger.warning("Failed to extract features from %s: %s", path, exc)
                continue

            new_row = pd.DataFrame({"images_paths": [path], "features": [feature]})
            self.image_data = pd.concat([self.image_data, new_row], axis=0, ignore_index=True)
            index.add(np.uint64(next_key), feature.astype(np.float32))
            next_key += 1

        # Save updated metadata
        paths = self.image_data["images_paths"].to_list()
        features = np.vstack(self.image_data["features"].values).astype(np.float32)
        np.save(self._features_file, features)
        with open(self._paths_file, "w", encoding="utf-8") as f:
            json.dump(paths, f)
        index.save(str(self._index_file))
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
