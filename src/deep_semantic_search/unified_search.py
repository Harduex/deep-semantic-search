"""Cross-modal unified search using SigLIP embeddings for images and text."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import (
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_METADATA_DIR,
    SIGLIP_IMAGE_SIZE,
    SIGLIP_MODEL_DEFAULT,
    UNIFIED_METADATA_FILE,
    UNIFIED_USEARCH_INDEX_FILE,
)
from .exceptions import IndexNotFoundError, ModelLoadError

logger = logging.getLogger("deep_semantic_search")


class UnifiedIndexer:
    """Index images and texts in a shared SigLIP embedding space.

    Parameters
    ----------
    model_name : str
        HuggingFace SigLIP model identifier.
    metadata_dir : str | Path | None
        Directory for storing index/metadata files.
    """

    def __init__(
        self,
        model_name: str = SIGLIP_MODEL_DEFAULT,
        metadata_dir: str | Path | None = None,
    ):
        import os

        self.model_name = model_name

        if metadata_dir is not None:
            self._metadata_dir = Path(metadata_dir)
        else:
            model_name_safe = model_name.replace("/", os.sep) if os.sep != "/" else model_name
            self._metadata_dir = DEFAULT_METADATA_DIR / "unified_metadata" / model_name_safe

        self._metadata_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._metadata_dir / UNIFIED_USEARCH_INDEX_FILE
        self._metadata_file = self._metadata_dir / UNIFIED_METADATA_FILE

        self._model = None
        self._processor = None
        self._model_loaded = False

        self._entries: list[dict] = []
        self._vectors: list[np.ndarray] = []

    @property
    def index_path(self) -> Path:
        return self._index_file

    @property
    def metadata_dir(self) -> Path:
        return self._metadata_dir

    def _load_model(self) -> None:
        if self._model_loaded:
            return
        try:
            from transformers import SiglipModel, SiglipProcessor

            self._processor = SiglipProcessor.from_pretrained(self.model_name)
            self._model = SiglipModel.from_pretrained(self.model_name)
            self._model_loaded = True
        except Exception as exc:
            raise ModelLoadError(f"Failed to load SigLIP model: {exc}") from exc

    @property
    def model(self):
        if not self._model_loaded:
            self._load_model()
        return self._model

    @property
    def processor(self):
        if not self._model_loaded:
            self._load_model()
        return self._processor

    def add_images(self, image_paths: list[str]) -> None:
        """Add images to the unified index.

        Parameters
        ----------
        image_paths : list[str]
            Paths to images to embed.
        """
        for path in image_paths:
            try:
                img = Image.open(path).resize(
                    (SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE)
                ).convert("RGB")
                inputs = self.processor(images=img, return_tensors="pt", padding=True)
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                vec = features.detach().cpu().numpy().flatten()
                vec = vec / np.linalg.norm(vec)
                self._entries.append({"type": "image", "source": path})
                self._vectors.append(vec)
            except Exception as exc:
                logger.warning("Failed to embed image %s: %s", path, exc)

    def add_texts(self, texts: list[str], labels: list[str] | None = None) -> None:
        """Add texts to the unified index.

        Parameters
        ----------
        texts : list[str]
            Text strings to embed.
        labels : list[str] | None
            Optional labels for each text. Defaults to the text itself.
        """
        if labels is None:
            labels = texts

        for text, label in zip(texts, labels):
            try:
                inputs = self.processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64,
                )
                with torch.no_grad():
                    features = self.model.get_text_features(**inputs)
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                vec = features.detach().cpu().numpy().flatten()
                vec = vec / np.linalg.norm(vec)
                self._entries.append({"type": "text", "source": label})
                self._vectors.append(vec)
            except Exception as exc:
                logger.warning("Failed to embed text '%s': %s", label[:50], exc)

    def build_index(self) -> None:
        """Build and save the USearch index and metadata."""
        if not self._vectors:
            raise IndexNotFoundError("No entries to index. Add images or texts first.")

        from usearch.index import Index

        features = np.vstack(self._vectors).astype(np.float32)
        ndim = features.shape[1]
        index = Index(ndim=ndim, metric="cos", dtype="f32")
        keys = np.arange(len(features), dtype=np.uint64)
        index.add(keys, features)
        index.save(str(self._index_file))

        with open(self._metadata_file, "w", encoding="utf-8") as f:
            json.dump(self._entries, f)

        logger.info("Unified index saved: %d entries", len(self._entries))

    def get_metadata(self) -> list[dict]:
        """Return the metadata entries."""
        if not self._entries and self._metadata_file.exists():
            with open(self._metadata_file, "r", encoding="utf-8") as f:
                self._entries = json.load(f)
        return self._entries


class UnifiedSearcher:
    """Search across images and texts in a shared SigLIP embedding space.

    Parameters
    ----------
    indexer : UnifiedIndexer
        A configured and indexed ``UnifiedIndexer`` instance.
    """

    def __init__(self, indexer: UnifiedIndexer):
        self._indexer = indexer
        self._cached_index = None

    @property
    def _index(self):
        if self._cached_index is None:
            from usearch.index import Index

            self._cached_index = Index.restore(str(self._indexer.index_path))
        return self._cached_index

    def _search_by_vector(
        self, vector: np.ndarray, n: int, modality_filter: str | None = None
    ) -> list[dict]:
        metadata = self._indexer.get_metadata()
        # Fetch all to enable per-modality score normalization
        fetch_n = len(metadata)
        matches = self._index.search(vector.astype(np.float32), fetch_n)

        # Collect raw results grouped by modality
        by_modality: dict[str, list[dict]] = {"image": [], "text": []}
        for key, distance in zip(matches.keys, matches.distances):
            idx = int(key)
            entry = metadata[idx]
            raw_score = float(1.0 - distance)
            by_modality.setdefault(entry["type"], []).append({
                "type": entry["type"],
                "source": entry["source"],
                "raw_score": raw_score,
            })

        # Normalize scores within each modality to [0, 1]
        all_results = []
        for mod, items in by_modality.items():
            if not items:
                continue
            scores = [r["raw_score"] for r in items]
            min_s, max_s = min(scores), max(scores)
            score_range = max_s - min_s
            for item in items:
                item["score"] = (
                    (item["raw_score"] - min_s) / score_range if score_range > 0 else 1.0
                )
                all_results.append(item)

        # Sort by normalized score, filter modality, and take top n
        all_results.sort(key=lambda x: -x["score"])
        results = []
        for r in all_results:
            if modality_filter and r["type"] != modality_filter:
                continue
            results.append({
                "rank": len(results) + 1,
                "type": r["type"],
                "source": r["source"],
                "score": r["score"],
            })
            if len(results) >= n:
                break

        return results

    def search(
        self, query: str, n: int = 10, modality_filter: str | None = None
    ) -> list[dict]:
        """Search by text query.

        Parameters
        ----------
        query : str
            Text query string.
        n : int
            Number of results.
        modality_filter : str | None
            Filter by ``"image"`` or ``"text"``. None returns both.

        Returns
        -------
        list[dict]
            Results with keys ``rank``, ``type``, ``source``, ``score``.
        """
        inputs = self._indexer.processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        with torch.no_grad():
            features = self._indexer.model.get_text_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        vec = features.detach().cpu().numpy().flatten()
        vec = vec / np.linalg.norm(vec)
        return self._search_by_vector(vec, n, modality_filter)

    def search_by_image(
        self, image_path: str, n: int = 10, modality_filter: str | None = None
    ) -> list[dict]:
        """Search by image query.

        Parameters
        ----------
        image_path : str
            Path to the query image.
        n : int
            Number of results.
        modality_filter : str | None
            Filter by ``"image"`` or ``"text"``. None returns both.

        Returns
        -------
        list[dict]
            Results with keys ``rank``, ``type``, ``source``, ``score``.
        """
        img = Image.open(image_path).resize(
            (SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE)
        ).convert("RGB")
        inputs = self._indexer.processor(images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            features = self._indexer.model.get_image_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        vec = features.detach().cpu().numpy().flatten()
        vec = vec / np.linalg.norm(vec)
        return self._search_by_vector(vec, n, modality_filter)

    def find_duplicates(
        self, threshold: float = DEFAULT_DUPLICATE_THRESHOLD
    ) -> list[tuple[str, str, float]]:
        """Find near-duplicate entries above the similarity threshold.

        Returns
        -------
        list[tuple[str, str, float]]
            Sorted list of (source1, source2, similarity) tuples.
        """
        metadata = self._indexer.get_metadata()
        if not metadata:
            return []

        vectors = np.vstack(self._indexer._vectors).astype(np.float32)
        duplicates = []
        for i in range(len(vectors)):
            matches = self._index.search(vectors[i], min(len(vectors), 50))
            for key, dist in zip(matches.keys, matches.distances):
                j = int(key)
                sim = 1.0 - float(dist)
                if j > i and sim >= threshold:
                    duplicates.append(
                        (metadata[i]["source"], metadata[j]["source"], sim)
                    )
        return sorted(duplicates, key=lambda x: -x[2])
