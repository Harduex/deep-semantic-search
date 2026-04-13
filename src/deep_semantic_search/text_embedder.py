"""Text embedding using Sentence Transformers."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch

from .config import (
    CORPUS_EMBEDDINGS_DATA_FILE,
    CORPUS_LIST_DATA_FILE,
    DEFAULT_METADATA_DIR,
    TEXT_MODEL_DEFAULT,
)
from .exceptions import EmbeddingError

logger = logging.getLogger("deep_semantic_search")


class TextEmbedder:
    """Embed text corpora using Sentence Transformers and persist to disk.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for sentence embeddings.
    metadata_dir : str | Path | None
        Directory for storing embeddings. Defaults to
        ``~/.deep-semantic-search/nli-mpnet-base-v2_metadata``.
    """

    def __init__(
        self,
        model_name: str = TEXT_MODEL_DEFAULT,
        metadata_dir: str | Path | None = None,
    ):
        self.model_name = model_name

        if metadata_dir is not None:
            self._metadata_dir = Path(metadata_dir)
        else:
            model_dir_name = model_name.replace("/", "_").replace("\\", "_")
            self._metadata_dir = DEFAULT_METADATA_DIR / f"{model_dir_name}_metadata"

        self._metadata_dir.mkdir(parents=True, exist_ok=True)

        self._embeddings_file = self._metadata_dir / CORPUS_EMBEDDINGS_DATA_FILE
        self._corpus_file = self._metadata_dir / CORPUS_LIST_DATA_FILE

        # Legacy pickle paths for migration
        self._legacy_embeddings_file = self._metadata_dir / "corpus_embeddings_data.pickle"
        self._legacy_corpus_file = self._metadata_dir / "corpus_list_data.pickle"

        self._embedder = None
        self.corpus_embeddings = None

    @property
    def embedder(self):
        """Lazy-loaded SentenceTransformer model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def _migrate_legacy_pickle(self) -> bool:
        """Migrate legacy .pickle format to numpy/json if needed."""
        if (
            self._legacy_embeddings_file.exists()
            and self._legacy_corpus_file.exists()
            and not self._embeddings_file.exists()
        ):
            try:
                with open(self._legacy_embeddings_file, "rb") as f:
                    embeddings = pickle.load(f)
                with open(self._legacy_corpus_file, "rb") as f:
                    corpus_dict = pickle.load(f)

                # Convert tensor to numpy if needed
                if hasattr(embeddings, "cpu"):
                    embeddings_np = embeddings.cpu().numpy()
                else:
                    embeddings_np = np.array(embeddings)
                np.save(self._embeddings_file, embeddings_np)

                with open(self._corpus_file, "w", encoding="utf-8") as f:
                    # Convert keys to strings for JSON
                    json.dump({str(k): v for k, v in corpus_dict.items()}, f)
                logger.info("Migrated legacy pickle to numpy/json format.")
                return True
            except Exception as exc:
                logger.warning("Failed to migrate legacy pickle: %s", exc)
        return False

    def embed(self, corpus_dict: dict, reindex: bool = False) -> None:
        """Compute and save embeddings for a text corpus.

        Parameters
        ----------
        corpus_dict : dict
            Mapping of identifiers to text strings.
        reindex : bool
            If True, recompute even if embeddings exist.
        """
        has_data = self._embeddings_file.exists() and self._corpus_file.exists()
        if not has_data and not reindex:
            # Check for legacy pickle files to migrate
            has_data = self._migrate_legacy_pickle()
        if not has_data or reindex:
            self.corpus_embeddings = self.embedder.encode(
                list(corpus_dict.values()),
                convert_to_tensor=True,
                show_progress_bar=True,
            )
            # Convert tensor to numpy for saving
            if hasattr(self.corpus_embeddings, "cpu"):
                embeddings_np = self.corpus_embeddings.cpu().numpy()
            else:
                embeddings_np = np.array(self.corpus_embeddings)
            np.save(self._embeddings_file, embeddings_np)
            with open(self._corpus_file, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in corpus_dict.items()}, f)
            logger.info("Embeddings saved to %s", self._metadata_dir)
        else:
            logger.info("Embeddings already present at %s, skipping.", self._metadata_dir)

    def load_embedding(self):
        """Load previously saved embeddings.

        Returns
        -------
        tuple
            (corpus_embeddings, corpus_dict)

        Raises
        ------
        EmbeddingError
            If no saved embeddings are found.
        """
        if not self._embeddings_file.exists():
            # Try migration from legacy format
            if not self._migrate_legacy_pickle():
                raise EmbeddingError("No embedding data found. Run embed() first.")

        embeddings_np = np.load(self._embeddings_file)
        embeddings = torch.from_numpy(embeddings_np)
        with open(self._corpus_file, "r", encoding="utf-8") as f:
            corpus_dict = json.load(f)

        logger.info("Embeddings loaded from %s", self._metadata_dir)
        return embeddings, corpus_dict
