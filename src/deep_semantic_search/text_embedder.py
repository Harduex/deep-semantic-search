"""Text embedding using Sentence Transformers."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

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

        from sentence_transformers import SentenceTransformer

        self.embedder = SentenceTransformer(model_name)
        self.corpus_embeddings = None

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
        if not has_data or reindex:
            self.corpus_embeddings = self.embedder.encode(
                list(corpus_dict.values()),
                convert_to_tensor=True,
                show_progress_bar=True,
            )
            with open(self._embeddings_file, "wb") as f:
                pickle.dump(self.corpus_embeddings, f)
            with open(self._corpus_file, "wb") as f:
                pickle.dump(corpus_dict, f)
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
            raise EmbeddingError("No embedding data found. Run embed() first.")

        with open(self._embeddings_file, "rb") as f:
            embeddings = pickle.load(f)
        with open(self._corpus_file, "rb") as f:
            corpus_dict = pickle.load(f)

        logger.info("Embeddings loaded from %s", self._metadata_dir)
        return embeddings, corpus_dict
