"""Text embedding using BGE-M3 (dense + sparse vectors) and USearch."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

from .config import (
    BGE_M3_MODEL_DEFAULT,
    CORPUS_EMBEDDINGS_DATA_FILE,
    CORPUS_LIST_DATA_FILE,
    DEFAULT_METADATA_DIR,
    TEXT_SPARSE_VECTORS_FILE,
    TEXT_USEARCH_INDEX_FILE,
)
from .exceptions import EmbeddingError

logger = logging.getLogger("deep_semantic_search")


class TextEmbedder:
    """Embed text corpora using BGE-M3 and persist dense + sparse vectors.

    Parameters
    ----------
    model_name : str
        HuggingFace BGE-M3 model identifier.
    metadata_dir : str | Path | None
        Directory for storing embeddings. Defaults to
        ``~/.deep-semantic-search/{model_name}_metadata``.
    """

    def __init__(
        self,
        model_name: str = BGE_M3_MODEL_DEFAULT,
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
        self._sparse_file = self._metadata_dir / TEXT_SPARSE_VECTORS_FILE
        self._index_file = self._metadata_dir / TEXT_USEARCH_INDEX_FILE

        # Legacy pickle paths for migration
        self._legacy_embeddings_file = self._metadata_dir / "corpus_embeddings_data.pickle"
        self._legacy_corpus_file = self._metadata_dir / "corpus_list_data.pickle"

        self._model = None
        self._supports_sparse = True
        self.corpus_embeddings = None

    @property
    def model(self):
        """Lazy-loaded embedding model.

        Prefers FlagEmbedding BGEM3 for dense+sparse output. Falls back to
        SentenceTransformer dense-only mode when FlagEmbedding isn't usable
        in the current environment.
        """
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel

                logger.info("Loading BGE-M3 model: %s", self.model_name)
                self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
                self._supports_sparse = True
                logger.info("Model loaded successfully: %s", self.model_name)
            except Exception as exc:
                logger.warning(
                    "FlagEmbedding load failed (%s). Falling back to SentenceTransformer dense-only mode.",
                    exc,
                )
                try:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name)
                    self._supports_sparse = False
                except Exception as fallback_exc:
                    raise EmbeddingError(
                        f"Failed to load embedding model '{self.model_name}': {fallback_exc}"
                    ) from fallback_exc
        return self._model

    @property
    def supports_sparse(self) -> bool:
        """Whether the active embedding backend supports sparse vectors."""
        # Touch model to ensure backend is initialized
        _ = self.model
        return self._supports_sparse

    @property
    def index_path(self) -> Path:
        """Path to the USearch index file."""
        return self._index_file

    @property
    def metadata_dir(self) -> Path:
        """Path to the metadata directory."""
        return self._metadata_dir

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

                if hasattr(embeddings, "cpu"):
                    embeddings_np = embeddings.cpu().numpy()
                else:
                    embeddings_np = np.array(embeddings)
                np.save(self._embeddings_file, embeddings_np)

                with open(self._corpus_file, "w", encoding="utf-8") as f:
                    json.dump({str(k): v for k, v in corpus_dict.items()}, f)
                logger.info("Migrated legacy pickle to numpy/json format.")
                return True
            except Exception as exc:
                logger.warning("Failed to migrate legacy pickle: %s", exc)
        return False

    def _build_usearch_index(self, features: np.ndarray) -> None:
        """Build and save a USearch index from dense feature vectors."""
        from usearch.index import Index

        d = features.shape[1]
        index = Index(ndim=d, metric="cos", dtype="f32")
        keys = np.arange(len(features), dtype=np.uint64)
        index.add(keys, features)
        index.save(str(self._index_file))
        logger.info("USearch text index saved: %s", self._index_file)

    def _save_sparse(self, sparse_vectors: list[dict] | None) -> None:
        """Serialize sparse vectors to JSON."""
        if sparse_vectors is None:
            if self._sparse_file.exists():
                self._sparse_file.unlink()
            return
        serializable = []
        for sv in sparse_vectors:
            serializable.append({str(k): float(v) for k, v in sv.items()})
        with open(self._sparse_file, "w", encoding="utf-8") as f:
            json.dump(serializable, f)

    def _load_sparse(self) -> list[dict] | None:
        """Load sparse vectors from JSON, or None if not present."""
        if not self._sparse_file.exists():
            return None
        with open(self._sparse_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [{int(k): float(v) for k, v in sv.items()} for sv in raw]

    def embed(self, corpus_dict: dict, reindex: bool = False) -> None:
        """Compute and save embeddings for a text corpus.

        Uses BGE-M3 to produce both dense and sparse vectors.

        Parameters
        ----------
        corpus_dict : dict
            Mapping of identifiers to text strings.
        reindex : bool
            If True, recompute even if embeddings exist.
        """
        has_data = self._embeddings_file.exists() and self._corpus_file.exists()
        if not has_data and not reindex:
            has_data = self._migrate_legacy_pickle()
        if not has_data or reindex:
            texts = list(corpus_dict.values())
            if self.supports_sparse:
                output = self.model.encode(
                    texts,
                    return_dense=True,
                    return_sparse=True,
                )
                dense = np.array(output["dense_vecs"], dtype=np.float32)
                sparse = output["lexical_weights"]
            else:
                dense = np.array(
                    self.model.encode(
                        texts,
                        convert_to_numpy=True,
                        show_progress_bar=True,
                        normalize_embeddings=True,
                    ),
                    dtype=np.float32,
                )
                sparse = None

            np.save(self._embeddings_file, dense)
            self._save_sparse(sparse)
            with open(self._corpus_file, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in corpus_dict.items()}, f)

            self._build_usearch_index(dense)
            self.corpus_embeddings = dense
            logger.info("Embeddings saved to %s", self._metadata_dir)
        else:
            # Build USearch index from existing dense vectors if missing
            if not self._index_file.exists() and self._embeddings_file.exists():
                features = np.load(self._embeddings_file).astype(np.float32)
                self._build_usearch_index(features)
            logger.info("Embeddings already present at %s, skipping.", self._metadata_dir)

    def load_embedding(self):
        """Load previously saved embeddings.

        Returns
        -------
        tuple
            (dense_embeddings_ndarray, sparse_vectors_or_None, corpus_dict)

        Raises
        ------
        EmbeddingError
            If no saved embeddings are found.
        """
        if not self._embeddings_file.exists():
            if not self._migrate_legacy_pickle():
                raise EmbeddingError("No embedding data found. Run embed() first.")

        embeddings = np.load(self._embeddings_file).astype(np.float32)
        sparse = self._load_sparse()
        with open(self._corpus_file, "r", encoding="utf-8") as f:
            corpus_dict = json.load(f)

        # Ensure USearch index exists
        if not self._index_file.exists():
            self._build_usearch_index(embeddings)

        logger.info("Embeddings loaded from %s", self._metadata_dir)
        return embeddings, sparse, corpus_dict
