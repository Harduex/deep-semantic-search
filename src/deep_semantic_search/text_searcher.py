"""Text similarity search using BGE-M3 + USearch with optional hybrid and reranking."""

from __future__ import annotations

import logging

import numpy as np

from .config import DEFAULT_DUPLICATE_THRESHOLD, DEFAULT_RERANK_MODEL
from .exceptions import SearchError
from .text_embedder import TextEmbedder

logger = logging.getLogger("deep_semantic_search")


class TextSearch:
    """Search for similar texts using pre-computed BGE-M3 embeddings.

    Parameters
    ----------
    embedder : TextEmbedder
        A ``TextEmbedder`` with saved embeddings.
    """

    def __init__(self, embedder: TextEmbedder):
        self._embedder = embedder
        self._cached_index = None
        self._reranker = None
        try:
            self._dense, self._sparse, self._corpus_dict = embedder.load_embedding()
        except Exception as exc:
            raise SearchError(f"Failed to load embeddings: {exc}") from exc

    @property
    def _index(self):
        """Lazy-load and cache the USearch index."""
        if self._cached_index is None:
            from usearch.index import Index

            self._cached_index = Index.restore(str(self._embedder.index_path))
        return self._cached_index

    def _get_reranker(self):
        """Lazy-load cross-encoder reranker."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(DEFAULT_RERANK_MODEL)
            except ImportError:
                raise ImportError(
                    "'sentence-transformers' is required for reranking. "
                    "Install it with: pip install deep-semantic-search"
                ) from None
        return self._reranker

    def _sparse_score(self, query_sparse: dict, doc_idx: int) -> float:
        """Compute sparse token-overlap dot-product score."""
        if self._sparse is None:
            return 0.0
        doc_sparse = self._sparse[doc_idx]
        score = 0.0
        for token_id, weight in query_sparse.items():
            if token_id in doc_sparse:
                score += weight * doc_sparse[token_id]
        return score

    def find_similar(
        self,
        query_text: str,
        top_n: int = 10,
        rerank: bool = False,
        hybrid: bool = True,
        sparse_weight: float = 0.3,
    ) -> list[dict]:
        """Find texts most similar to a query.

        Parameters
        ----------
        query_text : str
            The search query.
        top_n : int
            Number of results to return.
        rerank : bool
            If True, rerank candidates using a cross-encoder model.
        hybrid : bool
            If True and sparse vectors are available, fuse dense + sparse scores.
        sparse_weight : float
            Weight for sparse scores in hybrid fusion (dense weight = 1 - sparse_weight).

        Returns
        -------
        list[dict]
            Each dict contains keys: ``index``, ``text``, ``path``, ``score``.
        """
        # Encode query
        if self._embedder.supports_sparse:
            output = self._embedder.model.encode(
                [query_text],
                return_dense=True,
                return_sparse=True,
            )
            query_dense = np.array(output["dense_vecs"][0], dtype=np.float32)
            query_sparse = output["lexical_weights"][0]
        else:
            query_dense = np.array(
                self._embedder.model.encode(
                    query_text,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
            query_sparse = {}

        # Dense search via USearch — fetch more candidates for reranking/hybrid
        fetch_n = min(max(top_n * 3, 50), len(self._dense))
        matches = self._index.search(query_dense, fetch_n)

        values = list(self._corpus_dict.values())
        keys = list(self._corpus_dict.keys())

        candidates = []
        for key, distance in zip(matches.keys, matches.distances):
            idx = int(key)
            dense_score = float(1.0 - distance)

            if hybrid and self._sparse is not None:
                sp_score = self._sparse_score(query_sparse, idx)
                score = (1 - sparse_weight) * dense_score + sparse_weight * sp_score
            else:
                score = dense_score

            candidates.append({
                "index": idx,
                "text": values[idx],
                "path": keys[idx],
                "score": score,
            })

        # Sort by fused score
        candidates.sort(key=lambda x: -x["score"])

        # Rerank top candidates if requested
        if rerank:
            reranker = self._get_reranker()
            top_candidates = candidates[:fetch_n]
            pairs = [(query_text, c["text"]) for c in top_candidates]
            rerank_scores = reranker.predict(pairs)
            for c, rs in zip(top_candidates, rerank_scores):
                c["score"] = float(rs)
            top_candidates.sort(key=lambda x: -x["score"])
            candidates = top_candidates

        return candidates[:top_n]

    def find_duplicates(
        self, threshold: float = DEFAULT_DUPLICATE_THRESHOLD
    ) -> list[tuple[str, str, float]]:
        """Find near-duplicate text pairs above the similarity threshold.

        Parameters
        ----------
        threshold : float
            Minimum cosine similarity to consider a pair as duplicates.

        Returns
        -------
        list[tuple[str, str, float]]
            Sorted list of (path1, path2, similarity) tuples.
        """
        keys = list(self._corpus_dict.keys())
        duplicates = []
        for i in range(len(self._dense)):
            matches = self._index.search(self._dense[i], min(len(self._dense), 50))
            for key, dist in zip(matches.keys, matches.distances):
                j = int(key)
                sim = 1.0 - float(dist)
                if j > i and sim >= threshold:
                    duplicates.append((keys[i], keys[j], sim))
        return sorted(duplicates, key=lambda x: -x[2])
