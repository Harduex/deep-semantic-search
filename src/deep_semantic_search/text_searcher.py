"""Text similarity search using Sentence Transformer embeddings."""

from __future__ import annotations

import logging

import numpy as np

from .exceptions import SearchError
from .text_embedder import TextEmbedder

logger = logging.getLogger("deep_semantic_search")


class TextSearch:
    """Search for similar texts using pre-computed sentence embeddings.

    Parameters
    ----------
    embedder : TextEmbedder
        A ``TextEmbedder`` with saved embeddings.
    """

    def __init__(self, embedder: TextEmbedder):
        self._embedder = embedder
        try:
            self._corpus_embeddings, self._corpus_dict = embedder.load_embedding()
        except Exception as exc:
            raise SearchError(f"Failed to load embeddings: {exc}") from exc

    def find_similar(self, query_text: str, top_n: int = 10) -> list[dict]:
        """Find texts most similar to a query.

        Parameters
        ----------
        query_text : str
            The search query.
        top_n : int
            Number of results to return.

        Returns
        -------
        list[dict]
            Each dict contains keys: ``index``, ``text``, ``path``, ``score``.
        """
        query_embedding = self._embedder.embedder.encode(query_text, convert_to_tensor=True)
        from sentence_transformers import util

        cos_scores = util.pytorch_cos_sim(query_embedding, self._corpus_embeddings)[0].cpu().data.numpy()
        sorted_indices = np.argsort(-cos_scores)

        results: list[dict] = []
        values = list(self._corpus_dict.values())
        keys = list(self._corpus_dict.keys())

        # Skip index 0 if it's the query itself in the corpus
        for idx in sorted_indices[:top_n + 1]:
            idx = int(idx)
            if len(results) >= top_n:
                break
            results.append({
                "index": idx,
                "text": values[idx],
                "path": keys[idx],
                "score": float(cos_scores[idx]),
            })

        return results
