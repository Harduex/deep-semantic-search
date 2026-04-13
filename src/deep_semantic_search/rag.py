"""Retrieval-Augmented Generation (RAG) for question answering."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

import numpy as np

from .config import BGE_M3_MODEL_DEFAULT, DEFAULT_OLLAMA_MODEL, DEFAULT_RERANK_MODEL
from .exceptions import SearchError

logger = logging.getLogger("deep_semantic_search")

DEFAULT_SYSTEM_PROMPT = (
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "Use three sentences maximum and keep the answer as concise as possible."
)


class RAG:
    """Retrieval-Augmented Generation using BGE-M3 + USearch + LiteLLM.

    Parameters
    ----------
    model_name : str | None
        Ollama model name. Defaults to env ``OLLAMA_LLM_MODEL`` or ``gemma4:e4b``.
    rerank : bool
        If True, rerank retrieved chunks with a cross-encoder.
    """

    def __init__(self, model_name: str | None = None, rerank: bool = False):
        self.model_name = model_name or os.getenv("OLLAMA_LLM_MODEL") or DEFAULT_OLLAMA_MODEL
        self.rerank = rerank
        self._embed_model = None
        self._reranker = None

    def _get_embed_model(self):
        """Lazy-load BGE-M3 model."""
        if self._embed_model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel

                self._embed_model = BGEM3FlagModel(BGE_M3_MODEL_DEFAULT, use_fp16=False)
            except ImportError:
                raise ImportError(
                    "'FlagEmbedding' is required for RAG. "
                    "Install it with: pip install deep-semantic-search"
                ) from None
        return self._embed_model

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

    def _fixed_chunk(self, texts: list[str], chunk_size: int = 1500) -> list[str]:
        """Split texts into fixed-size character chunks."""
        chunks = []
        for text in texts:
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
        return chunks

    def _semantic_chunk(self, texts: list[str], max_chunk_size: int = 1500) -> list[str]:
        """Split texts into semantically coherent chunks.

        Embeds individual sentences with BGE-M3, then splits at
        low-similarity boundaries between consecutive sentences.
        """
        import re

        all_sentences = []
        for text in texts:
            sents = re.split(r"(?<=[.!?])\s+", text.strip())
            all_sentences.extend(s for s in sents if s.strip())

        if not all_sentences:
            return texts

        model = self._get_embed_model()
        out = model.encode(all_sentences, return_dense=True, return_sparse=False)
        embeddings = np.array(out["dense_vecs"], dtype=np.float32)

        # Compute cosine similarity between consecutive sentences
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = embeddings / norms

        similarities = []
        for i in range(len(normed) - 1):
            sim = float(np.dot(normed[i], normed[i + 1]))
            similarities.append(sim)

        # Find split points: below median similarity
        if similarities:
            threshold = float(np.median(similarities))
        else:
            threshold = 0.5

        chunks = []
        current_chunk = [all_sentences[0]]
        current_len = len(all_sentences[0])

        for i, sent in enumerate(all_sentences[1:]):
            if (
                current_len + len(sent) > max_chunk_size
                or (i < len(similarities) and similarities[i] < threshold)
            ):
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_len = len(sent)
            else:
                current_chunk.append(sent)
                current_len += len(sent)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def ask(
        self,
        text_data: list[str],
        question: str,
        chunk_size: int = 1500,
        semantic_chunking: bool = True,
        top_k: int = 5,
        llm_fn: Callable[[str], str] | None = None,
    ) -> str:
        """Answer a question using RAG over the provided text data.

        Parameters
        ----------
        text_data : list[str]
            List of text documents to search over.
        question : str
            The question to answer.
        chunk_size : int
            Maximum chunk size in characters.
        semantic_chunking : bool
            If True, use embedding-based semantic chunking.
        top_k : int
            Number of top chunks to retrieve.
        llm_fn : Callable[[str], str] | None
            Custom LLM callable. If provided, used instead of LiteLLM.

        Returns
        -------
        str
            The generated answer.
        """
        if not text_data:
            raise SearchError("text_data must be a non-empty list of strings.")

        # 1. Chunk
        if semantic_chunking:
            chunks = self._semantic_chunk(text_data, max_chunk_size=chunk_size)
        else:
            chunks = self._fixed_chunk(text_data, chunk_size=chunk_size)

        if not chunks:
            raise SearchError("No text chunks produced from text_data.")

        # 2. Embed chunks
        model = self._get_embed_model()
        out = model.encode(chunks, return_dense=True, return_sparse=False)
        chunk_embeddings = np.array(out["dense_vecs"], dtype=np.float32)

        # 3. Build ephemeral USearch index
        from usearch.index import Index

        ndim = chunk_embeddings.shape[1]
        index = Index(ndim=ndim, metric="cos", dtype="f32")
        keys = np.arange(len(chunks), dtype=np.uint64)
        index.add(keys, chunk_embeddings)

        # 4. Embed query and search
        q_out = model.encode([question], return_dense=True, return_sparse=False)
        query_vec = np.array(q_out["dense_vecs"][0], dtype=np.float32)
        matches = index.search(query_vec, min(top_k, len(chunks)))

        retrieved = [chunks[int(k)] for k in matches.keys]

        # 5. Optionally rerank
        if self.rerank and retrieved:
            reranker = self._get_reranker()
            pairs = [(question, c) for c in retrieved]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(retrieved, scores), key=lambda x: -x[1])
            retrieved = [c for c, _ in ranked]

        # 6. Generate answer
        context = "\n\n".join(retrieved)

        if llm_fn is not None:
            prompt = (
                f"{DEFAULT_SYSTEM_PROMPT}\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Helpful Answer:"
            )
            return llm_fn(prompt)

        try:
            from litellm import completion

            response = completion(
                model=f"ollama/{self.model_name}",
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError(
                "'litellm' is required for RAG. "
                "Install it with: pip install deep-semantic-search[llm]"
            ) from None


def ask_question(
    text_data: list[str],
    question: str,
    chunk_size: int = 1500,
    llm_fn: Callable[[str], str] | None = None,
    model_name: str | None = None,
    **kwargs,
) -> str:
    """Backward-compatible wrapper around :class:`RAG`.

    Parameters
    ----------
    text_data : list[str]
        List of text documents.
    question : str
        The question to answer.
    chunk_size : int
        Maximum chunk size.
    llm_fn : Callable[[str], str] | None
        Custom LLM callable.
    model_name : str | None
        Ollama model name.

    Returns
    -------
    str
        The generated answer.
    """
    rag = RAG(model_name=model_name)
    return rag.ask(
        text_data=text_data,
        question=question,
        chunk_size=chunk_size,
        llm_fn=llm_fn,
        **kwargs,
    )
