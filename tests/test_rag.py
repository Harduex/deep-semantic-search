"""Tests for RAG with mocked BGE-M3 and LiteLLM."""

import pytest

from deep_semantic_search.exceptions import SearchError
from deep_semantic_search.rag import RAG, ask_question


def test_ask_question_empty_data():
    with pytest.raises(SearchError, match="non-empty"):
        ask_question([], "What is this?")


def test_rag_ask_with_llm_fn(mock_bge_m3_model):
    """Test RAG.ask with a custom llm_fn (no LiteLLM needed)."""
    def custom_fn(prompt):
        return "custom response"

    rag = RAG()
    result = rag.ask(
        text_data=["Machine learning is a subset of AI. Deep learning uses neural networks."],
        question="What is ML?",
        llm_fn=custom_fn,
        semantic_chunking=False,
    )
    assert result == "custom response"


def test_rag_semantic_chunking(mock_bge_m3_model):
    """Test that semantic chunking produces chunks."""
    rag = RAG()
    chunks = rag._semantic_chunk(
        ["First sentence. Second sentence. Third sentence about something else."],
        max_chunk_size=100,
    )
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_rag_fixed_chunking():
    """Test fixed-size chunking."""
    rag = RAG()
    chunks = rag._fixed_chunk(["A" * 100], chunk_size=30)
    assert len(chunks) == 4  # 100/30 = 3.33 → ceil = 4


def test_ask_question_backward_compat(mock_bge_m3_model):
    """Test backward-compatible ask_question wrapper."""
    def custom_fn(prompt):
        return "compat answer"

    result = ask_question(
        text_data=["Some text about AI."],
        question="What is AI?",
        llm_fn=custom_fn,
        semantic_chunking=False,
    )
    assert result == "compat answer"
