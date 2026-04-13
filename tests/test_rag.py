"""Tests for RAG ask_question with mocked dependencies."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from deep_semantic_search.exceptions import SearchError
from deep_semantic_search.rag import ask_question


def test_ask_question_empty_data():
    with pytest.raises(SearchError, match="non-empty"):
        ask_question([], "What is this?")


def test_ask_question_basic():
    """Test that ask_question runs end-to-end with fully mocked langchain deps."""
    mock_chroma = MagicMock()
    mock_retriever = MagicMock()
    mock_chroma.from_documents.return_value.as_retriever.return_value = mock_retriever

    mock_text_splitters = MagicMock()
    mock_text_splitters.RecursiveCharacterTextSplitter.return_value.split_documents.return_value = [
        MagicMock(page_content="chunk")
    ]

    # Build a mock LCEL chain that returns a real string
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Test answer"

    mock_prompt = MagicMock()
    # pipe operators: {dict} | prompt | llm | parser → chain
    # The final result of all | operations should be mock_chain
    mock_prompt.__ror__ = MagicMock(
        return_value=MagicMock(
            __or__=MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))
        )
    )

    mock_documents = MagicMock()
    mock_output_parsers = MagicMock()
    mock_runnables = MagicMock()

    with patch.dict(sys.modules, {
        "langchain_huggingface": MagicMock(HuggingFaceEmbeddings=MagicMock()),
        "langchain_chroma": MagicMock(Chroma=mock_chroma),
        "langchain_text_splitters": mock_text_splitters,
        "langchain_core.documents": mock_documents,
        "langchain_core.output_parsers": mock_output_parsers,
        "langchain_core.runnables": mock_runnables,
    }), patch("deep_semantic_search.rag._get_default_llm", return_value=MagicMock()), \
         patch("deep_semantic_search.rag._get_default_prompt", return_value=mock_prompt):
        result = ask_question(
            ["Some text data about AI."],
            "What is AI?",
        )

    assert result == "Test answer"


def test_ask_question_with_custom_llm_fn():
    """Test that llm_fn callback is accepted and used."""
    def custom_fn(text):
        return "custom response"

    mock_chroma = MagicMock()
    mock_retriever = MagicMock()
    mock_chroma.from_documents.return_value.as_retriever.return_value = mock_retriever

    mock_text_splitters = MagicMock()
    mock_text_splitters.RecursiveCharacterTextSplitter.return_value.split_documents.return_value = [
        MagicMock(page_content="chunk")
    ]

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "custom answer"

    mock_prompt = MagicMock()
    mock_prompt.__ror__ = MagicMock(
        return_value=MagicMock(
            __or__=MagicMock(return_value=MagicMock(__or__=MagicMock(return_value=mock_chain)))
        )
    )

    mock_documents = MagicMock()
    mock_output_parsers = MagicMock()
    mock_runnables = MagicMock()
    mock_lc_models = MagicMock()

    with patch.dict(sys.modules, {
        "langchain_huggingface": MagicMock(HuggingFaceEmbeddings=MagicMock()),
        "langchain_chroma": MagicMock(Chroma=mock_chroma),
        "langchain_text_splitters": mock_text_splitters,
        "langchain_core.documents": mock_documents,
        "langchain_core.output_parsers": mock_output_parsers,
        "langchain_core.runnables": mock_runnables,
        "langchain_core.language_models": mock_lc_models,
        "langchain_core.outputs": MagicMock(),
    }), patch("deep_semantic_search.rag._get_default_prompt", return_value=mock_prompt):
        result = ask_question(
            ["test data"],
            "question",
            llm_fn=custom_fn,
        )

    assert result == "custom answer"
