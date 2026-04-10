"""Tests for exceptions module."""

from deep_semantic_search.exceptions import (
    ClusteringError,
    DeepSemanticSearchError,
    EmbeddingError,
    IndexNotFoundError,
    ModelLoadError,
    SearchError,
)


def test_base_exception_hierarchy():
    assert issubclass(IndexNotFoundError, DeepSemanticSearchError)
    assert issubclass(ModelLoadError, DeepSemanticSearchError)
    assert issubclass(SearchError, DeepSemanticSearchError)
    assert issubclass(EmbeddingError, DeepSemanticSearchError)
    assert issubclass(ClusteringError, DeepSemanticSearchError)


def test_exceptions_are_catchable():
    try:
        raise IndexNotFoundError("test")
    except DeepSemanticSearchError as e:
        assert str(e) == "test"


def test_all_exceptions_inherit_from_base():
    for exc_cls in [IndexNotFoundError, ModelLoadError, SearchError, EmbeddingError, ClusteringError]:
        assert issubclass(exc_cls, Exception)
