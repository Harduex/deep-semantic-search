"""Custom exceptions for deep-semantic-search."""


class DeepSemanticSearchError(Exception):
    """Base exception for all deep-semantic-search errors."""


class IndexNotFoundError(DeepSemanticSearchError):
    """Raised when a vector index or metadata file does not exist."""


class ModelLoadError(DeepSemanticSearchError):
    """Raised when a model fails to download or load."""


class SearchError(DeepSemanticSearchError):
    """Raised when a search operation fails."""


class EmbeddingError(DeepSemanticSearchError):
    """Raised when an embedding operation fails."""


class ClusteringError(DeepSemanticSearchError):
    """Raised when a clustering operation fails."""


class MigrationError(DeepSemanticSearchError):
    """Raised when auto-migration of legacy index format fails."""


class CaptioningError(DeepSemanticSearchError):
    """Raised when an image captioning operation fails."""
