"""Deep Semantic Search — embedding, indexing, and semantic search for text and image data."""

import logging

from .exceptions import (
    CaptioningError,
    ClusteringError,
    DeepSemanticSearchError,
    EmbeddingError,
    IndexNotFoundError,
    MigrationError,
    ModelLoadError,
    SearchError,
)
from .image_indexer import ImageIndexer
from .image_loader import LoadImageData
from .image_searcher import ImageSearcher
from .text_embedder import TextEmbedder
from .text_loader import LoadTextData
from .text_searcher import TextSearch

# These require optional extras
try:
    from .image_clusterer import ImageClusterer
except ImportError:
    pass
try:
    from .image_captioner import ImageCaptioner
except ImportError:
    pass
try:
    from .rag import RAG, ask_question
except ImportError:
    pass
try:
    from .unified_search import UnifiedIndexer, UnifiedSearcher
except ImportError:
    pass

__version__ = "3.0.1"

__all__ = [
    # Image
    "LoadImageData",
    "ImageIndexer",
    "ImageSearcher",
    "ImageClusterer",
    "ImageCaptioner",
    # Text
    "LoadTextData",
    "TextEmbedder",
    "TextSearch",
    # Unified
    "UnifiedIndexer",
    "UnifiedSearcher",
    # RAG
    "RAG",
    "ask_question",
    # Exceptions
    "DeepSemanticSearchError",
    "IndexNotFoundError",
    "ModelLoadError",
    "SearchError",
    "EmbeddingError",
    "ClusteringError",
    "MigrationError",
    "CaptioningError",
]

# Library best practice: NullHandler so users control logging
logging.getLogger("deep_semantic_search").addHandler(logging.NullHandler())
