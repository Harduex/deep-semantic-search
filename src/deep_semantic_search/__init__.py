"""Deep Semantic Search — embedding, indexing, and semantic search for text and image data."""

import logging

from .exceptions import (
    ClusteringError,
    DeepSemanticSearchError,
    EmbeddingError,
    IndexNotFoundError,
    ModelLoadError,
    SearchError,
)
from .image_captioner import ImageCaptioner
from .image_clusterer import ImageClusterer
from .image_indexer import ImageIndexer
from .image_loader import LoadImageData
from .image_searcher import ImageSearcher
from .rag import ask_question
from .text_embedder import TextEmbedder
from .text_loader import LoadTextData
from .text_searcher import TextSearch

__version__ = "1.1.0"

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
    # RAG
    "ask_question",
    # Exceptions
    "DeepSemanticSearchError",
    "IndexNotFoundError",
    "ModelLoadError",
    "SearchError",
    "EmbeddingError",
    "ClusteringError",
]

# Library best practice: NullHandler so users control logging
logging.getLogger("deep_semantic_search").addHandler(logging.NullHandler())
