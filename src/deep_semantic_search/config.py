"""Shared configuration defaults for deep-semantic-search."""

import warnings
from pathlib import Path

DEFAULT_METADATA_DIR = Path.home() / ".deep-semantic-search"

# Model defaults
SIGLIP_MODEL_DEFAULT = "google/siglip-so400m-patch14-384"
FLORENCE_MODEL_DEFAULT = "microsoft/Florence-2-large"
BGE_M3_MODEL_DEFAULT = "BAAI/bge-m3"
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# SigLIP dimensions
SIGLIP_EMBEDDING_DIM = 1152
SIGLIP_IMAGE_SIZE = 384

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp")

# Persistence file names
IMAGE_DATA_FEATURES_FILE = "image_features.npy"
IMAGE_DATA_PATHS_FILE = "image_paths.json"
IMAGE_USEARCH_INDEX_FILE = "image_features.usearch"
CORPUS_LIST_DATA_FILE = "corpus_metadata.json"
CORPUS_EMBEDDINGS_DATA_FILE = "corpus_embeddings.npy"
TEXT_USEARCH_INDEX_FILE = "text_embeddings.usearch"
TEXT_SPARSE_VECTORS_FILE = "text_sparse_vectors.json"
UNIFIED_USEARCH_INDEX_FILE = "unified_features.usearch"
UNIFIED_METADATA_FILE = "unified_metadata.json"

# Duplicate detection
DEFAULT_DUPLICATE_THRESHOLD = 0.95

# Deprecated v2 constant aliases
_DEPRECATED_NAMES = {
    "CLIP_MODEL_DEFAULT": ("SIGLIP_MODEL_DEFAULT", SIGLIP_MODEL_DEFAULT),
    "BLIP_MODEL_DEFAULT": ("FLORENCE_MODEL_DEFAULT", FLORENCE_MODEL_DEFAULT),
    "TEXT_MODEL_DEFAULT": ("BGE_M3_MODEL_DEFAULT", BGE_M3_MODEL_DEFAULT),
    "IMAGE_FEATURES_VECTORS_FILE": ("IMAGE_USEARCH_INDEX_FILE", IMAGE_USEARCH_INDEX_FILE),
}


def __getattr__(name):
    if name in _DEPRECATED_NAMES:
        new_name, value = _DEPRECATED_NAMES[name]
        warnings.warn(
            f"{name} is deprecated, use {new_name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_device():
    """Return CUDA device if available, otherwise CPU."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
