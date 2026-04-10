"""Shared configuration defaults for deep-semantic-search."""

from pathlib import Path

DEFAULT_METADATA_DIR = Path.home() / ".deep-semantic-search"

CLIP_MODEL_DEFAULT = "openai/clip-vit-base-patch32"
BLIP_MODEL_DEFAULT = "Salesforce/blip-image-captioning-large"
TEXT_MODEL_DEFAULT = "sentence-transformers/nli-mpnet-base-v2"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp")

IMAGE_DATA_FEATURES_FILE = "image_data_features.pkl"
IMAGE_FEATURES_VECTORS_FILE = "image_features_vectors.idx"
CORPUS_LIST_DATA_FILE = "corpus_list_data.pickle"
CORPUS_EMBEDDINGS_DATA_FILE = "corpus_embeddings_data.pickle"
