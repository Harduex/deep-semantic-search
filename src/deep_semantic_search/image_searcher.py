"""Image similarity search using CLIP + FAISS."""

import logging
import math

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from .image_indexer import ImageIndexer

logger = logging.getLogger("deep_semantic_search")


class ImageSearcher:
    """Search for similar images using a pre-built CLIP/FAISS index.

    Parameters
    ----------
    indexer : ImageIndexer
        A configured and indexed ``ImageIndexer`` instance.
    """

    def __init__(self, indexer: ImageIndexer):
        self._indexer = indexer

    def _search_by_vector(self, vector: np.ndarray, n: int) -> dict[str, float]:
        """Search FAISS index by feature vector."""
        index = faiss.read_index(str(self._indexer._index_file))
        D, I = index.search(np.array([vector], dtype=np.float32), n)
        image_data = self._indexer.get_metadata()
        paths = image_data.iloc[I[0]]["images_paths"].to_list()
        return dict(zip(paths, D[0].tolist()))

    def search_by_image(self, image_path: str, n: int = 10) -> dict[str, float]:
        """Find images most similar to a query image.

        Parameters
        ----------
        image_path : str
            Path to the query image.
        n : int
            Number of results to return.

        Returns
        -------
        dict[str, float]
            Mapping of image path to distance score.
        """
        query_vector = self._indexer._extract(Image.open(image_path))
        return self._search_by_vector(query_vector, n)

    def search_by_text(self, text: str, n: int = 10) -> dict[str, float]:
        """Find images most similar to a text query using CLIP.

        Parameters
        ----------
        text : str
            Text query.
        n : int
            Number of results to return.

        Returns
        -------
        dict[str, float]
            Mapping of image path to similarity score (higher = more similar).
        """
        inputs = self._indexer.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self._indexer.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_embeddings = self._indexer.model.get_text_features(**inputs)
        text_embeddings = text_embeddings.detach().cpu().numpy()

        image_data = self._indexer.get_metadata()
        image_embeddings = np.vstack(image_data["features"].values)
        similarity_scores = np.inner(image_embeddings, text_embeddings).flatten()
        sorted_indices = np.argsort(similarity_scores)[::-1][:n]

        similar = image_data.iloc[sorted_indices]
        return dict(zip(similar["images_paths"], similarity_scores[sorted_indices].tolist()))

    def plot_similar_images(self, image_path: str, n: int = 6) -> None:
        """Display a query image and its most similar matches.

        Parameters
        ----------
        image_path : str
            Path to the query image.
        n : int
            Number of similar images to show.
        """
        input_img = Image.open(image_path)
        input_img_resized = ImageOps.fit(input_img, (224, 224), Image.LANCZOS)
        plt.figure(figsize=(5, 5))
        plt.axis("off")
        plt.title("Input Image", fontsize=18)
        plt.imshow(input_img_resized)
        plt.show()

        results = self.search_by_image(image_path, n)
        img_list = list(results.keys())

        grid_size = math.ceil(math.sqrt(n))
        fig = plt.figure(figsize=(20, 15))
        for i in range(min(n, len(img_list))):
            fig.add_subplot(grid_size, grid_size, i + 1)
            plt.axis("off")
            img = Image.open(img_list[i])
            img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
            plt.imshow(img_resized)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        fig.suptitle("Similar Results", fontsize=22)
        plt.show()
