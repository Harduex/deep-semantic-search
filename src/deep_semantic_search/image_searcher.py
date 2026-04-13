"""Image similarity search using SigLIP + USearch."""

import logging
import math

import numpy as np
import torch
from PIL import Image, ImageOps

from .config import DEFAULT_DUPLICATE_THRESHOLD, SIGLIP_IMAGE_SIZE
from .image_indexer import ImageIndexer

logger = logging.getLogger("deep_semantic_search")


class ImageSearcher:
    """Search for similar images using a pre-built SigLIP/USearch index.

    Parameters
    ----------
    indexer : ImageIndexer
        A configured and indexed ``ImageIndexer`` instance.
    """

    def __init__(self, indexer: ImageIndexer):
        self._indexer = indexer
        self._cached_index = None

    @property
    def _index(self):
        """Lazy-load and cache the USearch index."""
        if self._cached_index is None:
            from usearch.index import Index

            self._cached_index = Index.restore(str(self._indexer.index_path))
        return self._cached_index

    def _search_by_vector(self, vector: np.ndarray, n: int) -> list[dict]:
        """Search USearch index by feature vector, return cosine similarity scores."""
        matches = self._index.search(vector.astype(np.float32), n)
        image_data = self._indexer.get_metadata()
        results = []
        for i, (key, distance) in enumerate(zip(matches.keys, matches.distances)):
            results.append({
                "rank": i + 1,
                "path": image_data["images_paths"].iloc[int(key)],
                "score": float(1.0 - distance),
            })
        return results

    def search_by_image(self, image_path: str, n: int = 10) -> list[dict]:
        """Find images most similar to a query image.

        Parameters
        ----------
        image_path : str
            Path to the query image.
        n : int
            Number of results to return.

        Returns
        -------
        list[dict]
            List of dicts with keys ``rank``, ``path``, ``score`` (cosine similarity).
        """
        query_vector = self._indexer._extract(Image.open(image_path))
        return self._search_by_vector(query_vector, n)

    def search_by_text(self, text: str, n: int = 10) -> list[dict]:
        """Find images most similar to a text query using SigLIP.

        Parameters
        ----------
        text : str
            Text query.
        n : int
            Number of results to return.

        Returns
        -------
        list[dict]
            List of dicts with keys ``rank``, ``path``, ``score`` (cosine similarity).
        """
        inputs = self._indexer.processor(
            text=[text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(self._indexer.device) for k, v in inputs.items()}
        with torch.no_grad():
            text_features = self._indexer.model.get_text_features(**inputs)
        if hasattr(text_features, "pooler_output"):
            text_features = text_features.pooler_output
        text_vector = text_features.detach().cpu().numpy().flatten()
        text_vector = text_vector / np.linalg.norm(text_vector)
        return self._search_by_vector(text_vector, n)

    def find_duplicates(self, threshold: float = DEFAULT_DUPLICATE_THRESHOLD) -> list[tuple[str, str, float]]:
        """Find near-duplicate image pairs above the similarity threshold.

        Parameters
        ----------
        threshold : float
            Minimum cosine similarity to consider a pair as duplicates.

        Returns
        -------
        list[tuple[str, str, float]]
            Sorted list of (path1, path2, similarity) tuples.
        """
        image_data = self._indexer.get_metadata()
        features = np.vstack(image_data["features"].values).astype(np.float32)
        paths = image_data["images_paths"].to_list()
        duplicates = []
        for i in range(len(features)):
            matches = self._index.search(features[i], min(len(features), 50))
            for key, dist in zip(matches.keys, matches.distances):
                j = int(key)
                sim = 1.0 - float(dist)
                if j > i and sim >= threshold:
                    duplicates.append((paths[i], paths[j], sim))
        return sorted(duplicates, key=lambda x: -x[2])

    def plot_similar_images(self, image_path: str, n: int = 6) -> None:
        """Display a query image and its most similar matches.

        Parameters
        ----------
        image_path : str
            Path to the query image.
        n : int
            Number of similar images to show.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "'matplotlib' is required for plotting. "
                "Install it with: pip install deep-semantic-search[viz]"
            ) from None

        img_size = (SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE)

        input_img = Image.open(image_path)
        input_img_resized = ImageOps.fit(input_img, img_size, Image.LANCZOS)
        plt.figure(figsize=(5, 5))
        plt.axis("off")
        plt.title("Input Image", fontsize=18)
        plt.imshow(input_img_resized)
        plt.show()

        results = self.search_by_image(image_path, n)
        img_list = [r["path"] for r in results]

        grid_size = math.ceil(math.sqrt(n))
        fig = plt.figure(figsize=(20, 15))
        for i in range(min(n, len(img_list))):
            fig.add_subplot(grid_size, grid_size, i + 1)
            plt.axis("off")
            img = Image.open(img_list[i])
            img_resized = ImageOps.fit(img, img_size, Image.LANCZOS)
            plt.imshow(img_resized)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        fig.suptitle("Similar Results", fontsize=22)
        plt.show()
