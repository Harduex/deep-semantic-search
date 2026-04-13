"""Image clustering using KMeans or HDBSCAN on SigLIP embeddings."""

from __future__ import annotations

import logging
import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .config import DEFAULT_OLLAMA_MODEL, SIGLIP_IMAGE_SIZE
from .exceptions import ClusteringError
from .image_indexer import ImageIndexer

if TYPE_CHECKING:
    from .image_captioner import ImageCaptioner

logger = logging.getLogger("deep_semantic_search")


def _default_topic_fn(texts: list[str]) -> list[str]:
    """Try LiteLLM/Ollama for topic extraction; fall back to generic label."""
    llm_model = os.getenv("OLLAMA_LLM_MODEL") or DEFAULT_OLLAMA_MODEL
    try:
        import ast
        import re

        from litellm import completion

        prompt_text = f"""
You have been provided with the following list of descriptions of images:
descriptions: {texts}

What is the best topic that describes these texts?
If you think the texts are about multiple topics, write them in a python list like this:
"Topic: ['landscapes', 'people', 'rocks']"
Don't pick more than a 3 topics.

Write the topic/topics in lowercase without any other special characters or spaces.
Make sure to write the topic in the desired format.

Answer in a python list format only.
Topic: ['topic']

Don't include any other information in your response. No clarifications or additional information.
"""

        response = completion(
            model=f"ollama/{llm_model}",
            messages=[
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": "What is the best topic for these texts?"},
            ],
        )
        answer = response.choices[0].message.content

        match = re.search(r"Topic: (\[.*\])", answer)
        if match:
            return ast.literal_eval(match.group(1))
    except ImportError:
        logger.warning(
            "LiteLLM not installed. Install with: pip install deep-semantic-search[llm]"
        )
    except Exception as exc:
        logger.warning("LLM topic extraction failed: %s. Using fallback.", exc)

    return ["other"]


class ImageClusterer:
    """Cluster images using KMeans or HDBSCAN on feature vectors.

    Parameters
    ----------
    indexer : ImageIndexer
        A configured and indexed ``ImageIndexer`` instance.
    llm_fn : Callable[[list[str]], list[str]] | None
        A callable that takes a list of caption texts and returns topic labels.
        Defaults to using LiteLLM/Ollama if available, else returns ``["other"]``.
    """

    def __init__(
        self,
        indexer: ImageIndexer,
        llm_fn: Callable[[list[str]], list[str]] | None = None,
    ):
        self._indexer = indexer
        self._llm_fn = llm_fn or _default_topic_fn
        self._image_data: pd.DataFrame = pd.DataFrame()

    def cluster(
        self,
        n_clusters: int | None = None,
        captioner: ImageCaptioner | None = None,
        min_cluster_size: int = 5,
    ) -> pd.DataFrame:
        """Cluster indexed images into groups and assign topic labels.

        If ``n_clusters`` is provided, uses KMeans. Otherwise, uses HDBSCAN
        for automatic cluster count detection.

        Parameters
        ----------
        n_clusters : int | None
            Number of clusters. If None, HDBSCAN auto-selects.
        captioner : ImageCaptioner | None
            Optional captioner for generating topic labels.
        min_cluster_size : int
            Minimum cluster size for HDBSCAN (ignored with KMeans).

        Returns
        -------
        pd.DataFrame
            Image data with ``cluster`` and ``topic`` columns.
        """
        image_data = self._indexer.get_metadata()
        if image_data.empty:
            raise ClusteringError("No indexed images found. Run indexer.run_index() first.")

        try:
            from sklearn.cluster import HDBSCAN, KMeans
        except ImportError:
            raise ImportError(
                "'scikit-learn' is required for clustering. "
                "Install it with: pip install deep-semantic-search[clustering]"
            ) from None

        features = np.vstack(image_data["features"].values).astype(np.float32)
        image_data = image_data.copy()

        if n_clusters is not None:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            image_data["cluster"] = km.fit_predict(features)
        else:
            hdb = HDBSCAN(min_cluster_size=min_cluster_size)
            image_data["cluster"] = hdb.fit_predict(features)

        unique_clusters = sorted(image_data["cluster"].unique())

        if captioner is not None:
            for cid in unique_clusters:
                if cid == -1:
                    image_data.loc[image_data["cluster"] == -1, "topic"] = "noise"
                    continue
                cluster_paths = image_data[image_data["cluster"] == cid]["images_paths"].to_list()
                sample_paths = cluster_paths[:15]
                captions_df = captioner.caption(sample_paths)
                topics = self._llm_fn(captions_df["caption"].to_list())
                image_data.loc[image_data["cluster"] == cid, "topic"] = topics[0]
        else:
            for cid in unique_clusters:
                if cid == -1:
                    image_data.loc[image_data["cluster"] == -1, "topic"] = "noise"
                else:
                    image_data.loc[image_data["cluster"] == cid, "topic"] = f"cluster_{cid}"

        self._image_data = image_data
        return image_data

    def get_cluster_images(self, cluster_id: int) -> list[str]:
        """Return image paths belonging to a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster identifier.

        Returns
        -------
        list[str]
            Image paths in the cluster.
        """
        if self._image_data.empty:
            raise ClusteringError("No clustering data. Run cluster() first.")
        return self._image_data[self._image_data["cluster"] == cluster_id]["images_paths"].to_list()

    def save_clusters(self, save_dir: str | Path) -> None:
        """Save clustered images to disk organized as ``{id}_{topic}/``.

        Parameters
        ----------
        save_dir : str | Path
            Directory to save clustered images into.
        """
        if self._image_data.empty:
            raise ClusteringError("No clustering data. Run cluster() first.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for cluster_id in self._image_data["cluster"].unique():
            cluster_data = self._image_data[self._image_data["cluster"] == cluster_id]
            for topic in cluster_data["topic"].unique():
                topic_dir = save_dir / f"{cluster_id}_{topic}"
                topic_dir.mkdir(parents=True, exist_ok=True)
                for img_path in cluster_data[cluster_data["topic"] == topic]["images_paths"]:
                    shutil.copy(img_path, topic_dir)

        logger.info("Clusters saved to %s", save_dir)

    def plot_cluster(self, cluster_id: int, n: int | None = None) -> None:
        """Plot images from a specific cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster identifier.
        n : int | None
            Number of images to plot. None means all.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "'matplotlib' is required for plotting. "
                "Install it with: pip install deep-semantic-search[viz]"
            ) from None

        from PIL import Image, ImageOps

        img_size = (SIGLIP_IMAGE_SIZE, SIGLIP_IMAGE_SIZE)
        img_list = self.get_cluster_images(cluster_id)
        count = n or len(img_list)
        count = min(count, len(img_list))

        grid_size = math.ceil(math.sqrt(count))
        fig = plt.figure(figsize=(20, 15))
        for i in range(count):
            fig.add_subplot(grid_size, grid_size, i + 1)
            plt.axis("off")
            img = Image.open(img_list[i])
            img_resized = ImageOps.fit(img, img_size, Image.LANCZOS)
            plt.imshow(img_resized)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        fig.suptitle(f"Cluster {cluster_id}", fontsize=22)
        plt.show()
