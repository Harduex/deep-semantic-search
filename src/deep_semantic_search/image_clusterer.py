"""Image clustering using KMeans on CLIP embeddings."""

import logging
import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from kmeans_pytorch import kmeans
from PIL import Image, ImageOps

from .exceptions import ClusteringError
from .image_indexer import ImageIndexer

logger = logging.getLogger("deep_semantic_search")


def _default_topic_fn(texts: list[str]) -> list[str]:
    """Try Ollama for topic extraction; fall back to generic label."""
    llm_model = os.getenv("OLLAMA_LLM_MODEL") or "mistral:7b"
    try:
        import ast
        import re

        from langchain.schema import HumanMessage, SystemMessage
        from langchain_community.chat_models import ChatOllama

        chat = ChatOllama(model=llm_model, temperature=0.8)

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

        answer = chat(
            [
                SystemMessage(content=prompt_text),
                HumanMessage(content="What is the best topic for these texts?"),
            ]
        ).content

        match = re.search(r"Topic: (\[.*\])", answer)
        if match:
            return ast.literal_eval(match.group(1))
    except Exception as exc:
        logger.warning("Ollama topic extraction failed: %s. Using fallback.", exc)

    return ["other"]


class ImageClusterer:
    """Cluster images using KMeans on CLIP feature vectors.

    Parameters
    ----------
    indexer : ImageIndexer
        A configured and indexed ``ImageIndexer`` instance.
    llm_fn : Callable[[list[str]], list[str]] | None
        A callable that takes a list of caption texts and returns topic labels.
        Defaults to using Ollama if available, else returns ``["other"]``.
    """

    def __init__(
        self,
        indexer: ImageIndexer,
        llm_fn: Callable[[list[str]], list[str]] | None = None,
    ):
        self._indexer = indexer
        self._llm_fn = llm_fn or _default_topic_fn
        self._image_data: pd.DataFrame = pd.DataFrame()

    def cluster(self, n_clusters: int, captioner: "ImageCaptioner | None" = None) -> pd.DataFrame:
        """Cluster indexed images into groups and assign topic labels.

        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        captioner : ImageCaptioner | None
            Optional captioner for generating topic labels. If None, topics
            are generated from a basic captioner import.

        Returns
        -------
        pd.DataFrame
            Image data with ``cluster`` and ``topic`` columns.
        """
        image_data = self._indexer.get_metadata()
        if image_data.empty:
            raise ClusteringError("No indexed images found. Run indexer.run_index() first.")

        features = np.vstack(image_data["features"].values)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = torch.from_numpy(features).float().to(device)

        cluster_ids, _centers = kmeans(
            X=data,
            num_clusters=n_clusters,
            distance="euclidean",
            device=device,
        )

        image_data = image_data.copy()
        image_data["cluster"] = cluster_ids.cpu().numpy()

        if captioner is not None:
            for i in range(n_clusters):
                cluster_paths = image_data[image_data["cluster"] == i]["images_paths"].to_list()
                sample_paths = cluster_paths[:15]
                captions_df = captioner.caption(sample_paths)
                topics = self._llm_fn(captions_df["caption"].to_list())
                image_data.loc[image_data["cluster"] == i, "topic"] = topics[0]
        else:
            for i in range(n_clusters):
                image_data.loc[image_data["cluster"] == i, "topic"] = f"cluster_{i}"

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
        img_list = self.get_cluster_images(cluster_id)
        count = n or len(img_list)
        count = min(count, len(img_list))

        grid_size = math.ceil(math.sqrt(count))
        fig = plt.figure(figsize=(20, 15))
        for i in range(count):
            fig.add_subplot(grid_size, grid_size, i + 1)
            plt.axis("off")
            img = Image.open(img_list[i])
            img_resized = ImageOps.fit(img, (224, 224), Image.LANCZOS)
            plt.imshow(img_resized)
        fig.tight_layout()
        fig.subplots_adjust(top=0.93)
        fig.suptitle(f"Cluster {cluster_id}", fontsize=22)
        plt.show()


# Avoid circular import — ImageCaptioner is only used as type hint
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from .image_captioner import ImageCaptioner
