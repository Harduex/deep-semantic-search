"""Image data loading utilities."""

import logging
import os
import random
from pathlib import Path

import pandas as pd

from .config import IMAGE_EXTENSIONS

logger = logging.getLogger("deep_semantic_search")


class LoadImageData:
    """Load image paths from folders or CSV files."""

    def from_folder(self, folder_list: list[str | Path], shuffle: bool = False) -> list[str]:
        """Collect image paths from one or more folders recursively.

        Parameters
        ----------
        folder_list : list[str | Path]
            Folders to scan for images.
        shuffle : bool
            If True, shuffle the returned paths.

        Returns
        -------
        list[str]
            Absolute paths to discovered images.
        """
        image_paths: list[str] = []
        for folder in folder_list:
            for root, _dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(IMAGE_EXTENSIONS):
                        image_paths.append(os.path.join(root, file))
        if shuffle:
            random.shuffle(image_paths)
        return image_paths

    def from_csv(self, csv_file_path: str | Path, images_column_name: str) -> list[str]:
        """Load image paths from a CSV column.

        Parameters
        ----------
        csv_file_path : str | Path
            Path to the CSV file.
        images_column_name : str
            Column name containing image paths.

        Returns
        -------
        list[str]
            Image paths from the CSV.
        """
        return pd.read_csv(csv_file_path)[images_column_name].to_list()
