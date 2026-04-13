"""Text data loading utilities."""

import logging
import os
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger("deep_semantic_search")


class LoadTextData:
    """Load text data from folders or CSV files into a dictionary."""

    def __init__(self) -> None:
        self.corpus_dict: dict[str, str] = {}

    def from_csv(self, file_path: str | Path, column_name: str) -> dict[int, str]:
        """Load text data from a CSV column.

        Parameters
        ----------
        file_path : str | Path
            Path to the CSV file.
        column_name : str
            Name of the column containing text data.

        Returns
        -------
        dict[int, str]
            Mapping of row index to text content.
        """
        self.corpus_dict = {}
        csv_data = pd.read_csv(file_path, encoding="latin1")
        self.corpus_dict = csv_data[column_name].dropna().to_dict()
        return self.corpus_dict

    def from_folder(self, folder_path: str | Path, corpus_count: int | None = None) -> dict[str, str]:
        """Load text from ``.txt`` and ``.html`` files in a folder recursively.

        Parameters
        ----------
        folder_path : str | Path
            Root folder to scan.
        corpus_count : int | None
            Maximum number of documents to load. None means all.

        Returns
        -------
        dict[str, str]
            Mapping of file path to text content.
        """
        self.corpus_dict = {}
        count = 0
        for dirpath, _dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                if corpus_count is not None and count >= corpus_count:
                    return self.corpus_dict
                file_path = os.path.join(dirpath, filename)
                try:
                    if filename.endswith(".txt"):
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            self.corpus_dict[file_path] = f.read()
                            count += 1
                    elif filename.endswith(".html"):
                        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                            soup = BeautifulSoup(f, "html.parser")
                            self.corpus_dict[file_path] = soup.get_text()
                            count += 1
                except Exception as exc:
                    logger.warning("Failed to read %s: %s", file_path, exc)
        return self.corpus_dict
