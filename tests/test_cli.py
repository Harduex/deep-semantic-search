"""Tests for the CLI module."""

from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from deep_semantic_search.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def text_folder(tmp_path):
    for i in range(3):
        (tmp_path / f"doc_{i}.txt").write_text(f"Document {i} content about testing.", encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def image_folder(tmp_path):
    from PIL import Image

    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 80, 0, 0))
        img.save(tmp_path / f"img_{i}.jpg")
    return str(tmp_path)


class TestCLIBase:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Deep Semantic Search CLI" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output


class TestImageSearch:
    def test_help(self, runner):
        result = runner.invoke(cli, ["image-search", "--help"])
        assert result.exit_code == 0
        assert "--folder" in result.output
        assert "--query" in result.output

    @patch("deep_semantic_search.image_searcher.ImageSearcher")
    @patch("deep_semantic_search.image_indexer.ImageIndexer")
    @patch("deep_semantic_search.image_loader.LoadImageData")
    def test_search_by_text(self, mock_loader_cls, mock_indexer_cls, mock_searcher_cls, runner, image_folder):
        mock_loader_cls.return_value.from_folder.return_value = ["img1.jpg", "img2.jpg"]
        mock_indexer_cls.return_value.run_index.return_value = None
        mock_searcher_cls.return_value.search_by_text.return_value = [
            {"rank": 1, "path": "img1.jpg", "score": 0.95},
            {"rank": 2, "path": "img2.jpg", "score": 0.80},
        ]

        result = runner.invoke(cli, ["image-search", "-f", image_folder, "--query", "sunset", "--top", "2"])
        assert result.exit_code == 0
        assert "img1.jpg" in result.output
        mock_indexer_cls.return_value.run_index.assert_called_once_with(reindex=False)

    @patch("deep_semantic_search.image_loader.LoadImageData")
    def test_no_images_found(self, mock_loader_cls, runner, image_folder):
        mock_loader_cls.return_value.from_folder.return_value = []
        result = runner.invoke(cli, ["image-search", "-f", image_folder, "--query", "test"])
        assert result.exit_code != 0
        assert "No images found" in result.output

    @patch("deep_semantic_search.image_searcher.ImageSearcher")
    @patch("deep_semantic_search.image_indexer.ImageIndexer")
    @patch("deep_semantic_search.image_loader.LoadImageData")
    def test_json_output(self, mock_loader_cls, mock_indexer_cls, mock_searcher_cls, runner, image_folder):
        mock_loader_cls.return_value.from_folder.return_value = ["a.jpg"]
        mock_indexer_cls.return_value.run_index.return_value = None
        mock_searcher_cls.return_value.search_by_text.return_value = [
            {"rank": 1, "path": "a.jpg", "score": 0.9},
        ]

        result = runner.invoke(cli, ["image-search", "-f", image_folder, "-q", "cat", "--format", "json"])
        assert result.exit_code == 0
        assert '"path": "a.jpg"' in result.output


class TestTextSearch:
    def test_help(self, runner):
        result = runner.invoke(cli, ["text-search", "--help"])
        assert result.exit_code == 0
        assert "--folder" in result.output
        assert "--rerank" in result.output
        assert "--hybrid" in result.output

    @patch("deep_semantic_search.text_searcher.TextSearch")
    @patch("deep_semantic_search.text_embedder.TextEmbedder")
    @patch("deep_semantic_search.text_loader.LoadTextData")
    def test_search(self, mock_loader_cls, mock_embedder_cls, mock_search_cls, runner, text_folder):
        mock_loader_cls.return_value.from_folder.return_value = {"doc.txt": "some text"}
        mock_embedder_cls.return_value.embed.return_value = None
        mock_search_cls.return_value.find_similar.return_value = [
            {"index": 0, "text": "some text", "path": "doc.txt", "score": 0.92}
        ]

        result = runner.invoke(cli, ["text-search", "-f", text_folder, "test query"])
        assert result.exit_code == 0
        assert "doc.txt" in result.output

    @patch("deep_semantic_search.text_loader.LoadTextData")
    def test_no_files(self, mock_loader_cls, runner, text_folder):
        mock_loader_cls.return_value.from_folder.return_value = {}
        result = runner.invoke(cli, ["text-search", "-f", text_folder, "query"])
        assert result.exit_code != 0
        assert "No text files" in result.output


class TestImageCluster:
    def test_help(self, runner):
        result = runner.invoke(cli, ["image-cluster", "--help"])
        assert result.exit_code == 0
        assert "--clusters" in result.output
        assert "--min-cluster-size" in result.output

    @patch("deep_semantic_search.image_clusterer.ImageClusterer")
    @patch("deep_semantic_search.image_indexer.ImageIndexer")
    @patch("deep_semantic_search.image_loader.LoadImageData")
    def test_cluster_with_k(self, mock_loader_cls, mock_indexer_cls, mock_clusterer_cls, runner, image_folder):
        mock_loader_cls.return_value.from_folder.return_value = ["img.jpg"]
        mock_indexer_cls.return_value.run_index.return_value = None

        result_df = pd.DataFrame({"images_paths": ["img.jpg"], "cluster": [0], "topic": ["nature"]})
        mock_clusterer_cls.return_value.cluster.return_value = result_df

        result = runner.invoke(cli, ["image-cluster", "-f", image_folder, "-k", "2"])
        assert result.exit_code == 0
        assert "nature" in result.output

    @patch("deep_semantic_search.image_clusterer.ImageClusterer")
    @patch("deep_semantic_search.image_indexer.ImageIndexer")
    @patch("deep_semantic_search.image_loader.LoadImageData")
    def test_cluster_hdbscan(self, mock_loader_cls, mock_indexer_cls, mock_clusterer_cls, runner, image_folder):
        """No -k flag → HDBSCAN auto."""
        mock_loader_cls.return_value.from_folder.return_value = ["img.jpg"]
        mock_indexer_cls.return_value.run_index.return_value = None

        result_df = pd.DataFrame({"images_paths": ["img.jpg"], "cluster": [0], "topic": ["cluster_0"]})
        mock_clusterer_cls.return_value.cluster.return_value = result_df

        result = runner.invoke(cli, ["image-cluster", "-f", image_folder])
        assert result.exit_code == 0
        assert "HDBSCAN" in result.output


class TestAsk:
    def test_help(self, runner):
        result = runner.invoke(cli, ["ask", "--help"])
        assert result.exit_code == 0
        assert "--folder" in result.output
        assert "--rerank" in result.output
        assert "--semantic-chunking" in result.output

    @patch("deep_semantic_search.rag.RAG")
    @patch("deep_semantic_search.text_loader.LoadTextData")
    def test_ask(self, mock_loader_cls, mock_rag_cls, runner, text_folder):
        mock_loader_cls.return_value.from_folder.return_value = {"doc.txt": "some content"}
        mock_rag_cls.return_value.ask.return_value = "The answer is 42."

        result = runner.invoke(cli, ["ask", "-f", text_folder, "What is the meaning?"])
        assert result.exit_code == 0
        assert "42" in result.output

    @patch("deep_semantic_search.text_loader.LoadTextData")
    def test_no_files(self, mock_loader_cls, runner, text_folder):
        mock_loader_cls.return_value.from_folder.return_value = {}
        result = runner.invoke(cli, ["ask", "-f", text_folder, "question"])
        assert result.exit_code != 0
        assert "No text files" in result.output


class TestUnifiedSearch:
    def test_help(self, runner):
        result = runner.invoke(cli, ["unified-search", "--help"])
        assert result.exit_code == 0
        assert "--image-folder" in result.output
        assert "--text-folder" in result.output
        assert "--filter" in result.output

    def test_no_folders(self, runner):
        result = runner.invoke(cli, ["unified-search", "-q", "test"])
        assert result.exit_code != 0


class TestFindDuplicates:
    def test_help(self, runner):
        result = runner.invoke(cli, ["find-duplicates", "--help"])
        assert result.exit_code == 0
        assert "--threshold" in result.output

    @patch("deep_semantic_search.image_searcher.ImageSearcher")
    @patch("deep_semantic_search.image_indexer.ImageIndexer")
    @patch("deep_semantic_search.image_loader.LoadImageData")
    def test_no_duplicates(self, mock_loader_cls, mock_indexer_cls, mock_searcher_cls, runner, image_folder):
        mock_loader_cls.return_value.from_folder.return_value = ["img.jpg"]
        mock_indexer_cls.return_value.run_index.return_value = None
        mock_searcher_cls.return_value.find_duplicates.return_value = []

        result = runner.invoke(cli, ["find-duplicates", "-f", image_folder])
        assert result.exit_code == 0
        assert "No duplicates" in result.output
