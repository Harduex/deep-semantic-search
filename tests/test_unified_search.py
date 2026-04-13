"""Tests for UnifiedIndexer and UnifiedSearcher."""

from deep_semantic_search.unified_search import UnifiedIndexer, UnifiedSearcher


def test_unified_add_images_and_texts(sample_images, tmp_metadata_dir, mock_siglip_model):
    indexer = UnifiedIndexer(metadata_dir=tmp_metadata_dir / "unified")
    indexer.add_images(sample_images[:2])
    indexer.add_texts(["a red image", "a blue sky"], labels=["red", "blue"])

    assert len(indexer._entries) == 4
    assert indexer._entries[0]["type"] == "image"
    assert indexer._entries[2]["type"] == "text"


def test_unified_build_and_search(sample_images, tmp_metadata_dir, mock_siglip_model):
    indexer = UnifiedIndexer(metadata_dir=tmp_metadata_dir / "unified")
    indexer.add_images(sample_images[:2])
    indexer.add_texts(["a red image"], labels=["red_text"])
    indexer.build_index()

    searcher = UnifiedSearcher(indexer)
    results = searcher.search("red", n=3)

    assert isinstance(results, list)
    assert len(results) <= 3
    assert all("type" in r and "source" in r and "score" in r for r in results)


def test_unified_modality_filter(sample_images, tmp_metadata_dir, mock_siglip_model):
    indexer = UnifiedIndexer(metadata_dir=tmp_metadata_dir / "unified")
    indexer.add_images(sample_images[:2])
    indexer.add_texts(["a red image"], labels=["red_text"])
    indexer.build_index()

    searcher = UnifiedSearcher(indexer)

    image_results = searcher.search("red", n=10, modality_filter="image")
    assert all(r["type"] == "image" for r in image_results)

    text_results = searcher.search("red", n=10, modality_filter="text")
    assert all(r["type"] == "text" for r in text_results)


def test_unified_search_by_image(sample_images, tmp_metadata_dir, mock_siglip_model):
    indexer = UnifiedIndexer(metadata_dir=tmp_metadata_dir / "unified")
    indexer.add_images(sample_images[:3])
    indexer.build_index()

    searcher = UnifiedSearcher(indexer)
    results = searcher.search_by_image(sample_images[0], n=2)

    assert isinstance(results, list)
    assert len(results) <= 2


def test_unified_find_duplicates(sample_images, tmp_metadata_dir, mock_siglip_model):
    indexer = UnifiedIndexer(metadata_dir=tmp_metadata_dir / "unified")
    indexer.add_images(sample_images[:3])
    indexer.build_index()

    searcher = UnifiedSearcher(indexer)
    duplicates = searcher.find_duplicates(threshold=0.5)

    assert isinstance(duplicates, list)
    for src1, src2, sim in duplicates:
        assert sim >= 0.5

    sims = [s for _, _, s in duplicates]
    assert sims == sorted(sims, reverse=True)
