"""Tests for ImageSearcher with mocked dependencies."""

from deep_semantic_search.image_searcher import ImageSearcher


def _build_indexed(sample_images, tmp_metadata_dir, mock_siglip_model):
    """Helper: build an indexed ImageIndexer + ImageSearcher pair."""
    from deep_semantic_search.image_indexer import ImageIndexer

    indexer = ImageIndexer(
        image_list=sample_images,
        metadata_dir=tmp_metadata_dir / "siglip",
    )
    indexer.run_index()
    return ImageSearcher(indexer)


def test_search_by_image(sample_images, tmp_metadata_dir, mock_siglip_model):
    searcher = _build_indexed(sample_images, tmp_metadata_dir, mock_siglip_model)
    results = searcher.search_by_image(sample_images[0], n=3)

    assert isinstance(results, list)
    assert len(results) <= 3
    assert all(isinstance(r, dict) for r in results)
    assert all("rank" in r and "path" in r and "score" in r for r in results)


def test_search_by_text(sample_images, tmp_metadata_dir, mock_siglip_model):
    searcher = _build_indexed(sample_images, tmp_metadata_dir, mock_siglip_model)
    results = searcher.search_by_text("a red image", n=2)

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(r, dict) for r in results)
    assert all("rank" in r and "path" in r and "score" in r for r in results)


def test_search_score_range(sample_images, tmp_metadata_dir, mock_siglip_model):
    """Cosine similarity scores should be in [-1, 1]."""
    searcher = _build_indexed(sample_images, tmp_metadata_dir, mock_siglip_model)
    results = searcher.search_by_image(sample_images[0], n=5)

    for r in results:
        assert -1.0 <= r["score"] <= 1.0, f"Score {r['score']} out of cosine range"


def test_find_duplicates(sample_images, tmp_metadata_dir, mock_siglip_model):
    searcher = _build_indexed(sample_images, tmp_metadata_dir, mock_siglip_model)
    duplicates = searcher.find_duplicates(threshold=0.5)

    assert isinstance(duplicates, list)
    for path1, path2, sim in duplicates:
        assert isinstance(path1, str)
        assert isinstance(path2, str)
        assert sim >= 0.5

    # Should be sorted descending by similarity
    sims = [s for _, _, s in duplicates]
    assert sims == sorted(sims, reverse=True)
