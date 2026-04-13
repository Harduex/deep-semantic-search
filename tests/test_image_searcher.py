"""Tests for ImageSearcher with mocked dependencies."""



from deep_semantic_search.image_searcher import ImageSearcher


def test_search_by_image(sample_images, tmp_metadata_dir, mock_clip_model):
    from deep_semantic_search.image_indexer import ImageIndexer

    indexer = ImageIndexer(
        image_list=sample_images,
        metadata_dir=tmp_metadata_dir / "clip",
    )
    indexer.run_index()

    searcher = ImageSearcher(indexer)
    results = searcher.search_by_image(sample_images[0], n=3)

    assert isinstance(results, list)
    assert len(results) <= 3
    assert all(isinstance(r, dict) for r in results)
    assert all("rank" in r and "path" in r and "score" in r for r in results)


def test_search_by_text(sample_images, tmp_metadata_dir, mock_clip_model):
    from deep_semantic_search.image_indexer import ImageIndexer

    indexer = ImageIndexer(
        image_list=sample_images,
        metadata_dir=tmp_metadata_dir / "clip",
    )
    indexer.run_index()

    searcher = ImageSearcher(indexer)
    results = searcher.search_by_text("a red image", n=2)

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(r, dict) for r in results)
    assert all("rank" in r and "path" in r and "score" in r for r in results)
