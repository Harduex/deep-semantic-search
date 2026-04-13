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

    assert isinstance(results, dict)
    assert len(results) <= 3


def test_search_by_text(sample_images, tmp_metadata_dir, mock_clip_model):
    from deep_semantic_search.image_indexer import ImageIndexer

    indexer = ImageIndexer(
        image_list=sample_images,
        metadata_dir=tmp_metadata_dir / "clip",
    )
    indexer.run_index()

    searcher = ImageSearcher(indexer)
    results = searcher.search_by_text("a red image", n=2)

    assert isinstance(results, dict)
    assert len(results) <= 2
