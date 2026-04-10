"""Tests for LoadTextData."""

from deep_semantic_search.text_loader import LoadTextData


def test_from_folder(tmp_path):
    (tmp_path / "a.txt").write_text("Hello world", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Foo bar", encoding="utf-8")
    (tmp_path / "c.csv").write_text("not loaded", encoding="utf-8")

    loader = LoadTextData()
    result = loader.from_folder(str(tmp_path))

    assert len(result) == 2
    assert any("Hello world" in v for v in result.values())


def test_from_folder_with_limit(tmp_path):
    for i in range(10):
        (tmp_path / f"doc_{i}.txt").write_text(f"Document {i}", encoding="utf-8")

    loader = LoadTextData()
    result = loader.from_folder(str(tmp_path), corpus_count=3)

    assert len(result) == 3


def test_from_csv(tmp_path):
    import pandas as pd

    csv_path = tmp_path / "texts.csv"
    pd.DataFrame({"content": ["text1", "text2", None, "text3"]}).to_csv(csv_path, index=False)

    loader = LoadTextData()
    result = loader.from_csv(str(csv_path), column_name="content")

    assert len(result) == 3  # None dropped


def test_from_folder_html(tmp_path):
    (tmp_path / "page.html").write_text("<html><body><p>HTML content</p></body></html>", encoding="utf-8")

    loader = LoadTextData()
    result = loader.from_folder(str(tmp_path))

    assert len(result) == 1
    assert "HTML content" in list(result.values())[0]
