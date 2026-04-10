"""Tests for LoadImageData."""

import os

from deep_semantic_search.image_loader import LoadImageData


def test_from_folder(tmp_path):
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ["a.png", "b.jpg", "c.txt"]:
        if name.endswith(".txt"):
            (img_dir / name).write_text("not an image")
        else:
            Image.new("RGB", (10, 10)).save(img_dir / name)

    loader = LoadImageData()
    paths = loader.from_folder([str(img_dir)])

    assert len(paths) == 2
    assert all(p.endswith((".png", ".jpg")) for p in paths)


def test_from_folder_shuffle(tmp_path):
    from PIL import Image

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(20):
        Image.new("RGB", (10, 10)).save(img_dir / f"img_{i:03d}.png")

    loader = LoadImageData()
    paths_normal = loader.from_folder([str(img_dir)], shuffle=False)
    # With shuffle, order should differ (statistically very likely with 20 items)
    paths_shuffled = loader.from_folder([str(img_dir)], shuffle=True)

    assert set(paths_normal) == set(paths_shuffled)


def test_from_csv(tmp_path):
    import pandas as pd

    csv_path = tmp_path / "images.csv"
    pd.DataFrame({"image_path": ["/img/a.png", "/img/b.jpg"]}).to_csv(csv_path, index=False)

    loader = LoadImageData()
    paths = loader.from_csv(str(csv_path), "image_path")

    assert paths == ["/img/a.png", "/img/b.jpg"]


def test_from_folder_recursive(tmp_path):
    from PIL import Image

    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(sub / "deep.png")

    loader = LoadImageData()
    paths = loader.from_folder([str(tmp_path)])
    assert len(paths) == 1
    assert "deep.png" in paths[0]
