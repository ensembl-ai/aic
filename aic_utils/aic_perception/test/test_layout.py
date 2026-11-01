"""Dataset layout and deterministic train/val assignment."""

from __future__ import annotations

from pathlib import Path

from aic_perception.config import load_config
from aic_perception.layout import (
    completed_ids,
    image_path,
    label_path,
    make_directories,
    scene_id_from_name,
    split_of,
)

CONFIG = load_config("collect")


def test_split_is_deterministic_and_matches_fraction() -> None:
    """Every worker assigns a scene to the same split, at the configured rate."""

    splits = [split_of(CONFIG, scene_id) for scene_id in range(5000)]
    assert splits == [split_of(CONFIG, scene_id) for scene_id in range(5000)]
    fraction = splits.count("val") / len(splits)
    assert abs(fraction - CONFIG["dataset"]["validation_fraction"]) < 0.02


def test_flat_paths_and_completion(tmp_path: Path) -> None:
    """Images and labels live flat per split; only labeled scenes count as done."""

    make_directories(tmp_path)
    image = image_path(tmp_path, "train", 42, "left")
    assert image == tmp_path / "train" / "images" / "scene_0000042_left.jpg"
    assert scene_id_from_name(image.name) == 42
    image.write_bytes(b"")
    assert completed_ids(tmp_path, "train") == []
    label_path(tmp_path, "train", 42).write_text("{}")
    (tmp_path / "train" / "labels" / ".scene_0000043.json.tmp").write_text("{}")
    assert completed_ids(tmp_path, "train") == [42]
