"""On-disk dataset layout shared by the collector, curation, and training.

```text
<dataset>/
  provenance.json
  manifest.json                         (written by curation)
  train/images/scene_0000123_left.jpg   (flat: one file per scene and camera)
  train/labels/scene_0000123.json       (written last; marks the scene complete)
  val/images/...
  val/labels/...
```

The split of a scene is a pure function of ``(seed, scene_id)``, so workers
write directly into their split and resumed collections stay consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

SPLITS = ("train", "val")
SPLIT_STREAM = 1


def scene_name(scene_id: int) -> str:
    """Return the file stem shared by every file of one scene."""

    return f"scene_{scene_id:07d}"


def scene_id_from_name(name: str) -> int:
    """Parse a scene ID from a label or image file name."""

    return int(name.split("_")[1].split(".")[0])


def split_of(config: dict[str, Any], scene_id: int) -> str:
    """Assign a scene to ``train`` or ``val`` deterministically.

    Args:
        config: Configuration with ``collection.seed`` and
            ``dataset.validation_fraction``.
        scene_id: Global scene index.

    Returns:
        Split name.
    """

    draw = np.random.default_rng(
        [config["collection"]["seed"], scene_id, SPLIT_STREAM]
    ).uniform()
    return "val" if draw < config["dataset"]["validation_fraction"] else "train"


def image_path(root: str | Path, split: str, scene_id: int, camera: str) -> Path:
    """Path of one camera image."""

    return Path(root) / split / "images" / f"{scene_name(scene_id)}_{camera}.jpg"


def label_path(root: str | Path, split: str, scene_id: int) -> Path:
    """Path of one scene label."""

    return Path(root) / split / "labels" / f"{scene_name(scene_id)}.json"


def make_directories(root: str | Path) -> None:
    """Create the image and label directories of every split."""

    for split in SPLITS:
        for kind in ("images", "labels"):
            (Path(root) / split / kind).mkdir(parents=True, exist_ok=True)


def completed_ids(root: str | Path, split: str) -> list[int]:
    """Sorted IDs of the scenes in a split whose label has been written."""

    labels = Path(root) / split / "labels"
    if not labels.is_dir():
        return []
    return sorted(
        scene_id_from_name(path.name)
        for path in labels.iterdir()
        if path.suffix == ".json"
    )
