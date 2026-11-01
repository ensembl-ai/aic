"""Torch dataset over curated multiview scenes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_jpeg, read_file
from torchvision.transforms import v2

from aic_perception.layout import image_path, label_path
from aic_perception.schema import BoardSchema


def read_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """Read the curated manifest and check it matches the configured schema.

    Args:
        config: Validated training configuration.

    Returns:
        Parsed manifest.
    """

    path = Path(config["dataset"]["directory"]) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; run scripts/curate_dataset.py first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema"] != config["schema"]["name"]:
        raise ValueError(f"Manifest schema {manifest['schema']} != {config['schema']['name']}")
    return manifest


class SceneDataset(Dataset):
    """Three RGB images, camera geometry, and per-target pose labels per scene.

    Images stay ``uint8``; normalization happens on the GPU inside the model.
    """

    def __init__(self, config: dict[str, Any], split: str, scene_ids: list[int], augment: bool):
        """Index the usable scenes of one split.

        Args:
            config: Validated training configuration.
            split: ``train`` or ``val``.
            scene_ids: Usable scene IDs of the split from the manifest.
            augment: Whether to apply photometric augmentation.
        """

        self.root = Path(config["dataset"]["directory"])
        self.split = split
        self.schema = BoardSchema.from_config(config)
        self.scene_ids = scene_ids
        settings = config["augmentation"]
        self.jitter = (
            v2.ColorJitter(settings["brightness"], settings["contrast"], settings["saturation"])
            if augment
            else None
        )

    def __len__(self) -> int:
        """Number of scenes in the split."""

        return len(self.scene_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one scene.

        Args:
            index: Position in the split.

        Returns:
            ``images`` (C, 3, H, W) uint8, ``intrinsics`` (C, 3, 3),
            ``base_from_camera`` (C, 4, 4), ``present`` (Q,), and
            ``base_from_target`` (Q, 4, 4; identity where not present).
        """

        scene_id = self.scene_ids[index]
        label = json.loads(label_path(self.root, self.split, scene_id).read_text(encoding="utf-8"))
        images = []
        for name in self.schema.cameras:
            image = decode_jpeg(read_file(str(image_path(self.root, self.split, scene_id, name))))
            images.append(self.jitter(image) if self.jitter else image)
        cameras = [label["cameras"][name] for name in self.schema.cameras]
        targets = [label["targets"][target.id] for target in self.schema.targets]
        poses = np.stack(
            [np.asarray(t["base_from_target"]) if t["visible"] else np.eye(4) for t in targets]
        )
        return {
            "images": torch.stack(images),
            "intrinsics": torch.tensor([c["intrinsics"] for c in cameras], dtype=torch.float32),
            "base_from_camera": torch.tensor(
                [c["base_from_camera"] for c in cameras], dtype=torch.float32
            ),
            "present": torch.tensor([float(t["visible"]) for t in targets]),
            "base_from_target": torch.from_numpy(poses).float(),
        }
