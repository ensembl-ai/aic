"""Dataset curation: validate scenes, summarize labels, and draw previews.

The train/val split is fixed at collection time. Curation checks every
scene of both splits, rejects frames repeated across scenes (a frozen
renderer), writes ``manifest.json`` (usable scene IDs per split,
translation normalization from the train split, and label statistics), and
saves projection previews that overlay every visible ground-truth target on
its images for a visual label check.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from aic_perception.layout import SPLITS, completed_ids, image_path, label_path, scene_name
from aic_perception.scene_sampler import rail_order_names
from aic_perception.schema import BoardSchema

TYPE_COLORS = {"board": (255, 64, 64), "sfp_port": (64, 220, 64), "sc_port": (64, 160, 255)}


def check_scene(
    root: Path, split: str, scene_id: int, cameras: tuple[str, ...], board_id: str
) -> tuple[str, str]:
    """Validate one scene.

    Args:
        root: Dataset root.
        split: Split name.
        scene_id: Scene ID.
        cameras: Required camera names.
        board_id: Target ID of the board, which must be visible.

    Returns:
        Status (``ok``, ``board_not_visible``, ``sc_ports_out_of_rail_order``,
        or ``corrupt``) and a digest of
        the image bytes used to find frames repeated across scenes.
    """

    digest = hashlib.md5()
    try:
        label = json.loads(label_path(root, split, scene_id).read_text(encoding="utf-8"))
        for name in cameras:
            path = image_path(root, split, scene_id, name)
            digest.update(path.read_bytes())
            with Image.open(path) as image:
                image.verify()
    except (OSError, ValueError):
        return "corrupt", ""
    components = label["scene"]["components"]
    if any(old != new for old, new in rail_order_names(components).items()):
        return "sc_ports_out_of_rail_order", digest.hexdigest()
    status = "ok" if label["targets"][board_id]["visible"] else "board_not_visible"
    return status, digest.hexdigest()


def draw_preview(root: Path, split: str, scene_id: int, schema: BoardSchema, output: Path) -> None:
    """Save the camera images side by side with projected target origins.

    Args:
        root: Dataset root.
        split: Split name.
        scene_id: Scene ID.
        schema: Board schema.
        output: Preview image path.
    """

    label = json.loads(label_path(root, split, scene_id).read_text(encoding="utf-8"))
    panels = []
    for name in schema.cameras:
        image = Image.open(image_path(root, split, scene_id, name)).convert("RGB")
        draw = ImageDraw.Draw(image)
        for target in schema.targets:
            item = label["targets"][target.id]
            if not item["spawned"]:
                continue
            u, v, depth = item["pixels"][name]
            if depth <= 0.0 or not (0 <= u < image.width and 0 <= v < image.height):
                continue
            color = TYPE_COLORS.get(target.type, (255, 255, 0))
            draw.ellipse((u - 4, v - 4, u + 4, v + 4), outline=color, width=2)
            draw.text((u + 6, v - 6), target.id.replace("nic_card_mount_", "nic"), fill=color)
        draw.text((6, 6), name, fill=(255, 255, 0))
        panels.append(image)
    sheet = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)))
    offset = 0
    for panel in panels:
        sheet.paste(panel, (offset, 0))
        offset += panel.width
    sheet.save(output, quality=90)


def label_statistics(
    root: Path, schema: BoardSchema, scene_ids: list[int]
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Summarize target visibility and translation normalization of the train split.

    Args:
        root: Dataset root.
        schema: Board schema.
        scene_ids: Usable train scene IDs.

    Returns:
        Per-target visible fractions and the translation mean/std over all
        visible targets in the base frame.
    """

    visible = {target.id: 0 for target in schema.targets}
    translations = []
    for scene_id in scene_ids:
        label = json.loads(label_path(root, "train", scene_id).read_text(encoding="utf-8"))
        for target in schema.targets:
            item = label["targets"][target.id]
            if item["visible"]:
                visible[target.id] += 1
                translations.append(np.asarray(item["base_from_target"])[:3, 3])
    points = np.asarray(translations)
    fractions = {key: count / max(1, len(scene_ids)) for key, count in visible.items()}
    normalization = {
        "mean": points.mean(axis=0).tolist(),
        "std": np.maximum(points.std(axis=0), 1e-3).tolist(),
    }
    return fractions, normalization


def curate(config: dict[str, Any]) -> dict[str, Any]:
    """Validate both splits and write ``manifest.json`` and previews.

    Args:
        config: Validated training configuration.

    Returns:
        The written manifest.
    """

    root = Path(config["dataset"]["directory"])
    schema = BoardSchema.from_config(config)
    board_id = next(target.id for target in schema.targets if target.type == "board")
    checked: dict[str, list[tuple[int, str, str]]] = {}
    with ProcessPoolExecutor() as pool:
        for split in SPLITS:
            ids = completed_ids(root, split)
            count = len(ids)
            results = list(
                pool.map(
                    check_scene,
                    [root] * count,
                    [split] * count,
                    ids,
                    [schema.cameras] * count,
                    [board_id] * count,
                    chunksize=64,
                )
            )
            checked[split] = [(i, status, digest) for i, (status, digest) in zip(ids, results)]
    occurrences = Counter(digest for rows in checked.values() for _, _, digest in rows if digest)
    usable: dict[str, list[int]] = {}
    rejected: dict[str, dict[str, int]] = {}
    for split, rows in checked.items():
        usable[split], rejected[split] = [], {}
        for scene_id, status, digest in rows:
            if status == "ok" and occurrences[digest] > 1:
                status = "repeated_frame"
            if status == "ok":
                usable[split].append(scene_id)
            else:
                rejected[split][status] = rejected[split].get(status, 0) + 1
    if not usable["train"] or not usable["val"]:
        raise ValueError(f"Both splits need usable scenes; found {[len(v) for v in usable.values()]}")

    visible_fraction, normalization = label_statistics(root, schema, usable["train"])
    manifest = {
        "schema": schema.name,
        "train": usable["train"],
        "val": usable["val"],
        "normalization": normalization,
        "statistics": {
            "usable": {split: len(ids) for split, ids in usable.items()},
            "rejected": rejected,
            "train_visible_fraction": visible_fraction,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    previews = root / "previews"
    previews.mkdir(exist_ok=True)
    for scene_id in usable["val"][: config["curation"]["preview_count"]]:
        draw_preview(root, "val", scene_id, schema, previews / f"{scene_name(scene_id)}.jpg")
    return manifest
