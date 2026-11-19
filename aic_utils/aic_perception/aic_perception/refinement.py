"""Close-range refiner: where the target port is relative to the held plug.

The scene estimator gives absolute port poses to a few millimetres. The final
approach needs the plug-to-port offset, which the wrist cameras see directly
once the plug hovers a few centimetres above the port. This module learns that
offset from the aic_v3 close views, whose labels hold the true plug and port
poses.

Inputs: from each camera, a fixed ``crop.size`` square of the stored image.
The plug is rigid in the gripper, so it projects to almost the same pixel in
every image of one plug type; the crop windows are centred there (median over
the training set), so both the plug tip and a port within the selected range
are inside every crop.

Outputs, all in the optical frame of ``selection.reference_camera``, which is
rigid with the gripper:

- ``offset``: port entrance minus plug tip (m). Moving the TCP by this
  offset puts the plug on the port, whatever the grasp calibration says.
- ``plug``: plug tip position (m), i.e. the current grasp.
- ``rotation``: rotation vector (rad) taking the plug frame onto the port
  frame (plug frame to port frame).
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_jpeg, read_file
from torchvision.transforms import v2
from transforms3d.axangles import mat2axangle

from aic_perception.backbones import build_backbone
from aic_perception.config import PLUG_TYPES
from aic_perception.dataset import read_manifest
from aic_perception.layout import image_path, label_path
from aic_perception.training import append_history, should_stop

LATERAL_BANDS_MM = (2.0, 5.0, 10.0)
HEIGHT_BANDS_M = (0.02, 0.04)


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a 3x3 rotation, in radians."""

    axis, angle = mat2axangle(rotation)
    return np.asarray(axis) * angle


def label_sample(label: dict[str, Any], camera: str) -> dict[str, Any] | None:
    """Refiner targets of one labeled scene, or ``None`` if it has no plug or port.

    Args:
        label: Scene label written by the collector.
        camera: Reference camera name.

    Returns:
        Plug type, targets in the reference camera frame, the plug tip pixel
        in every camera, and the plug height and lateral distance above the
        port.
    """

    scene = label["scene"]
    held = label.get("held_plug")
    if scene.get("viewpoint", {}).get("mode") != "close" or not held or held["base_from_plug"] is None:
        return None
    target = label["targets"][scene["target"]]
    if not target["spawned"]:
        return None
    plug = np.asarray(held["base_from_plug"])
    port = np.asarray(target["base_from_target"])
    base_from_camera = np.asarray(label["cameras"][camera]["base_from_camera"])
    rotation = base_from_camera[:3, :3].T
    relative = port[:3, 3] - plug[:3, 3]
    pixels = {}
    for name, info in label["cameras"].items():
        camera_from_base = np.linalg.inv(np.asarray(info["base_from_camera"]))
        point = camera_from_base[:3, :3] @ plug[:3, 3] + camera_from_base[:3, 3]
        uv = np.asarray(info["intrinsics"]) @ point
        pixels[name] = (uv[:2] / uv[2]).tolist()
    plug_to_port = rotation_vector(plug[:3, :3].T @ port[:3, :3])
    return {
        "plug_type": held["plug_type"],
        "offset": (rotation @ relative).tolist(),
        "plug": (rotation @ (plug[:3, 3] - base_from_camera[:3, 3])).tolist(),
        "rotation": (rotation @ plug[:3, :3] @ plug_to_port).tolist(),
        "pixels": pixels,
        "height": float(-relative[2]),
        "lateral": float(np.hypot(relative[0], relative[1])),
        "angle": float(np.linalg.norm(plug_to_port)),
    }


def select_samples(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    """Close views of one split, over every configured dataset, that the refiner can learn from.

    Keeps usable scenes (per each dataset's curated manifest) with the plug
    above the target port within ``selection.max_height_m`` and
    ``selection.max_lateral_m``, and aligned within
    ``selection.max_rotation_deg``; views where the arm did not reach its
    planned pose fall outside these bounds. Datasets are collected with
    different seeds, so their boards, and hence their splits, never overlap.

    Args:
        config: Validated refiner configuration.
        split: ``train`` or ``val``.

    Returns:
        Samples with their dataset ``root`` and scene ID.
    """

    selection = config["selection"]
    samples = []
    for directory in config["datasets"]:
        root = Path(directory)
        for scene_id in read_manifest(config, root)[split]:
            label = json.loads(label_path(root, split, scene_id).read_text(encoding="utf-8"))
            sample = label_sample(label, selection["reference_camera"])
            if (
                sample is not None
                and 0.0 < sample["height"] <= selection["max_height_m"]
                and sample["lateral"] <= selection["max_lateral_m"]
                and math.degrees(sample["angle"]) <= selection["max_rotation_deg"]
            ):
                samples.append({**sample, "root": str(root), "scene_id": scene_id})
    return samples


def crop_windows(
    samples: list[dict[str, Any]], cameras: list[str], width: int, height: int, size: int
) -> dict[str, dict[str, list[int]]]:
    """Top-left corner of each camera's crop, per plug type.

    Each window is centred on the median plug tip pixel of the training
    samples and shifted, if needed, to stay inside the image.

    Returns:
        ``{plug_type: {camera: [u0, v0]}}``.
    """

    windows: dict[str, dict[str, list[int]]] = {}
    for plug_type in PLUG_TYPES:
        pixels = [s["pixels"] for s in samples if s["plug_type"] == plug_type]
        if not pixels:
            raise ValueError(f"No training samples hold a {plug_type} plug")
        windows[plug_type] = {}
        for camera in cameras:
            u, v = np.median([p[camera] for p in pixels], axis=0)
            windows[plug_type][camera] = [
                int(np.clip(round(u - size / 2), 0, width - size)),
                int(np.clip(round(v - size / 2), 0, height - size)),
            ]
    return windows


def crop(image: torch.Tensor, window: list[int], size: int) -> torch.Tensor:
    """Cut a ``size`` square at ``window`` from a ``(3, H, W)`` image."""

    u0, v0 = window
    return image[:, v0 : v0 + size, u0 : u0 + size]


def target_statistics(samples: list[dict[str, Any]]) -> dict[str, list[float]]:
    """Per-axis mean and standard deviation of the offset and plug targets."""

    statistics = {}
    for key in ("offset", "plug"):
        values = np.asarray([s[key] for s in samples])
        statistics[f"{key}_mean"] = values.mean(0).tolist()
        statistics[f"{key}_std"] = np.maximum(values.std(0), 1e-4).tolist()
    return statistics


class RefinerDataset(Dataset):
    """Camera crops around the plug with plug-to-port targets."""

    def __init__(
        self,
        config: dict[str, Any],
        split: str,
        samples: list[dict[str, Any]],
        windows: dict[str, dict[str, list[int]]],
        augment: bool,
    ):
        """Index selected samples of one split.

        Args:
            config: Validated refiner configuration.
            split: ``train`` or ``val``.
            samples: Output of ``select_samples``.
            windows: Output of ``crop_windows``.
            augment: Whether to apply photometric augmentation.
        """

        self.split = split
        self.samples = samples
        self.windows = windows
        self.size = config["crop"]["size"]
        self.cameras = config["schema"]["cameras"]
        settings = config["augmentation"]
        self.jitter = (
            v2.ColorJitter(settings["brightness"], settings["contrast"], settings["saturation"])
            if augment
            else None
        )

    def __len__(self) -> int:
        """Number of samples."""

        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Load one sample.

        Returns:
            ``crops`` (C, 3, S, S) uint8, ``plug_index``, ``offset`` and
            ``plug`` (3,) in meters, ``rotation`` (3,) in radians, and the
            true ``height`` and ``lateral`` distance for grouped errors.
        """

        sample = self.samples[index]
        windows = self.windows[sample["plug_type"]]
        crops = []
        for camera in self.cameras:
            image = decode_jpeg(read_file(str(image_path(sample["root"], self.split, sample["scene_id"], camera))))
            patch = crop(image, windows[camera], self.size)
            crops.append(self.jitter(patch) if self.jitter else patch)
        return {
            "crops": torch.stack(crops),
            "plug_index": torch.tensor(PLUG_TYPES.index(sample["plug_type"])),
            "offset": torch.tensor(sample["offset"], dtype=torch.float32),
            "plug": torch.tensor(sample["plug"], dtype=torch.float32),
            "rotation": torch.tensor(sample["rotation"], dtype=torch.float32),
            "height": torch.tensor(sample["height"]),
            "lateral": torch.tensor(sample["lateral"]),
        }


class Refiner(nn.Module):
    """Shared backbone per crop, transformer fusion across cameras, one readout."""

    def __init__(
        self,
        model_config: dict[str, Any],
        cameras: int,
        crop_size: int,
        statistics: dict[str, list[float]],
    ):
        """Build the network.

        Args:
            model_config: ``model`` section of the refiner configuration.
            cameras: Number of cameras (crops per sample).
            crop_size: Crop side length in pixels.
            statistics: Output of ``target_statistics``.
        """

        super().__init__()
        d_model = model_config["d_model"]
        self.backbone = build_backbone(model_config)
        with torch.no_grad():
            cells = self.backbone(torch.zeros(1, 3, crop_size, crop_size, dtype=torch.uint8)).shape[-2:]
        self.projection = nn.Conv2d(self.backbone.channels, d_model, kernel_size=1)
        self.position = nn.Parameter(torch.zeros(cameras, cells[0] * cells[1], d_model))
        self.plug_embedding = nn.Embedding(len(PLUG_TYPES), d_model)
        self.readout = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.readout, std=0.02)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=model_config["heads"],
                dim_feedforward=model_config["feedforward"],
                dropout=model_config["dropout"],
                batch_first=True,
                norm_first=True,
            ),
            num_layers=model_config["layers"],
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 9))
        for key, value in statistics.items():
            self.register_buffer(key, torch.tensor(value))

    def backbone_parameters(self) -> list[nn.Parameter]:
        """Parameters of the pretrained image encoder."""

        return list(self.backbone.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        """Parameters trained from scratch."""

        backbone = {id(parameter) for parameter in self.backbone.parameters()}
        return [p for p in self.parameters() if id(p) not in backbone]

    def forward(self, crops: torch.Tensor, plug_index: torch.Tensor) -> dict[str, torch.Tensor]:
        """Estimate the plug-to-port offset, the plug position, and the rotation.

        Args:
            crops: ``(B, C, 3, S, S)`` uint8 crops.
            plug_index: ``(B,)`` index into ``PLUG_TYPES``.

        Returns:
            ``offset`` and ``plug`` (B, 3) in meters and ``rotation`` (B, 3)
            in radians, in the reference camera frame.
        """

        batch, cameras = crops.shape[:2]
        features = self.projection(self.backbone(crops.flatten(0, 1)))
        tokens = features.flatten(2).transpose(1, 2).reshape(batch, cameras, -1, features.shape[1])
        tokens = (tokens + self.position).flatten(1, 2) + self.plug_embedding(plug_index)[:, None]
        tokens = torch.cat((self.readout.expand(batch, -1, -1), tokens), dim=1)
        outputs = self.head(self.encoder(tokens)[:, 0]).float()
        return {
            "offset": self.offset_mean + outputs[:, 0:3] * self.offset_std,
            "plug": self.plug_mean + outputs[:, 3:6] * self.plug_std,
            "rotation": outputs[:, 6:9] * 0.1,
        }


def refiner_losses(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], weights: dict[str, float]
) -> tuple[torch.Tensor, dict[str, float]]:
    """L1 losses in millimetres and degrees, weighted per output."""

    offset = (outputs["offset"] - batch["offset"]).abs().mean() * 1000.0
    plug = (outputs["plug"] - batch["plug"]).abs().mean() * 1000.0
    rotation = torch.rad2deg((outputs["rotation"] - batch["rotation"]).abs()).mean()
    total = weights["offset"] * offset + weights["plug"] * plug + weights["rotation"] * rotation
    return total, {
        "loss/total": total.item(),
        "loss/offset_mm": offset.item(),
        "loss/plug_mm": plug.item(),
        "loss/rotation_deg": rotation.item(),
    }


def band_name(value: float, edges: tuple[float, ...], unit: str, scale: float) -> str:
    """Readable band label of a value, e.g. ``2-5mm``."""

    low = 0.0
    for edge in edges:
        if value < edge:
            return f"{low * scale:g}-{edge * scale:g}{unit}"
        low = edge
    return f">{low * scale:g}{unit}"


@torch.no_grad()
def evaluate_refiner(model: Refiner, loader: DataLoader, device: torch.device, bf16: bool) -> dict[str, Any]:
    """Errors of the refiner per plug type, starting misalignment, and height.

    ``uncorrected_mm`` is the true lateral misalignment the refiner had to
    measure; ``lateral_mm`` is what would remain after moving by the
    estimate. Lateral and vertical are taken along the base frame, where the
    board is level.

    Returns:
        Summary with ``per_type``, ``per_misalignment``, and ``per_height``.
    """

    model.eval()
    rows = []
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            outputs = model(batch["crops"], batch["plug_index"])
        error = (outputs["offset"] - batch["offset"]).float().cpu().numpy()
        plug = (outputs["plug"] - batch["plug"]).float().norm(dim=-1).cpu().numpy()
        rotation = torch.rad2deg((outputs["rotation"] - batch["rotation"]).float().norm(dim=-1)).cpu().numpy()
        for index in range(error.shape[0]):
            rows.append(
                {
                    "type": PLUG_TYPES[int(batch["plug_index"][index])],
                    "offset_mm": 1000.0 * float(np.linalg.norm(error[index])),
                    "plug_mm": 1000.0 * float(plug[index]),
                    "rotation_deg": float(rotation[index]),
                    "uncorrected_mm": 1000.0 * float(batch["lateral"][index]),
                    "height": float(batch["height"][index]),
                }
            )

    def describe(group: list[dict[str, Any]]) -> dict[str, float]:
        """Mean, median, and p90 of the errors in one group."""

        if not group:
            return {"samples": 0}
        offset = np.asarray([r["offset_mm"] for r in group])
        return {
            "samples": len(group),
            "offset_mean_mm": float(offset.mean()),
            "offset_median_mm": float(np.median(offset)),
            "offset_p90_mm": float(np.percentile(offset, 90)),
            "offset_within_0_5mm": float((offset < 0.5).mean()),
            "offset_within_1mm": float((offset < 1.0).mean()),
            "plug_mean_mm": float(np.mean([r["plug_mm"] for r in group])),
            "rotation_mean_deg": float(np.mean([r["rotation_deg"] for r in group])),
            "uncorrected_mean_mm": float(np.mean([r["uncorrected_mm"] for r in group])),
        }

    summary: dict[str, Any] = {"samples": len(rows), "per_type": {}, "per_misalignment": {}, "per_height": {}}
    for plug_type in PLUG_TYPES:
        group = [r for r in rows if r["type"] == plug_type]
        summary["per_type"][plug_type] = describe(group)
        for row in group:
            row["misalignment"] = band_name(row["uncorrected_mm"] / 1000.0, tuple(e / 1000.0 for e in LATERAL_BANDS_MM), "mm", 1000.0)
            row["height_band"] = band_name(row["height"], HEIGHT_BANDS_M, "cm", 100.0)
        for key, target in (("misalignment", "per_misalignment"), ("height_band", "per_height")):
            for band in dict.fromkeys(r[key] for r in group):
                summary[target][f"{plug_type}/{band}"] = describe([r for r in group if r[key] == band])
    return summary


def build_refiner(config: dict[str, Any], statistics: dict[str, list[float]]) -> Refiner:
    """Construct the refiner for a configuration."""

    return Refiner(config["model"], len(config["schema"]["cameras"]), config["crop"]["size"], statistics)


def load_refiner(path: str | Path, device: str) -> tuple[Refiner, dict[str, Any]]:
    """Load a trained refiner and its training-time metadata.

    Returns:
        Model in eval mode and the checkpoint metadata: ``config``,
        ``windows``, and ``statistics``.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config["model"]["pretrained_backbone"] = False
    model = build_refiner(config, checkpoint["statistics"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def make_refiner_loader(
    config: dict[str, Any], split: str, samples: list[dict[str, Any]], windows: dict[str, dict[str, list[int]]]
) -> DataLoader:
    """Data loader of one split (shuffled and augmented for training)."""

    train = split == "train"
    optimization = config["optimization"]
    workers = optimization["num_workers"]
    return DataLoader(
        RefinerDataset(config, split, samples, windows, augment=train),
        batch_size=optimization["batch_size"],
        shuffle=train,
        drop_last=train and len(samples) > optimization["batch_size"],
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def flatten_refiner_summary(summary: dict[str, Any]) -> dict[str, float]:
    """Headline refiner metrics for W&B."""

    flat = {}
    for group in ("per_type", "per_misalignment"):
        for name, metrics in summary[group].items():
            for key in ("offset_mean_mm", "offset_within_1mm", "plug_mean_mm", "rotation_mean_deg"):
                if key in metrics:
                    flat[f"validation/{name}/{key}"] = metrics[key]
    return flat


def train_refiner(config: dict[str, Any]) -> Path:
    """Train the refiner and write checkpoints and metrics under the run directory.

    Args:
        config: Validated refiner configuration.

    Returns:
        Path of the best checkpoint.
    """

    run = config["run"]
    random.seed(run["seed"])
    np.random.seed(run["seed"])
    torch.manual_seed(run["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(run["device"])
    optimization = config["optimization"]
    bf16 = optimization["bf16"]

    samples = {split: select_samples(config, split) for split in ("train", "val")}
    images = config["images"]
    windows = crop_windows(samples["train"], config["schema"]["cameras"], images["width"], images["height"], config["crop"]["size"])
    statistics = target_statistics(samples["train"])
    train_loader = make_refiner_loader(config, "train", samples["train"], windows)
    validation_loader = make_refiner_loader(config, "val", samples["val"], windows)
    model = build_refiner(config, statistics).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone_parameters(), "lr": optimization["backbone_learning_rate"]},
            {"params": model.head_parameters(), "lr": optimization["learning_rate"]},
        ],
        weight_decay=optimization["weight_decay"],
    )
    total_steps = optimization["epochs"] * len(train_loader)
    warmup = optimization["warmup_steps"]

    def schedule(step: int) -> float:
        """Linear warmup followed by cosine decay to zero."""

        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    run_directory = Path(run["directory"]) / run["name"]
    checkpoints = run_directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    (run_directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    counts = {split: {t: sum(s["plug_type"] == t for s in samples[split]) for t in PLUG_TYPES} for split in samples}
    logging = config["logging"]
    wandb.init(project=logging["wandb_project"], name=run["name"], mode=logging["wandb_mode"], dir=str(run_directory), config=config)
    print(f"Refiner samples {counts}, crop windows {windows}, {total_steps} steps, run directory {run_directory}")

    history = run_directory / "history.jsonl"
    append_history(history, {"kind": "samples", "counts": counts, "windows": windows})
    best = math.inf
    errors: list[float] = []
    step = 0
    for epoch in range(optimization["epochs"]):
        model.train()
        started = time.perf_counter()
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                outputs = model(batch["crops"], batch["plug_index"])
            loss, values = refiner_losses(outputs, batch, config["loss"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optimization["gradient_clip"])
            optimizer.step()
            scheduler.step()
            step += 1
            if step % logging["log_every_steps"] == 0:
                wandb.log(
                    {**values, "train/gradient_norm": gradient_norm.item(), "train/learning_rate": scheduler.get_last_lr()[1], "train/epoch": epoch},
                    step=step,
                )
                append_history(
                    history,
                    {"kind": "train", "epoch": epoch, "step": step, "learning_rate": scheduler.get_last_lr()[1], **values},
                )
                print(
                    f"epoch {epoch} step {step}/{total_steps} loss {values['loss/total']:.3f} offset {values['loss/offset_mm']:.2f} mm "
                    f"plug {values['loss/plug_mm']:.2f} mm rotation {values['loss/rotation_deg']:.2f} deg "
                    f"({time.perf_counter() - started:.0f} s)",
                    flush=True,
                )
        summary = evaluate_refiner(model, validation_loader, device, bf16)
        error = float(np.mean([summary["per_type"][t]["offset_mean_mm"] for t in PLUG_TYPES if summary["per_type"][t]["samples"]]))
        wandb.log({**flatten_refiner_summary(summary), "validation/offset_mm": error}, step=step)
        append_history(history, {"kind": "validation", "epoch": epoch, "step": step, "offset_mm": error, **summary})
        print(
            f"epoch {epoch} validation offset error "
            + ", ".join(f"{t} {summary['per_type'][t].get('offset_mean_mm', float('nan')):.2f} mm" for t in PLUG_TYPES),
            flush=True,
        )
        state = {
            "model": model.state_dict(),
            "config": config,
            "windows": windows,
            "statistics": statistics,
            "epoch": epoch,
            "validation": summary,
        }
        torch.save(state, checkpoints / "last.pt")
        if error < best:
            best = error
            torch.save(state, checkpoints / "best.pt")
            (run_directory / "best_validation.json").write_text(json.dumps({"epoch": epoch, **summary}, indent=2) + "\n", encoding="utf-8")
        errors.append(error)
        if should_stop(errors, optimization["early_stopping_patience"], optimization["early_stopping_min_delta"]):
            print(f"Early stop after epoch {epoch}: best {best:.3f} mm", flush=True)
            append_history(history, {"kind": "early_stop", "epoch": epoch, "best_mm": best})
            break
    wandb.finish()
    return checkpoints / "best.pt"


class RefinerRuntime:
    """Loaded refiner with the same preprocessing as the collector."""

    def __init__(self, checkpoint: str | Path, device: str):
        """Load a checkpoint.

        Args:
            checkpoint: Path to a refiner checkpoint.
            device: Torch device.
        """

        self.device = torch.device(device)
        self.model, metadata = load_refiner(checkpoint, device)
        self.config = metadata["config"]
        self.windows = metadata["windows"]
        self.cameras = self.config["schema"]["cameras"]
        self.reference = self.cameras.index(self.config["selection"]["reference_camera"])
        self.size = self.config["crop"]["size"]
        self.width = self.config["images"]["width"]
        self.height = self.config["images"]["height"]

    @torch.no_grad()
    def predict(self, images: list[np.ndarray], base_from_camera: list[np.ndarray], plug_type: str) -> dict[str, np.ndarray]:
        """Measure the port relative to the held plug from one observation.

        Args:
            images: ``(H, W, 3)`` uint8 RGB images in schema camera order, at
                the camera's native resolution.
            base_from_camera: 4x4 optical-frame poses at the image stamp.
            plug_type: ``sfp`` or ``sc``.

        Returns:
            In ``base_link``: ``offset`` (port minus plug), ``plug`` and
            ``port`` positions in meters, and ``rotation`` (plug frame to
            port frame, as a rotation vector in radians).
        """

        crops = []
        for camera, image in zip(self.cameras, images, strict=True):
            pixels = torch.from_numpy(np.ascontiguousarray(image)).to(self.device).permute(2, 0, 1)[None].float()
            resized = F.interpolate(pixels, size=(self.height, self.width), mode="area")
            stored = resized.round().clamp(0, 255).to(torch.uint8)[0]
            crops.append(crop(stored, self.windows[plug_type][camera], self.size))
        outputs = self.model(torch.stack(crops)[None], torch.tensor([PLUG_TYPES.index(plug_type)], device=self.device))
        reference = np.asarray(base_from_camera[self.reference])
        rotation = reference[:3, :3]
        offset = rotation @ outputs["offset"][0].cpu().numpy()
        plug = rotation @ outputs["plug"][0].cpu().numpy() + reference[:3, 3]
        return {
            "offset": offset,
            "plug": plug,
            "port": plug + offset,
            "rotation": rotation @ outputs["rotation"][0].cpu().numpy(),
        }
