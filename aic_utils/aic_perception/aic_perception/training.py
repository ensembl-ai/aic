"""Supervised training of the scene estimator with Weights & Biases logging."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from aic_perception.dataset import SceneDataset, read_manifest
from aic_perception.losses import ErrorAccumulator, compute_losses
from aic_perception.model import SceneEstimator
from aic_perception.schema import BoardSchema

PORT_TYPES = ("sfp_port", "sc_port")


def build_model(
    config: dict[str, Any], normalization: dict[str, list[float]]
) -> SceneEstimator:
    """Construct the estimator for a configuration.

    Args:
        config: Training configuration stored with the checkpoint.
        normalization: Translation ``mean`` and ``std`` from curation.

    Returns:
        Untrained model.
    """

    return SceneEstimator(
        config["model"],
        num_targets=len(config["schema"]["targets"]),
        translation_mean=normalization["mean"],
        translation_std=normalization["std"],
    )


def load_checkpoint(
    path: str | Path, device: str
) -> tuple[SceneEstimator, dict[str, Any]]:
    """Load a trained estimator and the configuration it was trained with.

    Args:
        path: Checkpoint file.
        device: Torch device.

    Returns:
        Model in eval mode and its training configuration.
    """

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    config["model"]["pretrained_backbone"] = False
    model = build_model(config, checkpoint["normalization"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config


def to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move a batch to the training device without blocking."""

    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def make_loader(
    config: dict[str, Any], manifest: dict[str, Any], split: str
) -> DataLoader:
    """Create the data loader of one split.

    Args:
        config: Validated training configuration.
        manifest: Curated manifest with usable scene IDs per split.
        split: ``train`` (shuffled, augmented) or ``val``.

    Returns:
        Data loader.
    """

    train = split == "train"
    scene_ids = manifest[split]
    optimization = config["optimization"]
    workers = optimization["num_workers"]
    return DataLoader(
        SceneDataset(config, split, scene_ids, augment=train),
        batch_size=optimization["batch_size"],
        shuffle=train,
        drop_last=train and len(scene_ids) > optimization["batch_size"],
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(
    model: SceneEstimator,
    loader: DataLoader,
    schema: BoardSchema,
    device: torch.device,
    bf16: bool,
) -> dict[str, Any]:
    """Compute per-target and per-type errors over a split.

    Args:
        model: Estimator.
        loader: Split loader.
        schema: Board schema.
        device: Torch device.
        bf16: Whether to run under bfloat16 autocast.

    Returns:
        Error summary from ``ErrorAccumulator``.
    """

    model.eval()
    errors = ErrorAccumulator(schema.ids, [target.type for target in schema.targets])
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            outputs = model(
                batch["images"], batch["intrinsics"], batch["base_from_camera"]
            )
        errors.add(outputs, batch)
    return errors.summary()


def port_error_mm(summary: dict[str, Any]) -> float:
    """Mean port translation error used to select the best checkpoint."""

    values = [summary["per_type"][kind] for kind in PORT_TYPES]
    total = sum(item["samples"] for item in values)
    if total == 0:
        return math.inf
    return (
        sum(
            item["translation_mean_mm"] * item["samples"]
            for item in values
            if item["samples"]
        )
        / total
    )


def flatten_summary(summary: dict[str, Any], prefix: str) -> dict[str, float]:
    """Flatten the per-type part of an error summary for W&B."""

    return {
        f"{prefix}/{kind}/{metric}": value
        for kind, metrics in summary["per_type"].items()
        for metric, value in metrics.items()
    }


def train(config: dict[str, Any]) -> Path:
    """Train the estimator and write checkpoints and metrics under the run directory.

    Args:
        config: Validated training configuration.

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

    manifest = read_manifest(config)
    schema = BoardSchema.from_config(config)
    train_loader = make_loader(config, manifest, "train")
    validation_loader = make_loader(config, manifest, "val")
    model = build_model(config, manifest["normalization"]).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone_parameters(),
                "lr": optimization["backbone_learning_rate"],
            },
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
    (run_directory / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    logging = config["logging"]
    wandb.init(
        project=logging["wandb_project"],
        name=run["name"],
        mode=logging["wandb_mode"],
        dir=str(run_directory),
        config=config,
    )
    print(
        f"Training on {len(manifest['train'])} scenes, validating on "
        f"{len(manifest['val'])}, {total_steps} steps, run directory {run_directory}"
    )

    best = math.inf
    step = 0
    for epoch in range(optimization["epochs"]):
        model.train()
        started = time.perf_counter()
        for batch in train_loader:
            batch = to_device(batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
                outputs = model(
                    batch["images"], batch["intrinsics"], batch["base_from_camera"]
                )
            loss, values = compute_losses(outputs, batch, config["loss"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), optimization["gradient_clip"]
            )
            optimizer.step()
            scheduler.step()
            step += 1
            if step % logging["log_every_steps"] == 0:
                elapsed = time.perf_counter() - started
                wandb.log(
                    {
                        **values,
                        "train/gradient_norm": gradient_norm.item(),
                        "train/learning_rate": scheduler.get_last_lr()[1],
                        "train/epoch": epoch,
                    },
                    step=step,
                )
                print(
                    f"epoch {epoch} step {step}/{total_steps} loss {values['loss/total']:.4f} "
                    f"translation {values['loss/translation_cm']:.3f} cm ({elapsed:.0f} s)",
                    flush=True,
                )

        summary = evaluate(model, validation_loader, schema, device, bf16)
        error = port_error_mm(summary)
        wandb.log(
            {
                **flatten_summary(summary, "validation"),
                "validation/port_error_mm": error,
            },
            step=step,
        )
        print(f"epoch {epoch} validation port error {error:.2f} mm", flush=True)
        state = {
            "model": model.state_dict(),
            "config": config,
            "normalization": manifest["normalization"],
            "epoch": epoch,
            "validation": summary,
        }
        torch.save(state, checkpoints / "last.pt")
        if error < best:
            best = error
            torch.save(state, checkpoints / "best.pt")
            (run_directory / "best_validation.json").write_text(
                json.dumps({"epoch": epoch, **summary}, indent=2) + "\n",
                encoding="utf-8",
            )
    wandb.finish()
    return checkpoints / "best.pt"
