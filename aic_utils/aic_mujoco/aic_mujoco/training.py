"""Supervised ACT policy training with Weights & Biases observability."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from aic_mujoco.config import load_training_config
from aic_mujoco.policy import ACTPolicy, camera_feature_name, create_policy
from aic_mujoco.training_data import (
    EpisodeRecord,
    TrajectoryDataset,
    compute_normalization_statistics,
    discover_episodes,
)


AMP_DTYPES = {"bfloat16": torch.bfloat16}


def validate_training_device(device_name: str) -> None:
    """Reject unavailable or unsupported training devices without fallback.

    Args:
        device_name: Explicit PyTorch device from the training configuration.

    Raises:
        RuntimeError: If the requested CUDA device cannot execute this build.
    """

    device = torch.device(device_name)
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(f"Configured CUDA device is unavailable: {device_name}")
    device_index = torch.cuda.current_device() if device.index is None else device.index
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"Configured CUDA device is unavailable: {device_name}")
    major, minor = torch.cuda.get_device_capability(device_index)
    architecture = f"sm_{major}{minor}"
    supported_architectures = torch.cuda.get_arch_list()
    if supported_architectures and architecture not in supported_architectures:
        raise RuntimeError(
            f"PyTorch {torch.__version__} does not support {architecture} on "
            f"{torch.cuda.get_device_name(device_index)}; install the locked Pixi "
            "environment before training"
        )


class PolicyBatchNormalizer:
    """Move batches to the training device and apply explicit normalization."""

    def __init__(
        self,
        config: dict[str, Any],
        statistics: dict[str, dict[str, np.ndarray]],
    ):
        """Create reusable device normalization tensors.

        Args:
            config: Strict merged policy-training configuration.
            statistics: Train-split state and action mean/std arrays.
        """

        self.device = torch.device(config["training"]["device"])
        self.amp_dtype = AMP_DTYPES[config["training"]["amp_dtype"]]
        self.camera_names = config["policy"]["camera_names"]
        self.image_mean = torch.tensor(
            config["policy"]["image_mean"], device=self.device
        ).view(1, 3, 1, 1)
        self.image_std = torch.tensor(
            config["policy"]["image_std"], device=self.device
        ).view(1, 3, 1, 1)
        self.state_mean = torch.from_numpy(
            statistics["observation.state"]["mean"]
        ).to(self.device)
        self.state_std = torch.from_numpy(
            statistics["observation.state"]["std"]
        ).to(self.device)
        self.action_mean = torch.from_numpy(statistics["action"]["mean"]).to(
            self.device
        )
        self.action_std = torch.from_numpy(statistics["action"]["std"]).to(
            self.device
        )

    def prepare(self, host_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transfer and normalize one collated training or validation batch."""

        batch: dict[str, torch.Tensor] = {}
        for camera_name in self.camera_names:
            key = camera_feature_name(camera_name)
            image = host_batch[key].to(self.device, non_blocking=True).float()
            image = image.div(255.0)
            batch[key] = (image - self.image_mean) / self.image_std
        state = host_batch["observation.state"].to(
            self.device, non_blocking=True
        )
        action = host_batch["action"].to(self.device, non_blocking=True)
        batch["observation.state"] = (state - self.state_mean) / self.state_std
        batch["action"] = (action - self.action_mean) / self.action_std
        batch["action_is_pad"] = host_batch["action_is_pad"].to(
            self.device, non_blocking=True
        )
        return batch

    def action_error_radians(
        self, predicted: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Convert normalized absolute action error back to joint radians."""

        return torch.abs(predicted - target) * self.action_std


def json_statistics(
    statistics: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, list[float]]]:
    """Convert normalization arrays into a portable JSON object."""

    return {
        feature: {name: values.tolist() for name, values in feature_stats.items()}
        for feature, feature_stats in statistics.items()
    }


def create_dataloader(
    config: dict[str, Any],
    records: list[EpisodeRecord],
    epoch: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    """Create one finite episode-streaming data loader.

    Args:
        config: Strict merged policy-training configuration.
        records: Immutable split snapshot.
        epoch: Epoch number controlling deterministic shuffling.
        shuffle: Whether to shuffle episodes and frame indices.
        drop_last: Whether to omit the final incomplete batch.

    Returns:
        Batched PyTorch loader.
    """

    training = config["training"]
    dataset = TrajectoryDataset(config, records, epoch, shuffle)
    common = {
        "dataset": dataset,
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "pin_memory": training["device"].startswith("cuda"),
        "drop_last": drop_last,
    }
    if training["num_workers"] == 0:
        return DataLoader(**common)
    return DataLoader(
        **common,
        prefetch_factor=training["prefetch_factor"],
        persistent_workers=False,
    )


@torch.no_grad()
def validate_policy(
    config: dict[str, Any],
    policy: ACTPolicy,
    normalizer: PolicyBatchNormalizer,
    records: list[EpisodeRecord],
) -> dict[str, float]:
    """Measure deterministic action-chunk error on validation episodes.

    Args:
        config: Strict merged policy-training configuration.
        policy: Policy being optimized.
        normalizer: Train-statistics batch normalizer.
        records: Fixed validation episode snapshot.

    Returns:
        Normalized and physical-unit mean absolute action errors.
    """

    loader = create_dataloader(config, records, 0, False, False)
    normalized_error = 0.0
    physical_error = 0.0
    value_count = 0
    policy.eval()
    for batch_index, host_batch in enumerate(loader):
        if batch_index >= config["training"]["validation_batches"]:
            break
        batch = normalizer.prepare(host_batch)
        with torch.autocast(
            device_type=normalizer.device.type,
            dtype=normalizer.amp_dtype,
            enabled=config["training"]["use_amp"],
        ):
            predicted = policy.predict_action_chunk(batch)
        valid = (~batch["action_is_pad"]).unsqueeze(-1)
        normalized = torch.abs(predicted - batch["action"])
        physical = normalizer.action_error_radians(predicted, batch["action"])
        normalized_error += float((normalized * valid).sum().item())
        physical_error += float((physical * valid).sum().item())
        value_count += int(valid.sum().item()) * predicted.shape[-1]
    if value_count == 0:
        raise RuntimeError("Validation loader produced no non-padding actions")
    policy.train()
    return {
        "validation/action_mae_normalized": normalized_error / value_count,
        "validation/action_mae_rad": physical_error / value_count,
    }


def save_checkpoint(
    output_directory: Path,
    step: int,
    policy: ACTPolicy,
    statistics: dict[str, dict[str, np.ndarray]],
) -> Path:
    """Atomically save policy weights and normalization.

    Args:
        output_directory: Training run root.
        step: Completed optimizer step.
        policy: Policy whose weights will be saved.
        statistics: Train-split normalization statistics.

    Returns:
        Final checkpoint directory.
    """

    checkpoints = output_directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    final_path = checkpoints / f"step_{step:08d}"
    temporary_path = final_path.with_name(final_path.name + ".incomplete")
    if final_path.exists() or temporary_path.exists():
        raise FileExistsError(f"Checkpoint already exists: {final_path}")
    temporary_path.mkdir()
    policy.save_pretrained(temporary_path / "policy", push_to_hub=False)
    (temporary_path / "normalization.json").write_text(
        json.dumps(json_statistics(statistics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.rename(final_path)
    return final_path


def initialize_wandb(
    config: dict[str, Any], output_directory: Path, policy: ACTPolicy
) -> Any:
    """Start the configured W&B run and optional gradient/parameter watching."""

    settings = config["wandb"]
    run = wandb.init(
        project=settings["project"],
        name=settings["run_name"],
        mode=settings["mode"],
        dir=str(output_directory),
        config=config,
    )
    if run is None:
        raise RuntimeError("Weights & Biases did not create a run")
    if settings["watch"] != "none":
        wandb.watch(
            policy,
            log=settings["watch"],
            log_freq=settings["watch_log_frequency"],
            log_graph=False,
        )
    return run


def train(config: dict[str, Any]) -> None:
    """Train ACT on the available successful synthetic demonstrations."""

    training = config["training"]
    dataset = config["dataset"]
    validate_training_device(training["device"])
    train_records = discover_episodes(
        config, "train", training["minimum_train_episodes"]
    )
    validation_records = discover_episodes(
        config, "validation", training["minimum_validation_episodes"]
    )
    train_sample_count = sum(record.steps for record in train_records)
    if train_sample_count < training["batch_size"]:
        raise RuntimeError(
            f"Training snapshot contains {train_sample_count} samples, fewer than "
            f"training.batch_size={training['batch_size']}"
        )
    statistics = compute_normalization_statistics(
        train_records,
        config["policy"]["state_fields"],
        config["policy"]["action_field"],
        len(config["scene"]["names"]["joints"]),
        training["normalization_minimum_std"],
    )

    random.seed(training["seed"])
    np.random.seed(training["seed"])
    torch.manual_seed(training["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training["seed"])

    output_directory = Path(training["output_directory"])
    if output_directory.exists():
        raise FileExistsError(
            f"Training output already exists; choose a new configured directory: "
            f"{output_directory}"
        )
    output_directory.mkdir(parents=True)
    (output_directory / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "normalization.json").write_text(
        json.dumps(json_statistics(statistics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    policy = create_policy(config)
    normalizer = PolicyBatchNormalizer(config, statistics)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    run = initialize_wandb(config, output_directory, policy)
    run.summary["dataset/train_episodes"] = len(train_records)
    run.summary["dataset/validation_episodes"] = len(validation_records)
    run.summary["dataset/train_frames"] = train_sample_count
    run.summary["model/parameters"] = parameter_count
    print(
        f"Training {config['policy']['type']} on {len(train_records)} train and "
        f"{len(validation_records)} validation episodes from "
        f"{dataset['output_directory']}"
    )
    print(f"Parameters: {parameter_count:,}; device: {training['device']}")

    step = 0
    epoch = 0
    try:
        policy.train()
        while step < training["steps"]:
            loader = create_dataloader(
                config, train_records, epoch, True, True
            )
            iterator = iter(loader)
            while step < training["steps"]:
                data_start = time.perf_counter()
                try:
                    host_batch = next(iterator)
                except StopIteration:
                    break
                data_seconds = time.perf_counter() - data_start
                update_start = time.perf_counter()
                batch = normalizer.prepare(host_batch)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=normalizer.device.type,
                    dtype=normalizer.amp_dtype,
                    enabled=training["use_amp"],
                ):
                    loss, loss_components = policy(batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step {step + 1}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), training["gradient_clip_norm"]
                )
                optimizer.step()
                step += 1
                update_seconds = time.perf_counter() - update_start

                if step % training["log_every_steps"] == 0:
                    metrics = {
                        "train/loss": float(loss.item()),
                        "train/l1_loss": float(loss_components["l1_loss"]),
                        "train/gradient_norm": float(gradient_norm.item()),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/backbone_learning_rate": optimizer.param_groups[1][
                            "lr"
                        ],
                        "throughput/samples_per_second": training["batch_size"]
                        / (data_seconds + update_seconds),
                        "timing/data_seconds": data_seconds,
                        "timing/update_seconds": update_seconds,
                        "epoch": epoch,
                    }
                    if "kld_loss" in loss_components:
                        metrics["train/kld_loss"] = float(
                            loss_components["kld_loss"]
                        )
                    run.log(metrics, step=step)
                    print(
                        f"step {step}/{training['steps']} "
                        f"loss={metrics['train/loss']:.5f} "
                        f"grad={metrics['train/gradient_norm']:.3f}"
                    )

                if step % training["validate_every_steps"] == 0:
                    validation = validate_policy(
                        config, policy, normalizer, validation_records
                    )
                    run.log(validation, step=step)
                    print(
                        f"validation step {step}: "
                        f"action_mae={validation['validation/action_mae_rad']:.6f} rad"
                    )

                if (
                    step % training["checkpoint_every_steps"] == 0
                    or step == training["steps"]
                ):
                    checkpoint = save_checkpoint(
                        output_directory, step, policy, statistics
                    )
                    run.summary["checkpoint/latest"] = str(checkpoint)
            epoch += 1
    finally:
        run.finish()


def main() -> None:
    """Load the canonical JSON overlays and run supervised policy training."""

    train(load_training_config())
