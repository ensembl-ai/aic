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
from aic_mujoco.config import load_training_config
from aic_mujoco.evaluation import evaluate_action_prediction
from aic_mujoco.policy import (
    ACTPolicy,
    PolicyNormalizer,
    create_policy,
    validate_policy_device,
)
from aic_mujoco.training_data import (
    compute_normalization_statistics,
    create_dataloader,
    discover_episodes,
)
from aic_mujoco.utils.metrics import TrainingMetricsRecorder


def json_statistics(
    statistics: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, list[float]]]:
    """Convert normalization arrays into a portable JSON object."""

    return {
        feature: {name: values.tolist() for name, values in feature_stats.items()}
        for feature, feature_stats in statistics.items()
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
    validate_policy_device(training["device"])
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
    normalizer = PolicyNormalizer(config, statistics)
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    run = initialize_wandb(config, output_directory, policy)
    local_metrics = TrainingMetricsRecorder(output_directory)
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
    latest_checkpoint: Path | None = None
    completed = False
    try:
        policy.train()
        while step < training["steps"]:
            loader = create_dataloader(
                config,
                train_records,
                epoch=epoch,
                shuffle=True,
                drop_last=True,
                batch_size=training["batch_size"],
                num_workers=training["num_workers"],
                prefetch_factor=training["prefetch_factor"],
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
                    local_metrics.record(step, "train", metrics)
                    run.log(metrics, step=step)
                    print(
                        f"step {step}/{training['steps']} "
                        f"loss={metrics['train/loss']:.5f} "
                        f"grad={metrics['train/gradient_norm']:.3f}"
                    )

                if step % training["validate_every_steps"] == 0:
                    validation_result = evaluate_action_prediction(
                        config,
                        policy,
                        normalizer,
                        validation_records,
                        batch_size=training["batch_size"],
                        num_workers=training["num_workers"],
                        prefetch_factor=training["prefetch_factor"],
                        maximum_batches=training["validation_batches"],
                    )
                    validation = {
                        "validation/action_mae_normalized": validation_result[
                            "action_mae_normalized"
                        ],
                        "validation/action_mae_rad": validation_result[
                            "action_mae_rad"
                        ],
                    }
                    local_metrics.record(step, "validation", validation)
                    run.log(validation, step=step)
                    print(
                        f"validation step {step}: "
                        f"action_mae={validation['validation/action_mae_rad']:.6f} rad"
                    )

                if (
                    step % training["checkpoint_every_steps"] == 0
                    or step == training["steps"]
                ):
                    latest_checkpoint = save_checkpoint(
                        output_directory, step, policy, statistics
                    )
                    run.summary["checkpoint/latest"] = str(latest_checkpoint)
            epoch += 1
        completed = True
    finally:
        local_metrics.finish(
            {
                "status": "completed" if completed else "interrupted_or_failed",
                "final_step": step,
                "final_epoch": epoch,
                "configured_steps": training["steps"],
                "train_episodes": len(train_records),
                "validation_episodes": len(validation_records),
                "train_frames": train_sample_count,
                "validation_frames": sum(
                    record.steps for record in validation_records
                ),
                "model_parameters": parameter_count,
                "wandb": {
                    "entity": run.entity,
                    "project": run.project,
                    "run_id": run.id,
                    "run_name": run.name,
                },
                "latest_checkpoint": (
                    latest_checkpoint.name if latest_checkpoint is not None else None
                ),
            }
        )
        run.finish()


def main() -> None:
    """Load the canonical JSON overlays and run supervised policy training."""

    train(load_training_config())
