"""Held-out action validation for trained AIC manipulation policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from aic_mujoco.config import load_evaluation_config
from aic_mujoco.policy import ACTPolicy, PolicyNormalizer, load_policy_checkpoint
from aic_mujoco.training_data import EpisodeRecord, create_dataloader, discover_episodes


@torch.no_grad()
def evaluate_action_prediction(
    config: dict[str, Any],
    policy: ACTPolicy,
    normalizer: PolicyNormalizer,
    records: list[EpisodeRecord],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    maximum_batches: int | None,
) -> dict[str, Any]:
    """Measure action-chunk error against privileged-teacher demonstrations.

    Args:
        config: Strict merged policy configuration.
        policy: Trained or currently optimizing ACT policy.
        normalizer: Train-split observation and action normalizer.
        records: Fixed validation or test episode snapshot.
        batch_size: Observations evaluated together.
        num_workers: Episode-decoding worker process count.
        prefetch_factor: Batches prefetched by each worker.
        maximum_batches: Optional training-time validation limit. ``None``
            evaluates every sample in ``records``.

    Returns:
        Normalized, physical-unit, and per-joint mean absolute errors plus
        evaluated sample counts.

    Raises:
        RuntimeError: If the loader produces no valid action targets.
    """

    loader = create_dataloader(
        config,
        records,
        epoch=0,
        shuffle=False,
        drop_last=False,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    joint_count = len(config["scene"]["names"]["joints"])
    normalized_sum = 0.0
    physical_sum = torch.zeros(
        joint_count, dtype=torch.float64, device=normalizer.device
    )
    action_vector_count = 0
    observation_count = 0
    was_training = policy.training
    policy.eval()
    try:
        for batch_index, host_batch in enumerate(loader):
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
            batch = normalizer.prepare(host_batch)
            with torch.autocast(
                device_type=normalizer.device.type,
                dtype=normalizer.amp_dtype,
                enabled=config["training"]["use_amp"],
            ):
                predicted = policy.predict_action_chunk(batch)
            valid = (~batch["action_is_pad"]).unsqueeze(-1)
            normalized_error = torch.abs(predicted - batch["action"])
            physical_error = normalizer.action_error_radians(
                predicted, batch["action"]
            )
            normalized_sum += float((normalized_error * valid).sum().item())
            physical_sum += (physical_error * valid).sum(dim=(0, 1)).double()
            action_vector_count += int(valid.sum().item())
            observation_count += int(predicted.shape[0])
    finally:
        policy.train(was_training)
    if action_vector_count == 0:
        raise RuntimeError("Evaluation loader produced no non-padding actions")
    physical_by_joint = (physical_sum / action_vector_count).cpu().tolist()
    value_count = action_vector_count * joint_count
    return {
        "action_mae_normalized": normalized_sum / value_count,
        "action_mae_rad": sum(physical_by_joint) / joint_count,
        "action_mae_rad_by_joint": physical_by_joint,
        "observations": observation_count,
        "action_vectors": action_vector_count,
    }


def validate_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the explicit checkpoint on the complete configured held-out split.

    Args:
        config: Strict merged evaluation configuration.

    Returns:
        Serializable held-out metric report.
    """

    settings = config["evaluation"]
    split = settings["dataset_split"]
    records = discover_episodes(config, split, config["dataset"]["splits"][split])
    policy, statistics = load_policy_checkpoint(config)
    normalizer = PolicyNormalizer(config, statistics)
    metrics = evaluate_action_prediction(
        config,
        policy,
        normalizer,
        records,
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        prefetch_factor=settings["prefetch_factor"],
        maximum_batches=None,
    )
    joint_names = config["scene"]["names"]["joints"]
    checkpoint = Path(settings["checkpoint_directory"])
    return {
        "run": checkpoint.parent.parent.name,
        "checkpoint": checkpoint.name,
        "dataset_split": split,
        "episodes": len(records),
        "frames": sum(record.steps for record in records),
        "action_mae_normalized": metrics["action_mae_normalized"],
        "action_mae_rad": metrics["action_mae_rad"],
        "action_mae_rad_by_joint": dict(
            zip(joint_names, metrics["action_mae_rad_by_joint"], strict=True)
        ),
        "observations": metrics["observations"],
        "action_vectors": metrics["action_vectors"],
    }


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    """Atomically write one JSON validation report.

    Args:
        path: Explicit report destination.
        metrics: Serializable validation result.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".incomplete")
    temporary.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    """Load canonical overlays and evaluate the configured checkpoint."""

    config = load_evaluation_config()
    metrics = validate_checkpoint(config)
    write_metrics(Path(config["evaluation"]["metrics_output"]), metrics)
    print(
        f"{metrics['dataset_split']}: {metrics['episodes']} episodes, "
        f"{metrics['frames']} frames"
    )
    print(f"Action MAE: {metrics['action_mae_rad']:.8f} rad")
    for joint_name, error in metrics["action_mae_rad_by_joint"].items():
        print(f"  {joint_name}: {error:.8f} rad")
    print(f"Metrics: {config['evaluation']['metrics_output']}")
