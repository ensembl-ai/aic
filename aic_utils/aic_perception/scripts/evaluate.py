"""Evaluate the best checkpoint of the configured run on the val split.

Writes ``evaluation.json`` with per-target and per-type errors next to the
run's checkpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_perception.config import load_config  # noqa: E402
from aic_perception.dataset import read_manifest  # noqa: E402
from aic_perception.schema import BoardSchema  # noqa: E402
from aic_perception.training import evaluate, load_checkpoint, make_loader  # noqa: E402


def main() -> None:
    """Evaluate and print the per-type summary."""

    config = load_config("train")
    run_directory = Path(config["run"]["directory"]) / config["run"]["name"]
    model, trained_config = load_checkpoint(run_directory / "checkpoints" / "best.pt", config["run"]["device"])
    if trained_config["schema"] != config["schema"]:
        raise ValueError("The checkpoint was trained with a different board schema")
    loader = make_loader(config, read_manifest(config), "val")
    summary = evaluate(
        model, loader, BoardSchema.from_config(config), model.translation_mean.device, config["optimization"]["bf16"]
    )
    (run_directory / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["per_type"], indent=2))
    print(f"Full report: {run_directory / 'evaluation.json'}")


if __name__ == "__main__":
    main()
