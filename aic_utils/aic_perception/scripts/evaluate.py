"""Evaluate the best checkpoint of the configured run on the val split.

Prints errors per target type, per viewpoint distance band, presence
precision and recall, and the confidence calibration, and writes the full
report (including every target) to ``evaluation.json`` in the run directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_perception.config import load_config  # noqa: E402
from aic_perception.dataset import read_manifest  # noqa: E402
from aic_perception.schema import BoardSchema  # noqa: E402
from aic_perception.training import (  # noqa: E402
    confidence_tolerance,
    evaluate,
    load_checkpoint,
    make_loader,
)


def print_report(summary: dict) -> None:
    """Print the summary as readable tables."""

    print(f"{'type':10} {'samples':>8} {'mean mm':>8} {'median':>7} {'p90':>7} {'<2mm':>6} {'<5mm':>6} {'rot deg':>8} {'prec':>6} {'recall':>6}")
    for kind, m in summary["per_type"].items():
        if m["samples"]:
            print(
                f"{kind:10} {m['samples']:8d} {m['translation_mean_mm']:8.2f} {m['translation_median_mm']:7.2f} "
                f"{m['translation_p90_mm']:7.2f} {m['translation_within_2mm']:6.2f} {m['translation_within_5mm']:6.2f} "
                f"{m['rotation_mean_deg']:8.2f} {m['presence_precision']:6.3f} {m['presence_recall']:6.3f}"
            )
    print("\nBy viewpoint (home = start pose; far >= 10 cm, mid 3-10 cm, close < 3 cm above the target port):")
    print(f"{'band':6} {'type':10} {'samples':>8} {'mean mm':>8} {'median':>7} {'p90':>7} {'<2mm':>6}")
    for band, types in summary["per_band"].items():
        for kind, m in types.items():
            if m["samples"]:
                print(
                    f"{band:6} {kind:10} {m['samples']:8d} {m['translation_mean_mm']:8.2f} "
                    f"{m['translation_median_mm']:7.2f} {m['translation_p90_mm']:7.2f} {m['translation_within_2mm']:6.2f}"
                )
    print("\nConfidence calibration (fraction really within tolerance per predicted-confidence bin):")
    for row in summary["confidence_calibration"]:
        print(f"  {row['confidence']}: {row['samples']:7d} estimates, {row['accurate_fraction']:.2f} accurate")


def main() -> None:
    """Evaluate and print the report."""

    config = load_config("train")
    run_directory = Path(config["run"]["directory"]) / config["run"]["name"]
    model, trained_config = load_checkpoint(
        run_directory / "checkpoints" / "best.pt", config["run"]["device"]
    )
    if trained_config["schema"] != config["schema"]:
        raise ValueError("The checkpoint was trained with a different board schema")
    loader = make_loader(config, read_manifest(config), "val")
    summary = evaluate(
        model,
        loader,
        BoardSchema.from_config(config),
        model.translation_mean.device,
        config["optimization"]["bf16"],
        confidence_tolerance(config),
    )
    (run_directory / "evaluation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print_report(summary)
    print(f"Full report: {run_directory / 'evaluation.json'}")


if __name__ == "__main__":
    main()
