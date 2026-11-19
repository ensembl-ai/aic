"""Evaluate the best refiner checkpoint of the configured run on the val split.

Prints the plug-to-port offset error per plug type, per starting
misalignment, and per plug height, next to the misalignment that the
refiner corrects, and writes ``evaluation.json`` to the run directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_perception.config import load_config  # noqa: E402
from aic_perception.refinement import (  # noqa: E402
    evaluate_refiner,
    load_refiner,
    make_refiner_loader,
    select_samples,
)


def main() -> None:
    """Evaluate and print the report."""

    config = load_config("refine")
    run_directory = Path(config["run"]["directory"]) / config["run"]["name"]
    model, checkpoint = load_refiner(
        run_directory / "checkpoints" / "best.pt", config["run"]["device"]
    )
    samples = select_samples(config, "val")
    loader = make_refiner_loader(config, "val", samples, checkpoint["windows"])
    summary = evaluate_refiner(
        model, loader, model.offset_mean.device, config["optimization"]["bf16"]
    )
    (run_directory / "evaluation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Best epoch {checkpoint['epoch']}, {summary['samples']} val samples")
    print(
        "Offset = port minus plug (what the policy corrects). 'uncorrected' is the true misalignment before correcting."
    )
    header = f"{'group':22} {'samples':>7} {'offset mean':>11} {'median':>7} {'p90':>6} {'<0.5mm':>7} {'<1mm':>6} {'plug mm':>8} {'rot deg':>8} {'uncorrected':>11}"
    for title, group in (
        ("Per plug type", "per_type"),
        ("Per starting misalignment", "per_misalignment"),
        ("Per plug height", "per_height"),
    ):
        print(f"\n{title}:\n{header}")
        for name, m in sorted(summary[group].items()):
            if m["samples"]:
                print(
                    f"{name:22} {m['samples']:7d} {m['offset_mean_mm']:11.2f} {m['offset_median_mm']:7.2f} {m['offset_p90_mm']:6.2f} "
                    f"{m['offset_within_0_5mm']:7.2f} {m['offset_within_1mm']:6.2f} {m['plug_mean_mm']:8.2f} "
                    f"{m['rotation_mean_deg']:8.2f} {m['uncorrected_mean_mm']:9.2f}mm"
                )
    print(f"\nFull report: {run_directory / 'evaluation.json'}")


if __name__ == "__main__":
    main()
