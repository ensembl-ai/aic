"""Summarize the newest trial log written by the insertion policies.

Reads the newest ``trials_*.jsonl`` in ``logging.trial_directory`` of
``configs/infer.json`` and prints one row per trial plus totals per plug
type: outcome, where it failed, the perception error, the plug
misalignment after the close-range servo, how the descent and spiral went,
and the final depth. Ground-truth columns are filled only when the trials ran
with ``ground_truth:=true`` and ``insertion.log_ground_truth``.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_perception.config import load_config  # noqa: E402


def phase(record: dict[str, Any], name: str, last: bool = False) -> dict[str, Any]:
    """First (or last) entry of one phase in a trial record, or an empty dict."""

    matches = [p for p in record["phases"] if p["phase"] == name]
    return (matches[-1] if last else matches[0]) if matches else {}


def number(value: Any, digits: int = 2) -> str:
    """Format an optional number."""

    return "-" if value is None else f"{value:.{digits}f}"


def main() -> None:
    """Print the trial table and totals."""

    directory = Path(load_config("infer")["logging"]["trial_directory"])
    logs = sorted(directory.glob("trials_*.jsonl"))
    if not logs:
        print(f"No trial logs in {directory}")
        return
    records = [json.loads(line) for line in logs[-1].read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{logs[-1]} ({len(records)} trials; {len(logs) - 1} older logs in {directory})\n")
    print("Columns in mm: perception = port error of the scene estimator; servo = port offset the refiner measured")
    print("first and last; aligned = true plug off the port axis before the descent; depth = final insertion depth.\n")
    header = f"{'#':>2} {'type':4} {'target':28} {'ok':3} {'time s':>6} {'percept':>7} {'servo 1st':>9} {'last':>6} {'iters':>5} {'aligned':>7} {'descent':>14} {'spiral':>10} {'depth':>6}  failure"
    print(header)
    totals: dict[str, Counter] = {}
    for index, record in enumerate(records, 1):
        task = record["task"]
        servo = [p for p in record["phases"] if p["phase"] == "servo"]
        first = (sum(v * v for v in servo[0]["offset_mm"][:2]) ** 0.5) if servo else None
        last = (sum(v * v for v in servo[-1]["offset_mm"][:2]) ** 0.5) if servo else None
        descent = phase(record, "descent")
        spiral = phase(record, "spiral", last=True)
        press = phase(record, "press")
        target = phase(record, "target").get("target_id", f"{task['target_module_name']}/{task['port_name']}")
        descent_text = f"{descent['reason']} {'in' if descent['inside'] else 'surf'}" if descent else "-"
        spiral_text = ("in r" if spiral.get("entered") else "miss r") + number(spiral.get("radius_mm"), 1) if spiral else "-"
        print(
            f"{index:2d} {task['plug_type']:4} {target:28} {'yes' if record['success'] else 'no':3} {record['duration_s']:6.1f} "
            f"{number(phase(record, 'truth perception').get('port_error_mm')):>7} {number(first):>9} {number(last):>6} {len(servo):5d} "
            f"{number(phase(record, 'truth aligned').get('plug_off_axis_mm')):>7} {descent_text:>14} {spiral_text:>10} "
            f"{number(press.get('depth_mm'), 1):>6}  {record['failure'] or ''}"
        )
        counts = totals.setdefault(task["plug_type"], Counter())
        counts["trials"] += 1
        counts["seated"] += bool(record["success"])
        if record["failure"]:
            counts[f"failure: {record['failure'].split(';')[0][:60]}"] += 1
    print()
    for plug_type, counts in totals.items():
        print(f"{plug_type}: {counts['seated']}/{counts['trials']} seated")
        for key, value in counts.items():
            if key.startswith("failure"):
                print(f"   {value}x {key}")


if __name__ == "__main__":
    main()
