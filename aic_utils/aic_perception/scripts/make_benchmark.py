"""Write a randomized aic_engine benchmark: ``benchmark.trials`` trials in ``benchmark.batches`` files.

Trials come from the collector's scene sampler (``randomization`` in
``configs/collect.json``) with the benchmark seed, so they follow the same
engine limits as the training data but are different boards. Exactly
``sc_trials`` are SC tasks; SC and SFP trials are spread evenly over the
batches. Each batch is a complete engine config (scoring topics, limits and
robot copied from ``benchmark.template``), run like ``engine_trials.yaml``.

SC slots: two SC ports look identical, and the policy maps the task's slot
onto a rail rank with ``sc_rank_for_slot`` (``sc_port_1`` is the lower
translation, as in the phase-1 eval trials). The sampler names ports by rank,
so rank 0 is written to ``sc_rail_1`` and rank 1 to ``sc_rail_0``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_perception.config import load_config  # noqa: E402
from aic_perception.scene_sampler import (
    FIXTURE_MOUNTS,
    NIC_MOUNTS,
    sample_scene,
)  # noqa: E402

ENGINE_SC_SLOT = {"sc_port_0": "sc_port_1", "sc_port_1": "sc_port_0"}
PLUG_NAMES = {"sfp": "sfp_tip", "sc": "sc_tip"}


def entity(name: str, component: dict[str, float] | None) -> dict[str, Any]:
    """Engine rail entry for one slot."""

    if component is None:
        return {"entity_present": False}
    return {
        "entity_present": True,
        "entity_name": name,
        "entity_pose": {
            "translation": round(component["translation"], 5),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": round(component["yaw"], 5),
        },
    }


def engine_trial(
    config: dict[str, Any], scene: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    """Convert a sampled scene into one aic_engine trial."""

    components = scene["components"]
    board = {
        "pose": {
            "x": round(scene["board"]["x"], 5),
            "y": round(scene["board"]["y"], 5),
            "z": scene["board"]["z"],
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": round(scene["board"]["yaw"], 5),
        }
    }
    for index, mount in enumerate(NIC_MOUNTS):
        board[f"nic_rail_{index}"] = entity(f"nic_card_{index}", components.get(mount))
    for rank_name, slot in ENGINE_SC_SLOT.items():
        rail = slot[-1]
        board[f"sc_rail_{rail}"] = entity(f"sc_mount_{rail}", components.get(rank_name))
    for name in FIXTURE_MOUNTS:
        kind, rail = name.split("_mount_rail_")
        board[name] = entity(f"{kind}_mount_{rail}", components.get(name))
    task = scene["task"]
    plug_type = task["port_type"]
    cable = config["cable"]
    offsets = cable["gripper_offsets"][plug_type]
    x, y, z = offsets[int(rng.integers(len(offsets)))]
    roll, pitch, yaw = cable["rpy"]
    cable_name = f"cable_{int(plug_type == 'sc')}"
    target = (
        ENGINE_SC_SLOT[task["target_module_name"]]
        if plug_type == "sc"
        else task["target_module_name"]
    )
    return {
        "scene": {
            "task_board": board,
            "cables": {
                cable_name: {
                    "pose": {
                        "gripper_offset": {"x": x, "y": y, "z": z},
                        "roll": roll,
                        "pitch": pitch,
                        "yaw": yaw,
                    },
                    "attach_cable_to_gripper": True,
                    "cable_type": cable["types"][plug_type],
                }
            },
        },
        "tasks": {
            "task_1": {
                "cable_type": "sfp_sc",
                "cable_name": cable_name,
                "plug_type": plug_type,
                "plug_name": PLUG_NAMES[plug_type],
                "port_type": plug_type,
                "port_name": task["port_name"],
                "target_module_name": target,
                "time_limit": 180,
            }
        },
    }


def main() -> None:
    """Sample the trials and write the batch files."""

    config = load_config("collect")
    settings = config["benchmark"]
    config["collection"]["seed"] = settings["seed"]
    quota = {
        "sc": settings["sc_trials"],
        "sfp": settings["trials"] - settings["sc_trials"],
    }
    rng = np.random.default_rng(settings["seed"])
    chosen: dict[str, list[dict[str, Any]]] = {"sc": [], "sfp": []}
    board_id = 0
    while any(len(chosen[kind]) < quota[kind] for kind in quota):
        scene = sample_scene(config, board_id)
        kind = scene["task"]["port_type"]
        if len(chosen[kind]) < quota[kind]:
            chosen[kind].append(scene)
        board_id += 1
    ordered = chosen["sc"] + chosen["sfp"]
    template = yaml.safe_load(Path(settings["template"]).read_text())
    directory = Path(settings["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    for batch in range(settings["batches"]):
        trials = {}
        for index in range(batch, len(ordered), settings["batches"]):
            scene = ordered[index]
            trials[f"bench_{index:02d}_{scene['task']['port_type']}"] = engine_trial(
                config, scene, rng
            )
        output = {**template, "trials": trials}
        path = directory / f"batch_{batch}.yaml"
        path.write_text(yaml.safe_dump(output, sort_keys=False))
        kinds = [name.rsplit("_", 1)[1] for name in trials]
        print(
            f"{path}: {len(trials)} trials ({kinds.count('sfp')} SFP, {kinds.count('sc')} SC)"
        )


if __name__ == "__main__":
    main()
