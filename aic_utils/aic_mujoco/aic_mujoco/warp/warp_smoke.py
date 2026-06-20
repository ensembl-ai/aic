"""MuJoCo Warp smoke test helpers for the AIC training scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco_warp
import warp as wp


@dataclass(frozen=True)
class WarpSmokeConfig:
    """Direct Warp model/data smoke-test config."""

    xml_path: Path
    num_envs: int = 32
    steps: int = 1


def run_warp_smoke(cfg: WarpSmokeConfig) -> dict[str, float | int | str]:
    """Compile a MuJoCo model into Warp data and run zero-control steps.

    This is intentionally direct: no MJLab, no manager layer, no hidden scene
    composition. If this fails, the failure is in the MuJoCo XML, MuJoCo Warp,
    CUDA/Warp cache, or the local driver/runtime setup.
    """

    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    data = mujoco.MjData(model)
    warp_model = mujoco_warp.put_model(model)
    warp_data = mujoco_warp.put_data(model, data, nworld=int(cfg.num_envs))
    for _ in range(int(cfg.steps)):
        mujoco_warp.step(warp_model, warp_data)
    wp.synchronize()
    out = mujoco.MjData(model)
    mujoco_warp.get_data_into(out, model, warp_data, world_id=0)
    return {
        "xml": str(cfg.xml_path),
        "num_envs": int(cfg.num_envs),
        "steps": int(cfg.steps),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
        "sim_time_env0": float(out.time),
    }
