#!/usr/bin/env python3
"""Visualize sampled AIC training environments while physics runs on CUDA.

Fresh run from a new ``aic_eval`` distrobox terminal:

cd /home/rmalhan/Software/ws_aic/src/aic
pixi shell
export PYTHONNOUSERSITE=1
export MUJOCO_PLUGIN_PATH=/home/rmalhan/Software/ws_aic/install/opt/mujoco_vendor/lib
export XDG_CACHE_HOME=/tmp/$USER-cache
mkdir -p "$XDG_CACHE_HOME"

python3 aic_utils/aic_mujoco/scripts/prepare_training_scene.py

PYTHONPATH=/home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco \
python3 aic_utils/aic_mujoco/scripts/visualize.py \
--config /home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco/configs/experiments/train.json \
--num-envs 4096 \
--display-envs 16 \
--steps-per-frame 4 \
--fps 30 \
--device cuda

Physics is stepped by ``mujoco_warp`` on the requested device for all
``--num-envs`` worlds. The host seed state is initialized from ``reset_q`` in
the same training config used by ``scripts/train.py`` before it is uploaded to
Warp, so the browser shows the same starting posture as headless training.
Viser is only a debug renderer: every frame it downloads the selected
``--display-envs`` worlds into host ``MjData`` objects and updates useful visual
geoms in a flat grid. Collision geoms, floor/wall shells, and enclosure shells
are intentionally hidden. ``--steps-per-frame`` controls playback smoothness:
smaller values look smoother, larger values fast-forward more sim time per
browser frame.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco_warp
import numpy as np
import viser
import warp as wp

from aic_mujoco.utils import quaternion_from_rotation
from aic_mujoco.warp import (
    apply_debug_joint_policy,
    initialize_data_from_config,
    make_debug_joint_policy,
    sample_observations,
    task_entity_ids,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_XML = PACKAGE_ROOT / "mjcf" / "scene_warp.xml"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "experiments" / "train.json"


def make_parser() -> argparse.ArgumentParser:
    """Build CLI for Viser visualization backed by MuJoCo Warp physics."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--display-envs", type=int, default=16)
    parser.add_argument("--steps-per-frame", type=int, default=4)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--spacing", type=float, default=3.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--motion-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    """Run batched training physics and stream selected envs to Viser."""

    args = make_parser().parse_args()
    xml_path = Path(args.xml).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    wp.set_device(str(args.device))

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    host_seed = mujoco.MjData(model)
    initialize_data_from_config(model, host_seed, config_path)
    policy = make_debug_joint_policy(config_path, str(args.device))
    task_ids = task_entity_ids(model)
    warp_model = mujoco_warp.put_model(model)
    warp_data = mujoco_warp.put_data(model, host_seed, nworld=int(args.num_envs))
    wp.synchronize()

    display_envs = min(int(args.display_envs), int(args.num_envs))
    host_datas = [mujoco.MjData(model) for _ in range(display_envs)]
    server = viser.ViserServer(port=int(args.port))
    cols = int(math.ceil(math.sqrt(display_envs)))
    handles: dict[tuple[int, int], object] = {}

    add_grid(server, display_envs, cols, float(args.spacing))
    for env_id in range(display_envs):
        mujoco_warp.get_data_into(host_datas[env_id], model, warp_data, world_id=env_id)
        add_env_geoms(
            server=server,
            model=model,
            data=host_datas[env_id],
            env_id=env_id,
            cols=cols,
            spacing=float(args.spacing),
            handles=handles,
        )

    print(f"[visualize] Viser: http://localhost:{int(args.port)}", flush=True)
    print(
        "[visualize] "
        f"physics_device={args.device} num_envs={int(args.num_envs)} "
        f"display_envs={display_envs} steps_per_frame={int(args.steps_per_frame)} "
        f"motion_scale={float(args.motion_scale):.3f}",
        flush=True,
    )

    period = 1.0 / max(float(args.fps), 1.0)
    total_steps = 0
    last_log_time = time.perf_counter()
    while True:
        frame_t0 = time.perf_counter()
        for _ in range(int(args.steps_per_frame)):
            apply_debug_joint_policy(
                warp_data=warp_data,
                policy=policy,
                num_envs=int(args.num_envs),
                step_idx=total_steps + 1,
                motion_scale=float(args.motion_scale),
            )
            mujoco_warp.step(warp_model, warp_data)
            total_steps += 1
        wp.synchronize()

        for env_id in range(display_envs):
            mujoco_warp.get_data_into(
                host_datas[env_id], model, warp_data, world_id=env_id
            )
            update_env_geoms(
                model=model,
                data=host_datas[env_id],
                env_id=env_id,
                cols=cols,
                spacing=float(args.spacing),
                handles=handles,
            )

        now = time.perf_counter()
        if now - last_log_time >= 1.0:
            physics_steps = int(args.num_envs) * total_steps
            sim_s = physics_steps * float(model.opt.timestep)
            obs = sample_observations(
                model=model,
                warp_data=warp_data,
                sample_datas=host_datas,
                task_ids=task_ids,
            )
            print(
                "[visualize] "
                f"total_steps={total_steps} "
                f"physics_steps={physics_steps} "
                f"aggregate_sim_s={sim_s:.3f} "
                f"reward={obs.reward_mean:+.4f} "
                f"lat_err={obs.lateral_error_mean:.4f}m "
                f"max_pen={obs.max_penetration:.5f}m "
                f"force={obs.force_norm_max:.3f}N "
                f"qstd={obs.qpos_std_mean:.5f}",
                flush=True,
            )
            last_log_time = now

        elapsed = time.perf_counter() - frame_t0
        time.sleep(max(0.0, period - elapsed))


def add_grid(
    server: viser.ViserServer, display_envs: int, cols: int, spacing: float
) -> None:
    """Add a flat XY floor grid sized to the displayed env cells."""

    rows = int(math.ceil(display_envs / cols))
    server.scene.add_grid(
        "/grid",
        width=max(cols * spacing, spacing),
        height=max(rows * spacing, spacing),
        plane="xy",
        cell_size=spacing,
        section_size=spacing,
        position=((cols - 1) * spacing / 2.0, (rows - 1) * spacing / 2.0, -0.03),
    )


def add_env_geoms(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    env_id: int,
    cols: int,
    spacing: float,
    handles: dict[tuple[int, int], object],
) -> None:
    """Create Viser geometry handles for one sampled environment."""

    offset = env_offset(env_id, cols, spacing)
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if not is_visual_geom(model, geom_id):
            continue
        handle = add_geom(server, model, data, geom_id, env_id, offset)
        if handle is not None:
            handles[(env_id, geom_id)] = handle


def update_env_geoms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    env_id: int,
    cols: int,
    spacing: float,
    handles: dict[tuple[int, int], object],
) -> None:
    """Update Viser geometry poses for one sampled environment."""

    offset = env_offset(env_id, cols, spacing)
    for geom_id in range(model.ngeom):
        handle = handles.get((env_id, geom_id))
        if handle is None:
            continue
        handle.position = data.geom_xpos[geom_id] + offset
        handle.wxyz = quaternion_from_rotation(data.geom_xmat[geom_id].reshape(3, 3))


def add_geom(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    env_id: int,
    offset: np.ndarray,
) -> object | None:
    """Add one MuJoCo geom to Viser using mesh or primitive geometry."""

    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    pos = data.geom_xpos[geom_id] + offset
    quat = quaternion_from_rotation(data.geom_xmat[geom_id].reshape(3, 3))
    color, opacity = geom_color(model, geom_id)
    node = f"/env_{env_id:03d}/geom_{geom_id:04d}"

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        vertices, faces = mesh_vertices_faces(model, mesh_id)
        return server.scene.add_mesh_simple(
            node,
            vertices=vertices,
            faces=faces,
            color=color,
            opacity=opacity,
            flat_shading=False,
            side="double",
            position=pos,
            wxyz=quat,
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return server.scene.add_icosphere(
            node,
            radius=max(float(size[0]), 1e-4),
            color=color,
            subdivisions=1,
            opacity=opacity,
            position=pos,
            wxyz=quat,
        )
    if geom_type in {mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE}:
        return server.scene.add_cylinder(
            node,
            radius=max(float(size[0]), 1e-4),
            height=max(2.0 * float(size[1]), 1e-4),
            color=color,
            radial_segments=12,
            opacity=opacity,
            position=pos,
            wxyz=quat,
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        return server.scene.add_box(
            node,
            dimensions=np.maximum(2.0 * size[:3], 1e-4),
            color=color,
            opacity=opacity,
            position=pos,
            wxyz=quat,
        )
    return None


def is_visual_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    """Return true for display geometry and false for collision-only geometry.

    Args:
        model: Compiled MuJoCo model.
        geom_id: MuJoCo geom index to classify.

    MuJoCo conventionally marks visual-only geoms with zero contact type and
    affinity. The converted AIC assets also carry ``visual`` in many geom
    names, so the name check keeps those visible even if a converter preserved
    contact flags more aggressively than expected.
    """

    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    lowered = f"{body_name} {name}".lower()
    hidden_visual_shells = (
        "enclosure",
        "floor_link",
        "floor_visual",
        "walls_visual",
        "light_visual",
    )
    if any(token in lowered for token in hidden_visual_shells):
        return False
    return (
        int(model.geom_contype[geom_id]) == 0
        and int(model.geom_conaffinity[geom_id]) == 0
    ) or "visual" in name.lower()


def env_offset(env_id: int, cols: int, spacing: float) -> np.ndarray:
    """Return a flat XY offset for an environment cell."""

    return np.array(
        [(env_id % cols) * spacing, (env_id // cols) * spacing, 0.02],
        dtype=float,
    )


def mesh_vertices_faces(
    model: mujoco.MjModel,
    mesh_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice one MuJoCo mesh asset into Viser vertex/face arrays."""

    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vert_adr : vert_adr + vert_num],
        dtype=np.float32,
    )
    faces = np.asarray(
        model.mesh_face[face_adr : face_adr + face_num],
        dtype=np.uint32,
    )
    return vertices, faces


def geom_color(
    model: mujoco.MjModel, geom_id: int
) -> tuple[tuple[int, int, int], float]:
    """Return display color/opacity from material or geom RGBA."""

    mat_id = int(model.geom_matid[geom_id])
    rgba = (
        np.asarray(model.mat_rgba[mat_id], dtype=float)
        if mat_id >= 0
        else np.asarray(model.geom_rgba[geom_id], dtype=float)
    )
    color = tuple(int(np.clip(round(c * 255.0), 0, 255)) for c in rgba[:3])
    opacity = float(np.clip(rgba[3] if rgba[3] > 0.0 else 1.0, 0.05, 1.0))
    return color, opacity


if __name__ == "__main__":
    raise SystemExit(main())
