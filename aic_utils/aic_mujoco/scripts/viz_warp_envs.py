#!/usr/bin/env python3
"""Visualize direct prototype envs.

This is a debug path, not a training path. It runs prototype env copies with
the same reset/action/observation/reward code as training.

Layouts:
  ``mujoco``:
    mirror one selected env into the normal MuJoCo viewer.

  ``viser``:
    show all envs in a flat floor grid using the real MuJoCo visual geoms.
    ``--num-envs`` controls the grid size. There is no per-env geom cap.
    The Viser view intentionally shows only the robot, held plug/SFP part,
    NIC/task board, and enclosure visuals. It does not draw collision geoms,
    frames, labels, fake boxes, force arrows, floor, room walls, or cable tail.
    The default demo moves downward by 0.15 m, then holds. The environment
    clips each action component to [-1, 1], so ``abs(--action-z) > 1`` is used
    as a visual speed multiplier: ``--action-z -5`` runs five clipped env steps
    per rendered frame. Increase ``--fps`` for faster visual playback.
    This is for debugging and visual sanity checks, not fast training.

Fresh run from a new ``aic_eval`` distrobox terminal:

  cd /home/rmalhan/Software/ws_aic/src/aic
  pixi shell
  source /opt/ros/kilted/setup.bash
  source /home/rmalhan/Software/ws_aic/install/setup.bash
  export PYTHONNOUSERSITE=1
  export MUJOCO_PLUGIN_PATH=/home/rmalhan/Software/ws_aic/install/opt/mujoco_vendor/lib

  PYTHONPATH=/home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco:/home/rmalhan/Software/ws_aic/src/aic/aic_model \
  python3 aic_utils/aic_mujoco/scripts/viz_warp_envs.py --num-envs 16

  # Faster visual debug:
  PYTHONPATH=/home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco:/home/rmalhan/Software/ws_aic/src/aic/aic_model \
  python3 aic_utils/aic_mujoco/scripts/viz_warp_envs.py --num-envs 16 --action-z -5 --fps 40

  PYTHONPATH=/home/rmalhan/Software/ws_aic/src/aic/aic_utils/aic_mujoco:/home/rmalhan/Software/ws_aic/src/aic/aic_model \
  python3 aic_utils/aic_mujoco/scripts/viz_warp_envs.py --layout mujoco --env-id 0
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from aic_mujoco.warp import AicInsertionVecEnv, AicInsertionVecEnvConfig
from aic_mujoco.utils import quaternion_from_rotation

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_XML = PACKAGE_ROOT / "mjcf" / "scene.xml"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "experiments" / "train_warp_smoke.json"


def make_parser() -> argparse.ArgumentParser:
    """Build CLI for Viser-grid and single-env MuJoCo visualization."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action-z", type=float, default=-1.0)
    parser.add_argument("--down-distance", type=float, default=0.15)
    parser.add_argument("--layout", choices=("mujoco", "viser"), default="viser")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--spacing", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument(
        "--show-grid", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--show-geoms", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--geom-opacity", type=float, default=1.0)
    parser.add_argument(
        "--geom-mode",
        choices=("visual", "collision", "all"),
        default="visual",
    )
    return parser


def main() -> int:
    """Create the prototype env and dispatch to Viser or native MuJoCo view."""

    args = make_parser().parse_args()
    env = AicInsertionVecEnv(
        AicInsertionVecEnvConfig.from_files(
            xml_path=Path(args.xml).expanduser().resolve(),
            config_path=Path(args.config).expanduser().resolve(),
            num_envs=int(args.num_envs),
            device=str(args.device),
        )
    )
    env_id = int(args.env_id)
    action = torch.zeros(
        (env.num_envs, env.num_actions),
        dtype=torch.float32,
        device=env.device,
    )

    if args.layout == "viser":
        run_viser_grid(env, action, args)
        return 0

    print("Close viewer to stop.")
    print(f"Visualizing env {env_id} of {env.num_envs}.")
    traveled = np.zeros(env.num_envs, dtype=float)
    with mujoco.viewer.launch_passive(env.model, env.datas[env_id]) as viewer:
        while viewer.is_running():
            step_visual_policy(
                env,
                action,
                traveled,
                action_z=float(args.action_z),
                down_distance=float(args.down_distance),
            )
            viewer.sync()
            time.sleep(env.physics_dt)
    return 0


def run_viser_grid(
    env: AicInsertionVecEnv,
    action: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    """Run all envs and visualize their real visual geoms in Viser.

    Args:
        env: Direct prototype vector env to step.
        action: Reused action tensor; mutated in place each frame.
        args: Parsed CLI namespace with layout, speed, and display controls.

    Viser is deliberately debug-only. It steps the same env as training, but it
    renders only selected visual geoms and sleeps to a target browser FPS.
    """

    import viser

    server = viser.ViserServer(port=int(args.port))
    cols = int(math.ceil(math.sqrt(env.num_envs)))
    rows = int(math.ceil(env.num_envs / cols))
    spacing = float(args.spacing)
    handles = {}

    if bool(args.show_grid):
        server.scene.add_grid(
            "/grid",
            width=max(cols * spacing, spacing),
            height=max(rows * spacing, spacing),
            plane="xy",
            cell_size=spacing,
            cell_thickness=0.8,
            section_size=spacing,
            section_thickness=2.0,
            cell_color=(150, 150, 150),
            section_color=(40, 40, 40),
            plane_color=(245, 245, 245),
            plane_opacity=0.18,
            position=((cols - 1) * spacing / 2.0, (rows - 1) * spacing / 2.0, -0.03),
        )

    for env_id in range(env.num_envs):
        offset = env_offset(env_id, cols, spacing)
        if bool(args.show_geoms):
            add_env_geoms(
                server=server,
                env=env,
                env_id=env_id,
                offset=offset,
                handles=handles,
                opacity=float(args.geom_opacity),
                geom_mode=str(args.geom_mode),
            )

    print(f"Viser grid running at http://localhost:{int(args.port)}")
    print("Ctrl-C to stop.")
    period = 1.0 / max(float(args.fps), 1.0)
    traveled = np.zeros(env.num_envs, dtype=float)
    while True:
        t0 = time.perf_counter()
        step_visual_policy(
            env,
            action,
            traveled,
            action_z=float(args.action_z),
            down_distance=float(args.down_distance),
        )
        for env_id in range(env.num_envs):
            update_env_geoms(env, env_id, handles, env_offset(env_id, cols, spacing))
        dt = time.perf_counter() - t0
        time.sleep(max(0.0, period - dt))


def step_visual_policy(
    env: AicInsertionVecEnv,
    action: torch.Tensor,
    traveled: np.ndarray,
    action_z: float,
    down_distance: float,
) -> None:
    """Advance the visual demo policy for one rendered frame.

    Args:
        env: Prototype env to step.
        action: Preallocated action tensor.
        traveled: Per-env accumulated downward travel in meters.
        action_z: User speed command. Values outside [-1, 1] become repeated
            clipped env steps because the env itself clips policy actions.
        down_distance: Total downward travel before holding still.
    """

    repeats = max(1, int(math.ceil(abs(action_z))))
    clipped_action_z = float(np.clip(action_z, -1.0, 1.0))
    for _ in range(repeats):
        set_downward_actions(action, traveled, clipped_action_z, down_distance)
        _, _, dones, _ = env.step(action)
        update_downward_travel(
            traveled,
            dones.detach().cpu().numpy(),
            clipped_action_z,
            down_distance,
            env.action_scale,
            env.decimation,
        )


def set_downward_actions(
    action: torch.Tensor,
    traveled: np.ndarray,
    action_z: float,
    down_distance: float,
) -> None:
    """Write the downward Cartesian action for envs still in motion.

    ``action_z`` should already be clipped to the env's normalized action
    range. Env copies that reached ``down_distance`` receive zero action and
    therefore hold through the impedance controller.
    """

    action.zero_()
    active = traveled < abs(down_distance)
    if np.any(active):
        action[torch.as_tensor(active, dtype=torch.bool, device=action.device), 2] = (
            action_z
        )


def update_downward_travel(
    traveled: np.ndarray,
    dones: np.ndarray,
    action_z: float,
    down_distance: float,
    action_scale: float,
    decimation: int,
) -> None:
    """Update the visual policy's travel bookkeeping.

    The distance estimate mirrors what the env actually receives after action
    clipping: ``abs(action_z) * action_scale * decimation`` per policy step.
    Done envs are reset to zero so the next episode repeats the same 15 cm
    insertion debug motion.
    """

    step_distance = abs(action_z) * abs(action_scale) * int(decimation)
    active = traveled < abs(down_distance)
    traveled[active] = np.minimum(abs(down_distance), traveled[active] + step_distance)
    traveled[dones.astype(bool)] = 0.0


def env_offset(env_id: int, cols: int, spacing: float) -> np.ndarray:
    """Return a flat XY grid offset for an env copy in Viser."""

    row = env_id // cols
    col = env_id % cols
    return np.array([col * spacing, row * spacing, 0.02], dtype=float)


def add_env_geoms(
    server,
    env: AicInsertionVecEnv,
    env_id: int,
    offset: np.ndarray,
    handles: dict,
    opacity: float,
    geom_mode: str,
) -> None:
    """Add MuJoCo geoms for one Viser env cell.

    Args:
        server: Active ``viser.ViserServer``.
        env: Prototype env containing model/data.
        env_id: Env copy to draw.
        offset: XY grid offset for this env.
        handles: Mutable map from geom ids to Viser handles.
        opacity: Alpha cap for displayed geoms.
        geom_mode: ``visual``, ``collision``, or ``all`` selection.

    Mesh geoms are shown as actual mesh vertices/faces. Primitive geoms are
    shown with matching primitive approximations. The default filter avoids
    collision bodies and room walls so the view stays robot + part + board.
    """

    model = env.model
    data = env.datas[env_id]
    for geom_id in range(model.ngeom):
        geom_type = int(model.geom_type[geom_id])
        if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            continue

        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not geom_name:
            geom_name = f"geom_{geom_id}"
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not include_geom(body_name, geom_name, geom_mode):
            continue
        color, geom_alpha = geom_color_alpha(
            model, geom_id, body_name, geom_name, opacity
        )
        size = np.asarray(model.geom_size[geom_id], dtype=float)
        pos = data.geom_xpos[geom_id] + offset
        quat = quaternion_from_rotation(data.geom_xmat[geom_id].reshape(3, 3))
        node = f"/env_{env_id:03d}/geoms/{sanitize_name(geom_name)}"

        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            vertices, faces = mesh_vertices_faces(model, mesh_id)
            handle = server.scene.add_mesh_simple(
                node,
                vertices=vertices,
                faces=faces,
                color=color,
                opacity=geom_alpha,
                flat_shading=False,
                side="double",
                position=pos,
                wxyz=quat,
            )
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            handle = server.scene.add_icosphere(
                node,
                radius=max(float(size[0]), 1e-4),
                color=color,
                subdivisions=1,
                opacity=geom_alpha,
                position=pos,
                wxyz=quat,
            )
        elif geom_type in {
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
        }:
            handle = server.scene.add_cylinder(
                node,
                radius=max(float(size[0]), 1e-4),
                height=max(2.0 * float(size[1]), 1e-4),
                color=color,
                radial_segments=12,
                opacity=geom_alpha,
                position=pos,
                wxyz=quat,
            )
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            handle = server.scene.add_box(
                node,
                color=color,
                dimensions=np.maximum(2.0 * size[:3], 1e-4),
                opacity=geom_alpha,
                position=pos,
                wxyz=quat,
            )
        else:
            continue

        handles[(env_id, "geom", geom_id)] = handle


def update_env_geoms(
    env: AicInsertionVecEnv,
    env_id: int,
    handles: dict,
    offset: np.ndarray,
) -> None:
    """Refresh Viser geom poses from current MuJoCo state."""

    data = env.datas[env_id]
    for geom_id in range(env.model.ngeom):
        handle = handles.get((env_id, "geom", geom_id))
        if handle is None:
            continue
        handle.position = data.geom_xpos[geom_id] + offset
        handle.wxyz = quaternion_from_rotation(data.geom_xmat[geom_id].reshape(3, 3))


def include_geom(body_name: str, geom_name: str, geom_mode: str) -> bool:
    """Return whether a MuJoCo geom belongs in the Viser debug scene.

    Args:
        body_name: Parent MuJoCo body name.
        geom_name: MuJoCo geom name.
        geom_mode: User-selected mode: ``visual``, ``collision``, or ``all``.

    The default visual path intentionally excludes floor/walls, cameras, cable
    tail, SC plug, and collision boxes. The goal is a clean robot + held SFP
    part + NIC/task board + enclosure view.
    """

    text = f"{body_name} {geom_name}".lower()
    if any(k in text for k in ("floor_link", "walls_visual", "light_visual")):
        return False
    if any(k in text for k in ("basler", "camera", "cam_mount")):
        return False
    if any(k in text for k in ("cable_connection", "sc_plug")):
        return False
    if geom_mode == "all":
        return True
    if geom_mode == "visual":
        return "visual" in geom_name.lower()
    if geom_mode == "collision":
        name = geom_name.lower()
        return "collision" in name or "collider" in name
    raise ValueError(f"Unsupported geom_mode: {geom_mode}")


def mesh_vertices_faces(
    model: mujoco.MjModel,
    mesh_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices/faces for a MuJoCo mesh asset.

    MuJoCo stores all mesh vertices/faces in flat global arrays; this slices the
    segment for one mesh id into arrays Viser can draw.
    """

    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])
    vertices = np.asarray(
        model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float32
    )
    faces = np.asarray(model.mesh_face[face_adr : face_adr + face_num], dtype=np.uint32)
    return vertices, faces


def geom_color_alpha(
    model: mujoco.MjModel,
    geom_id: int,
    body_name: str,
    geom_name: str,
    opacity: float,
) -> tuple[tuple[int, int, int], float]:
    """Choose display color/alpha for a MuJoCo geom.

    Material colors are preferred to preserve exported scene colors. When the
    converter leaves a neutral default gray, semantic colors make robot, plug,
    board, and enclosure easier to distinguish in Viser.
    """

    mat_id = int(model.geom_matid[geom_id])
    if mat_id >= 0:
        rgba = np.asarray(model.mat_rgba[mat_id], dtype=float)
        color = tuple(int(np.clip(round(c * 255.0), 0, 255)) for c in rgba[:3])
        alpha = float(np.clip(rgba[3] if rgba[3] > 0.0 else opacity, 0.05, opacity))
        return color, alpha

    rgba = np.asarray(model.geom_rgba[geom_id], dtype=float)
    if float(np.max(rgba[:3])) > 0.0 and not np.allclose(rgba[:3], 0.5):
        color = tuple(int(np.clip(round(c * 255.0), 0, 255)) for c in rgba[:3])
    else:
        color = semantic_color(body_name, geom_name)
    alpha = float(np.clip(rgba[3] if rgba[3] > 0.0 else opacity, 0.05, opacity))
    return color, alpha


def semantic_color(body_name: str, geom_name: str) -> tuple[int, int, int]:
    """Return a stable debug color when material color is uninformative."""

    text = f"{body_name} {geom_name}".lower()
    if any(
        k in text
        for k in (
            "shoulder",
            "upper_arm",
            "forearm",
            "wrist",
            "gripper",
            "finger",
            "tool",
            "ati",
        )
    ):
        return (180, 180, 190)
    if any(k in text for k in ("sfp", "plug", "lc_plug", "sc_plug")):
        return (255, 175, 40)
    if any(k in text for k in ("nic", "port", "task_board", "enclosure")):
        return (80, 185, 255)
    if any(k in text for k in ("table", "floor", "wall")):
        return (80, 90, 95)
    return (140, 145, 150)


def sanitize_name(name: str) -> str:
    """Make a MuJoCo name safe as a Viser scene-tree path component."""

    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


if __name__ == "__main__":
    raise SystemExit(main())
