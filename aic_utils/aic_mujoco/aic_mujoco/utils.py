from __future__ import annotations

import mujoco
import numpy as np


def transform_from_rotation_translation(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    return np.linalg.inv(transform)


def relative_transform(frame_T_a: np.ndarray, frame_T_b: np.ndarray) -> np.ndarray:
    """Return a_T_b from frame_T_a and frame_T_b."""

    return inverse_transform(frame_T_a) @ frame_T_b


def quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    quaternion = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, dtype=float).reshape(9))
    return quaternion


def transform_from_translation_quaternion(
    translation: np.ndarray,
    quaternion: np.ndarray,
) -> np.ndarray:
    rotation = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion, dtype=float).reshape(4))
    return transform_from_rotation_translation(
        rotation.reshape(3, 3),
        np.asarray(translation, dtype=float),
    )


def body_transform(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise RuntimeError(f"Body not found in MuJoCo model: {body_name!r}")
    return transform_from_rotation_translation(data.xmat[body_id], data.xpos[body_id])


def site_transform(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
) -> np.ndarray:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"Site not found in MuJoCo model: {site_name!r}")
    return transform_from_rotation_translation(
        data.site_xmat[site_id],
        data.site_xpos[site_id],
    )


def set_freejoint_transform(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    world_T_body: np.ndarray,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Freejoint not found in MuJoCo model: {joint_name!r}")
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError(f"Joint {joint_name!r} is not a freejoint.")

    qpos_addr = int(model.jnt_qposadr[joint_id])
    qvel_addr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_addr : qpos_addr + 3] = world_T_body[:3, 3]
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quaternion_from_rotation(
        world_T_body[:3, :3]
    )
    data.qvel[qvel_addr : qvel_addr + 6] = 0.0
    mujoco.mj_forward(model, data)


def weld_body_transform(
    model: mujoco.MjModel,
    body_a_name: str,
    body_b_name: str,
) -> np.ndarray | None:
    """Return a_T_b from a weld equality when present."""

    body_a_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_a_name)
    body_b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_b_name)
    if body_a_id < 0 or body_b_id < 0:
        return None

    for eq_id in range(model.neq):
        if model.eq_type[eq_id] != mujoco.mjtEq.mjEQ_WELD:
            continue
        obj1_id = int(model.eq_obj1id[eq_id])
        obj2_id = int(model.eq_obj2id[eq_id])
        # MuJoCo stores weld relpose in eq_data as:
        # [anchor_xyz, relpose_xyz, relpose_quat, torquescale].
        obj1_T_obj2 = transform_from_translation_quaternion(
            model.eq_data[eq_id, 3:6],
            model.eq_data[eq_id, 6:10],
        )
        if obj1_id == body_a_id and obj2_id == body_b_id:
            return obj1_T_obj2
        if obj1_id == body_b_id and obj2_id == body_a_id:
            return inverse_transform(obj1_T_obj2)

    return None


def print_transform(label: str, transform: np.ndarray) -> None:
    print(f"  {label}:")
    print(f"    p: {transform[:3, 3].tolist()}")
    print("    R:")
    for row in transform[:3, :3]:
        print(f"      {row.tolist()}")


def compute_preinsert_joint_target(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled,
    home_q: np.ndarray,
    port_body: str,
    tcp_site: str,
    sfp_tip_body: str,
    tool_body: str,
    weld_child_body: str,
    height: float,
    payload_root_body: str | None = None,
    payload_root_freejoint: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray | str | float | int | None]]:
    """Solve IK for an SFP-tip pre-insertion pose.

    Transform convention:
      A_T_B means transform of frame B with respect to frame A.

    Frame names in the generated AIC MuJoCo scene:
      world_T_port:      body ``sfp_port_0_link_entrance`` by default.
      world_T_tcp:       site ``gripper_tcp``.
      world_T_tool:      body ``ati/tool_link``.
      world_T_weld_child body ``lc_plug_link``.
      world_T_sfp_tip:   body ``sfp_tip_link``.

    The robust held-plug transform is derived from topology, not current free-body
    placement:

      tcp_T_sfp_tip =
          inv(tool_T_tcp)
          @ tool_T_weld_child
          @ weld_child_T_sfp_tip

    ``tool_T_weld_child`` comes from the MuJoCo weld equality between
    ``ati/tool_link`` and ``lc_plug_link``. This avoids using the raw current
    ``world_T_sfp_tip`` pose before the cable free body has been aligned.

    Desired pre-insertion target:

      desired_world_T_sfp_tip = world_T_port
      desired_world_T_sfp_tip.p.z += height

    IK target:

      desired_world_T_tcp =
          desired_world_T_sfp_tip @ inv(tcp_T_sfp_tip)

      desired_base_T_tcp =
          inv(world_T_base) @ desired_world_T_tcp

    ``EnsemblRobot.ComputeIK`` targets ``gripper/tcp``, so the TCP transform is
    passed directly to the AIC/Tesseract IK solver.
    """

    from aic_model.robot import EnsemblRobot

    home_q = np.asarray(home_q, dtype=float)
    if home_q.shape != (controlled.n,):
        raise ValueError(
            f"home_q must have shape ({controlled.n},), got {home_q.shape}"
        )

    world_T_port = body_transform(model, data, port_body)

    controlled.set_q(data, home_q, zero_velocity=True)
    mujoco.mj_forward(model, data)

    robot = EnsemblRobot()
    robot.SetActiveDOFValues(home_q)
    base_T_tcp_home = robot.ComputeFK()
    world_T_tcp_home = site_transform(model, data, tcp_site)
    world_T_base = world_T_tcp_home @ inverse_transform(base_T_tcp_home)

    world_T_tcp = site_transform(model, data, tcp_site)
    world_T_tool = body_transform(model, data, tool_body)
    world_T_weld_child = body_transform(model, data, weld_child_body)
    world_T_sfp_tip = body_transform(model, data, sfp_tip_body)

    tool_T_tcp = relative_transform(world_T_tool, world_T_tcp)
    weld_child_T_sfp_tip = relative_transform(world_T_weld_child, world_T_sfp_tip)
    tool_T_weld_child = weld_body_transform(model, tool_body, weld_child_body)

    if tool_T_weld_child is None:
        raise RuntimeError(
            f"Required weld equality not found between {tool_body!r} and "
            f"{weld_child_body!r}."
        )
    tcp_T_sfp_tip = (
        inverse_transform(tool_T_tcp)
        @ tool_T_weld_child
        @ weld_child_T_sfp_tip
    )

    desired_world_T_sfp_tip = world_T_port.copy()
    desired_world_T_sfp_tip[2, 3] += float(height)

    desired_world_T_tcp = desired_world_T_sfp_tip @ inverse_transform(tcp_T_sfp_tip)
    desired_base_T_tcp = relative_transform(world_T_base, desired_world_T_tcp)

    raw_solutions = robot.ComputeIK(
        desired_base_T_tcp,
        return_all=True,
        check_collision=False,
    )
    if raw_solutions is None:
        solution_count = 0
        q_target = None
    else:
        solutions = np.atleast_2d(np.asarray(raw_solutions, dtype=float))
        solution_count = len(solutions)
        q_target = solutions[np.argmin(np.linalg.norm(solutions - home_q, axis=1))]

    diagnostics: dict[str, np.ndarray | str | float | int | None] = {
        "port_body": port_body,
        "tcp_site": tcp_site,
        "sfp_tip_body": sfp_tip_body,
        "tool_body": tool_body,
        "weld_child_body": weld_child_body,
        "payload_root_body": payload_root_body,
        "payload_root_freejoint": payload_root_freejoint,
        "height": float(height),
        "solution_count": solution_count,
        "world_T_port": world_T_port,
        "world_T_base": world_T_base,
        "world_T_tool": world_T_tool,
        "world_T_weld_child": world_T_weld_child,
        "world_T_sfp_tip": world_T_sfp_tip,
        "tool_T_tcp": tool_T_tcp,
        "weld_child_T_sfp_tip": weld_child_T_sfp_tip,
        "tool_T_weld_child": tool_T_weld_child,
        "tcp_T_sfp_tip": tcp_T_sfp_tip,
        "desired_world_T_sfp_tip": desired_world_T_sfp_tip,
        "desired_world_T_tcp": desired_world_T_tcp,
        "desired_base_T_tcp": desired_base_T_tcp,
    }

    if q_target is None:
        raise RuntimeError(
            "EnsemblRobot IK failed for SFP-tip pre-insertion target. This is "
            "raw IK, with no environment or cable collision filtering involved."
        )

    controlled.set_q(data, q_target, zero_velocity=True)
    mujoco.mj_forward(model, data)

    achieved_world_T_tcp = site_transform(model, data, tcp_site)
    virtual_world_T_sfp_tip = achieved_world_T_tcp @ tcp_T_sfp_tip
    diagnostics["achieved_world_T_tcp"] = achieved_world_T_tcp
    diagnostics["virtual_world_T_sfp_tip"] = virtual_world_T_sfp_tip

    if payload_root_body and payload_root_freejoint:
        actual_world_T_sfp_tip_before = body_transform(model, data, sfp_tip_body)
        world_T_payload_root = body_transform(model, data, payload_root_body)
        desired_world_T_payload_root = (
            desired_world_T_sfp_tip
            @ inverse_transform(actual_world_T_sfp_tip_before)
            @ world_T_payload_root
        )
        set_freejoint_transform(
            model,
            data,
            payload_root_freejoint,
            desired_world_T_payload_root,
        )
        diagnostics["actual_world_T_sfp_tip_before_payload_set"] = (
            actual_world_T_sfp_tip_before
        )

    diagnostics["actual_world_T_sfp_tip_after_payload_set"] = body_transform(
        model,
        data,
        sfp_tip_body,
    )
    return q_target, diagnostics
