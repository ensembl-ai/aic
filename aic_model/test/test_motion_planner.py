#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np
import tqdm
from transforms3d.euler import euler2mat, mat2euler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_model.robot import EnsemblRobot
from tesseract_robotics.tesseract_command_language import (
    InstructionPoly_as_MoveInstructionPoly,
    WaypointPoly_as_StateWaypointPoly,
)


HOME_JOINT_POSITIONS = np.array(
    [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110],
    dtype=np.float64,
)
POSITION_ROI_OFFSETS_METERS = np.array(
    [
        [-0.2, 0.2],
        [0.4, -0.4],
        [0.2, -0.3],
    ],
    dtype=np.float64,
)
ORIENTATION_ROI_HALF_WIDTH_RADIANS = np.deg2rad(20.0)


def transform_to_position_euler(transform):
    return (
        transform[:3, 3],
        np.array(mat2euler(transform[:3, :3])),
    )


def position_euler_to_transform(position, euler):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler2mat(*euler)
    transform[:3, 3] = position
    return transform


def sample_roi_transform(
    rng,
    anchor_transform,
    position_offsets=POSITION_ROI_OFFSETS_METERS,
    orientation_half_width=ORIENTATION_ROI_HALF_WIDTH_RADIANS,
):
    anchor_position, anchor_euler = transform_to_position_euler(anchor_transform)
    position_offsets = np.asarray(position_offsets, dtype=np.float64)
    position_low = np.min(position_offsets, axis=1)
    position_high = np.max(position_offsets, axis=1)
    return position_euler_to_transform(
        anchor_position + rng.uniform(position_low, position_high),
        anchor_euler
        + rng.uniform(-orientation_half_width, orientation_half_width, size=3),
    )


def closest_joint_solution(solutions, reference_config):
    solutions = np.atleast_2d(solutions)
    return solutions[np.argmin(np.linalg.norm(solutions - reference_config, axis=1))]


def max_abs_joint_delta(a, b):
    return float(np.max(np.abs(a - b)))


def format_vector(values, precision=3):
    return np.array2string(
        np.asarray(values, dtype=np.float64),
        precision=precision,
        suppress_small=True,
    )


def format_percent(count, total):
    if total == 0:
        return "n/a"
    return f"{count / total:.2%}"


def enabled_label(enabled):
    return "enabled" if enabled else "disabled"


def sample_reachable_roi_target(
    robot,
    rng,
    anchor_transform,
    reference_config,
    orientation_half_width,
    max_joint_delta,
    max_attempts=1000,
):
    for _ in range(max_attempts):
        robot.SetActiveDOFValues(reference_config)
        target_transform = sample_roi_transform(
            rng,
            anchor_transform,
            orientation_half_width=orientation_half_width,
        )
        solutions = robot.ComputeIK(
            target_transform,
            return_all=True,
            check_collision=True,
        )
        if solutions is None:
            continue

        target_config = closest_joint_solution(solutions, reference_config)
        if max_abs_joint_delta(target_config, reference_config) > max_joint_delta:
            continue
        return target_transform, target_config

    raise RuntimeError(
        f"Unable to find a reachable ROI target after {max_attempts} attempts."
    )


def rotation_error_radians(actual, expected):
    rotation_delta = actual[:3, :3].T @ expected[:3, :3]
    trace = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(trace))


def extract_state_waypoints(program):
    state_waypoints = []
    for instruction in program.flatten():
        move = InstructionPoly_as_MoveInstructionPoly(instruction)
        waypoint = move.getWaypoint()
        if not waypoint.isStateWaypoint():
            continue
        state_waypoint = WaypointPoly_as_StateWaypointPoly(waypoint)
        state_waypoints.append(
            np.asarray(state_waypoint.getPosition(), dtype=np.float64).reshape(-1)
        )
    return state_waypoints


def max_joint_step(path):
    if len(path) < 2:
        return 0.0
    return float(
        max(
            np.max(np.abs(next_waypoint - waypoint))
            for waypoint, next_waypoint in zip(path[:-1], path[1:])
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", type=int, default=20000)
    parser.add_argument(
        "--num-planner-samples",
        type=int,
        default=250,
        help="Number of reachable ROI target planning samples to validate.",
    )
    parser.add_argument(
        "--planner-max-delta",
        type=float,
        default=float("inf"),
        help=(
            "Maximum absolute per-joint delta in radians from the home "
            "configuration for accepted planning IK goals."
        ),
    )
    parser.add_argument(
        "--orientation-roi-half-width-deg",
        type=float,
        default=np.rad2deg(ORIENTATION_ROI_HALF_WIDTH_RADIANS),
        help="Half-width in degrees for RPY target sampling around home FK.",
    )
    parser.add_argument(
        "--reachable-sample-attempts",
        type=int,
        default=1000,
        help="Maximum ROI samples to try for each reachable planning target.",
    )
    parser.add_argument(
        "--collision-check-ik",
        action="store_true",
        help="Enable collision filtering for the standalone IK/FK sample test.",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    orientation_roi_half_width = np.deg2rad(args.orientation_roi_half_width_deg)

    robot = EnsemblRobot()
    home_config = HOME_JOINT_POSITIONS.copy()
    robot.SetActiveDOFValues(home_config)
    if robot.CheckCollision():
        raise RuntimeError("Home joint positions are in collision.")
    home_transform = robot.ComputeFK()
    home_position, home_euler = transform_to_position_euler(home_transform)

    print("\nHome configuration")
    print("------------------")
    print(f"joints:     {np.array2string(home_config, precision=4)}")
    print(f"position:   {np.array2string(home_position, precision=4)}")
    print(f"euler_xyz:  {np.array2string(home_euler, precision=4)}")

    ik_failures = 0
    checked_solutions = 0
    bad_solutions = []
    position_tolerance = 1e-4
    orientation_tolerance = 1e-3
    for sample_index in tqdm.tqdm(range(args.num_samples), desc="IK/FK"):
        robot.SetActiveDOFValues(home_config)
        target_transform = sample_roi_transform(
            rng,
            home_transform,
            orientation_half_width=orientation_roi_half_width,
        )
        solutions = robot.ComputeIK(
            target_transform,
            return_all=True,
            check_collision=args.collision_check_ik,
        )
        if solutions is None:
            ik_failures += 1
            continue

        closest_solution = closest_joint_solution(solutions, home_config)
        robot.SetActiveDOFValues(closest_solution)
        fk = robot.ComputeFK()
        position_error = float(
            np.linalg.norm(fk[:3, 3] - target_transform[:3, 3])
        )
        orientation_error = rotation_error_radians(fk, target_transform)
        checked_solutions += 1
        if (
            position_error > position_tolerance
            or orientation_error > orientation_tolerance
        ):
            bad_solutions.append(
                (sample_index, 0, position_error, orientation_error)
            )

    successful_samples = args.num_samples - ik_failures
    print("\nIK/FK sampled validation")
    print("------------------------")
    print(f"samples:           {args.num_samples}")
    print(f"ik successes:      {successful_samples}")
    print(f"ik failures:       {ik_failures}")
    print(f"ik success rate:   {successful_samples / args.num_samples:.2%}")
    print(f"collision filter:  {enabled_label(args.collision_check_ik)}")
    print(f"closest solutions: {checked_solutions}")
    print(f"bad FK matches:    {len(bad_solutions)}")

    planner_failures = []
    planner_path_failures = []
    planner_goal_failures = []
    planner_collision_failures = []
    planner_goal_position_tolerance = 1e-4
    planner_goal_orientation_tolerance = 1e-3
    planner_goal_joint_tolerance = 1e-6
    planner_step_limit = np.pi
    planner_start_config = home_config.copy()

    for sample_index in tqdm.tqdm(
        range(args.num_planner_samples),
        desc="Motion planning",
    ):
        start_config = planner_start_config.copy()
        try:
            goal_transform, goal_config = sample_reachable_roi_target(
                robot,
                rng,
                home_transform,
                start_config,
                orientation_roi_half_width,
                args.planner_max_delta,
                max_attempts=args.reachable_sample_attempts,
            )
        except RuntimeError as exc:
            planner_failures.append(
                {
                    "sample_index": sample_index,
                    "reason": str(exc),
                    "goal_config": None,
                    "goal_position": None,
                    "goal_euler": None,
                    "max_joint_delta": None,
                    "goal_collision": None,
                }
            )
            continue

        robot.SetActiveDOFValues(start_config)
        plan = robot.PlanToConfiguration(goal_config)
        if plan is None:
            goal_position, goal_euler = transform_to_position_euler(goal_transform)
            robot.SetActiveDOFValues(goal_config)
            planner_failures.append(
                {
                    "sample_index": sample_index,
                    "reason": (
                        robot._planner.last_failure_reason
                        or "PlanToConfiguration returned None."
                    ),
                    "goal_config": goal_config.copy(),
                    "goal_position": goal_position.copy(),
                    "goal_euler": goal_euler.copy(),
                    "max_joint_delta": max_abs_joint_delta(
                        goal_config,
                        start_config,
                    ),
                    "goal_collision": bool(robot.CheckCollision()),
                }
            )
            continue

        state_waypoints = extract_state_waypoints(plan.results)
        if len(state_waypoints) < 2:
            planner_path_failures.append(
                {
                    "sample_index": sample_index,
                    "reason": "planned result has fewer than two state waypoints",
                    "state_waypoint_count": len(state_waypoints),
                    "max_joint_step": None,
                    "goal_config": goal_config.copy(),
                }
            )
            continue

        path_max_step = max_joint_step(state_waypoints)
        if path_max_step > planner_step_limit:
            planner_path_failures.append(
                {
                    "sample_index": sample_index,
                    "reason": "path max joint step exceeds limit",
                    "state_waypoint_count": len(state_waypoints),
                    "max_joint_step": path_max_step,
                    "goal_config": goal_config.copy(),
                }
            )
            continue

        in_collision = False
        for waypoint_index, waypoint in enumerate(state_waypoints):
            robot.SetActiveDOFValues(waypoint)
            in_collision, collision_report = robot.CheckCollision(report=True)
            if in_collision:
                planner_collision_failures.append(
                    {
                        "sample_index": sample_index,
                        "waypoint_index": waypoint_index,
                        "waypoint": waypoint.copy(),
                        "contacts": collision_report[:3],
                    }
                )
                in_collision = True
                break
        if in_collision:
            continue

        final_waypoint = state_waypoints[-1]
        joint_error = float(np.max(np.abs(final_waypoint - goal_config)))
        if joint_error > planner_goal_joint_tolerance:
            planner_goal_failures.append(
                {
                    "sample_index": sample_index,
                    "reason": "final waypoint does not match target joints",
                    "joint_error": joint_error,
                    "position_error": None,
                    "orientation_error": None,
                    "goal_config": goal_config.copy(),
                    "final_config": final_waypoint.copy(),
                }
            )
            continue

        robot.SetActiveDOFValues(final_waypoint)
        final_transform = robot.ComputeFK()
        goal_position, goal_euler = transform_to_position_euler(goal_transform)
        final_position, final_euler = transform_to_position_euler(final_transform)
        position_error = float(
            np.linalg.norm(final_transform[:3, 3] - goal_transform[:3, 3])
        )
        orientation_error = rotation_error_radians(final_transform, goal_transform)
        if (
            position_error > planner_goal_position_tolerance
            or orientation_error > planner_goal_orientation_tolerance
        ):
            planner_goal_failures.append(
                {
                    "sample_index": sample_index,
                    "reason": "final FK does not match target transform",
                    "joint_error": joint_error,
                    "position_error": position_error,
                    "orientation_error": orientation_error,
                    "goal_position": goal_position.copy(),
                    "goal_euler": goal_euler.copy(),
                    "final_position": final_position.copy(),
                    "final_euler": final_euler.copy(),
                }
            )

    total_planner_failures = (
        len(planner_failures)
        + len(planner_path_failures)
        + len(planner_collision_failures)
        + len(planner_goal_failures)
    )
    planner_successes = args.num_planner_samples - total_planner_failures

    print("\nMotion planner validation")
    print("-------------------------")
    print(f"samples:              {args.num_planner_samples}")
    print("collision checking:   enabled")
    print(
        f"successes:            {planner_successes} "
        f"({format_percent(planner_successes, args.num_planner_samples)})"
    )
    print(
        f"planner failures:     {len(planner_failures)} "
        f"({format_percent(len(planner_failures), args.num_planner_samples)})"
    )
    print(
        f"path shape failures:  {len(planner_path_failures)} "
        f"({format_percent(len(planner_path_failures), args.num_planner_samples)})"
    )
    print(
        f"collision failures:   {len(planner_collision_failures)} "
        f"({format_percent(len(planner_collision_failures), args.num_planner_samples)})"
    )
    print(
        f"goal mismatch count:  {len(planner_goal_failures)} "
        f"({format_percent(len(planner_goal_failures), args.num_planner_samples)})"
    )

    if bad_solutions:
        print("\nFirst bad FK matches:")
        for (
            sample_index,
            solution_index,
            position_error,
            orientation_error,
        ) in bad_solutions[:20]:
            print(
                f"  sample={sample_index} solution={solution_index} "
                f"pos_err={position_error:.3e} rot_err={orientation_error:.3e}"
            )

    if planner_failures:
        print("\nFirst planner failures:")
        for failure in planner_failures[:10]:
            print(
                f"  sample={failure['sample_index']} "
                f"reason={failure['reason']}"
            )
            if failure["goal_config"] is None:
                continue
            print(
                f"    max_joint_delta={failure['max_joint_delta']:.3f} "
                f"goal_collision={failure['goal_collision']}"
            )
            print(f"    goal_joints={format_vector(failure['goal_config'])}")
            print(
                f"    goal_position={format_vector(failure['goal_position'])} "
                f"goal_euler_xyz={format_vector(failure['goal_euler'])}"
            )

    if planner_path_failures:
        print("\nFirst planner path failures:")
        for failure in planner_path_failures[:10]:
            print(
                f"  sample={failure['sample_index']} "
                f"reason={failure['reason']}"
            )
            print(
                f"    state_waypoints={failure['state_waypoint_count']} "
                f"max_joint_step={failure['max_joint_step']}"
            )
            print(f"    goal_joints={format_vector(failure['goal_config'])}")

    if planner_collision_failures:
        print("\nFirst planner collision failures:")
        for failure in planner_collision_failures[:10]:
            print(
                f"  sample={failure['sample_index']} "
                f"waypoint_index={failure['waypoint_index']}"
            )
            print(f"    waypoint={format_vector(failure['waypoint'])}")
            for contact in failure["contacts"]:
                print(
                    f"    contact_pair={contact['pair']} "
                    f"distance={contact['distance']:.3e}"
                )

    if planner_goal_failures:
        print("\nFirst planner goal mismatches:")
        for failure in planner_goal_failures[:10]:
            print(
                f"  sample={failure['sample_index']} "
                f"reason={failure['reason']}"
            )
            print(f"    joint_err={failure['joint_error']:.3e}")
            if failure["position_error"] is None:
                print(f"    goal_joints={format_vector(failure['goal_config'])}")
                print(f"    final_joints={format_vector(failure['final_config'])}")
            else:
                print(
                    f"    pos_err={failure['position_error']:.3e} "
                    f"rot_err={failure['orientation_error']:.3e}"
                )
                print(
                    f"    goal_position={format_vector(failure['goal_position'])} "
                    f"goal_euler_xyz={format_vector(failure['goal_euler'])}"
                )
                print(
                    f"    final_position={format_vector(failure['final_position'])} "
                    f"final_euler_xyz={format_vector(failure['final_euler'])}"
                )

    if (
        bad_solutions
        or planner_failures
        or planner_path_failures
        or planner_collision_failures
        or planner_goal_failures
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
