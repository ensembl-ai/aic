#!/usr/bin/env python3

import argparse
from collections import defaultdict
from functools import wraps
from pathlib import Path
import sys
import time
from typing import Any, cast

import numpy as np
import tqdm
from transforms3d.euler import euler2mat, mat2euler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_model.robot import EnsemblRobot
from tesseract_robotics.tesseract_command_language import (
    InstructionPoly_as_MoveInstructionPoly,
    WaypointPoly_as_StateWaypointPoly,
)
from tesseract_robotics.tesseract_common import Isometry3d


def transform_to_position_euler(transform):
    iso = Isometry3d()
    iso.setMatrix(transform)
    return (
        np.asarray(iso.translation(), dtype=np.float64).reshape(3),
        np.asarray(mat2euler(iso.rotation()), dtype=np.float64),
    )


def position_euler_to_transform(position, euler):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = euler2mat(*euler)
    transform[:3, 3] = position
    iso = Isometry3d()
    iso.setMatrix(transform)
    return np.asarray(iso.matrix(), dtype=np.float64)


def rotation_error_radians(actual, expected):
    rotation_delta = actual[:3, :3].T @ expected[:3, :3]
    trace = np.clip((np.trace(rotation_delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(trace))


def trace_robot_methods(robot, method_names):
    stats = defaultdict(
        lambda: {
            "calls": 0,
            "total_seconds": 0.0,
            "min_seconds": float("inf"),
            "max_seconds": 0.0,
        }
    )

    for method_name in method_names:
        method = getattr(robot, method_name)

        @wraps(method)
        def traced_method(*args, __method=method, __name=method_name, **kwargs):
            start = time.perf_counter()
            try:
                result = __method(*args, **kwargs)
            except Exception:
                elapsed = time.perf_counter() - start
                method_stats = stats[__name]
                method_stats["calls"] += 1
                method_stats["total_seconds"] += elapsed
                method_stats["min_seconds"] = min(method_stats["min_seconds"], elapsed)
                method_stats["max_seconds"] = max(method_stats["max_seconds"], elapsed)
                raise
            elapsed = time.perf_counter() - start
            method_stats = stats[__name]
            method_stats["calls"] += 1
            method_stats["total_seconds"] += elapsed
            method_stats["min_seconds"] = min(method_stats["min_seconds"], elapsed)
            method_stats["max_seconds"] = max(method_stats["max_seconds"], elapsed)
            return result

        setattr(robot, method_name, traced_method)

    return stats


def print_trace_summary(stats):
    print("\nRobot trace summary")
    print("-------------------")
    for method_name, method_stats in stats.items():
        calls = method_stats["calls"]
        total_seconds = method_stats["total_seconds"]
        mean_seconds = (total_seconds / calls) if calls else 0.0
        min_seconds = method_stats["min_seconds"] if calls else 0.0
        max_seconds = method_stats["max_seconds"] if calls else 0.0
        print(
            f"{method_name}: mean={mean_seconds:.3e} s "
            f"max={max_seconds:.3e} s min={min_seconds:.3e} s"
        )


def sample_collision_free_config(robot, rng, limits, max_attempts=1000):
    for _ in range(max_attempts):
        config = rng.uniform(limits[0], limits[1])
        robot.SetActiveDOFValues(config)
        if not robot.CheckCollision():
            return config
    raise RuntimeError(
        f"Unable to find collision-free sample after {max_attempts} attempts."
    )


def sample_collision_free_neighbor(
    robot,
    rng,
    limits,
    anchor_config,
    max_delta=0.35,
    max_attempts=1000,
):
    anchor_config = np.asarray(anchor_config, dtype=np.float64).reshape(-1)
    for _ in range(max_attempts):
        candidate = anchor_config + rng.uniform(
            -max_delta,
            max_delta,
            size=anchor_config.shape,
        )
        candidate = np.clip(candidate, limits[0], limits[1])
        robot.SetActiveDOFValues(candidate)
        if not robot.CheckCollision():
            return candidate
    raise RuntimeError(
        f"Unable to find nearby collision-free sample after {max_attempts} attempts."
    )


def extract_state_waypoints(program):
    state_waypoints = []
    for instruction in program.flatten():
        move = InstructionPoly_as_MoveInstructionPoly(instruction)
        waypoint = move.getWaypoint()
        if not waypoint.isStateWaypoint():
            continue
        state_waypoint = WaypointPoly_as_StateWaypointPoly(waypoint)
        state_waypoints.append(
            np.asarray(
                state_waypoint.getPosition(),
                dtype=np.float64,
            ).reshape(-1)
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
        help="Number of collision-free start/goal planning samples to validate.",
    )
    parser.add_argument(
        "--planner-max-delta",
        type=float,
        default=0.1,
        help="Maximum per-joint delta in radians for sampled planning goals.",
    )
    parser.add_argument("--check-collision", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    robot = cast(Any, EnsemblRobot())
    trace_stats = trace_robot_methods(
        robot,
        (
            "ComputeFK",
            "ComputeIK",
            "CheckCollision",
            "PlanToConfiguration",
            "PlanToTarget",
        ),
    )
    limits = robot.GetActiveDOFLimits()

    ik_failures = 0
    checked_solutions = 0
    bad_solutions = []
    position_tolerance = 1e-4
    orientation_tolerance = 1e-3
    for sample_index in tqdm.tqdm(range(args.num_samples), desc="IK/FK"):
        config = rng.uniform(limits[0], limits[1])
        robot.SetActiveDOFValues(config)

        position, euler = transform_to_position_euler(robot.ComputeFK())
        target_transform = position_euler_to_transform(
            position + rng.uniform(-0.050, 0.050, size=3),
            euler + rng.uniform(-np.deg2rad(10.0), np.deg2rad(10.0), size=3),
        )
        solutions = robot.ComputeIK(
            target_transform,
            return_all=True,
            check_collision=False,
        )
        if solutions is None:
            ik_failures += 1
            continue

        if args.check_collision:
            collision_free_solutions = []
            for solution in np.atleast_2d(solutions):
                robot.SetActiveDOFValues(solution)
                if not robot.CheckCollision():
                    collision_free_solutions.append(solution)
            if not collision_free_solutions:
                ik_failures += 1
                continue
            solutions = np.vstack(collision_free_solutions)

        for solution_index, solution in enumerate(np.atleast_2d(solutions)):
            robot.SetActiveDOFValues(solution)
            fk = robot.ComputeFK()
            position_error = float(np.linalg.norm(fk[:3, 3] - target_transform[:3, 3]))
            orientation_error = rotation_error_radians(fk, target_transform)
            checked_solutions += 1
            if (
                position_error > position_tolerance
                or orientation_error > orientation_tolerance
            ):
                bad_solutions.append(
                    (sample_index, solution_index, position_error, orientation_error)
                )

    successful_samples = args.num_samples - ik_failures
    print("\nIK/FK sampled validation")
    print("------------------------")
    print(f"samples:           {args.num_samples}")
    print(f"ik successes:      {successful_samples}")
    print(f"ik failures:       {ik_failures}")
    print(f"ik success rate:   {successful_samples / args.num_samples:.2%}")
    print(f"solutions checked: {checked_solutions}")
    print(f"bad FK matches:    {len(bad_solutions)}")

    planner_failures = []
    planner_path_failures = []
    planner_goal_failures = []
    planner_collision_failures = []
    planner_goal_position_tolerance = 1e-4
    planner_goal_orientation_tolerance = 1e-3
    planner_goal_joint_tolerance = 1e-6
    planner_step_limit = np.pi
    planner_start_config = np.clip(
        np.zeros_like(limits[0]),
        limits[0],
        limits[1],
    )
    robot.SetActiveDOFValues(planner_start_config)
    if robot.CheckCollision():
        planner_start_config = sample_collision_free_config(robot, rng, limits)

    for sample_index in tqdm.tqdm(
        range(args.num_planner_samples),
        desc="Motion planning",
    ):
        start_config = planner_start_config.copy()
        goal_config = sample_collision_free_neighbor(
            robot,
            rng,
            limits,
            start_config,
            max_delta=args.planner_max_delta,
        )

        robot.SetActiveDOFValues(goal_config)
        goal_transform = robot.ComputeFK()
        robot.SetActiveDOFValues(start_config)
        plan = robot.PlanToConfiguration(goal_config)
        if plan is None:
            planner_failures.append((sample_index, "PlanToConfiguration returned None"))
            continue

        state_waypoints = extract_state_waypoints(plan.results)
        if len(state_waypoints) < 2:
            planner_path_failures.append((sample_index, len(state_waypoints)))
            continue

        if max_joint_step(state_waypoints) > planner_step_limit:
            planner_path_failures.append((sample_index, len(state_waypoints)))
            continue

        if args.check_collision:
            in_collision = False
            for waypoint in state_waypoints:
                robot.SetActiveDOFValues(waypoint)
                if robot.CheckCollision():
                    planner_collision_failures.append(sample_index)
                    in_collision = True
                    break
            if in_collision:
                continue

        final_waypoint = state_waypoints[-1]
        joint_error = float(np.max(np.abs(final_waypoint - goal_config)))
        if joint_error > planner_goal_joint_tolerance:
            planner_goal_failures.append((sample_index, joint_error, None))
            continue

        robot.SetActiveDOFValues(final_waypoint)
        final_transform = robot.ComputeFK()
        position_error = float(
            np.linalg.norm(final_transform[:3, 3] - goal_transform[:3, 3])
        )
        orientation_error = rotation_error_radians(final_transform, goal_transform)
        if (
            position_error > planner_goal_position_tolerance
            or orientation_error > planner_goal_orientation_tolerance
        ):
            planner_goal_failures.append(
                (
                    sample_index,
                    position_error,
                    orientation_error,
                )
            )

    print("\nMotion planner validation")
    print("-------------------------")
    print(f"samples:              {args.num_planner_samples}")
    print(f"planner failures:     {len(planner_failures)}")
    print(f"path shape failures:  {len(planner_path_failures)}")
    print(f"collision failures:   {len(planner_collision_failures)}")
    print(f"goal mismatch count:  {len(planner_goal_failures)}")

    print_trace_summary(trace_stats)

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
        for sample_index, message in planner_failures[:20]:
            print(f"  sample={sample_index} error={message}")

    if planner_path_failures:
        print("\nFirst planner path failures:")
        for sample_index, num_waypoints in planner_path_failures[:20]:
            print(f"  sample={sample_index} state_waypoints={num_waypoints}")

    if planner_collision_failures:
        print("\nFirst planner collision failures:")
        for sample_index in planner_collision_failures[:20]:
            print(f"  sample={sample_index}")

    if planner_goal_failures:
        print("\nFirst planner goal mismatches:")
        for sample_index, position_error, orientation_error in planner_goal_failures[
            :20
        ]:
            if orientation_error is None:
                print(f"  sample={sample_index} joint_err={position_error:.3e}")
            else:
                print(
                    f"  sample={sample_index} "
                    f"pos_err={position_error:.3e} rot_err={orientation_error:.3e}"
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
