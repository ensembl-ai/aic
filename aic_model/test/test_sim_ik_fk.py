#!/usr/bin/env python3

import time
import argparse
from collections import defaultdict
from functools import wraps
from pathlib import Path
import sys
import tqdm
import numpy as np
from transforms3d.euler import euler2mat, mat2euler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_model.robot import EnsemblRobot
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
    print("\nKinematics trace summary")
    print("------------------------")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", type=int, default=20000)
    parser.add_argument("--check-collision", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    robot = EnsemblRobot()
    trace_stats = trace_robot_methods(
        robot, ("ComputeFK", "ComputeIK", "CheckCollision")
    )
    limits = robot.GetActiveDOFLimits()

    ik_failures = 0
    checked_solutions = 0
    bad_solutions = []
    position_tolerance = 1e-4
    orientation_tolerance = 1e-3
    for sample_index in tqdm.tqdm(range(args.num_samples)):
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
        raise SystemExit(1)


if __name__ == "__main__":
    main()
