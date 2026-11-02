"""Host-side launcher for parallel, isolated Gazebo collection workers.

Each worker is one ``aic_eval`` container with its own network namespace, so
every Gazebo server, Zenoh router, and ROS graph is fully isolated without
any domain-ID or partition bookkeeping. The repository is mounted read-only
at its host path and the dataset read-write at its host path, so configured
paths resolve identically inside and outside the containers.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from aic_perception.config import CONFIG_DIR, PACKAGE_DIR
from aic_perception.layout import SPLITS, completed_ids, make_directories

REPOSITORY_DIR = PACKAGE_DIR.parents[1]
WORKER_SCRIPT = PACKAGE_DIR / "scripts" / "collect_worker.py"
BRINGUP_ARGUMENTS = (
    "gazebo_gui:=false launch_rviz:=false ground_truth:=true "
    "start_aic_engine:=false spawn_task_board:=false"
)
RENDERING_FLAGS = {
    "wsl_gpu": [
        "--device",
        "/dev/dxg",
        "-v",
        "/usr/lib/wsl:/usr/lib/wsl:ro",
        "-e",
        "LD_LIBRARY_PATH=/usr/lib/wsl/lib",
        "-e",
        "GALLIUM_DRIVER=d3d12",
        "-e",
        "MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA",
    ],
    "nvidia_gpu": ["--gpus", "all", "-e", "NVIDIA_DRIVER_CAPABILITIES=all"],
    "cpu": [],
}
PROVENANCE_KEYS = ("schema", "images", "randomization")


def total_memory_gb() -> float:
    """Return the host's physical memory in GiB."""

    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def worker_count(config: dict[str, Any]) -> int:
    """Resolve the number of parallel workers.

    ``auto`` takes the largest count that fits the host's CPU cores and 80% of
    its memory at the configured per-worker cost, capped by
    ``max_gpu_workers`` when rendering on the GPU.

    Args:
        config: Validated collection configuration.

    Returns:
        Worker count, never more than the number of scenes.
    """

    configured = config["collection"]["workers"]
    docker = config["docker"]
    if configured == "auto":
        count = min(
            math.floor((os.cpu_count() or 1) / docker["cpus_per_worker"]),
            math.floor(0.8 * total_memory_gb() / docker["memory_per_worker_gb"]),
        )
        if docker["rendering"] != "cpu":
            count = min(count, docker["max_gpu_workers"])
    else:
        count = configured
    return max(1, min(count, config["collection"]["scenes"]))


def provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Return the configuration values that define what a scene contains."""

    return {
        "seed": config["collection"]["seed"],
        "validation_fraction": config["dataset"]["validation_fraction"],
        **{key: config[key] for key in PROVENANCE_KEYS},
    }


def check_provenance(config: dict[str, Any]) -> None:
    """Record or verify the scene-defining configuration of the dataset.

    Resuming with a different seed, schema, image size, or randomization
    would silently mix incompatible scenes, so it is rejected.

    Args:
        config: Validated collection configuration.

    Raises:
        ValueError: If the dataset was collected with different settings.
    """

    path = Path(config["dataset"]["directory"]) / "provenance.json"
    expected = provenance(config)
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded != expected:
            raise ValueError(
                f"{path} was collected with different scene settings; use a new "
                "dataset directory or restore the original settings"
            )
    else:
        path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")


def container_command(config: dict[str, Any], index: int, count: int) -> list[str]:
    """Build the ``docker run`` command for one worker.

    Args:
        config: Validated collection configuration.
        index: Worker index.
        count: Total worker count.

    Returns:
        Command line.
    """

    docker = config["docker"]
    dataset = Path(config["dataset"]["directory"])
    script = (
        f"/entrypoint.sh {BRINGUP_ARGUMENTS} > /tmp/bringup.log 2>&1 & "
        ". /ws_aic/install/setup.bash && export RMW_IMPLEMENTATION=rmw_zenoh_cpp && "
        f"exec python3 {WORKER_SCRIPT}"
    )
    return [
        "docker",
        "run",
        "-d",
        "--name",
        f"{docker['container_prefix']}_{index}",
        "--cpus",
        str(docker["cpus_per_worker"]),
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "-e",
        f"AIC_PERCEPTION_WORKER_INDEX={index}",
        "-e",
        f"AIC_PERCEPTION_WORKER_COUNT={count}",
        "-v",
        f"{REPOSITORY_DIR}:{REPOSITORY_DIR}:ro",
        "-v",
        f"{dataset}:{dataset}",
        *RENDERING_FLAGS[docker["rendering"]],
        "--entrypoint",
        "bash",
        docker["image"],
        "-c",
        script,
    ]


def container_states(names: list[str]) -> dict[str, tuple[str, int]]:
    """Return ``(status, exit_code)`` for each container.

    Args:
        names: Container names.

    Returns:
        State per container name.
    """

    output = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            "{{.Name}} {{.State.Status}} {{.State.ExitCode}}",
            *names,
        ],
        capture_output=True,
        text=True,
    ).stdout
    states = {}
    for line in output.splitlines():
        name, status, code = line.split()
        states[name.lstrip("/")] = (status, int(code))
    return states


def remove_containers(names: list[str]) -> None:
    """Force-remove containers, ignoring ones that do not exist."""

    subprocess.run(["docker", "rm", "-f", *names], capture_output=True)


def completed_scenes(config: dict[str, Any]) -> int:
    """Count scenes whose label has been written, across both splits."""

    root = config["dataset"]["directory"]
    return sum(len(completed_ids(root, split)) for split in SPLITS)


def run_collection(config: dict[str, Any]) -> int:
    """Launch, monitor, and clean up all workers.

    Args:
        config: Validated collection configuration.

    Returns:
        Exit code: 0 when every worker finished cleanly.
    """

    dataset = Path(config["dataset"]["directory"])
    make_directories(dataset)
    check_provenance(config)
    count = worker_count(config)
    names = [
        f"{config['docker']['container_prefix']}_{index}" for index in range(count)
    ]
    remove_containers(names)
    total = config["collection"]["scenes"]
    initial = completed_scenes(config)
    print(f"Dataset: {dataset}")
    print(f"Config: {CONFIG_DIR}")
    print(
        f"Workers: {count} ({config['docker']['rendering']} rendering), scenes: {initial}/{total} done"
    )
    restarts = {name: 0 for name in names}
    started = time.monotonic()
    try:
        for index in range(count):
            if index:
                time.sleep(config["docker"]["launch_interval_s"])
            subprocess.run(
                container_command(config, index, count), check=True, capture_output=True
            )
        while True:
            time.sleep(config["collection"]["progress_interval_s"])
            states = container_states(names)
            for index, name in enumerate(names):
                status, code = states[name]
                if (
                    status == "exited"
                    and code != 0
                    and restarts[name] < config["docker"]["max_restarts"]
                ):
                    restarts[name] += 1
                    print(
                        f"Worker {name} exited with code {code}; restart {restarts[name]}",
                        flush=True,
                    )
                    remove_containers([name])
                    subprocess.run(
                        container_command(config, index, count),
                        check=True,
                        capture_output=True,
                    )
                    states[name] = ("running", 0)
            done = completed_scenes(config)
            elapsed = time.monotonic() - started
            rate = (done - initial) / elapsed
            remaining = (total - done) / rate if rate > 0 else float("inf")
            running = sum(status == "running" for status, _ in states.values())
            print(
                f"[{elapsed / 60:6.1f} min] {done}/{total} scenes, "
                f"{rate * 60:.1f} scenes/min, ETA {remaining / 60:.1f} min, "
                f"{running}/{count} workers running",
                flush=True,
            )
            if running == 0:
                break
    except KeyboardInterrupt:
        print("\nStopping workers; completed scenes are kept and the next run resumes.")
        remove_containers(names)
        return 130
    failed = [name for name, (_, code) in container_states(names).items() if code != 0]
    for name in failed:
        print(f"Worker {name} failed; inspect with: docker logs {name}")
    remove_containers([name for name in names if name not in failed])
    print(f"Finished: {completed_scenes(config)}/{total} scenes in {dataset}")
    return 1 if failed else 0
