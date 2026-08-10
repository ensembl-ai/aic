"""Run the configured AIC HOLD scene until interrupted with Ctrl+C."""

from __future__ import annotations

import time

from aic_mujoco.config import load_config
from aic_mujoco.outputs import RuntimeOutputs
from aic_mujoco.runtime import AICWarpRuntime
from aic_mujoco.scene import prepare_scene
from aic_mujoco.utils.timing import wait_for_realtime


def main() -> None:
    """Run the complete no-CLI foundation from base.json plus run.json."""

    config = load_config()
    scene_path = prepare_scene(config)
    runtime = AICWarpRuntime(config)
    observations = runtime.observations()
    print(f"Scene: {scene_path}")
    print(f"Worlds: {runtime.num_envs} on {runtime.device}")
    print(
        "Observations:",
        {name: tensor.shape for name, tensor in observations["rgb"].items()},
        "wrench",
        observations["wrench"]["tared"].shape,
    )
    outputs = RuntimeOutputs(config, runtime)
    timestep = config["physics"]["timestep"]
    started = time.perf_counter()
    step_index = 0
    print("Running continuously; press Ctrl+C to stop.")
    try:
        while True:
            events = runtime.step()
            if events.camera:
                outputs.update(runtime)
            step_index += 1
            if config["visualization"]["enabled"] and config["visualization"]["realtime"]:
                wait_for_realtime(started, step_index, timestep)
    except KeyboardInterrupt:
        print("\nStopping AIC MuJoCo-Warp cleanly.")
    finally:
        outputs.close()


if __name__ == "__main__":
    main()
