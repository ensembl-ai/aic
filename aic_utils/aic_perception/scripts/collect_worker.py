"""Container entry point for one collection worker.

Started by ``collect_data.py`` inside an ``aic_eval`` container; it is not
meant to be run by hand. The worker index and count arrive through the
``AIC_PERCEPTION_WORKER_INDEX`` and ``AIC_PERCEPTION_WORKER_COUNT`` variables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rclpy  # noqa: E402

from aic_perception.collector import SceneCollector  # noqa: E402
from aic_perception.config import load_config  # noqa: E402


def main() -> int:
    """Collect this worker's share of the configured scenes."""

    config = load_config("collect")
    rclpy.init()
    collector = SceneCollector(
        config,
        worker_index=int(os.environ["AIC_PERCEPTION_WORKER_INDEX"]),
        worker_count=int(os.environ["AIC_PERCEPTION_WORKER_COUNT"]),
    )
    try:
        return collector.run()
    finally:
        collector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
