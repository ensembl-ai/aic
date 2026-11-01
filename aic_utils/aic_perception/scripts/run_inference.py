"""Publish scene estimates from live AIC observations.

Configured by ``configs/base.json`` and ``configs/infer.json``. Start the
AIC simulation first; estimates appear on the configured topic and as TF
frames under the configured prefix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rclpy  # noqa: E402

from aic_perception.config import load_config  # noqa: E402
from aic_perception.inference_node import SceneEstimatorNode  # noqa: E402


def main() -> None:
    """Spin the estimator node until Ctrl+C."""

    rclpy.init()
    node = SceneEstimatorNode(load_config("infer"))
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
