"""Collect the multiview scene dataset with parallel Gazebo containers.

Configured entirely by ``configs/base.json`` and ``configs/collect.json``.
Re-running resumes: finished scenes are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_perception.config import load_config  # noqa: E402
from aic_perception.orchestrator import run_collection  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_collection(load_config("collect")))
