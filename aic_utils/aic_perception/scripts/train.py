"""Train the multiview scene estimator.

Configured by ``configs/base.json`` and ``configs/train.json``. Run
``scripts/curate_dataset.py`` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_perception.config import load_config  # noqa: E402
from aic_perception.training import train  # noqa: E402

if __name__ == "__main__":
    print(f"Best checkpoint: {train(load_config('train'))}")
