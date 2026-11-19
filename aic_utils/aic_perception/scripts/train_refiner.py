"""Train the close-range plug-to-port refiner.

Configured by ``configs/base.json`` and ``configs/refine.json``. Run
``scripts/curate_dataset.py`` first; the refiner trains on the close views of
the curated dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_perception.config import load_config  # noqa: E402
from aic_perception.refinement import train_refiner  # noqa: E402

if __name__ == "__main__":
    print(f"Best checkpoint: {train_refiner(load_config('refine'))}")
