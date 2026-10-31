#!/usr/bin/env python3
"""Run AIC synthetic demonstration collection from its JSON configuration."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from aic_mujoco.collection import main


if __name__ == "__main__":
    main()

