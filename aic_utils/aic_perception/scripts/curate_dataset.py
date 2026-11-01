"""Validate collected scenes, split them, and write label previews.

Configured by ``configs/base.json`` and the ``curation`` section of
``configs/train.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic_perception.config import load_config  # noqa: E402
from aic_perception.curation import curate  # noqa: E402

if __name__ == "__main__":
    config = load_config("train")
    manifest = curate(config)
    print(json.dumps(manifest["statistics"], indent=2))
    print(f"Usable scenes: train {len(manifest['train'])}, val {len(manifest['val'])}")
    print(f"Previews: {Path(config['dataset']['directory']) / 'previews'}")
