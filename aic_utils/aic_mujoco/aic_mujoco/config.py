"""Small JSON config loader for AIC MuJoCo R&D scripts.

Configs may include other configs:

  {
    "include": [
      "../common/ur5e_control.json",
      "../common/preinsert_sfp_nic.json"
    ],
    "duration": 10.0
  }

Includes are resolved relative to the file that declares them. Later includes
override earlier includes, and the including file overrides all included files.
Nested dictionaries are merged recursively.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_json_config(path: str | Path | None) -> dict[str, Any]:
    """Load a JSON config with optional recursive ``include`` support."""

    if path is None:
        return {}

    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    includes = raw.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise ValueError(f"'include' must be a string or list in {config_path}")

    cfg: dict[str, Any] = {}
    for include in includes:
        include_path = (config_path.parent / str(include)).resolve()
        cfg = deep_merge(cfg, load_json_config(include_path))

    return deep_merge(cfg, raw)
