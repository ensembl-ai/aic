"""The insertion policy's trial record stays JSON-serializable."""

from __future__ import annotations

import json

import numpy as np

from aic_perception.policies.insertion import rounded


def test_record_values_with_numpy_types_serialize() -> None:
    """NumPy booleans, integers, floats, and arrays become plain JSON values."""

    record = {
        "inside": np.bool_(True),
        "count": np.int64(3),
        "error_mm": np.float32(0.123456789),
        "offset_mm": np.array([1.0, -2.5, 0.25]),
        "phases": [{"entered": np.bool_(False), "radius": np.float64(0.5)}],
    }
    restored = json.loads(json.dumps(rounded(record)))
    assert restored == {
        "inside": True,
        "count": 3,
        "error_mm": 0.123457,
        "offset_mm": [1.0, -2.5, 0.25],
        "phases": [{"entered": False, "radius": 0.5}],
    }
