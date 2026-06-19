"""AIC MJLab termination placeholders."""

from __future__ import annotations


def insertion_success(env, **kwargs):
    """Placeholder success termination."""
    raise NotImplementedError("Define geometric/contact success condition first.")


def excessive_force(env, **kwargs):
    """Placeholder safety termination based on zeroed force/torque."""
    raise NotImplementedError("Choose force/torque thresholds first.")


def invalid_task_state(env, **kwargs):
    """Placeholder for out-of-bounds or unrecoverable task states."""
    raise NotImplementedError("Define invalid state conditions first.")
