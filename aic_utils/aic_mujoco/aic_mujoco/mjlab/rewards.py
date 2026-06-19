"""AIC MJLab reward placeholders.

Rewards define the training objective. Leave these unimplemented until the task
state, success condition, and shaping philosophy are agreed.
"""

from __future__ import annotations


def insertion_task_reward(env, **kwargs):
    """Placeholder for the main insertion reward."""
    raise NotImplementedError("Define the insertion objective before training.")


def action_regularization_reward(env, **kwargs):
    """Placeholder for action smoothness or magnitude penalties."""
    raise NotImplementedError("Choose action regularization deliberately.")


def force_safety_penalty(env, **kwargs):
    """Placeholder for excessive force/torque penalty."""
    raise NotImplementedError("Choose force limits and penalty shape first.")
