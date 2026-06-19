"""AIC-specific MJLab action skeletons.

The intended first policy action is a Cartesian delta at the TCP:

  action -> scaled Cartesian delta -> differential IK -> joint target

MJLab already provides ``DifferentialIKActionCfg`` for this path. The custom
classes below reserve a place for a later, closer AIC-style action term that
adds explicit impedance-like behavior after IK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    import torch
    from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
except Exception:  # pragma: no cover - allows importing without MJLab installed.
    torch = None
    ActionTerm = object
    ActionTermCfg = object

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _mjlab_required() -> None:
    if torch is None:
        raise RuntimeError(
            "MJLab is required for AIC MJLab action terms. Install or activate "
            "the MJLab environment before building the training task."
        )


@dataclass(kw_only=True)
class AicCartesianDeltaImpedanceActionCfg(ActionTermCfg):
    """Placeholder for a closer AIC-style Cartesian action term.

    This is not implemented yet because the exact policy action, stiffness,
    damping, force-feedback, and torque application semantics should be chosen
    deliberately. Use MJLab's built-in ``DifferentialIKActionCfg`` first unless
    we decide that closer AIC impedance behavior is needed.
    """

    entity_name: str
    actuator_names: tuple[str, ...] | list[str]
    frame_type: str = "site"
    frame_name: str = "gripper_tcp"
    action_dim: int = 3
    delta_pos_scale: float = 0.002
    delta_ori_scale: float = 0.02
    damping: float = 0.05
    max_dq: float = 0.05

    def build(self, env: "ManagerBasedRlEnv") -> "AicCartesianDeltaImpedanceAction":
        _mjlab_required()
        return AicCartesianDeltaImpedanceAction(self, env)


class AicCartesianDeltaImpedanceAction(ActionTerm):
    """Unimplemented AIC-like action term.

    Planned behavior:
      1. Receive Cartesian delta actions from the policy.
      2. Resolve TCP pose from the named frame.
      3. Use differential IK to compute a joint target.
      4. Apply joint target through an impedance-like controller or MJLab
         actuator targets.

    Until the exact control semantics are chosen, this class raises
    ``NotImplementedError`` to avoid accidental training with guessed behavior.
    """

    cfg: AicCartesianDeltaImpedanceActionCfg

    def __init__(self, cfg: AicCartesianDeltaImpedanceActionCfg, env: "ManagerBasedRlEnv"):
        _mjlab_required()
        super().__init__(cfg=cfg, env=env)
        raise NotImplementedError(
            "AicCartesianDeltaImpedanceAction is a placeholder. Use MJLab's "
            "DifferentialIKActionCfg for the first implementation, or fill this "
            "class once the AIC impedance semantics are finalized."
        )

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def raw_action(self) -> "torch.Tensor":
        raise NotImplementedError

    def process_actions(self, actions: "torch.Tensor") -> None:
        raise NotImplementedError

    def apply_actions(self) -> None:
        raise NotImplementedError
