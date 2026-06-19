from __future__ import annotations

from typing import Any

import numpy as np


class MjModel:
    nq: int
    nv: int
    nu: int
    njnt: int
    nbody: int
    nsite: int
    nsensor: int
    neq: int
    opt: Any
    jnt_type: Any
    jnt_qposadr: Any
    jnt_dofadr: Any
    jnt_limited: Any
    jnt_range: Any
    actuator_trntype: Any
    actuator_trnid: Any
    actuator_gear: Any
    sensor_adr: Any
    sensor_dim: Any
    eq_type: Any
    eq_obj1id: Any
    eq_obj2id: Any
    eq_data: Any

    @classmethod
    def from_xml_path(cls, filename: str) -> MjModel: ...


class MjData:
    qpos: Any
    qvel: Any
    ctrl: Any
    qfrc_applied: Any
    qfrc_bias: Any
    sensordata: Any
    xpos: Any
    xmat: Any
    site_xpos: Any
    site_xmat: Any

    def __init__(self, model: MjModel) -> None: ...


class Renderer:
    def __init__(self, model: MjModel, height: int, width: int) -> None: ...
    def update_scene(self, data: MjData, camera: str | int | None = None) -> None: ...
    def render(self) -> np.ndarray: ...
    def close(self) -> None: ...


mjtObj: Any
mjtJoint: Any
mjtTrn: Any
mjtEq: Any
viewer: Any


def mj_name2id(model: MjModel, objtype: Any, name: str) -> int: ...
def mj_id2name(model: MjModel, objtype: Any, id: int) -> str | None: ...
def mj_resetData(model: MjModel, data: MjData) -> None: ...
def mj_forward(model: MjModel, data: MjData) -> None: ...
def mj_step(model: MjModel, data: MjData) -> None: ...
def mj_jacSite(
    model: MjModel,
    data: MjData,
    jacp: np.ndarray,
    jacr: np.ndarray,
    site: int,
) -> None: ...
def mju_mat2Quat(quat: np.ndarray, mat: np.ndarray) -> None: ...
def mju_quat2Mat(mat: np.ndarray, quat: np.ndarray) -> None: ...


def __getattr__(name: str) -> Any: ...
