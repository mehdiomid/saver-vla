"""Imperfect sensing and actuation, for testing robustness beyond privileged state.

On a real robot the object position comes from cameras and is never exact. `Perception`
turns the true simulator state into the noisy estimate the supervisor would get, and
`noisy_action` perturbs executed commands. Fault triggers and task success keep using
the true state, so noise can cause false alarms but cannot fake success.
"""

from __future__ import annotations

import numpy as np

from .config import NoiseConfig
from .types import RobotState


class Perception:
    def __init__(self, cfg: NoiseConfig | None = None, rng: np.random.Generator | None = None):
        self.cfg = cfg or NoiseConfig()
        self.rng = rng or np.random.default_rng(0)
        self.bias = np.asarray(self.cfg.object_position_bias_m, dtype=float).reshape(3)

    @property
    def active(self) -> bool:
        c = self.cfg
        return bool(c.object_position_std_m or c.eef_position_std_m or self.bias.any())

    def observe(self, true_state: RobotState) -> RobotState:
        if not self.active:
            return true_state
        c = self.cfg
        obj = true_state.object_xyz + self.bias + self.rng.normal(0.0, c.object_position_std_m, 3)
        eef = true_state.eef_xyz + self.rng.normal(0.0, c.eef_position_std_m, 3)
        return RobotState(
            step=true_state.step, time=true_state.time, eef_xyz=eef, object_xyz=obj,
            target_xyz=true_state.target_xyz, gripper_command=true_state.gripper_command,
        )

    def noisy_action(self, action: np.ndarray) -> np.ndarray:
        if not self.cfg.action_std:
            return action
        out = np.asarray(action, dtype=float).copy()
        out[:3] = np.clip(out[:3] + self.rng.normal(0.0, self.cfg.action_std, 3), -1.0, 1.0)
        return out
