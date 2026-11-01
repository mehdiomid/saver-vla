"""The recovery skill library, ported from the notebook (Phases 3.4-6.5.1).

All skills keep the gripper open and command bounded, normalized Cartesian deltas.
Numeric parameters live in `SkillConfig` so each robot can have its own profile.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..config import ApproachParams, PregraspParams, ReassessParams, Reposition2DParams
from ..types import MotionCommand, RobotState
from .base import Skill


class MoveToPregrasp(Skill):
    """Settle, raise to clearance, align above the object, lower to a pregrasp height."""

    name = "MOVE_TO_PREGRASP"

    def __init__(self, params: PregraspParams | None = None, variant: str = "CENTER"):
        super().__init__(variant)
        self.p = params or PregraspParams()
        self.phase_steps = 0
        self.target: np.ndarray | None = None

    def start(self, state: RobotState) -> bool:
        if self.phase not in ("IDLE", "REACHED", "FAILED"):
            return False
        self.phase, self.steps, self.phase_steps = "SETTLE", 0, 0
        self.target, self.reason = None, None
        return True

    def _transition(self, phase: str) -> None:
        self.phase, self.phase_steps = phase, 0

    def _move(self, eef: np.ndarray, target: np.ndarray, allow_descend: bool = True) -> MotionCommand:
        p = self.p
        delta = np.clip((target - eef) / p.position_scale_m, -p.max_command, p.max_command)
        if not allow_descend:
            delta[2] = max(0.0, delta[2])
        if eef[2] <= p.min_eef_z_m + 0.01:
            delta[2] = max(0.0, delta[2])
        return MotionCommand(delta)

    def update(self, state: RobotState) -> MotionCommand:
        p, eef, obj = self.p, state.eef_xyz, state.object_xyz
        if self.phase in ("IDLE", "REACHED", "FAILED"):
            return MotionCommand.hold()
        self.steps += 1
        self.phase_steps += 1
        if self.steps > p.max_steps:
            return self._fail("TOTAL_TIMEOUT")
        if self.phase_steps > p.max_phase_steps:
            return self._fail("PHASE_TIMEOUT")

        if self.phase == "SETTLE":
            if self.phase_steps >= p.settle_steps:
                self._transition("RAISE")
            return MotionCommand.hold()

        if self.phase == "RAISE":
            if eef[2] >= p.clearance_height_m - p.tolerance_m:
                self._transition("ALIGN")
                return MotionCommand.hold()
            return self._move(eef, np.array([eef[0], eef[1], p.clearance_height_m]), allow_descend=False)

        if self.phase == "ALIGN":
            target = np.array([obj[0], obj[1], max(p.clearance_height_m, obj[2] + p.pregrasp_offset_m)])
            if np.linalg.norm(target[:2] - eef[:2]) <= p.tolerance_m:
                self._transition("LOWER")
                return MotionCommand.hold()
            return self._move(eef, target, allow_descend=False)

        if self.phase == "LOWER":
            target = obj.copy()
            target[2] += p.pregrasp_offset_m
            self.target = target
            if np.linalg.norm(target - eef) <= p.tolerance_m:
                self._transition("REACHED")
                return MotionCommand.hold()
            return self._move(eef, target)

        return self._fail("UNKNOWN_PHASE")

    def status(self) -> dict[str, Any]:
        s = super().status()
        s["target"] = None if self.target is None else self.target.tolist()
        return s


class ApproachObject(Skill):
    """Move to a demonstrated near-grasp pose above the object. Does not grasp.

    Variants: CENTER (APPROACH_OBJECT) or OFFSET_X_POS / OFFSET_X_NEG (ADJUST_GRASP_POSE).
    `target_bias` is a hidden calibration fault added to CENTER targets (Phase 6.4b);
    it is never reported to the agent.
    """

    name = "APPROACH_OBJECT"

    def __init__(
        self,
        params: ApproachParams | None = None,
        variant: str = "CENTER",
        target_bias: np.ndarray | None = None,
    ):
        super().__init__(variant)
        if variant not in ("CENTER", "OFFSET_X_POS", "OFFSET_X_NEG"):
            raise ValueError(f"Unknown approach variant: {variant}")
        self.name = "APPROACH_OBJECT" if variant == "CENTER" else "ADJUST_GRASP_POSE"
        self.p = params or ApproachParams()
        self.target_bias = np.zeros(3) if target_bias is None else np.asarray(target_bias, float)
        self.target: np.ndarray | None = None

    def start(self, state: RobotState) -> bool:
        p = self.p
        target = state.object_xyz + np.asarray(p.grasp_offset_m, dtype=float)
        if self.variant == "OFFSET_X_POS":
            target[0] += p.adjust_offset_x_m
        elif self.variant == "OFFSET_X_NEG":
            target[0] -= p.adjust_offset_x_m
        target[2] = max(float(target[2]), p.min_target_z_m)
        if not p.min_target_z_m <= target[2] <= p.max_target_z_m:
            return False
        if state.eef_object_distance > p.max_start_distance_m:
            return False
        if self.variant == "CENTER":
            target = target + self.target_bias
        self.target = target
        self.phase, self.steps, self.reason = "APPROACH", 0, None
        return True

    def update(self, state: RobotState) -> MotionCommand:
        p, eef = self.p, state.eef_xyz
        if self.phase != "APPROACH":
            return MotionCommand.hold()
        self.steps += 1
        delta = self.target - eef
        if float(np.linalg.norm(delta)) <= p.tolerance_m:
            self.phase = "REACHED"
        elif self.steps >= p.max_steps:
            self.phase, self.reason = "FAILED", "TIMEOUT"
        if self.phase != "APPROACH":
            return MotionCommand.hold()
        cmd = np.clip(delta * p.gain, -p.max_command, p.max_command)
        if eef[2] <= p.min_target_z_m and cmd[2] < 0:
            cmd[2] = 0.0
        return MotionCommand(cmd)

    def status(self) -> dict[str, Any]:
        s = super().status()
        s["variant"] = self.variant
        s["target"] = None if self.target is None else self.target.tolist()
        return s


class ReassessState(Skill):
    """Hold still with the gripper open for a few steps to gather fresh measurements."""

    name = "REASSESS_STATE"

    def __init__(self, params: ReassessParams | None = None, variant: str = "CENTER"):
        super().__init__(variant)
        self.p = params or ReassessParams()

    def start(self, state: RobotState) -> bool:
        self.phase, self.steps = "REASSESS", 0
        return True

    def update(self, state: RobotState) -> MotionCommand:
        if self.phase == "REASSESS":
            self.steps += 1
            if self.steps >= self.p.hold_steps:
                self.phase = "REACHED"
        return MotionCommand.hold()


class Reposition2D(Skill):
    """Height-aware 2D repositioning (Phase 6.5.1).

    Raise to clearance, align X and Y above the object, then descend to a pregrasp
    height while correcting small XY errors.
    """

    name = "REPOSITION_TO_OBJECT_2D"

    def __init__(self, params: Reposition2DParams | None = None, variant: str = "CENTER"):
        super().__init__(variant)
        self.p = params or Reposition2DParams()
        self.xy_error_m: float | None = None
        self.z_error_m: float | None = None
        self.target_z_m: float | None = None

    def start(self, state: RobotState) -> bool:
        self.phase, self.steps, self.reason = "RAISE", 0, None
        self.xy_error_m = self.z_error_m = self.target_z_m = None
        return True

    def update(self, state: RobotState) -> MotionCommand:
        p, eef, obj = self.p, state.eef_xyz, state.object_xyz
        cmd = np.zeros(3)
        if self.phase not in ("RAISE", "ALIGN", "LOWER_TO_PREGRASP"):
            return MotionCommand(cmd)
        self.steps += 1
        delta_xy = obj[:2] - eef[:2]
        self.xy_error_m = float(np.linalg.norm(delta_xy))
        target_z = obj[2] + p.pregrasp_offset_z_m
        self.target_z_m = float(target_z)
        self.z_error_m = float(target_z - eef[2])

        if not p.min_target_z_m <= target_z <= p.max_target_z_m:
            self.phase, self.reason = "FAILED", "TARGET_OUTSIDE_Z_BOUNDS"
            return MotionCommand(cmd)
        if self.steps >= p.max_steps:
            self.phase, self.reason = "FAILED", "TIMEOUT"
            return MotionCommand(cmd)

        if self.phase == "RAISE":
            dz = p.clearance_z_m - eef[2]
            if dz > p.z_tolerance_m:
                cmd[2] = np.clip(dz * 2.0, 0.0, 0.12)
                return MotionCommand(cmd)
            self.phase = "ALIGN"

        if self.phase == "ALIGN":
            if eef[2] < p.clearance_z_m - p.z_tolerance_m:
                self.phase = "RAISE"
                return MotionCommand(cmd)
            if self.xy_error_m > p.xy_tolerance_m:
                cmd[:2] = np.clip(delta_xy * 2.0, -0.12, 0.12)
                return MotionCommand(cmd)
            self.phase = "LOWER_TO_PREGRASP"

        if self.phase == "LOWER_TO_PREGRASP":
            if self.xy_error_m > p.realign_xy_m:
                self.phase = "RAISE"
                return MotionCommand(cmd)
            if self.xy_error_m <= p.xy_tolerance_m and abs(self.z_error_m) <= p.z_tolerance_m:
                self.phase = "REACHED"
                return MotionCommand(cmd)
            cmd[:2] = np.clip(delta_xy * 1.5, -0.06, 0.06)
            cmd[2] = np.clip(self.z_error_m * 2.0, -0.08, 0.08)
            if eef[2] <= target_z and cmd[2] < 0:
                cmd[2] = 0.0
        return MotionCommand(cmd)

    def status(self) -> dict[str, Any]:
        s = super().status()
        s.update(xy_error_m=self.xy_error_m, z_error_m=self.z_error_m, target_z_m=self.target_z_m)
        return s
