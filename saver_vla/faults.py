"""Controlled fault injection for reliability testing.

Two families:
  * Action faults (gripper release, first-trial gripper impairment) and target
    faults (approach calibration bias) only modify commands, so they work on any
    robot, including a real SO-101 for supervised testing.
  * Object faults (forced separation, velocity impulse) move the object directly
    and need a simulator backend that implements `inject_object_fault`.

Fault details are recorded in the event log for auditing but never passed to the
agent when `recovery.blind` is true.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from .config import FaultConfig
from .types import GRIPPER_OPEN, RobotState

TRANSPORT_FAULTS = ("none", "forced_separation", "gripper_release", "velocity_impulse")
OBJECT_FAULTS = ("forced_separation", "velocity_impulse")


class SupportsObjectFaults(Protocol):
    def inject_object_fault(self, kind: str, direction_xy: np.ndarray, cfg: FaultConfig) -> dict[str, Any]: ...


class FaultInjector:
    def __init__(self, cfg: FaultConfig | None = None, gripper_index: int = 6):
        self.cfg = cfg or FaultConfig()
        if self.cfg.transport not in TRANSPORT_FAULTS:
            raise ValueError(f"faults.transport must be one of {TRANSPORT_FAULTS}")
        self.gripper_index = gripper_index
        self.reset()

    def reset(self) -> None:
        self.initial_z: float | None = None
        self.lift_checks = 0
        self.lift_time: float | None = None
        self.injected = False
        self.injection_step: int | None = None
        self.release_remaining = 0
        self.trial_block_remaining = 0
        self.trial_block_steps = 0

    # ----------------------------------------------------------------- transport fault

    def before_step(self, state: RobotState, backend: Any) -> dict[str, Any] | None:
        """Check the trigger condition; inject at most once. Returns a fault record or None."""
        cfg, trig = self.cfg, self.cfg.trigger
        if self.initial_z is None:
            self.initial_z = float(state.object_xyz[2])
        if cfg.transport == "none" or self.injected:
            return None

        if state.object_xyz[2] - self.initial_z > trig.min_lift_m:
            self.lift_checks += 1
        else:
            self.lift_checks, self.lift_time = 0, None
            return None
        if self.lift_checks < trig.min_lift_checks:
            return None
        if self.lift_time is None:
            self.lift_time = state.time
            return None
        if state.time - self.lift_time < trig.min_time_after_lift_s:
            return None
        if state.target_xyz is None:
            return None
        direction = state.object_xyz[:2] - state.target_xyz[:2]
        if np.linalg.norm(direction) < trig.target_exclusion_radius_m:
            return None
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return None
        direction = direction / norm

        record: dict[str, Any] = {
            "fault": cfg.transport,
            "step": state.step,
            "time": state.time,
            "object_xyz": state.object_xyz.tolist(),
        }
        if cfg.transport == "gripper_release":
            if not 1 <= cfg.release_steps <= 40:
                raise ValueError("faults.release_steps must be between 1 and 40")
            self.release_remaining = cfg.release_steps
            record["release_steps"] = cfg.release_steps
        else:
            if not hasattr(backend, "inject_object_fault"):
                raise NotImplementedError(f"Backend cannot inject object fault '{cfg.transport}'")
            record.update(backend.inject_object_fault(cfg.transport, direction, cfg))
        self.injected = True
        self.injection_step = state.step
        return record

    # ----------------------------------------------------------------- action faults

    def begin_trial(self, trial_number: int) -> None:
        """Arm the first-trial gripper impairment (Phase 6.3)."""
        steps = max(0, min(150, int(self.cfg.trial_impairment_steps)))
        self.trial_block_remaining = steps if trial_number == 1 else 0
        self.trial_block_steps = 0

    def end_trial(self) -> dict[str, int]:
        info = {"grip_block_steps": self.trial_block_steps}
        self.trial_block_remaining = 0
        return info

    def apply_action(self, action: np.ndarray, in_trial: bool) -> np.ndarray:
        """Return the action actually sent to the robot after any active action fault."""
        out = np.asarray(action, dtype=float).copy()
        g = self.gripper_index
        if self.release_remaining > 0:
            out[g] = GRIPPER_OPEN
            self.release_remaining -= 1
        if in_trial and self.trial_block_remaining > 0:
            out[g] = GRIPPER_OPEN
            self.trial_block_remaining -= 1
            self.trial_block_steps += 1
        return out

    # ----------------------------------------------------------------- target faults

    def approach_bias(self) -> np.ndarray:
        return np.array([self.cfg.approach_bias_x_m, 0.0, 0.0])
