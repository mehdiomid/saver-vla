"""Controlled fault injection for reliability testing.

Two families:
  * Action faults (gripper release, first-trial gripper impairment) and target
    faults (approach calibration bias) only modify commands, so they work on any
    robot, including a real SO-101 for supervised testing.
  * Object faults (forced separation, velocity impulse) move the object directly
    and need a simulator backend that implements `inject_object_fault`.

Every numeric parameter may be a fixed value or a [low, high] range. Ranges are
sampled from a generator seeded by (seed, episode_index): episodes differ from each
other, yet any episode can be reproduced exactly. Sampled values are written to the
event log for auditing and never shown to the agent when `recovery.blind` is true.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Protocol

import numpy as np

from .config import FaultConfig
from .types import GRIPPER_OPEN, RobotState

TRANSPORT_FAULTS = ("none", "forced_separation", "gripper_release", "velocity_impulse")
OBJECT_FAULTS = ("forced_separation", "velocity_impulse")
DIRECTIONS = ("away_from_target", "random", "lateral")


class SupportsObjectFaults(Protocol):
    def inject_object_fault(self, kind: str, direction_xy: np.ndarray, cfg: FaultConfig) -> dict[str, Any]: ...


def sample(value: Any, rng: np.random.Generator, integer: bool = False) -> float | int:
    """A fixed value, or a uniform sample from [low, high]."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or value[0] > value[1]:
            raise ValueError(f"Range must be [low, high], got {value}")
        x = rng.uniform(float(value[0]), float(value[1]))
        return int(round(x)) if integer else float(x)
    return int(value) if integer else float(value)


def episode_rng(seed: int | None, episode_index: int, stream: int = 0) -> np.random.Generator:
    return np.random.default_rng([int(seed or 0), int(episode_index), int(stream)])


class FaultInjector:
    def __init__(self, cfg: FaultConfig | None = None, gripper_index: int = 6):
        self.cfg = cfg or FaultConfig()
        if self.cfg.transport not in TRANSPORT_FAULTS:
            raise ValueError(f"faults.transport must be one of {TRANSPORT_FAULTS}")
        if self.cfg.direction not in DIRECTIONS:
            raise ValueError(f"faults.direction must be one of {DIRECTIONS}")
        self.gripper_index = gripper_index
        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self, rng: np.random.Generator | None = None) -> None:
        if rng is not None:
            self.rng = rng
        self.initial_z: float | None = None
        self.lift_checks = 0
        self.lift_time: float | None = None
        self.injections = 0
        self.injection_steps: list[int] = []
        self.release_remaining = 0
        self.trial_block_remaining = 0
        self.trial_block_steps = 0
        # Values fixed for the whole episode.
        self.bias = np.array(
            [sample(self.cfg.approach_bias_x_m, self.rng), sample(self.cfg.approach_bias_y_m, self.rng), 0.0]
        )
        self.trial_impairment = sample(self.cfg.trial_impairment_steps, self.rng, integer=True)
        self._draw_next()

    def _draw_next(self) -> None:
        """Sample the parameters of the next transport fault."""
        c = self.cfg
        self.next = {
            "delay_s": sample(c.trigger.min_time_after_lift_s, self.rng),
            "displacement_m": sample(c.displacement_m, self.rng),
            "release_steps": sample(c.release_steps, self.rng, integer=True),
            "impulse_speed_mps": sample(c.impulse_speed_mps, self.rng),
            "jitter_rad": math.radians(self.rng.uniform(-c.direction_jitter_deg, c.direction_jitter_deg)),
            "random_angle": self.rng.uniform(0, 2 * math.pi),
            "side": 1.0 if self.rng.random() < 0.5 else -1.0,
        }

    def episode_plan(self) -> dict[str, Any]:
        """The sampled hidden settings, for the audit log."""
        return {
            "transport": self.cfg.transport,
            "approach_bias_m": self.bias.tolist(),
            "trial_impairment_steps": self.trial_impairment,
            "first_fault": dict(self.next),
            "max_injections": self.cfg.max_injections,
        }

    @property
    def injected(self) -> bool:
        return self.injections > 0

    @property
    def injection_step(self) -> int | None:
        return self.injection_steps[0] if self.injection_steps else None

    # ----------------------------------------------------------------- transport fault

    def _direction(self, away: np.ndarray) -> np.ndarray:
        n = self.next
        if self.cfg.direction == "random":
            angle = n["random_angle"]
            d = np.array([math.cos(angle), math.sin(angle)])
        elif self.cfg.direction == "lateral":
            d = n["side"] * np.array([-away[1], away[0]])
        else:
            d = away
        c, s = math.cos(n["jitter_rad"]), math.sin(n["jitter_rad"])
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

    def before_step(self, state: RobotState, backend: Any) -> dict[str, Any] | None:
        """Check the trigger condition; inject when it holds. Returns a fault record or None."""
        cfg, trig = self.cfg, self.cfg.trigger
        if self.initial_z is None:
            self.initial_z = float(state.object_xyz[2])
        if cfg.transport == "none" or self.injections >= cfg.max_injections:
            return None
        if self.injection_steps and state.step - self.injection_steps[-1] < cfg.min_steps_between_injections:
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
        if state.time - self.lift_time < self.next["delay_s"]:
            return None
        if state.target_xyz is None:
            return None
        away = state.object_xyz[:2] - state.target_xyz[:2]
        norm = float(np.linalg.norm(away))
        if norm < trig.target_exclusion_radius_m or norm < 1e-8:
            return None
        direction = self._direction(away / norm)

        params = self.next
        record: dict[str, Any] = {
            "fault": cfg.transport,
            "injection": self.injections + 1,
            "step": state.step,
            "time": state.time,
            "object_xyz": state.object_xyz.tolist(),
            "delay_after_lift_s": params["delay_s"],
        }
        if cfg.transport == "gripper_release":
            steps = params["release_steps"]
            if not 1 <= steps <= 40:
                raise ValueError("faults.release_steps must be between 1 and 40")
            self.release_remaining = steps
            record["release_steps"] = steps
        else:
            if not hasattr(backend, "inject_object_fault"):
                raise NotImplementedError(f"Backend cannot inject object fault '{cfg.transport}'")
            sampled = replace(cfg, displacement_m=params["displacement_m"], impulse_speed_mps=params["impulse_speed_mps"])
            record["direction_xy"] = direction.tolist()
            record.update(backend.inject_object_fault(cfg.transport, direction, sampled))
        self.injections += 1
        self.injection_steps.append(state.step)
        self.lift_checks, self.lift_time = 0, None  # the next fault needs a new sustained lift
        self._draw_next()
        return record

    # ----------------------------------------------------------------- action faults

    def begin_trial(self, trial_number: int) -> None:
        """Arm the first-trial gripper impairment (Phase 6.3)."""
        steps = max(0, min(150, int(self.trial_impairment)))
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
        return self.bias.copy()
