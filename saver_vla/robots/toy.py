"""A tiny kinematic pick-and-place world for fast, dependency-free tests.

It is NOT a physics simulator and makes no claim about real robot behavior. It
exists so the full supervisor loop (fault -> detection -> agent -> skills ->
grasp trials -> success) can be tested in seconds on any laptop.

Its stand-in policy imitates two SmolVLA behaviors that matter for recovery:
  * once it has started transporting, it keeps following its plan even if the
    object is lost (like a stale action chunk) until `reset()` is called;
  * when it starts close to the object but misaligned (> `misalign_tol` in XY),
    it attempts a grasp in place and misses, as in the Phase 6.5.1 first trial.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..types import RobotState
from . import PandaDeltaProfile

DT = 0.05  # 20 Hz
STEP_SCALE = 0.05  # metres per unit of normalized delta command
GRASP_OFFSET = np.array([0.0142, -0.0112, 0.0041])  # eef - object at a good grasp
REST_Z = 0.0384


class ToyWorld:
    def __init__(self, object_xyz=(-0.12, -0.13, REST_Z), target_xyz=(-0.30, 0.15, 0.04), eef_xyz=(0.0, 0.0, 0.30)):
        self.object0 = np.array(object_xyz, float)
        self.target = np.array(target_xyz, float)
        self.eef0 = np.array(eef_xyz, float)
        self.reset()

    def reset(self) -> None:
        self.eef = self.eef0.copy()
        self.obj = self.object0.copy()
        self.attached = False
        self.attach_offset = np.zeros(3)
        self.closed_steps = 0
        self.step_count = 0
        self.gripper = None

    def step(self, action: np.ndarray) -> None:
        action = np.asarray(action, float)
        self.eef = self.eef + np.clip(action[:3], -1, 1) * STEP_SCALE
        self.eef[2] = max(self.eef[2], 0.03)
        g = float(action[6])
        self.gripper = g
        if g > 0.5:
            self.closed_steps += 1
            aligned = np.linalg.norm(self.eef - (self.obj + GRASP_OFFSET)) < 0.02
            if not self.attached and aligned and self.closed_steps >= 3:
                self.attached = True
                self.attach_offset = self.obj - self.eef
        else:
            self.closed_steps = 0
            self.attached = False
        if self.attached:
            self.obj = self.eef + self.attach_offset
        elif self.obj[2] > REST_Z:
            self.obj[2] = max(REST_Z, self.obj[2] - 0.03)
        self.step_count += 1

    def success(self) -> bool:
        return (
            not self.attached
            and np.linalg.norm(self.obj[:2] - self.target[:2]) < 0.05
            and self.obj[2] < 0.08
        )


class ToyPolicy:
    """Scripted stand-in for a VLA policy (see module docstring)."""

    def __init__(self, misalign_tol: float = 0.03, near_radius: float = 0.10):
        self.misalign_tol = misalign_tol
        self.near_radius = near_radius
        self.reset()

    def reset(self) -> None:
        self.phase = None
        self.counter = 0

    def act(self, w: ToyWorld) -> np.ndarray:
        a = np.zeros(7)
        a[6] = -1.0
        grasp_point = w.obj + GRASP_OFFSET
        if self.phase is None:
            near = np.linalg.norm(w.eef - w.obj) < self.near_radius
            misaligned = np.linalg.norm(w.eef[:2] - grasp_point[:2]) > self.misalign_tol
            self.phase = "BLIND_GRASP" if (near and misaligned) else "ALIGN"
            self.counter = 0

        def move(target, gain=0.5, limit=1.0):
            return np.clip((target - w.eef) / STEP_SCALE * gain, -limit, limit)

        if self.phase == "BLIND_GRASP":  # descend where it is and close: misses the object
            target = np.array([w.eef[0], w.eef[1], grasp_point[2]])
            a[:3] = move(target)
            if abs(w.eef[2] - grasp_point[2]) < 0.01:
                a[6] = 1.0
                self.counter += 1
                if self.counter > 6:
                    self.phase, self.counter = "HOVER", 0
            return a
        if self.phase == "HOVER":
            a[:3] = move(np.array([w.eef[0], w.eef[1], 0.12]))
            return a
        if self.phase == "ALIGN":
            above = grasp_point + np.array([0, 0, 0.06])
            a[:3] = move(above)
            if np.linalg.norm(w.eef - above) < 0.01:
                self.phase = "DESCEND"
            return a
        if self.phase == "DESCEND":
            a[:3] = move(grasp_point)
            if np.linalg.norm(w.eef - grasp_point) < 0.006:
                self.phase = "CLOSE"
            return a
        if self.phase == "CLOSE":
            a[6] = 1.0
            self.counter += 1
            if self.counter >= 5:
                self.phase, self.counter = "LIFT", 0
            return a
        if self.phase == "LIFT":
            a[6] = 1.0
            a[:3] = move(np.array([w.eef[0], w.eef[1], 0.25]), limit=0.4)
            if w.eef[2] > 0.24:
                self.phase = "TRANSPORT"
            return a
        if self.phase == "TRANSPORT":  # follows its plan even if the object was lost
            a[6] = 1.0
            above_target = np.array([w.target[0], w.target[1], 0.25]) + GRASP_OFFSET * [1, 1, 0]
            a[:3] = move(above_target, limit=0.3)
            if np.linalg.norm(w.eef - above_target) < 0.01:
                self.phase = "RELEASE"
            return a
        if self.phase == "RELEASE":
            a[:3] = move(np.array([w.eef[0], w.eef[1], 0.12]), limit=0.3)
            return a
        return a


class ToyBackend:
    """Backend over `ToyWorld` + `ToyPolicy`. Supports every fault type."""

    def __init__(self, episode_length: int = 850, world: ToyWorld | None = None, policy: ToyPolicy | None = None):
        self.profile = PandaDeltaProfile()
        self.episode_length = episode_length
        self.world = world or ToyWorld()
        self.policy = policy or ToyPolicy()

    def reset(self, seed: int | None, episode_index: int) -> None:
        self.world.reset()
        self.policy.reset()

    def state(self) -> RobotState:
        w = self.world
        return RobotState(
            step=w.step_count, time=w.step_count * DT, eef_xyz=w.eef.copy(), object_xyz=w.obj.copy(),
            target_xyz=w.target.copy(), gripper_command=w.gripper,
        )

    def policy_action(self) -> np.ndarray:
        return self.policy.act(self.world)

    def reset_policy(self) -> None:
        self.policy.reset()

    def step(self, action: np.ndarray) -> tuple[bool, bool, dict[str, Any]]:
        self.world.step(action)
        success = self.world.success()
        terminated = success or self.world.step_count >= self.episode_length
        return terminated, success, {}

    def inject_object_fault(self, kind: str, direction_xy: np.ndarray, cfg: Any) -> dict[str, Any]:
        w = self.world
        before = w.obj.copy()
        w.attached = False
        if kind == "forced_separation":
            w.obj[:2] += cfg.displacement_m * np.asarray(direction_xy)
        elif kind == "velocity_impulse":
            w.obj[:2] += cfg.impulse_speed_mps * DT * np.asarray(direction_xy)
        return {"object_before": before.tolist(), "object_after": w.obj.tolist()}

    def close(self) -> None:
        pass
