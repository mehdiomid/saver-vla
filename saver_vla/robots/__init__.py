"""Robot adapters: the only robot-specific part of SAVER-VLA.

A *profile* turns a robot-agnostic `MotionCommand` into the robot's native action.
A *backend* runs episodes: it resets the world, exposes `RobotState`, queries the
policy, and executes actions. The supervisor, skills, agents and coordinator never
import a backend, so adding the SO-101 means adding a profile and a backend here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..types import MotionCommand, RobotState


@dataclass(frozen=True)
class RobotProfile:
    name: str
    action_dim: int
    gripper_index: int

    def to_action(self, cmd: MotionCommand) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class PandaDeltaProfile(RobotProfile):
    """Franka Panda under robosuite OSC_POSE in relative (delta) mode, as in LIBERO.

    Action = [dx, dy, dz, droll, dpitch, dyaw, gripper], each in [-1, 1].
    Recovery skills command translation only and keep orientation fixed.
    """

    name: str = "panda_libero"
    action_dim: int = 7
    gripper_index: int = 6

    def to_action(self, cmd: MotionCommand) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=float)
        action[:3] = np.clip(cmd.delta, -1.0, 1.0)
        action[self.gripper_index] = float(np.clip(cmd.gripper, -1.0, 1.0))
        return action


@dataclass(frozen=True)
class SO101Profile(RobotProfile):
    """Placeholder for the SO-101 (5 joints + gripper, joint-position control).

    The SO-101 adapter will convert end-effector targets into joint targets with
    inverse kinematics, and use its own skill parameters (reach, speeds, workspace).
    Not implemented yet; see README "Adding the SO-101".
    """

    name: str = "so101"
    action_dim: int = 6
    gripper_index: int = 5

    def to_action(self, cmd: MotionCommand) -> np.ndarray:
        raise NotImplementedError("SO-101 adapter not implemented yet (needs IK).")


PROFILES: dict[str, RobotProfile] = {
    "panda_libero": PandaDeltaProfile(),
    "so101": SO101Profile(),
}


def get_profile(name: str) -> RobotProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown robot profile '{name}'. Known: {sorted(PROFILES)}") from exc


class Backend(Protocol):
    """What the episode runner needs from a simulator or a real robot."""

    profile: RobotProfile

    def reset(self, seed: int | None, episode_index: int) -> None: ...

    def state(self) -> RobotState: ...

    def policy_action(self) -> np.ndarray: ...

    def reset_policy(self) -> None: ...

    def step(self, action: np.ndarray) -> tuple[bool, bool, dict[str, Any]]:
        """Execute one action. Returns (terminated, success, info)."""
        ...

    def close(self) -> None: ...


__all__ = ["Backend", "PROFILES", "PandaDeltaProfile", "RobotProfile", "SO101Profile", "get_profile"]
