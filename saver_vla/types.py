"""Robot-agnostic data types shared by every SAVER-VLA component.

Positions are world-frame metres. Gripper commands follow the LIBERO/Panda
convention used throughout the project: -1 = open, +1 = close.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

# Skill phases that end a skill execution.
TERMINAL_PHASES = ("REACHED", "FAILED")


@dataclass
class RobotState:
    """One snapshot of the quantities the recovery layer reasons about.

    In simulation these come from privileged simulator state. On a real robot
    `object_xyz` and `target_xyz` must come from a state estimator (cameras).
    """

    step: int
    time: float
    eef_xyz: np.ndarray
    object_xyz: np.ndarray
    target_xyz: np.ndarray | None = None
    # Gripper command actually applied on the most recent step (None before the first step).
    gripper_command: float | None = None

    def __post_init__(self) -> None:
        self.eef_xyz = np.asarray(self.eef_xyz, dtype=float).reshape(3)
        self.object_xyz = np.asarray(self.object_xyz, dtype=float).reshape(3)
        if self.target_xyz is not None:
            self.target_xyz = np.asarray(self.target_xyz, dtype=float).reshape(3)
        if not (np.isfinite(self.eef_xyz).all() and np.isfinite(self.object_xyz).all()):
            raise ValueError("RobotState contains non-finite positions")

    @property
    def eef_object_distance(self) -> float:
        return float(np.linalg.norm(self.eef_xyz - self.object_xyz))

    @property
    def eef_object_xy_offset(self) -> np.ndarray:
        return self.eef_xyz[:2] - self.object_xyz[:2]

    @property
    def eef_object_xy_distance(self) -> float:
        return float(np.linalg.norm(self.eef_object_xy_offset))

    @property
    def target_xy_distance(self) -> float | None:
        if self.target_xyz is None:
            return None
        return float(np.linalg.norm(self.object_xyz[:2] - self.target_xyz[:2]))


@dataclass
class MotionCommand:
    """A robot-agnostic end-effector command produced by a recovery skill.

    `delta` is a normalized Cartesian displacement command in [-1, 1] per axis.
    The robot adapter converts it into the robot's native action (for the Panda
    in LIBERO: a 7-D OSC delta action; for the SO-101: joint targets via IK).
    """

    delta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gripper: float = GRIPPER_OPEN

    def __post_init__(self) -> None:
        self.delta = np.asarray(self.delta, dtype=float).reshape(3)

    @classmethod
    def hold(cls, gripper: float = GRIPPER_OPEN) -> "MotionCommand":
        return cls(np.zeros(3), gripper)
