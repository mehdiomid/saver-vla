"""Base class for bounded recovery skills.

A skill is a small closed-loop controller with a clear start condition, a time
limit, and a terminal status (REACHED or FAILED). It reads a `RobotState` each
step and returns a `MotionCommand`. It never decides *which* skill to run; that
is the agent's job, checked by the coordinator.
"""

from __future__ import annotations

from typing import Any

from ..types import TERMINAL_PHASES, MotionCommand, RobotState


class Skill:
    name: str = "SKILL"

    def __init__(self, variant: str = "CENTER"):
        self.variant = variant
        self.phase = "IDLE"
        self.steps = 0
        self.reason: str | None = None

    @property
    def done(self) -> bool:
        return self.phase in TERMINAL_PHASES

    def start(self, state: RobotState) -> bool:
        """Check preconditions and begin. Return False to decline (nothing is executed)."""
        raise NotImplementedError

    def update(self, state: RobotState) -> MotionCommand:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {"skill": self.name, "phase": self.phase, "steps": self.steps, "reason": self.reason}

    def _fail(self, reason: str) -> MotionCommand:
        self.phase = "FAILED"
        self.reason = reason
        return MotionCommand.hold()
