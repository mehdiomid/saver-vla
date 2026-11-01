"""The coordinator: decides which actions the agent may take right now.

It enforces the same rules in two places, so the agent is never shown an option
the coordinator would then refuse (Phase 6.4c):
  * `availability()` builds the menu the agent sees;
  * `rejection_reason()` checks the agent's actual request.
"""

from __future__ import annotations

from .config import RecoveryConfig, SkillConfig
from .skills import ACTION_MENU, CONTROL_ACTIONS, NEAR_OBJECT_SKILLS, action_key, valid_variant


class Coordinator:
    def __init__(self, recovery: RecoveryConfig, skills: SkillConfig):
        self.cfg = recovery
        self.approach_max_distance = skills.approach.max_start_distance_m
        self.menu = [(s, v) for s, v in ACTION_MENU if s in recovery.skills]
        if not any(s == "ABORT_RECOVERY" for s, _ in self.menu):
            self.menu.append(("ABORT_RECOVERY", "CENTER"))

    def _rule_violation(self, skill: str, distance: float, grasp_verified: bool, attempts: int) -> str | None:
        if skill not in CONTROL_ACTIONS and attempts >= self.cfg.attempt_limit:
            return f"attempt limit {self.cfg.attempt_limit} reached"
        if skill in NEAR_OBJECT_SKILLS and distance > self.approach_max_distance:
            return f"EEF-object distance exceeds {self.approach_max_distance:.2f} m"
        if skill == "RESUME_POLICY" and distance > self.cfg.resume_max_distance_m and not grasp_verified:
            return (
                f"EEF-object distance exceeds {self.cfg.resume_max_distance_m:.3f} m "
                "without confirmed grasp"
            )
        return None

    def availability(
        self, distance: float, grasp_verified: bool, attempts: dict[str, int]
    ) -> tuple[list[str], dict[str, str]]:
        available: list[str] = []
        unavailable: dict[str, str] = {}
        for skill, variant in self.menu:
            key = action_key(skill, variant)
            reason = self._rule_violation(skill, distance, grasp_verified, attempts.get(key, 0))
            if reason:
                unavailable[key] = reason
            else:
                available.append(key)
        return available, unavailable

    def rejection_reason(
        self,
        skill: str,
        variant: str,
        distance: float,
        grasp_verified: bool,
        attempts: dict[str, int],
        available: list[str] | None = None,
        unavailable: dict[str, str] | None = None,
    ) -> str | None:
        key = action_key(str(skill), str(variant))
        if available is not None:
            if key not in available:
                return (unavailable or {}).get(key, "not in current allowed action set")
            return None
        if (skill, variant) not in self.menu or not valid_variant(skill, variant):
            return "invalid skill or variant"
        return self._rule_violation(skill, distance, grasp_verified, attempts.get(key, 0))
