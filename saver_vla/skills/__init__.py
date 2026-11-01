"""Skill registry: maps an agent's (skill, variant) request to a controller."""

from __future__ import annotations

import numpy as np

from ..config import SkillConfig
from .base import Skill
from .library import ApproachObject, MoveToPregrasp, ReassessState, Reposition2D

# Actions the agent may request, in the order they are shown to it.
# RESUME_POLICY and ABORT_RECOVERY are coordinator actions, not motion skills.
ACTION_MENU: list[tuple[str, str]] = [
    ("MOVE_TO_PREGRASP", "CENTER"),
    ("REPOSITION_TO_OBJECT_2D", "CENTER"),
    ("APPROACH_OBJECT", "CENTER"),
    ("REASSESS_STATE", "CENTER"),
    ("ADJUST_GRASP_POSE", "OFFSET_X_POS"),
    ("ADJUST_GRASP_POSE", "OFFSET_X_NEG"),
    ("RESUME_POLICY", "CENTER"),
    ("ABORT_RECOVERY", "CENTER"),
]
CONTROL_ACTIONS = ("RESUME_POLICY", "ABORT_RECOVERY")
MOTION_SKILLS = tuple(name for name, _ in ACTION_MENU if name not in CONTROL_ACTIONS)
VARIANTS = ("CENTER", "OFFSET_X_POS", "OFFSET_X_NEG")
# Skills that start near the object and are refused beyond approach range.
NEAR_OBJECT_SKILLS = ("APPROACH_OBJECT", "ADJUST_GRASP_POSE")


def action_key(skill: str, variant: str) -> str:
    return f"{skill}:{variant}"


def valid_variant(skill: str, variant: str) -> bool:
    if skill == "ADJUST_GRASP_POSE":
        return variant in ("OFFSET_X_POS", "OFFSET_X_NEG")
    return variant == "CENTER"


def make_skill(
    name: str,
    variant: str,
    cfg: SkillConfig,
    approach_bias: np.ndarray | None = None,
) -> Skill:
    if name == "MOVE_TO_PREGRASP":
        return MoveToPregrasp(cfg.pregrasp, variant)
    if name == "APPROACH_OBJECT":
        return ApproachObject(cfg.approach, "CENTER", target_bias=approach_bias)
    if name == "ADJUST_GRASP_POSE":
        return ApproachObject(cfg.approach, variant)
    if name == "REASSESS_STATE":
        return ReassessState(cfg.reassess, variant)
    if name == "REPOSITION_TO_OBJECT_2D":
        return Reposition2D(cfg.reposition_2d, variant)
    raise ValueError(f"Unknown motion skill: {name}")


__all__ = [
    "ACTION_MENU",
    "CONTROL_ACTIONS",
    "MOTION_SKILLS",
    "NEAR_OBJECT_SKILLS",
    "VARIANTS",
    "ApproachObject",
    "MoveToPregrasp",
    "ReassessState",
    "Reposition2D",
    "Skill",
    "action_key",
    "make_skill",
    "valid_variant",
]
