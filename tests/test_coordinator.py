"""The coordinator must offer exactly the actions recorded in the Phase 6.5.1 run.

Each row is one agent decision from the notebook's Phase 6.5.1 log (cell 660):
step, EEF-object distance, the logged `available_actions`, and the skill executed.
"""

from saver_vla.config import RecoveryConfig, SkillConfig
from saver_vla.coordinator import Coordinator
from saver_vla.skills import action_key

P, R, A, Q = "MOVE_TO_PREGRASP:CENTER", "REPOSITION_TO_OBJECT_2D:CENTER", "APPROACH_OBJECT:CENTER", "REASSESS_STATE:CENTER"
XP, XN, RES, AB = "ADJUST_GRASP_POSE:OFFSET_X_POS", "ADJUST_GRASP_POSE:OFFSET_X_NEG", "RESUME_POLICY:CENTER", "ABORT_RECOVERY:CENTER"

PHASE651 = [
    (80, 0.21948330938278582, [P, R, Q, AB], ("MOVE_TO_PREGRASP", "CENTER")),
    (127, 0.13317828205351362, [P, R, A, Q, XP, XN, AB], ("APPROACH_OBJECT", "CENTER")),
    (200, 0.09652024660435495, [P, R, A, Q, XP, XN, RES, AB], ("RESUME_POLICY", "CENTER")),
    (295, 0.3778217947691248, [P, R, Q, AB], ("REPOSITION_TO_OBJECT_2D", "CENTER")),
    (515, 0.24677734247617278, [P, R, Q, AB], ("MOVE_TO_PREGRASP", "CENTER")),
    (599, 0.13292761859584043, [R, A, Q, XP, XN, AB], ("APPROACH_OBJECT", "CENTER")),
    (672, 0.09628638394713396, [R, Q, XP, XN, RES, AB], ("ADJUST_GRASP_POSE", "OFFSET_X_NEG")),
    (740, 0.07120266903750813, [R, Q, XP, XN, RES, AB], ("RESUME_POLICY", "CENTER")),
]


def test_phase651_action_menus_match_the_logged_run():
    coord = Coordinator(RecoveryConfig(), SkillConfig())
    attempts: dict[str, int] = {}
    for step, distance, logged, (skill, variant) in PHASE651:
        available, unavailable = coord.availability(distance, grasp_verified=False, attempts=attempts)
        assert available == logged, f"step {step}"
        assert set(available) | set(unavailable) == {k for k in [P, R, A, Q, XP, XN, RES, AB]}
        assert coord.rejection_reason(skill, variant, distance, False, attempts, available, unavailable) is None
        if skill not in ("RESUME_POLICY", "ABORT_RECOVERY"):
            key = action_key(skill, variant)
            attempts[key] = attempts.get(key, 0) + 1


def test_unavailable_reasons_are_explained():
    coord = Coordinator(RecoveryConfig(), SkillConfig())
    _, unavailable = coord.availability(0.30, False, {P: 2})
    assert "attempt limit" in unavailable[P]
    assert "0.20" in unavailable[A]
    assert "without confirmed grasp" in unavailable[RES]


def test_resume_allowed_far_away_when_grasp_is_verified():
    coord = Coordinator(RecoveryConfig(), SkillConfig())
    available, _ = coord.availability(0.30, grasp_verified=True, attempts={})
    assert RES in available


def test_disabled_skills_are_never_offered():
    cfg = RecoveryConfig(skills=["MOVE_TO_PREGRASP", "APPROACH_OBJECT", "RESUME_POLICY", "ABORT_RECOVERY"])
    coord = Coordinator(cfg, SkillConfig())
    available, _ = coord.availability(0.09, False, {})
    assert available == [P, A, RES, AB]


def test_legacy_mode_checks_rules_without_a_menu():
    coord = Coordinator(RecoveryConfig(capability_aware=False), SkillConfig())
    assert coord.rejection_reason("ADJUST_GRASP_POSE", "CENTER", 0.1, False, {}) == "invalid skill or variant"
    assert coord.rejection_reason("APPROACH_OBJECT", "CENTER", 0.25, False, {}) is not None
    assert coord.rejection_reason("APPROACH_OBJECT", "CENTER", 0.15, False, {}) is None
