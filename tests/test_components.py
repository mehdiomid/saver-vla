"""Unit tests for skills, the verifier, the grasp tracker, faults and configuration."""

import numpy as np
import pytest

from saver_vla.config import ExperimentConfig, FaultConfig, config_from_dict, load_config
from saver_vla.faults import FaultInjector
from saver_vla.skills import ApproachObject, MoveToPregrasp, ReassessState, Reposition2D, make_skill
from saver_vla.config import SkillConfig
from saver_vla.types import GRIPPER_OPEN, RobotState
from saver_vla.verifier import GRASP_CONFIRMED, GRASP_LOST, GraspTracker, TransportVerifier


def state(eef, obj, step=0, target=(-0.3, 0.15, 0.04)):
    return RobotState(step=step, time=step * 0.05, eef_xyz=eef, object_xyz=obj, target_xyz=target)


def drive(skill, eef, obj, scale=0.05, max_steps=400):
    """Kinematic stand-in: the end effector moves `scale` metres per unit command."""
    eef = np.array(eef, float)
    for k in range(max_steps):
        cmd = skill.update(state(eef, obj, k))
        assert cmd.gripper == GRIPPER_OPEN
        assert np.all(np.abs(cmd.delta) <= 1.0)
        eef = eef + cmd.delta * scale
        if skill.done:
            break
    return eef


# ----------------------------------------------------------------------------- skills


def test_pregrasp_reaches_above_object():
    obj = np.array([-0.12, -0.13, 0.0384])
    skill = MoveToPregrasp()
    assert skill.start(state([-0.12, 0.02, 0.25], obj))
    eef = drive(skill, [-0.12, 0.02, 0.25], obj)
    assert skill.phase == "REACHED"
    assert np.linalg.norm(eef - (obj + [0, 0, 0.12])) <= 0.02


def test_pregrasp_times_out_when_robot_cannot_move():
    skill = MoveToPregrasp()
    skill.start(state([0, 0, 0.05], [0.3, 0.3, 0.04]))
    drive(skill, [0, 0, 0.05], [0.3, 0.3, 0.04], scale=0.0)
    assert skill.phase == "FAILED" and skill.reason in ("PHASE_TIMEOUT", "TOTAL_TIMEOUT")


def test_reposition_2d_matches_notebook_cell_644():
    """Same start state and simplified dynamics as the notebook's Phase 6.5 unit test."""
    obj = np.array([-0.1213, -0.1313, 0.0384])
    skill = Reposition2D()
    skill.start(state([-0.1210, 0.0215, 0.2526], obj))
    eef = drive(skill, [-0.1210, 0.0215, 0.2526], obj, scale=0.20)
    assert skill.phase == "REACHED"
    assert np.linalg.norm(eef[:2] - obj[:2]) <= 0.012
    assert abs(eef[2] - (obj[2] + 0.13)) <= 0.012  # height-aware: ends at pregrasp height


def test_approach_declines_when_far_from_object():
    skill = ApproachObject()
    assert not skill.start(state([0, 0, 0.40], [0, 0, 0.04]))
    assert skill.phase == "IDLE"


def test_approach_bias_applies_only_to_center_variant():
    obj = np.array([0.0, 0.0, 0.04])
    near = state([0.0, 0.0, 0.17], obj)
    bias = np.array([0.055, 0, 0])
    center = make_skill("APPROACH_OBJECT", "CENTER", SkillConfig(), approach_bias=bias)
    adjust = make_skill("ADJUST_GRASP_POSE", "OFFSET_X_NEG", SkillConfig(), approach_bias=bias)
    assert center.start(near) and adjust.start(near)
    assert center.target[0] == pytest.approx(0.0142 + 0.055)
    assert adjust.target[0] == pytest.approx(0.0142 - 0.020)
    assert adjust.name == "ADJUST_GRASP_POSE"


def test_approach_reaches_target():
    obj = np.array([0.0, 0.0, 0.04])
    skill = ApproachObject()
    skill.start(state([0, 0, 0.17], obj))
    drive(skill, [0, 0, 0.17], obj)
    assert skill.phase == "REACHED"


def test_reassess_holds_for_configured_steps():
    skill = ReassessState()
    skill.start(state([0, 0, 0.2], [0, 0, 0.04]))
    for _ in range(4):
        cmd = skill.update(state([0, 0, 0.2], [0, 0, 0.04]))
        assert not cmd.delta.any()
    assert skill.phase == "REACHED"


# ----------------------------------------------------------------------------- verifier


def test_verifier_detects_transport_separation():
    v = TransportVerifier()
    events = []
    obj = np.array([0.0, 0.0, 0.04])
    eef = np.array([0.0, 0.0, 0.08])
    for k in range(80):
        if k > 5:  # lift together
            eef[2] = min(eef[2] + 0.01, 0.30)
            if k < 40:
                obj = eef - [0, 0, 0.04]
        if k >= 40:  # object stays behind while the gripper carries on
            eef[0] += 0.01
            obj = np.array([obj[0], obj[1], max(0.04, obj[2] - 0.03)])
        events += v.update(state(eef, obj, k))
    kinds = [e["event_type"] for e in events]
    assert kinds[:2] == ["APPROACH_CONFIRMED", "LIFT_CONFIRMED"]
    sep = [e for e in events if e["event_type"] == "SUSPECTED_TRANSPORT_SEPARATION"]
    assert len(sep) == 1 and sep[0]["evidence"]["possible_drop"]


def test_verifier_ignores_release_at_target():
    v = TransportVerifier()
    target = (0.0, 0.0, 0.04)
    for k in range(10):
        events = v.update(state([0, 0, 0.1 + 0.02 * k], [0, 0, 0.06 + 0.02 * k], k, target=target))
    later = v.update(state([0.3, 0, 0.3], [0, 0, 0.04], 11, target=target))
    assert all(e["event_type"] != "SUSPECTED_TRANSPORT_SEPARATION" for e in events + later)


def test_grasp_tracker_confirms_then_loses_grasp():
    t = GraspTracker()
    for k in range(6):
        z = 0.10 + 0.01 * k
        s = t.update([0, 0, z], [0, 0, z - 0.03], 1.0)
    assert s == GRASP_CONFIRMED
    for k in range(3):
        s = t.update([0, 0, 0.2], [0.2, 0, 0.04], -1.0)
    assert s == GRASP_LOST


def test_grasp_tracker_requires_closed_gripper():
    t = GraspTracker()
    for k in range(10):
        z = 0.10 + 0.01 * k
        s = t.update([0, 0, z], [0, 0, z - 0.03], -1.0)
    assert s != GRASP_CONFIRMED


# ----------------------------------------------------------------------------- faults


def test_fault_trigger_waits_for_sustained_lift_and_injects_once():
    inj = FaultInjector(FaultConfig(transport="gripper_release", release_steps=5))
    records = []
    for k in range(40):
        z = 0.04 if k < 10 else 0.20
        rec = inj.before_step(state([0, 0, z + 0.04], [0, 0, z], k), backend=None)
        if rec:
            records.append(rec)
    assert len(records) == 1 and records[0]["step"] >= 10 + 1 + 6
    actions = [inj.apply_action(np.ones(7), in_trial=False)[6] for _ in range(7)]
    assert actions == [-1.0] * 5 + [1.0] * 2


def test_trial_impairment_only_affects_first_trial():
    inj = FaultInjector(FaultConfig(transport="none", trial_impairment_steps=3))
    inj.begin_trial(1)
    first = [inj.apply_action(np.ones(7), in_trial=True)[6] for _ in range(4)]
    assert first == [-1, -1, -1, 1] and inj.end_trial()["grip_block_steps"] == 3
    inj.begin_trial(2)
    assert inj.apply_action(np.ones(7), in_trial=True)[6] == 1


def test_object_fault_needs_capable_backend():
    inj = FaultInjector(FaultConfig(transport="forced_separation"))
    with pytest.raises(NotImplementedError):
        for k in range(40):
            inj.before_step(state([0, 0, 0.3], [0, 0, 0.04 if k < 5 else 0.2], k), backend=object())


# ----------------------------------------------------------------------------- config


def test_defaults_match_phase651():
    cfg = ExperimentConfig()
    assert cfg.faults.transport == "gripper_release" and cfg.faults.release_steps == 15
    assert cfg.faults.approach_bias_x_m == 0.055
    assert cfg.recovery.model == "gpt-5.5" and cfg.recovery.max_decisions == 8
    assert cfg.recovery.trial_steps == 95 and cfg.recovery.max_trials == 2
    assert cfg.env.episode_length == 850


def test_overrides_and_unknown_keys():
    cfg = config_from_dict({"recovery": {"agent": "scripted"}}, ["recovery.model=gpt-5.2", "seed=7"])
    assert cfg.recovery.agent == "scripted" and cfg.recovery.model == "gpt-5.2" and cfg.seed == 7
    with pytest.raises(KeyError):
        config_from_dict({"recovery": {"modle": "x"}})


def test_example_configs_load():
    from pathlib import Path

    for path in sorted(Path(__file__).parents[1].joinpath("configs").glob("*.yaml")):
        load_config(path)
