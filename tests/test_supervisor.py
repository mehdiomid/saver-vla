"""Supervisor behavior and full episodes in the toy world (no GPU, no simulator)."""

import json
from types import SimpleNamespace

import pytest

from saver_vla.agents import CallableAgent, LLMAgent, ReplayAgent, ScriptedAgent
from saver_vla.config import config_from_dict
from saver_vla.events import read_events
from saver_vla.runner import make_backend, run_episode


def toy(**sections):
    sections.setdefault("env", {})["backend"] = "toy"
    return config_from_dict(sections)


def geometry_rule(obs):
    """A rule-based stand-in for the GPT agent's Phase 6.5.1 behaviour."""
    last, res, avail = obs["last_skill"], obs["last_skill_result"], obs["available_actions"]

    def pick(*keys):
        for key in keys:
            if key in avail:
                skill, variant = key.split(":")
                return {"skill": skill, "variant": variant, "reason": "rule"}
        return {"skill": "ABORT_RECOVERY", "variant": "CENTER", "reason": "nothing available"}

    dx = obs["eef_object_xy_offset_m"][0]
    if last is None or res in ("NOT_VERIFIED", "FAILED"):
        return pick("MOVE_TO_PREGRASP:CENTER", "REPOSITION_TO_OBJECT_2D:CENTER")
    if last in ("MOVE_TO_PREGRASP", "REPOSITION_TO_OBJECT_2D"):
        return pick("APPROACH_OBJECT:CENTER")
    if last == "APPROACH_OBJECT" and obs["policy_trial_count"] >= 1 and abs(dx) > 0.03:
        return pick("ADJUST_GRASP_POSE:OFFSET_X_NEG" if dx > 0 else "ADJUST_GRASP_POSE:OFFSET_X_POS")
    return pick("RESUME_POLICY:CENTER")


# ----------------------------------------------------------------------------- baselines


def test_policy_alone_fails_after_drop():
    cfg = toy(faults={"transport": "gripper_release"}, recovery={"agent": "none"})
    r = run_episode(cfg, make_backend(cfg))
    assert not r["success"] and r["outcome"] == "failure"
    assert r["fault_injected"] and r["detection_latency_steps"] is not None
    assert r["decisions"] == 0


def test_no_fault_episode_is_nominal_success():
    cfg = toy(faults={"transport": "none"}, recovery={"agent": "scripted"})
    r = run_episode(cfg, make_backend(cfg))
    assert r["success"] and r["outcome"] == "nominal_success" and r["decisions"] == 0


@pytest.mark.parametrize("fault", ["gripper_release", "forced_separation", "velocity_impulse"])
def test_scripted_agent_recovers(fault):
    cfg = toy(faults={"transport": fault, "approach_bias_x_m": 0.0}, recovery={"agent": "scripted"})
    r = run_episode(cfg, make_backend(cfg))
    assert r["success"] and r["outcome"] == "agent_recovery"
    assert r["decisions"] == 3 and r["verified_trial"] == 1


# ----------------------------------------------------------------------------- Phase 6.5.1 pattern


def test_geometry_bias_needs_replanning(tmp_path):
    cfg = toy(faults={"transport": "gripper_release", "approach_bias_x_m": 0.055})
    r = run_episode(cfg, make_backend(cfg), out_dir=tmp_path, agent=CallableAgent(geometry_rule))
    assert r["success"] and r["outcome"] == "agent_recovery"
    assert r["policy_trials"] == 2 and r["verified_trial"] == 2
    assert r["skill_attempts"].get("ADJUST_GRASP_POSE:OFFSET_X_NEG") == 1
    events = read_events(tmp_path / "events.jsonl")
    trials = [e["phase"] for e in events if e["kind"] == "POLICY_TRIAL_OUTCOME"]
    assert trials == ["NOT_VERIFIED", "VERIFIED"]
    assert json.loads((tmp_path / "result.json").read_text())["success"]


def test_fixed_script_cannot_fix_geometry_bias():
    """The script aborts after the failed trial. Any later success belongs to the policy, not the agent."""
    cfg = toy(faults={"transport": "gripper_release", "approach_bias_x_m": 0.055}, recovery={"agent": "scripted"})
    r = run_episode(cfg, make_backend(cfg))
    assert r["recovery_outcome"] == "ABORTED" and r["verified_trial"] is None
    assert r["outcome"] in ("failure", "policy_after_abort")


def test_replay_reproduces_the_same_episode(tmp_path):
    cfg = toy(faults={"transport": "gripper_release"})
    first = run_episode(cfg, make_backend(cfg), out_dir=tmp_path / "a", agent=CallableAgent(geometry_rule))
    replay = ReplayAgent.from_log(tmp_path / "a" / "events.jsonl")
    second = run_episode(cfg, make_backend(cfg), out_dir=tmp_path / "b", agent=replay)

    def trace(path):
        return [(e["kind"], e.get("step")) for e in read_events(path) if e["kind"] != "EPISODE_END"]

    assert trace(tmp_path / "a" / "events.jsonl") == trace(tmp_path / "b" / "events.jsonl")
    assert first["steps"] == second["steps"]


# ----------------------------------------------------------------------------- coordinator interplay


def test_rejected_request_triggers_replan_and_is_logged(tmp_path):
    asked = []

    def stubborn_then_valid(obs):
        asked.append(obs)
        if len(asked) == 1:
            return {"skill": "ADJUST_GRASP_POSE", "variant": "CENTER", "reason": "not a valid variant"}
        return ScriptedAgent()._decide(obs)

    cfg = toy(faults={"transport": "gripper_release", "approach_bias_x_m": 0.0})
    r = run_episode(cfg, make_backend(cfg), out_dir=tmp_path, agent=CallableAgent(stubborn_then_valid))
    assert r["rejections"] == 1 and r["success"]
    assert "rejected_request" in asked[1]
    rejected = [e for e in read_events(tmp_path / "events.jsonl") if e["kind"] == "SKILL_REJECTED"]
    assert rejected[0]["requested_skill"] == "ADJUST_GRASP_POSE"


def test_legacy_mode_aborts_instead_of_replanning():
    cfg = toy(faults={"transport": "gripper_release"}, recovery={"capability_aware": False})
    agent = CallableAgent(lambda obs: {"skill": "ADJUST_GRASP_POSE", "variant": "CENTER"})
    r = run_episode(cfg, make_backend(cfg), agent=agent)
    assert r["recovery_outcome"] == "ABORTED" and r["agent_calls"] == 1 and r["rejections"] == 1


def test_agent_errors_end_recovery_safely():
    def broken(obs):
        raise RuntimeError("network down")

    cfg = toy(faults={"transport": "gripper_release"})
    r = run_episode(cfg, make_backend(cfg), agent=CallableAgent(broken))
    assert r["recovery_outcome"] == "ABORTED"


def test_decision_budget_is_enforced():
    cfg = toy(faults={"transport": "gripper_release"}, recovery={"max_decisions": 2})
    agent = CallableAgent(lambda obs: {"skill": "REASSESS_STATE", "variant": "CENTER"})
    r = run_episode(cfg, make_backend(cfg), agent=agent)
    assert r["agent_calls"] == 2 and r["recovery_outcome"] == "ABORTED"


# ----------------------------------------------------------------------------- blind diagnosis


def test_blind_mode_hides_fault_details(tmp_path):
    seen = []

    def watcher(obs):
        seen.append(obs)
        return geometry_rule(obs)

    cfg = toy(faults={"transport": "gripper_release", "trial_impairment_steps": 60, "approach_bias_x_m": 0.0})
    run_episode(cfg, make_backend(cfg), out_dir=tmp_path, agent=CallableAgent(watcher))
    assert all(o["fault_mode"] == "UNDISCLOSED" for o in seen)
    feedback = [o["policy_trial_feedback"] for o in seen if o["policy_trial_feedback"]]
    assert feedback, "the impaired first trial should be reported"
    assert "actuator_impairment_steps" not in feedback[0]
    assert feedback[0]["gripper_command_mismatch_steps"] > 0  # observable evidence remains
    audit = [e for e in read_events(tmp_path / "events.jsonl") if e["kind"] == "POLICY_TRIAL_OUTCOME"]
    assert audit[0]["actuator_impairment_steps"] == 60  # hidden from the agent, kept for the audit


def test_informed_mode_shows_fault_mode():
    seen = []
    cfg = toy(faults={"transport": "gripper_release"}, recovery={"blind": False})
    run_episode(cfg, make_backend(cfg), agent=CallableAgent(lambda o: seen.append(o) or geometry_rule(o)))
    assert seen[0]["fault_mode"] == "GRIPPER_RELEASE"


# ----------------------------------------------------------------------------- LLM agent (fake client)


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        usage = SimpleNamespace(input_tokens=100, output_tokens=20)
        return SimpleNamespace(output_text=json.dumps(self.payload), usage=usage)


def test_llm_agent_request_format_and_validation():
    responses = FakeResponses(
        {"skill": "MOVE_TO_PREGRASP", "variant": "CENTER", "reason": "r", "expected_outcome": "e"}
    )
    agent = LLMAgent(client=SimpleNamespace(responses=responses))
    decision = agent.decide({"step": 1, "available_actions": ["MOVE_TO_PREGRASP:CENTER"]})
    assert decision["skill"] == "MOVE_TO_PREGRASP"
    req = responses.requests[0]
    assert req["model"] == "gpt-5.5"
    assert req["text"]["format"]["strict"] is True
    assert "You are NOT told the source" in req["instructions"]
    assert "REPOSITION_TO_OBJECT_2D:CENTER appears" in req["instructions"]
    assert json.loads(req["input"])["available_skills"] == ["MOVE_TO_PREGRASP:CENTER"]
    assert agent.stats()["input_tokens"] == 100

    bad = LLMAgent(client=SimpleNamespace(responses=FakeResponses(
        {"skill": "ADJUST_GRASP_POSE", "variant": "CENTER", "reason": "", "expected_outcome": ""})))
    with pytest.raises(ValueError):
        bad.decide({"step": 1})
