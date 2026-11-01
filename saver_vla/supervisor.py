"""The recovery supervisor: SAVER-VLA's core closed loop.

The supervisor sits between the policy and the robot. Each control step:

    action = supervisor.before_step(state, policy_action)   # policy or active skill
    ... robot executes the (possibly fault-modified) action ...
    supervisor.after_step(new_state, applied_action)        # monitor, decide, verify

It is robot-agnostic: it only sees `RobotState`s and returns actions through a
`to_action` function supplied by the robot adapter. The agent is called only at
recovery boundaries: the first detected failure, each finished skill, and each
unverified policy grasp trial.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .agents import Agent
from .config import ExperimentConfig
from .coordinator import Coordinator
from .events import EventLog
from .faults import FaultInjector
from .skills import CONTROL_ACTIONS, Skill, action_key, make_skill
from .types import MotionCommand, RobotState
from .verifier import GRASP_HELD, GraspTracker, TransportVerifier

HIDDEN_KEYS = ("actuator_impairment_steps", "grip_block_steps", "injected_fault", "fault_mode")
MISMATCH_THRESHOLD = 0.25


def _abort(reason: str) -> dict[str, Any]:
    return {
        "skill": "ABORT_RECOVERY",
        "variant": "CENTER",
        "reason": reason,
        "expected_outcome": "End bounded intervention",
    }


class RecoverySupervisor:
    def __init__(
        self,
        cfg: ExperimentConfig,
        agent: Agent | None,
        to_action: Callable[[MotionCommand], np.ndarray],
        gripper_index: int,
        log: EventLog | None = None,
        faults: FaultInjector | None = None,
    ):
        self.cfg = cfg
        self.rc = cfg.recovery
        self.agent = agent
        self.to_action = to_action
        self.gripper_index = gripper_index
        self.log = log or EventLog()
        self.faults = faults
        self.coordinator = Coordinator(cfg.recovery, cfg.skills)
        self.verifier = TransportVerifier(cfg.verifier)
        self.grasp_tracker = GraspTracker(cfg.grasp_tracker)
        self.reset()

    # ------------------------------------------------------------------ lifecycle

    def reset(self) -> None:
        self.verifier.reset()
        self.grasp_tracker.reset()
        if self.agent is not None:
            self.agent.reset()
        self.state: RobotState | None = None
        self.skill: Skill | None = None
        self.recovery_started = False
        self.recovery_finished = False
        self.outcome: str | None = None  # VERIFIED | ABORTED | RESUMED | BUDGET_EXHAUSTED
        self.failure_type: str | None = None
        self.failure_step: int | None = None
        self.last_skill: str | None = None
        self.last_result_detail: dict[str, Any] | None = None
        self.decisions = 0
        self.agent_calls = 0
        self.rejections = 0
        self.skill_counts: dict[str, int] = {}
        self.trial_active = False
        self.trial_start: int | None = None
        self.trial_count = 0
        self.trial_steps = 0
        self.trial_mismatch_steps = 0
        self.verified_trial: int | None = None
        self._requested_gripper: float | None = None
        self.policy_reset_pending = False

    @property
    def controlling(self) -> bool:
        """True while a recovery skill (not the policy) commands the robot."""
        return self.skill is not None

    # ------------------------------------------------------------------ per-step API

    def before_step(self, state: RobotState, policy_action: np.ndarray | None) -> np.ndarray:
        self.state = state
        if self.skill is not None:
            action = np.asarray(self.to_action(self.skill.update(state)), dtype=float)
        else:
            if policy_action is None:
                raise ValueError("policy_action is required when no skill is active")
            action = np.asarray(policy_action, dtype=float)
        self._requested_gripper = float(action[self.gripper_index])
        return action

    def after_step(self, state: RobotState, applied_action: np.ndarray) -> None:
        self.state = state
        applied_gripper = float(np.asarray(applied_action)[self.gripper_index])
        self.grasp_tracker.update(state.eef_xyz, state.object_xyz, applied_gripper)

        if self.trial_active:
            self.trial_steps += 1
            if (
                self._requested_gripper is not None
                and abs(applied_gripper - self._requested_gripper) > MISMATCH_THRESHOLD
            ):
                self.trial_mismatch_steps += 1

        for event in self.verifier.update(state):
            self.log.write("VERIFIER_EVENT", **event)
            if event["event_type"] not in self.rc.trigger_events or self.recovery_started:
                continue
            if self.failure_step is None:
                self.failure_type = event["event_type"]
                self.failure_step = state.step
                self.log.write("FAILURE", event=event)
            if self.agent is not None:
                self.recovery_started = True
                self._decide(self.failure_type, None)
            elif self.rc.enabled and self.rc.agent == "queue_reset":
                # Phase 3 baseline: restart policy inference, no recovery skills.
                self.recovery_started = True
                self.recovery_finished = True
                self.outcome = "RESUMED"
                self.policy_reset_pending = True
                self.log.write("POLICY_QUEUE_RESET", step=state.step)

        if self.trial_active:
            self._check_trial(state)

        if self.skill is not None and self.skill.done:
            status = self.skill.status()
            self.skill = None
            self.log.write("SKILL_OUTCOME", status=status, step=state.step)
            if self.agent is not None:
                self.agent.record_result(status)
            self.last_result_detail = status
            self._decide(self.failure_type, status["phase"])

    def consume_policy_reset(self) -> bool:
        pending, self.policy_reset_pending = self.policy_reset_pending, False
        return pending

    # ------------------------------------------------------------------ policy trials

    def _check_trial(self, state: RobotState) -> None:
        elapsed = int(state.step - self.trial_start)
        if self.grasp_tracker.state in GRASP_HELD:
            info = self.faults.end_trial() if self.faults else {}
            self.trial_active = False
            self.verified_trial = self.trial_count
            self.recovery_finished = True
            self.outcome = "VERIFIED"
            self.log.write(
                "POLICY_TRIAL_OUTCOME", trial=self.trial_count, phase="VERIFIED", elapsed=elapsed,
                step=state.step, worker_info=info,
            )
            if self.agent is not None:
                self.agent.record_result({"skill": "POLICY_GRASP_TRIAL", "phase": "VERIFIED", "steps": elapsed})
            return
        if elapsed < self.rc.trial_steps:
            return
        info = self.faults.end_trial() if self.faults else {}
        self.trial_active = False
        outcome = {
            "skill": "POLICY_GRASP_TRIAL",
            "phase": "NOT_VERIFIED",
            "steps": elapsed,
            "actuator_impairment_steps": info.get("grip_block_steps", 0),
            "gripper_command_mismatch_steps": self.trial_mismatch_steps,
            "reason": "No sustained grasp evidence during trial window",
        }
        self.log.write("POLICY_TRIAL_OUTCOME", trial=self.trial_count, step=state.step, **outcome)
        agent_outcome = self._sanitize(outcome)
        if self.agent is not None:
            self.agent.record_result(agent_outcome)
        self.last_result_detail = agent_outcome
        if self.decisions < self.rc.max_decisions:
            self._decide("POLICY_GRASP_NOT_VERIFIED", "NOT_VERIFIED")
        else:
            self.log.write("REPLAN_SKIPPED", reason="decision budget exhausted", step=state.step)
            self.recovery_finished = True
            self.outcome = "BUDGET_EXHAUSTED"

    def _begin_trial(self, step: int) -> None:
        if self.trial_count >= self.rc.max_trials:
            self.log.write("POLICY_TRIAL_SKIPPED", reason="trial budget exhausted", step=step)
            self.recovery_finished = True
            self.outcome = "RESUMED"
            return
        self.trial_count += 1
        if self.faults:
            self.faults.begin_trial(self.trial_count)
        self.trial_active = True
        self.trial_start = step
        self.trial_steps = 0
        self.trial_mismatch_steps = 0
        self.log.write("POLICY_TRIAL_START", trial=self.trial_count, step=step, accepted=True)

    # ------------------------------------------------------------------ decisions

    def _sanitize(self, detail: Any) -> Any:
        if not self.rc.blind or not isinstance(detail, dict):
            return detail
        return {k: v for k, v in detail.items() if k not in HIDDEN_KEYS}

    def _observation(self, failure: str | None, result: str | None) -> dict[str, Any]:
        s = self.state
        grasp_state = self.grasp_tracker.state
        detail = self.last_result_detail
        feedback = detail if isinstance(detail, dict) and detail.get("skill") == "POLICY_GRASP_TRIAL" else None
        fault_mode = self.cfg.faults.transport.upper()
        obs: dict[str, Any] = {
            "step": s.step,
            "failure": failure,
            "eef_object_distance_m": s.eef_object_distance,
            "object_height_m": float(s.object_xyz[2]),
            "eef_xyz_m": s.eef_xyz.tolist(),
            "object_xyz_m": s.object_xyz.tolist(),
            "eef_object_xy_offset_m": [round(float(v), 5) for v in s.eef_object_xy_offset],
            "eef_object_xy_distance_m": round(s.eef_object_xy_distance, 5),
            "target_xy_distance_m": s.target_xy_distance,
            "gripper_command": s.gripper_command,
            "fault_mode": "UNDISCLOSED" if self.rc.blind else fault_mode,
            "grasp_state": grasp_state,
            "last_skill": self.last_skill,
            "last_skill_result": result,
            "last_skill_detail": self._sanitize(detail),
            "policy_trial_count": self.trial_count,
            "policy_trial_feedback": self._sanitize(feedback),
            "skill_attempts": dict(self.skill_counts),
            "grasp_verified": grasp_state in GRASP_HELD,
        }
        if self.rc.capability_aware:
            available, unavailable = self.coordinator.availability(
                s.eef_object_distance, obs["grasp_verified"], self.skill_counts
            )
            obs["available_actions"] = available
            obs["unavailable_actions"] = unavailable
            obs["remaining_policy_trials"] = max(0, self.rc.max_trials - self.trial_count) if self.rc.policy_trials else 0
            obs["remaining_agent_decisions"] = max(0, self.rc.max_decisions - self.decisions)
        return obs

    def _ask(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.agent_calls += 1
        try:
            decision = self.agent.decide(observation)
        except Exception as exc:  # LLM/network/parse errors end the intervention safely
            decision = _abort(f"Agent error: {str(exc)[:200]}")
            self.log.write("AGENT_ERROR", error=str(exc)[:500], step=observation["step"])
            return decision
        self.log.write("AGENT_CALL", step=observation["step"], decision=decision)
        return decision

    def _decide(self, failure: str | None, result: str | None) -> None:
        s = self.state
        observation = self._observation(failure, result)
        if self.decisions >= self.rc.max_decisions:
            choice = _abort("decision budget exhausted")
        else:
            choice = self._ask(observation)
        self.decisions += 1

        requested = dict(choice)
        available = observation.get("available_actions") if self.rc.capability_aware else None
        unavailable = observation.get("unavailable_actions") if self.rc.capability_aware else None
        replans = 0
        rejection: str | None = None
        started: Skill | None = None
        while True:
            skill = choice.get("skill", "ABORT_RECOVERY")
            variant = choice.get("variant", "CENTER")
            key = action_key(str(skill), str(variant))
            rejection = self.coordinator.rejection_reason(
                skill, variant, s.eef_object_distance, observation["grasp_verified"],
                self.skill_counts, available, unavailable,
            )
            if rejection is None and skill not in CONTROL_ACTIONS:
                bias = self.faults.approach_bias() if self.faults else None
                candidate = make_skill(skill, variant, self.cfg.skills, approach_bias=bias)
                try:
                    if candidate.start(s):
                        started = candidate
                        self.skill_counts[key] = self.skill_counts.get(key, 0) + 1
                    else:
                        rejection = "skill declined to start (preconditions not met)"
                except Exception as exc:
                    rejection = "skill start failed: " + str(exc)[:200]
            if rejection is None:
                break
            self.rejections += 1
            self.log.write(
                "SKILL_REJECTED", step=s.step, requested_skill=skill, requested_variant=variant,
                rejection_reason=rejection, retry_number=replans,
            )
            if (
                not self.rc.capability_aware
                or replans >= self.rc.max_replans
                or self.decisions >= self.rc.max_decisions
            ):
                choice = _abort("Coordinator fallback after rejected requests: " + rejection)
                skill, variant = "ABORT_RECOVERY", "CENTER"
                break
            replans += 1
            self.decisions += 1
            replan_obs = dict(observation)
            replan_obs["rejected_request"] = {"skill": skill, "variant": variant, "reason": rejection}
            choice = self._ask(replan_obs)

        self.log.write(
            "DECISION",
            step=s.step,
            observation=observation,
            decision=choice,
            initial_requested_decision=requested,
            executed_skill=skill,
            variant=variant,
            accepted=started is not None,
            replans=replans,
            rejection_reason=rejection,
        )
        self.policy_reset_pending = True

        if skill == "RESUME_POLICY":
            self.skill = None
            if self.rc.policy_trials:
                self._begin_trial(s.step)
            else:
                self.recovery_finished = True
                self.outcome = "RESUMED"
        elif skill == "ABORT_RECOVERY":
            self.skill = None
            self.recovery_finished = True
            self.outcome = "ABORTED"
            self.log.write(
                "RECOVERY_ABORTED", step=s.step,
                note="bounded intervention ended; the policy may continue",
            )
        else:
            self.skill = started
            self.last_skill = skill

    # ------------------------------------------------------------------ summary

    def summary(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type,
            "failure_step": self.failure_step,
            "recovery_started": self.recovery_started,
            "recovery_outcome": self.outcome,
            "decisions": self.decisions,
            "agent_calls": self.agent_calls,
            "rejections": self.rejections,
            "skill_attempts": dict(self.skill_counts),
            "policy_trials": self.trial_count,
            "verified_trial": self.verified_trial,
        }
