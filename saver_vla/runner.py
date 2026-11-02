"""Run one episode: backend + fault injector + recovery supervisor.

    for each control step:
        state  = backend.state()
        fault  = faults.before_step(state, backend)        # may inject once
        policy = backend.policy_action()                   # the VLA's proposal
        action = supervisor.before_step(state, policy)     # policy or recovery skill
        action = faults.apply_action(action)               # e.g. forced gripper opening
        backend.step(action)
        supervisor.after_step(backend.state(), action)     # monitor, decide, verify

Each episode writes `events.jsonl`, `result.json` and `config.yaml` to its folder.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from .agents import Agent, make_agent
from .config import ExperimentConfig
from .events import EventLog
from .faults import FaultInjector, episode_rng
from .perception import Perception
from .robots import Backend
from .supervisor import RecoverySupervisor


def make_backend(cfg: ExperimentConfig) -> Backend:
    if cfg.env.backend == "toy":
        from .robots.toy import ToyBackend

        return ToyBackend(episode_length=cfg.env.episode_length)
    if cfg.env.backend == "libero":
        from .sim.libero import LiberoBackend

        return LiberoBackend(cfg.env)
    raise ValueError(f"Unknown env.backend: {cfg.env.backend}")


def classify(success: bool, fault_injected: bool, summary: dict[str, Any]) -> str:
    """Attribute the episode outcome, so task success is not credited to the agent by default."""
    if summary.get("agent_errors"):
        return "agent_error"  # the agent was unreachable: the episode does not measure recovery
    if not success:
        return "failure" if fault_injected else "nominal_failure"
    if not fault_injected:
        # No injected fault, but the agent recovered from the policy's own failure (E5).
        if summary.get("recoveries") and summary.get("recovery_outcome") == "VERIFIED":
            return "agent_rescue"
        return "nominal_success"
    outcome = summary.get("recovery_outcome")
    recoveries = summary.get("recoveries", 0)
    if outcome == "VERIFIED" and summary.get("verified_recoveries", 0) == recoveries:
        return "agent_recovery"
    if outcome in ("ABORTED", "BUDGET_EXHAUSTED"):
        return "policy_after_abort"
    if outcome == "RESUMED":
        return "resumed_success"
    return "policy_only_success"


def run_episode(
    cfg: ExperimentConfig,
    backend: Backend,
    out_dir: str | Path | None = None,
    agent: Agent | None = None,
    echo: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    log = EventLog(out / "events.jsonl" if out else None, echo=echo)
    if agent is None:
        agent = make_agent(cfg.recovery)
    profile = backend.profile
    faults = FaultInjector(cfg.faults, gripper_index=profile.gripper_index)
    perception = Perception(cfg.noise, episode_rng(cfg.seed, cfg.episode_index, stream=1))
    supervisor = RecoverySupervisor(cfg, agent, profile.to_action, profile.gripper_index, log=log, faults=faults)

    started = time.perf_counter()
    backend.reset(cfg.seed, cfg.episode_index)
    if cfg.env.record_video and hasattr(backend, "start_recording"):
        backend.start_recording()
    supervisor.reset()
    faults.reset(episode_rng(cfg.seed, cfg.episode_index, stream=0))
    log.write("EPISODE_START", seed=cfg.seed, episode_index=cfg.episode_index, name=cfg.name)
    # Hidden settings sampled for this episode: audit only, never shown to the agent.
    log.write("FAULT_PLAN", **faults.episode_plan())

    success = False
    fault_record = None
    fault_records: list[dict[str, Any]] = []
    steps = 0
    for _ in range(cfg.env.episode_length):
        true_state = backend.state()
        record = faults.before_step(true_state, backend)
        if record is not None:
            fault_record = fault_record or record
            fault_records.append(record)
            log.write("FAULT", **record)
            true_state = backend.state()  # object faults change the world before the next action

        need_policy = (not supervisor.controlling) or cfg.env.query_policy_during_skills
        policy_action = backend.policy_action() if need_policy else None
        action = supervisor.before_step(perception.observe(true_state), policy_action)
        applied = faults.apply_action(action, in_trial=supervisor.trial_active)
        executed = perception.noisy_action(applied)
        terminated, success, _ = backend.step(executed)
        steps += 1
        supervisor.after_step(perception.observe(backend.state()), applied)
        if supervisor.consume_policy_reset():
            backend.reset_policy()
        if terminated:
            break

    if cfg.env.record_video and hasattr(backend, "save_recording") and out:
        backend.save_recording(out / "episode.mp4")

    summary = supervisor.summary()
    detection_latency = None
    false_alarm = summary["failure_step"] is not None and (
        fault_record is None or summary["failure_step"] < fault_record["step"]
    )
    if fault_record is not None and summary["failure_step"] is not None and not false_alarm:
        detection_latency = summary["failure_step"] - fault_record["step"]
    result: dict[str, Any] = {
        "name": cfg.name,
        "seed": cfg.seed,
        "episode_index": cfg.episode_index,
        "success": bool(success),
        "steps": steps,
        "fault": cfg.faults.transport,
        "fault_injected": fault_record is not None,
        "fault_step": None if fault_record is None else fault_record["step"],
        "fault_steps": [r["step"] for r in fault_records],
        "faults_injected": len(fault_records),
        "false_alarm": bool(false_alarm),
        "detection_latency_steps": detection_latency,
        **summary,
        "agent": None if agent is None else agent.stats(),
        "outcome": classify(bool(success), fault_record is not None, summary),
        "wall_time_s": round(time.perf_counter() - started, 2),
    }
    log.write("EPISODE_END", **{k: v for k, v in result.items() if k != "agent"})
    if out:
        (out / "result.json").write_text(json.dumps(result, indent=2))
    return result
