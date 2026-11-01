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
from .faults import FaultInjector
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
    if not success:
        return "failure"
    if not fault_injected:
        return "nominal_success"
    outcome = summary.get("recovery_outcome")
    if outcome == "VERIFIED":
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
    supervisor = RecoverySupervisor(cfg, agent, profile.to_action, profile.gripper_index, log=log, faults=faults)

    started = time.perf_counter()
    backend.reset(cfg.seed, cfg.episode_index)
    supervisor.reset()
    faults.reset()
    log.write("EPISODE_START", seed=cfg.seed, episode_index=cfg.episode_index, name=cfg.name)

    success = False
    fault_record = None
    steps = 0
    for _ in range(cfg.env.episode_length):
        state = backend.state()
        record = faults.before_step(state, backend)
        if record is not None:
            fault_record = record
            log.write("FAULT", **record)
            state = backend.state()  # object faults change the world before the next action

        need_policy = (not supervisor.controlling) or cfg.env.query_policy_during_skills
        policy_action = backend.policy_action() if need_policy else None
        action = supervisor.before_step(state, policy_action)
        applied = faults.apply_action(action, in_trial=supervisor.trial_active)
        terminated, success, _ = backend.step(applied)
        steps += 1
        supervisor.after_step(backend.state(), applied)
        if supervisor.consume_policy_reset():
            backend.reset_policy()
        if terminated:
            break

    summary = supervisor.summary()
    detection_latency = None
    if fault_record is not None and summary["failure_step"] is not None:
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
