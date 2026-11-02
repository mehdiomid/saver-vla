"""Run many episodes from one YAML: every grid combination x seeds x initial states.

    sweep:
      seeds: [1000, 1001, 1002]
      episode_indices: [0, 1]
      grid:
        recovery.agent: [none, scripted, llm]
        faults.transport: [forced_separation, gripper_release]
      scenarios:              # optional: named bundles of settings, crossed with the grid
        long_release: {faults.transport: gripper_release, faults.release_steps: [25, 40]}

Results go to <output_dir>/<name>/<condition>/s<seed>_e<episode>/. Finished
episodes are skipped on re-run, so an interrupted Colab session can resume.
The simulator and policy are loaded once and reused while the env config is unchanged.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig, apply_overrides, config_from_dict
from .metrics import write_report
from .runner import make_backend, run_episode


def _label(key: str, value: Any) -> str:
    return f"{key.split('.')[-1]}={value}"


def expand(raw: dict[str, Any], overrides: list[str] | None = None) -> list[tuple[str, ExperimentConfig]]:
    # Overrides may also target the sweep itself, e.g. `sweep.episode_indices=[2,5,9]`.
    raw = apply_overrides(dict(raw), overrides)
    spec = raw.pop("sweep", None) or {}
    base = raw
    seeds = spec.get("seeds", [base.get("seed", 1000)])
    episodes = spec.get("episode_indices", [base.get("episode_index", 0)])
    grid: dict[str, list[Any]] = spec.get("grid", {}) or {}
    scenarios: dict[str, dict[str, Any]] = spec.get("scenarios") or {"": {}}
    keys = list(grid)
    runs = []
    for scenario, settings in scenarios.items():
        scenario_overrides = [f"{k}={json.dumps(v)}" for k, v in (settings or {}).items()]
        for combo in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
            labels = ([f"scenario={scenario}"] if scenario else []) + [_label(k, v) for k, v in zip(keys, combo)]
            condition = ",".join(labels) or "base"
            combo_overrides = scenario_overrides + [f"{k}={json.dumps(v)}" for k, v in zip(keys, combo)]
            for seed in seeds:
                for ep in episodes:
                    cfg = config_from_dict(base, combo_overrides + [f"seed={seed}", f"episode_index={ep}"])
                    runs.append((condition, cfg))
    return runs


class AgentUnavailable(RuntimeError):
    """The LLM agent could not be reached; results would measure the API, not the method."""


def preflight_llm(cfg: ExperimentConfig) -> None:
    """One cheap test call before hours of simulation, so key/quota/model errors surface now."""
    from .agents import make_agent

    agent = make_agent(cfg.recovery)
    probe = {
        "step": 0,
        "failure": "PREFLIGHT_CHECK",
        "available_actions": ["ABORT_RECOVERY:CENTER"],
        "unavailable_actions": {},
    }
    try:
        decision = agent.decide(probe)
    except Exception as exc:
        raise AgentUnavailable(
            f"LLM preflight failed ({type(exc).__name__}: {exc}). Check OPENAI_API_KEY, quota, "
            f"and that model '{cfg.recovery.model}' is available to your account."
        ) from exc
    print(f"LLM preflight OK ({cfg.recovery.model}): {decision['skill']}")


def run_sweep(
    path: str | Path,
    overrides: list[str] | None = None,
    dry_run: bool = False,
    stop_on_agent_error: bool = True,
) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    runs = expand(raw, overrides)
    name = raw.get("name", Path(path).stem)
    root = Path(apply_overrides(raw, overrides).get("output_dir", "runs")) / name
    print(f"Sweep '{name}': {len(runs)} episodes -> {root}")
    if dry_run:
        for condition, cfg in runs:
            print(f"  {condition} seed={cfg.seed} episode={cfg.episode_index}")
        return {}

    llm_runs = [cfg for _, cfg in runs if cfg.recovery.enabled and cfg.recovery.agent == "llm"]
    if llm_runs:
        preflight_llm(llm_runs[0])

    backend, backend_key = None, None
    for i, (condition, cfg) in enumerate(runs, 1):
        out = root / condition / f"s{cfg.seed}_e{cfg.episode_index}"
        if (out / "result.json").exists():
            previous = json.loads((out / "result.json").read_text())
            if previous.get("outcome") != "agent_error" and not previous.get("agent_errors"):
                print(f"[{i}/{len(runs)}] {condition} seed={cfg.seed} ep={cfg.episode_index}: done, skipping")
                continue
            print(f"[{i}/{len(runs)}] {condition} seed={cfg.seed} ep={cfg.episode_index}: had agent errors, re-running")
        key = json.dumps(cfg.to_dict()["env"], sort_keys=True)
        if key != backend_key:
            if backend is not None:
                backend.close()
            backend, backend_key = make_backend(cfg), key
        result = run_episode(cfg, backend, out_dir=out)
        result["condition"] = condition
        (out / "result.json").write_text(json.dumps(result, indent=2))
        print(
            f"[{i}/{len(runs)}] {condition} seed={cfg.seed} ep={cfg.episode_index}: "
            f"success={result['success']} outcome={result['outcome']}"
        )
        if result.get("agent_errors") and stop_on_agent_error:
            if backend is not None:
                backend.close()
            raise AgentUnavailable(
                "Agent error during the sweep; stopping so no more episodes are wasted. "
                f"First error: {(result.get('agent_error_messages') or ['?'])[0]}. "
                "Fix the cause and re-run the same command: finished episodes are kept, "
                "and episodes with agent errors are re-run automatically."
            )
    if backend is not None:
        backend.close()
    summaries = write_report(root)
    print(f"Report: {root / 'report.md'}")
    return summaries
