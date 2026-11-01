"""Run many episodes from one YAML: every grid combination x seeds x initial states.

    sweep:
      seeds: [1000, 1001, 1002]
      episode_indices: [0, 1]
      grid:
        recovery.agent: [none, scripted, llm]
        faults.transport: [forced_separation, gripper_release]

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
    raw = dict(raw)
    spec = raw.pop("sweep", None) or {}
    base = apply_overrides(raw, overrides)
    seeds = spec.get("seeds", [base.get("seed", 1000)])
    episodes = spec.get("episode_indices", [base.get("episode_index", 0)])
    grid: dict[str, list[Any]] = spec.get("grid", {}) or {}
    keys = list(grid)
    runs = []
    for combo in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
        condition = ",".join(_label(k, v) for k, v in zip(keys, combo)) or "base"
        combo_overrides = [f"{k}={json.dumps(v)}" for k, v in zip(keys, combo)]
        for seed in seeds:
            for ep in episodes:
                cfg = config_from_dict(base, combo_overrides + [f"seed={seed}", f"episode_index={ep}"])
                runs.append((condition, cfg))
    return runs


def run_sweep(path: str | Path, overrides: list[str] | None = None, dry_run: bool = False) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    runs = expand(raw, overrides)
    name = raw.get("name", Path(path).stem)
    root = Path(apply_overrides(raw, overrides).get("output_dir", "runs")) / name
    print(f"Sweep '{name}': {len(runs)} episodes -> {root}")
    if dry_run:
        for condition, cfg in runs:
            print(f"  {condition} seed={cfg.seed} episode={cfg.episode_index}")
        return {}

    backend, backend_key = None, None
    for i, (condition, cfg) in enumerate(runs, 1):
        out = root / condition / f"s{cfg.seed}_e{cfg.episode_index}"
        if (out / "result.json").exists():
            print(f"[{i}/{len(runs)}] {condition} seed={cfg.seed} ep={cfg.episode_index}: done, skipping")
            continue
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
    if backend is not None:
        backend.close()
    summaries = write_report(root)
    print(f"Report: {root / 'report.md'}")
    return summaries
