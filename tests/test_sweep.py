"""Sweep expansion, reporting, and replay of notebook-era logs."""

import json

from saver_vla.agents import ReplayAgent
from saver_vla.metrics import summarize, wilson
from saver_vla.sweep import expand, run_sweep


def test_expand_builds_full_grid():
    raw = {
        "name": "x",
        "env": {"backend": "toy"},
        "sweep": {"seeds": [1, 2], "episode_indices": [0, 3], "grid": {"recovery.agent": ["none", "scripted"]}},
    }
    runs = expand(raw)
    assert len(runs) == 8
    assert {c for c, _ in runs} == {"agent=none", "agent=scripted"}
    assert {(cfg.seed, cfg.episode_index) for _, cfg in runs} == {(1, 0), (1, 3), (2, 0), (2, 3)}


def test_sweep_runs_reports_and_resumes(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        f"name: s\noutput_dir: {tmp_path}\nenv: {{backend: toy}}\nfaults: {{approach_bias_x_m: 0.0}}\n"
        "sweep:\n  seeds: [1, 2]\n  grid:\n    recovery.agent: [none, scripted]\n"
    )
    summaries = run_sweep(path)
    assert summaries["agent=none"]["task_success"]["k"] == 0
    assert summaries["agent=scripted"]["agent_recovery"]["k"] == 2
    assert (tmp_path / "s" / "report.md").exists() and (tmp_path / "s" / "summary.csv").exists()
    first = (tmp_path / "s" / "agent=none" / "s1_e0" / "result.json").stat().st_mtime
    run_sweep(path)  # finished episodes are skipped
    assert (tmp_path / "s" / "agent=none" / "s1_e0" / "result.json").stat().st_mtime == first


def test_wilson_interval_edges():
    lo, hi = wilson(0, 5)
    assert lo == 0.0 and 0.4 < hi < 0.5
    lo, hi = wilson(5, 5)
    assert hi == 1.0 and 0.5 < lo < 0.6
    assert summarize([])["task_success"]["n"] == 0


def test_replay_reads_notebook_agent_log(tmp_path):
    log = tmp_path / "agent_events.jsonl"
    rows = [
        {"kind": "FAILURE", "event": {}},
        {"kind": "DECISION", "decision": {"skill": "MOVE_TO_PREGRASP", "variant": "CENTER"},
         "initial_requested_decision": {"skill": "MOVE_TO_PREGRASP", "variant": "CENTER"}},
        {"kind": "SKILL_OUTCOME", "status": {}},
        {"kind": "DECISION", "decision": {"skill": "ABORT_RECOVERY", "variant": "CENTER"},
         "initial_requested_decision": {"skill": "APPROACH_OBJECT", "variant": "CENTER"}},
    ]
    log.write_text("\n".join(json.dumps(r) for r in rows))
    agent = ReplayAgent.from_log(log)
    skills = [agent.decide({"step": 0})["skill"] for _ in range(4)]
    assert skills == ["MOVE_TO_PREGRASP", "APPROACH_OBJECT", "ABORT_RECOVERY", "ABORT_RECOVERY"]
