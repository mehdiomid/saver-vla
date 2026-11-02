"""Reliability metrics over many episodes.

Rates come with 95% Wilson confidence intervals, which stay sensible for the
small sample sizes typical of robot experiments (including 0/n and n/n).
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [lo, hi]}


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    faulted = [r for r in results if r.get("fault_injected")]
    detected = [r for r in faulted if r.get("failure_step") is not None and not r.get("false_alarm")]
    started = [r for r in results if r.get("recovery_started")]
    agent_stats = [r.get("agent") or {} for r in results]
    decisions = sum(r.get("decisions", 0) for r in results)
    outcomes: dict[str, int] = defaultdict(int)
    for r in results:
        outcomes[r.get("outcome", "unknown")] += 1
    return {
        "episodes": n,
        "task_success": _rate(sum(r["success"] for r in results), n),
        # Episodes where the fault never fired (e.g. the object reached the target first)
        # are fault-free runs; they are excluded here so they cannot inflate the result.
        "fault_fired": _rate(len(faulted), n),
        "success_given_fault": _rate(sum(r["success"] for r in faulted), len(faulted)),
        "fault_detected": _rate(len(detected), len(faulted)),
        # Recovery success: episodes where the agent's supervised grasp trial was verified
        # and the task then succeeded, out of episodes where recovery started.
        "agent_recovery": _rate(sum(r.get("outcome") == "agent_recovery" for r in started), len(started)),
        "outcomes": dict(outcomes),
        "false_alarms": sum(bool(r.get("false_alarm")) for r in results),
        "agent_errors": sum(int(r.get("agent_errors") or 0) for r in results),
        "faults_injected_mean": _mean([r.get("faults_injected", int(bool(r.get("fault_injected")))) for r in results]),
        "recoveries_mean": _mean([r.get("recoveries", 0) for r in started]),
        "detection_latency_steps_mean": _mean([r.get("detection_latency_steps") for r in detected]),
        "decisions_mean": _mean([r.get("decisions", 0) for r in started]),
        "rejected_request_rate": (sum(r.get("rejections", 0) for r in results) / decisions) if decisions else None,
        "aborted": sum(r.get("recovery_outcome") == "ABORTED" for r in results),
        "agent_calls_mean": _mean([a.get("calls") for a in agent_stats if a]),
        "agent_latency_s_mean": _mean([a.get("latency_s") for a in agent_stats if a]),
        "tokens_mean": _mean([(a.get("input_tokens", 0) + a.get("output_tokens", 0)) for a in agent_stats if a]),
        "episode_steps_mean": _mean([r.get("steps") for r in results]),
        "wall_time_s_mean": _mean([r.get("wall_time_s") for r in results]),
    }


def load_results(run_dir: str | Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(Path(run_dir).rglob("result.json"))]


def group_by_condition(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[r.get("condition") or r.get("name", "run")].append(r)
    return dict(groups)


def _fmt_rate(rate: dict[str, Any]) -> str:
    if not rate["n"]:
        return "-"
    lo, hi = rate["ci95"]
    return f"{rate['k']}/{rate['n']} ({100 * rate['rate']:.0f}%, CI {100 * lo:.0f}-{100 * hi:.0f}%)"


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_report(run_dir: str | Path) -> dict[str, Any]:
    """Write summary.csv, summary.json and report.md for every result.json under run_dir."""
    run_dir = Path(run_dir)
    results = load_results(run_dir)
    groups = group_by_condition(results)
    summaries = {name: summarize(rs) for name, rs in groups.items()}

    columns = [
        "condition", "seed", "episode_index", "success", "outcome", "fault", "fault_step",
        "faults_injected", "false_alarm", "failure_step", "detection_latency_steps",
        "recovery_outcome", "recoveries", "verified_recoveries", "decisions", "agent_calls", "agent_errors",
        "rejections", "policy_trials", "verified_trial", "steps", "wall_time_s",
    ]
    with (run_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({**r, "condition": r.get("condition") or r.get("name")})
    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2))

    lines = [
        f"# Reliability report: {run_dir.name}",
        "",
        f"{len(results)} episodes. Rates show k/n with 95% Wilson confidence intervals.",
        "Success is counted only over episodes where the fault actually fired.",
        "",
        "| Condition | Fault fired | Success given fault | Fault detected | Agent recovery | False alarms | Agent errors | Mean decisions | Mean latency (steps) | Outcomes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in summaries.items():
        outcomes = ", ".join(f"{k}: {v}" for k, v in sorted(s["outcomes"].items()))
        lines.append(
            f"| {name} | {_fmt_rate(s['fault_fired'])} | {_fmt_rate(s['success_given_fault'])} | {_fmt_rate(s['fault_detected'])} | "
            f"{_fmt_rate(s['agent_recovery'])} | {s['false_alarms']} | {s['agent_errors']} | {_fmt(s['decisions_mean'])} | "
            f"{_fmt(s['detection_latency_steps_mean'])} | {outcomes} |"
        )
    lines += [
        "",
        "Outcome labels: `agent_recovery` = a supervised grasp trial was verified and the task succeeded;",
        "`policy_after_abort` = the task succeeded only after the agent ended its intervention;",
        "`resumed_success` = success after control was returned without a verified trial;",
        "`policy_only_success` = success with no recovery; `nominal_success` = no fault was injected;",
        "`nominal_failure` = the policy failed before any fault fired (not a recovery result);",
        "`agent_rescue` = no fault was injected, but the agent recovered a failure of the policy itself;",
        "`agent_error` = the agent could not be reached (API error); exclude and re-run.",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n")
    (run_dir / "table.tex").write_text(latex_table(summaries))
    return summaries


def latex_table(summaries: dict[str, Any]) -> str:
    """A booktabs table for papers: success given fault and agent recovery, with 95% CIs."""

    def cell(rate: dict[str, Any]) -> str:
        if not rate["n"]:
            return "--"
        lo, hi = rate["ci95"]
        return f"{rate['k']}/{rate['n']} ({100 * lo:.0f}--{100 * hi:.0f}\\%)"

    rows = [
        "\\begin{tabular}{@{}lccc@{}}",
        "\\toprule",
        "Condition & Success given fault & Agent recovery & Decisions \\\\",
        "\\midrule",
    ]
    for name, s in summaries.items():
        label = name.replace("_", "\\_").replace("=", ": ").replace(",", ", ")
        rows.append(
            f"{label} & {cell(s['success_given_fault'])} & {cell(s['agent_recovery'])} & {_fmt(s['decisions_mean'])} \\\\"
        )
    rows += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(rows)
