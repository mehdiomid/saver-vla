"""Statistical comparison of two controllers (exact tests, no SciPy needed).

* Fisher's exact test compares success rates of two independent groups.
* McNemar's exact test compares two controllers run on the *same* episodes
  (same seed and initial state, hence the same sampled fault). Pairing removes
  episode difficulty from the comparison, so it needs fewer episodes.
"""

from __future__ import annotations

from math import comb
from typing import Any

from .metrics import wilson


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher's exact p-value for successes k1/n1 versus k2/n2."""
    total_k, total_n = k1 + k2, n1 + n2
    denom = comb(total_n, total_k)

    def prob(a: int) -> float:
        return comb(n1, a) * comb(n2, total_k - a) / denom

    lo, hi = max(0, total_k - n2), min(total_k, n1)
    observed = prob(k1)
    return min(1.0, sum(p for a in range(lo, hi + 1) if (p := prob(a)) <= observed * (1 + 1e-9)))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value from the discordant pairs."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def _key(r: dict[str, Any]) -> tuple:
    return (r.get("seed"), r.get("episode_index"))


def compare(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare task success of condition A and B, counting only episodes where the fault fired."""
    a = [r for r in a if r.get("fault_injected") and r.get("outcome") != "agent_error"]
    b = [r for r in b if r.get("fault_injected") and r.get("outcome") != "agent_error"]
    ka, kb = sum(r["success"] for r in a), sum(r["success"] for r in b)
    result: dict[str, Any] = {
        "a": {"k": ka, "n": len(a), "ci95": wilson(ka, len(a))},
        "b": {"k": kb, "n": len(b), "ci95": wilson(kb, len(b))},
        "fisher_p": fisher_exact(ka, len(a), kb, len(b)) if a and b else None,
    }
    pairs = {_key(r): r for r in a}
    matched = [(pairs[_key(r)], r) for r in b if _key(r) in pairs]
    only_a = sum(ra["success"] and not rb["success"] for ra, rb in matched)
    only_b = sum(rb["success"] and not ra["success"] for ra, rb in matched)
    result["paired"] = {
        "pairs": len(matched), "only_a_succeeded": only_a, "only_b_succeeded": only_b,
        "mcnemar_p": mcnemar_exact(only_a, only_b) if matched else None,
    }
    return result


def _split_agent(condition: str) -> tuple[str, str | None]:
    parts = condition.split(",")
    agent = next((p.split("=", 1)[1] for p in parts if p.startswith("agent=")), None)
    rest = ",".join(p for p in parts if not p.startswith("agent="))
    return rest or "base", agent


def compare_conditions(run_dir: str, a: str | None = None, b: str | None = None) -> str:
    """Markdown table of controller comparisons with Fisher and paired McNemar p-values."""
    from .metrics import group_by_condition, load_results

    groups = group_by_condition(load_results(run_dir))
    pairs: list[tuple[str, str]] = []
    if a and b:
        pairs = [(a, b)]
    else:
        by_rest: dict[str, dict[str, str]] = {}
        for cond in groups:
            rest, agent = _split_agent(cond)
            if agent:
                by_rest.setdefault(rest, {})[agent] = cond
        order = ["llm", "scripted", "queue_reset", "none"]
        for rest, agents in sorted(by_rest.items()):
            names = [x for x in order if x in agents] + sorted(set(agents) - set(order))
            for i, x in enumerate(names):
                for y in names[i + 1 :]:
                    pairs.append((agents[x], agents[y]))

    def fmt(side: dict) -> str:
        lo, hi = side["ci95"]
        if not side["n"]:
            return "-"
        return f"{side['k']}/{side['n']} ({100 * lo:.0f}-{100 * hi:.0f}%)"

    lines = [
        "# Controller comparisons",
        "",
        "Task success over episodes where the fault fired. Paired episodes share seed and initial",
        "state (so the same sampled fault). p-values: two-sided exact tests; with many comparisons,",
        "correct for multiple testing (e.g. multiply by the number of comparisons, Bonferroni).",
        "",
        "| A | B | A success | B success | Fisher p | Pairs | Only A | Only B | McNemar p |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for x, y in pairs:
        if x not in groups or y not in groups:
            lines.append(f"| {x} | {y} | missing condition | | | | | | |")
            continue
        c = compare(groups[x], groups[y])
        p = c["paired"]
        fp = "-" if c["fisher_p"] is None else f"{c['fisher_p']:.3g}"
        mp = "-" if p["mcnemar_p"] is None else f"{p['mcnemar_p']:.3g}"
        lines.append(
            f"| {x} | {y} | {fmt(c['a'])} | {fmt(c['b'])} | {fp} | {p['pairs']} | "
            f"{p['only_a_succeeded']} | {p['only_b_succeeded']} | {mp} |"
        )
    return "\n".join(lines) + "\n"
