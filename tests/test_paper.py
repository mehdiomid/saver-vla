"""Paper experiment configs, sweep-level overrides, exact tests, and comparisons."""

import pytest
import yaml

from saver_vla.runner import classify
from saver_vla.stats import compare_conditions, fisher_exact, mcnemar_exact
from saver_vla.sweep import expand, run_sweep


@pytest.mark.parametrize("path,n", [
    ("configs/paper_E1_main.yaml", 128),
    ("configs/paper_E2_hard.yaml", 80),
    ("configs/paper_E3_ablations.yaml", 48),
    ("configs/paper_E5_policy_rescue.yaml", 24),
    ("configs/nominal_rescreen.yaml", 8),
    ("configs/sweep_hard.yaml", 72),
])
def test_paper_configs_expand(path, n):
    runs = expand(yaml.safe_load(open(path)))
    assert len(runs) == n
    assert all(cfg.episode_index != 3 for _, cfg in runs) or "rescue" in path or "rescreen" in path


def test_ablation_variants_apply():
    runs = {c: cfg for c, cfg in expand(yaml.safe_load(open("configs/paper_E3_ablations.yaml")))}
    by = {c.split("=")[1]: cfg for c, cfg in runs.items()}
    assert by["no_trial_verification"].recovery.policy_trials is False
    assert by["no_capability_info"].recovery.capability_aware is False
    assert "ADJUST_GRASP_POSE" not in by["no_geometry_skills"].recovery.skills
    assert by["fault_disclosed"].recovery.blind is False
    assert by["scripted_reference"].recovery.agent == "scripted"
    assert by["full"].recovery.agent == "llm"


def test_sweep_settings_can_be_overridden():
    raw = yaml.safe_load(open("configs/paper_E2_hard.yaml"))
    runs = expand(raw, ["sweep.episode_indices=[2, 5]", "sweep.seeds=[7]"])
    assert {(cfg.seed, cfg.episode_index) for _, cfg in runs} == {(7, 2), (7, 5)}


def test_exact_tests_match_known_values():
    # 0/10 vs 10/10: p = 2 / C(20, 10)
    assert fisher_exact(0, 10, 10, 10) == pytest.approx(2 / 184756)
    assert fisher_exact(5, 10, 5, 10) == pytest.approx(1.0)
    assert mcnemar_exact(0, 6) == pytest.approx(2 / 64)
    assert mcnemar_exact(3, 3) == pytest.approx(1.0)


def test_agent_rescue_label():
    assert classify(True, False, {"recoveries": 1, "recovery_outcome": "VERIFIED"}) == "agent_rescue"
    assert classify(True, False, {}) == "nominal_success"


def test_compare_pairs_agents_within_scenarios(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        f"name: s\noutput_dir: {tmp_path}\nenv: {{backend: toy}}\n"
        "faults: {transport: gripper_release, approach_bias_x_m: 0.055}\n"
        "sweep:\n  seeds: [1, 2, 3]\n  grid:\n    recovery.agent: [none, scripted]\n"
    )
    run_sweep(path)
    text = compare_conditions(str(tmp_path / "s"))
    assert "agent=scripted" in text and "agent=none" in text and "McNemar" in text
    assert (tmp_path / "s" / "table.tex").exists()
