"""Randomized and repeated faults, noisy sensing, multiple recoveries, scenario sweeps."""

import numpy as np
import pytest
import yaml

from saver_vla.agents import CallableAgent
from saver_vla.config import FaultConfig, NoiseConfig, config_from_dict
from saver_vla.events import read_events
from saver_vla.faults import FaultInjector, episode_rng, sample
from saver_vla.perception import Perception
from saver_vla.runner import make_backend, run_episode
from saver_vla.sweep import expand
from saver_vla.types import RobotState
from tests.test_supervisor import geometry_rule


def toy(**sections):
    sections.setdefault("env", {})["backend"] = "toy"
    return config_from_dict(sections)


def test_ranges_are_sampled_reproducibly():
    cfg = FaultConfig(displacement_m=[0.06, 0.22], release_steps=[25, 40], approach_bias_y_m=[-0.05, 0.05])
    a, b, c = (FaultInjector(cfg) for _ in range(3))
    a.reset(episode_rng(1, 0)); b.reset(episode_rng(1, 0)); c.reset(episode_rng(2, 0))
    assert a.episode_plan() == b.episode_plan()  # same seed and initial state -> same episode
    assert a.episode_plan() != c.episode_plan()  # different seed -> different faults
    assert 0.06 <= a.next["displacement_m"] <= 0.22 and 25 <= a.next["release_steps"] <= 40
    with pytest.raises(ValueError):
        sample([0.3, 0.1], np.random.default_rng(0))


@pytest.mark.parametrize("direction", ["away_from_target", "random", "lateral"])
def test_separation_directions_and_magnitudes_vary(direction, tmp_path):
    angles, moves = [], []
    for seed in range(6):
        cfg = toy(faults={"transport": "forced_separation", "displacement_m": [0.06, 0.22],
                          "direction": direction, "approach_bias_x_m": 0.0},
                  recovery={"agent": "none"}, seed=seed)
        run_episode(cfg, make_backend(cfg), out_dir=tmp_path / str(seed))
        fault = [e for e in read_events(tmp_path / str(seed) / "events.jsonl") if e["kind"] == "FAULT"][0]
        d = np.array(fault["direction_xy"])
        assert np.linalg.norm(d) == pytest.approx(1.0)
        angles.append(np.arctan2(d[1], d[0]))
        moves.append(np.linalg.norm(np.subtract(fault["object_after"], fault["object_before"])[:2]))
    assert all(0.06 - 1e-9 <= m <= 0.22 + 1e-9 for m in moves) and np.ptp(moves) > 0.02
    if direction == "away_from_target":
        assert np.ptp(angles) < 1e-6  # fixed direction
    else:
        assert np.ptp(angles) > 0.5  # direction changes between episodes


def test_repeated_faults_need_two_recoveries(tmp_path):
    cfg = toy(faults={"transport": "gripper_release", "approach_bias_x_m": 0.0, "max_injections": 2},
              recovery={"agent": "scripted", "max_recoveries": 2})
    r = run_episode(cfg, make_backend(cfg), out_dir=tmp_path)
    assert r["faults_injected"] == 2
    assert r["recoveries"] == 2 and r["verified_recoveries"] == 2
    assert r["success"] and r["outcome"] == "agent_recovery"
    failures = [e for e in read_events(tmp_path / "events.jsonl") if e["kind"] == "FAILURE"]
    assert [f["recovery"] for f in failures] == [1, 2]


def test_second_fault_fails_when_only_one_recovery_allowed():
    cfg = toy(faults={"transport": "gripper_release", "approach_bias_x_m": 0.0, "max_injections": 2},
              recovery={"agent": "scripted", "max_recoveries": 1})
    r = run_episode(cfg, make_backend(cfg))
    assert r["faults_injected"] == 2 and r["recoveries"] == 1
    assert r["outcome"] != "agent_recovery"


def test_perception_noise_changes_what_the_agent_sees_not_the_world():
    p = Perception(NoiseConfig(object_position_std_m=0.01, object_position_bias_m=[0.02, 0, 0]),
                   np.random.default_rng(0))
    true = RobotState(step=0, time=0.0, eef_xyz=[0, 0, 0.3], object_xyz=[0.1, 0.1, 0.04])
    seen = [p.observe(true).object_xyz for _ in range(500)]
    assert np.mean(seen, axis=0)[0] == pytest.approx(0.12, abs=0.002)
    assert np.std(seen, axis=0)[1] == pytest.approx(0.01, abs=0.002)
    assert np.allclose(true.object_xyz, [0.1, 0.1, 0.04])


def test_large_sensing_noise_causes_false_alarms_or_failures():
    cfg = toy(faults={"transport": "none"}, noise={"object_position_std_m": 0.08},
              recovery={"agent": "scripted"})
    r = run_episode(cfg, make_backend(cfg))
    assert r["false_alarm"] or not r["success"] or r["outcome"] == "nominal_success"


def test_y_bias_defeats_x_only_correction():
    cfg = toy(faults={"transport": "gripper_release", "approach_bias_x_m": 0.0, "approach_bias_y_m": 0.05})
    r = run_episode(cfg, make_backend(cfg), agent=CallableAgent(geometry_rule))
    assert r["outcome"] != "agent_recovery"  # the skill library cannot shift the grasp in Y


def test_hard_sweep_files_expand():
    for path, n in [("configs/sweep_hard.yaml", 108), ("configs/sweep_generalization.yaml", 81)]:
        runs = expand(yaml.safe_load(open(path)))
        assert len(runs) == n
        assert len({cfg.seed * 1000 + cfg.episode_index for _, cfg in runs}) > 1
