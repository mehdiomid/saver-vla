"""Typed configuration for SAVER-VLA experiments.

One YAML file describes one experiment. It replaces the ~60 SAVER_* environment
variables used in the notebook. Every field has the value used in the Phase 6.5.1
run as its default, so an empty YAML reproduces that configuration.

Override any field from the command line, e.g.
    saver run configs/phase651.yaml --set recovery.model=gpt-5.2 --set seed=1001
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


# --------------------------------------------------------------------------- environment


@dataclass
class EnvConfig:
    """Which simulator/robot backend to run and how."""

    backend: str = "libero"  # "libero" | "toy" (fast, dependency-free test world)
    suite: str = "libero_object"
    task_id: int = 0
    episode_length: int = 850
    object_name: str = "alphabet_soup_1"  # MuJoCo body of the object; "auto" = infer from the task sentence
    target_name: str = "basket_1"  # MuJoCo body of the placement target
    policy_path: str = "lerobot/smolvla_libero"
    device: str = "cuda"
    control_mode: str = "relative"
    rename_map: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.image": "observation.images.camera1",
            "observation.images.image2": "observation.images.camera2",
        }
    )
    # Call the policy on every step, even while a recovery skill controls the robot.
    # The notebook did this; it keeps the policy's random-number stream comparable.
    query_policy_during_skills: bool = True
    # Save an MP4 of each episode (LIBERO backend; needs the `av` package).
    record_video: bool = False


# --------------------------------------------------------------------------- faults


# Any numeric fault parameter may be a single value or a [low, high] range. Ranges are
# sampled independently for every injection from a generator seeded by (seed, episode_index),
# so each episode is different but every episode can be reproduced exactly.
Range = float | list[float]


@dataclass
class TriggerConfig:
    """When a transport fault fires: after a sustained lift, before the target."""

    min_lift_m: float = 0.03
    min_lift_checks: int = 2
    min_time_after_lift_s: Range = 0.30
    target_exclusion_radius_m: float = 0.12


@dataclass
class FaultConfig:
    """Injected faults. Hidden fault settings never reach the agent in blind mode."""

    # Transport fault during carrying: none | forced_separation | gripper_release | velocity_impulse
    transport: str = "gripper_release"
    displacement_m: Range = 0.12  # forced_separation (simulation only)
    release_steps: Range = 15  # gripper_release: force the gripper open for N steps (1-40)
    impulse_speed_mps: Range = 1.5  # velocity_impulse (simulation only)
    # Displacement / impulse direction: away_from_target | random | lateral
    direction: str = "away_from_target"
    direction_jitter_deg: float = 0.0  # random rotation of the direction, +/- degrees
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    # Repeated faults: inject again after each re-lift, up to this many times per episode.
    max_injections: int = 1
    min_steps_between_injections: int = 40
    # Force the gripper open during the first N steps of the first policy grasp trial (Phase 6.3).
    trial_impairment_steps: Range = 0
    # Calibration bias added to the CENTER approach target, metres (Phase 6.4b).
    approach_bias_x_m: Range = 0.055
    approach_bias_y_m: Range = 0.0


@dataclass
class NoiseConfig:
    """Imperfect sensing and actuation (all zero = privileged simulator state).

    Perception noise is applied to the state the supervisor, skills and agent see;
    faults and task success always use the true state. Action noise is added to the
    executed action of the policy and the skills alike.
    """

    object_position_std_m: float = 0.0  # per-step Gaussian noise on the object estimate
    object_position_bias_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    eef_position_std_m: float = 0.0
    action_std: float = 0.0  # Gaussian noise on translation commands (normalized units)


# --------------------------------------------------------------------------- verifier


@dataclass
class VerifierConfig:
    """Online transport verifier (Phase 2.3D) thresholds; frozen on the development set."""

    approach_radius_m: float = 0.10
    approach_persistence: int = 2
    lift_height_m: float = 0.03
    lift_persistence: int = 2
    lift_timeout_s: float = 3.0
    drift_threshold_m: float = 0.08
    separation_persistence: int = 3
    target_radius_m: float = 0.10
    drop_height_m: float = 0.04


@dataclass
class GraspTrackerConfig:
    """Persistent grasp tracker (Phase 6.2) used for grasp-trial verification."""

    window: int = 6
    min_lift_m: float = 0.025
    max_coupling_change_m: float = 0.015
    lost_distance_m: float = 0.09
    loss_persistence: int = 3
    close_command_threshold: float = 0.5


# --------------------------------------------------------------------------- skills


@dataclass
class PregraspParams:
    max_steps: int = 120
    max_phase_steps: int = 70
    max_command: float = 0.20
    position_scale_m: float = 0.05  # command = error / scale, then clipped
    tolerance_m: float = 0.015
    clearance_height_m: float = 0.25
    pregrasp_offset_m: float = 0.12
    min_eef_z_m: float = 0.10
    settle_steps: int = 6


@dataclass
class ApproachParams:
    # Demonstrated end-effector offset from a successful grasp, plus 4 cm clearance.
    grasp_offset_m: list[float] = field(default_factory=lambda: [0.0142, -0.0112, 0.0441])
    min_target_z_m: float = 0.105
    max_target_z_m: float = 0.50
    max_start_distance_m: float = 0.20
    tolerance_m: float = 0.015
    gain: float = 2.0
    max_command: float = 0.12
    max_steps: int = 110
    adjust_offset_x_m: float = 0.020  # ADJUST_GRASP_POSE OFFSET_X_POS / OFFSET_X_NEG


@dataclass
class ReassessParams:
    hold_steps: int = 4


@dataclass
class Reposition2DParams:
    """Height-aware 2D repositioning (Phase 6.5.1)."""

    clearance_z_m: float = 0.25
    pregrasp_offset_z_m: float = 0.13
    xy_tolerance_m: float = 0.012
    z_tolerance_m: float = 0.012
    max_steps: int = 220
    min_target_z_m: float = 0.105
    max_target_z_m: float = 0.35
    realign_xy_m: float = 0.04


@dataclass
class SkillConfig:
    pregrasp: PregraspParams = field(default_factory=PregraspParams)
    approach: ApproachParams = field(default_factory=ApproachParams)
    reassess: ReassessParams = field(default_factory=ReassessParams)
    reposition_2d: Reposition2DParams = field(default_factory=Reposition2DParams)


# --------------------------------------------------------------------------- recovery


@dataclass
class RecoveryConfig:
    """The agentic recovery layer."""

    enabled: bool = True
    agent: str = "llm"  # none | queue_reset | scripted | llm | replay
    model: str = "gpt-5.5"
    replay_log: str | None = None  # agent=replay: a previous events.jsonl
    blind: bool = True  # hide the injected fault's identity from the agent (Phase 6.4a)
    capability_aware: bool = True  # show available/blocked skills, replan on rejection (Phase 6.4c)
    skills: list[str] = field(
        default_factory=lambda: [
            "MOVE_TO_PREGRASP",
            "REPOSITION_TO_OBJECT_2D",
            "APPROACH_OBJECT",
            "REASSESS_STATE",
            "ADJUST_GRASP_POSE",
            "RESUME_POLICY",
            "ABORT_RECOVERY",
        ]
    )
    trigger_events: list[str] = field(default_factory=lambda: ["SUSPECTED_TRANSPORT_SEPARATION"])
    # How many separate failures one episode may recover from (budgets reset per recovery).
    max_recoveries: int = 1
    max_decisions: int = 8
    max_replans: int = 2
    attempt_limit: int = 2  # per skill:variant
    resume_max_distance_m: float = 0.105
    history_length: int = 7
    # Bounded SmolVLA grasp trials after RESUME_POLICY (Phase 6.3).
    policy_trials: bool = True
    trial_steps: int = 95
    max_trials: int = 2
    llm_timeout_s: float = 90.0
    llm_max_retries: int = 4  # retries with exponential backoff (rate limits, timeouts)


# --------------------------------------------------------------------------- top level


@dataclass
class ExperimentConfig:
    name: str = "phase651"
    output_dir: str = "runs"
    seed: int = 1000  # LeRobot eval default; also seeds torch for SmolVLA sampling
    episode_index: int = 0  # LIBERO initial-state index
    robot: str = "panda_libero"
    env: EnvConfig = field(default_factory=EnvConfig)
    faults: FaultConfig = field(default_factory=FaultConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    grasp_tracker: GraspTrackerConfig = field(default_factory=GraspTrackerConfig)
    skills: SkillConfig = field(default_factory=SkillConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------- loading


def _build(cls: type, data: dict[str, Any] | None, path: str = "") -> Any:
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    data = dict(data or {})
    hints = get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"Unknown config key(s) at '{path or cls.__name__}': {sorted(unknown)}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hint = hints[f.name]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            value = _build(hint, value, f"{path}.{f.name}".strip("."))
        kwargs[f.name] = value
    return cls(**kwargs)


def _parse_scalar(text: str) -> Any:
    return yaml.safe_load(text)


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply dotted `key=value` overrides (values parsed as YAML)."""
    data = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_scalar(raw)
    return data


def config_from_dict(data: dict[str, Any] | None, overrides: list[str] | None = None) -> ExperimentConfig:
    return _build(ExperimentConfig, apply_overrides(data or {}, overrides))


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw.pop("sweep", None)  # sweep definitions are read by saver_vla.sweep
    return config_from_dict(raw, overrides)
