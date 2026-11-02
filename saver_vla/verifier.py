"""Execution monitoring: the online transport verifier and the persistent grasp tracker.

Both are ports of the notebook modules (Phase 2.3D `SAVERVerifier`, Phase 6.2
`GraspTracker`). They only read positions, so they work unchanged on a real robot
once object positions come from perception instead of the simulator.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from .config import GraspTrackerConfig, VerifierConfig
from .types import RobotState

SEPARATION = "SUSPECTED_TRANSPORT_SEPARATION"
LIFT_TIMEOUT = "SUSPECTED_GRASP_LIFT_TIMEOUT"


class TransportVerifier:
    """Detects approach, lift, and unexpected object-gripper separation during transport.

    Emits events:
      APPROACH_CONFIRMED, LIFT_CONFIRMED, SUSPECTED_GRASP_LIFT_TIMEOUT,
      TARGET_VICINITY_REACHED, SUSPECTED_TRANSPORT_SEPARATION.
    """

    def __init__(self, cfg: VerifierConfig | None = None):
        self.cfg = cfg or VerifierConfig()
        self.reset()

    def reset(self) -> None:
        self.initial_z: float | None = None
        self.approach_count = 0
        self.approach_time: float | None = None
        self.lift_count = 0
        self.lift_confirmed = False
        self.lift_time: float | None = None
        self.reference_offset: np.ndarray | None = None
        self.separation_count = 0
        self.max_height: float | None = None
        self.transport_finished = False
        self.reported: set[str] = set()

    def rearm(self, state: RobotState) -> None:
        """Resume transport monitoring after a recovered grasp, so a later failure is detected."""
        self.lift_confirmed = True
        self.reference_offset = state.object_xyz - state.eef_xyz
        self.max_height = float(state.object_xyz[2])
        self.separation_count = 0
        self.transport_finished = False
        self.reported.discard(SEPARATION)

    def update(self, state: RobotState) -> list[dict[str, Any]]:
        c = self.cfg
        obj, eef, t, step = state.object_xyz, state.eef_xyz, state.time, state.step
        if self.initial_z is None:
            self.initial_z = float(obj[2])

        events: list[dict[str, Any]] = []
        eef_distance = state.eef_object_distance
        near_object = eef_distance < c.approach_radius_m
        lifted = (obj[2] - self.initial_z) > c.lift_height_m

        self.approach_count = self.approach_count + 1 if near_object else 0
        if self.approach_time is None and self.approach_count >= c.approach_persistence:
            self.approach_time = t
            events.append({"event_type": "APPROACH_CONFIRMED", "step": step, "time": t})

        if self.approach_time is not None and not self.lift_confirmed:
            self.lift_count = self.lift_count + 1 if lifted else 0
            if self.lift_count >= c.lift_persistence:
                self.lift_confirmed = True
                self.lift_time = t
                self.reference_offset = obj - eef
                self.max_height = float(obj[2])
                events.append(
                    {
                        "event_type": "LIFT_CONFIRMED",
                        "step": step,
                        "time": t,
                        "approach_to_lift_s": round(t - self.approach_time, 3),
                    }
                )
            elif t - self.approach_time >= c.lift_timeout_s and LIFT_TIMEOUT not in self.reported:
                self.reported.add(LIFT_TIMEOUT)
                events.append(
                    {
                        "event_type": LIFT_TIMEOUT,
                        "step": step,
                        "time": t,
                        "severity": "HIGH",
                        "evidence": {
                            "seconds_since_approach": round(t - self.approach_time, 3),
                            "eef_object_distance_m": eef_distance,
                        },
                    }
                )

        # Attachment is only monitored after a confirmed lift.
        if not self.lift_confirmed or self.transport_finished:
            return events

        self.max_height = max(self.max_height, float(obj[2]))

        # Intentional release may happen near the target.
        target_xy = state.target_xy_distance
        if target_xy is not None and target_xy < c.target_radius_m:
            self.transport_finished = True
            events.append({"event_type": "TARGET_VICINITY_REACHED", "step": step, "time": t})
            return events

        drift = float(np.linalg.norm((obj - eef) - self.reference_offset))
        height_loss = float(self.max_height - obj[2])
        self.separation_count = self.separation_count + 1 if drift > c.drift_threshold_m else 0
        if self.separation_count >= c.separation_persistence and SEPARATION not in self.reported:
            self.reported.add(SEPARATION)
            events.append(
                {
                    "event_type": SEPARATION,
                    "step": step,
                    "time": t,
                    "severity": "HIGH",
                    "evidence": {
                        "relative_drift_m": drift,
                        "eef_object_distance_m": eef_distance,
                        "downward_change_m": height_loss,
                        "possible_drop": height_loss > c.drop_height_m,
                    },
                }
            )
        return events


GRASP_NOT_ESTABLISHED = "GRASP_NOT_ESTABLISHED"
GRASP_CONFIRMED = "GRASP_CONFIRMED"
TRANSPORT_STABLE = "TRANSPORT_STABLE"
GRASP_LOST = "GRASP_LOST"
GRASP_HELD = (GRASP_CONFIRMED, TRANSPORT_STABLE)


class GraspTracker:
    """Persistent grasp state from object-gripper coupling, updated every step.

    A grasp is confirmed when, over `window` samples, the gripper is commanded
    closed, the object-gripper offset stays stable, and both rise by `min_lift_m`.
    It is lost after `loss_persistence` steps of drift or excessive distance.
    """

    def __init__(self, cfg: GraspTrackerConfig | None = None):
        self.cfg = cfg or GraspTrackerConfig()
        self.reset()

    def reset(self) -> None:
        self.samples: deque = deque(maxlen=self.cfg.window)
        self.confirmed = False
        self.reference: tuple[float, ...] | None = None
        self.loss_count = 0
        self.state = GRASP_NOT_ESTABLISHED

    def update(self, eef_xyz, object_xyz, gripper_command: float) -> str:
        c = self.cfg
        eef = tuple(map(float, eef_xyz))
        obj = tuple(map(float, object_xyz))
        grip = float(gripper_command)
        if len(eef) != 3 or len(obj) != 3 or not all(map(math.isfinite, (*eef, *obj, grip))):
            raise ValueError("Invalid grasp verification measurement")
        offset = tuple(o - e for o, e in zip(obj, eef))
        self.samples.append((eef, obj, offset, grip))

        if not self.confirmed and len(self.samples) == self.samples.maxlen:
            first = self.samples[0]
            eef_lift = eef[2] - first[0][2]
            obj_lift = obj[2] - first[1][2]
            closed = all(s[3] > c.close_command_threshold for s in self.samples)
            stable = all(math.dist(s[2], first[2]) <= c.max_coupling_change_m for s in self.samples)
            if closed and stable and eef_lift >= c.min_lift_m and obj_lift >= c.min_lift_m:
                self.confirmed = True
                self.reference = offset
                self.loss_count = 0
                self.state = GRASP_CONFIRMED
        elif self.confirmed:
            drift = math.dist(offset, self.reference)
            lost = drift > c.max_coupling_change_m or math.dist(eef, obj) > c.lost_distance_m
            self.loss_count = self.loss_count + 1 if lost else 0
            if self.loss_count >= c.loss_persistence:
                self.confirmed = False
                self.reference = None
                self.samples.clear()
                self.state = GRASP_LOST
            else:
                self.state = TRANSPORT_STABLE
        return self.state
