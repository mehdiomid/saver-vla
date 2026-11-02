"""LIBERO + SmolVLA backend, built on LeRobot v0.6.1's public APIs.

This replaces every in-place edit the notebook made to LeRobot (`libero.py`,
`lerobot_eval.py`). LeRobot stays untouched; instead:
  * `SaverLiberoEnv` subclasses LeRobot's `LiberoEnv` to read privileged object
    state and to inject object faults (forced separation, velocity impulse);
  * the episode loop lives in `saver_vla.runner`, in one process, so the
    supervisor can switch between the policy and recovery skills directly.

Policy loading and pre/post-processing mirror `lerobot_eval.eval_main` so SmolVLA
sees exactly the observations it sees under `lerobot-eval`.

Requires: `pip install -e ".[libero,smolvla]"` of LeRobot v0.6.1, MuJoCo 3.3.2,
and an NVIDIA GPU with EGL (e.g. Colab).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import EnvConfig, FaultConfig
from ..robots import PandaDeltaProfile
from ..types import RobotState


def _make_env_class():
    from lerobot.envs.libero import LiberoEnv

    class SaverLiberoEnv(LiberoEnv):
        """LeRobot's LiberoEnv plus privileged-state access and object fault injection."""

        saver_object_name = "alphabet_soup_1"
        saver_target_name = "basket_1"

        def reset(self, seed=None, **kwargs):
            observation, info = super().reset(seed=seed, **kwargs)
            self.saver_resolve_object()
            self.saver_step = 0
            self.saver_success = False
            self._saver_obs = observation
            return observation, info

        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            self.saver_step += 1
            self.saver_success = bool(info.get("is_success", False))
            self._saver_obs = observation
            return observation, reward, terminated, truncated, info

        # --- privileged simulator access -------------------------------------
        def _task_env(self):
            return getattr(self._env, "env", self._env)

        def saver_resolve_object(self) -> str:
            """Find the manipulated object's body from the task sentence when object_name is "auto".

            "pick up the cream cheese and place it in the basket" -> "cream_cheese_1".
            """
            if self.saver_object_name != "auto":
                return self.saver_object_name
            ids = getattr(self._task_env(), "obj_body_id", {}) or {}
            sentence = self.task_description.lower()
            candidates = [
                name for name in ids
                if name != self.saver_target_name and name.rsplit("_", 1)[0].replace("_", " ") in sentence
            ]
            if not candidates:
                raise RuntimeError(f"Cannot infer the object from {sentence!r}; set env.object_name. Bodies: {sorted(ids)}")
            self.saver_object_name = max(candidates, key=len)
            return self.saver_object_name

        def _body_id(self, name: str) -> int:
            task_env = self._task_env()
            ids = getattr(task_env, "obj_body_id", None)
            if ids and name in ids:
                return int(ids[name])
            return int(task_env.sim.model.body_name2id(name))

        def saver_state(self) -> dict[str, Any]:
            task_env = self._task_env()
            data = task_env.sim.data
            obj = np.asarray(data.body_xpos[self._body_id(self.saver_object_name)], float).copy()
            target = np.asarray(data.body_xpos[self._body_id(self.saver_target_name)], float).copy()
            eef = None
            robot_state = (self._saver_obs or {}).get("robot_state") if isinstance(self._saver_obs, dict) else None
            if robot_state is not None:
                eef = np.asarray(robot_state["eef"]["pos"], float).copy()
            if eef is None:  # obs_type="pixels": read the grip site directly
                robot = self._env.robots[0]
                eef = np.asarray(data.site_xpos[robot.eef_site_id], float).copy()
            return {"step": self.saver_step, "time": float(data.time), "eef": eef, "object": obj, "target": target}

        def saver_inject_object_fault(self, kind: str, direction_xy, displacement_m: float, impulse_speed: float):
            task_env = self._task_env()
            sim = task_env.sim
            model, data = sim.model, sim.data
            joint_id = int(model.joint_name2id(f"{self.saver_object_name}_joint0"))
            if int(model.jnt_type[joint_id]) != 0:
                raise RuntimeError("Object joint is not a free joint")
            qpos0 = int(model.jnt_qposadr[joint_id])
            qvel0 = int(model.jnt_dofadr[joint_id])
            direction = np.asarray(direction_xy, float)
            record: dict[str, Any] = {"joint": f"{self.saver_object_name}_joint0"}
            if kind == "forced_separation":
                # Controlled state displacement away from the target, NOT a natural drop.
                record["qpos_before"] = np.asarray(data.qpos[qpos0 : qpos0 + 7]).tolist()
                data.qpos[qpos0 : qpos0 + 2] += displacement_m * direction
                data.qvel[qvel0 : qvel0 + 6] = 0.0
                sim.forward()
                record["qpos_after"] = np.asarray(data.qpos[qpos0 : qpos0 + 7]).tolist()
                record["displacement_m"] = displacement_m
            elif kind == "velocity_impulse":
                record["velocity_before"] = np.asarray(data.qvel[qvel0 : qvel0 + 6]).tolist()
                data.qvel[qvel0 : qvel0 + 2] += impulse_speed * direction
                sim.forward()
                record["velocity_after"] = np.asarray(data.qvel[qvel0 : qvel0 + 6]).tolist()
                record["impulse_speed_mps"] = impulse_speed
            else:
                raise ValueError(f"Unknown object fault: {kind}")
            return record

    return SaverLiberoEnv


class LiberoBackend:
    """Runs SmolVLA in one LIBERO task through the Backend interface."""

    def __init__(self, cfg: EnvConfig):
        import gymnasium as gym
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.libero import _get_suite
        from lerobot.envs.utils import parse_camera_names
        from lerobot.policies import make_policy, make_pre_post_processors

        self.cfg = cfg
        self.profile = PandaDeltaProfile()
        self._torch = torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.env_cfg = LiberoEnvConfig(
            task=cfg.suite,
            task_ids=[cfg.task_id],
            episode_length=cfg.episode_length,
            control_mode=cfg.control_mode,
        )
        policy_cfg = PreTrainedConfig.from_pretrained(cfg.policy_path)
        policy_cfg.pretrained_path = Path(cfg.policy_path)
        policy_cfg.device = cfg.device
        self.policy = make_policy(cfg=policy_cfg, env_cfg=self.env_cfg, rename_map=cfg.rename_map)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)},
                "rename_observations_processor": {"rename_map": cfg.rename_map},
            },
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=self.env_cfg, policy_cfg=policy_cfg
        )

        env_cls = _make_env_class()
        env_cls.saver_object_name = cfg.object_name
        env_cls.saver_target_name = cfg.target_name
        gym_kwargs = dict(self.env_cfg.gym_kwargs)
        gym_kwargs.pop("task_ids", None)
        self.env = env_cls(
            task_suite=_get_suite(cfg.suite),
            task_id=cfg.task_id,
            task_suite_name=cfg.suite,
            camera_name=parse_camera_names(self.env_cfg.camera_name),
            init_states=self.env_cfg.init_states,
            episode_length=cfg.episode_length,
            episode_index=0,
            n_envs=1,
            control_mode=cfg.control_mode,
            camera_name_mapping=self.env_cfg.camera_name_mapping,
            is_libero_plus=self.env_cfg.is_libero_plus,
            **gym_kwargs,
        )
        # A one-env vector wrapper gives the batched observations LeRobot's processors expect.
        self.vec = gym.vector.SyncVectorEnv([lambda: self.env], autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        self.task_description = self.env.task_description
        self._obs = None

    # ------------------------------------------------------------------ Backend API

    def reset(self, seed: int | None, episode_index: int) -> None:
        from lerobot.envs.utils import NEW_ROLLOUT_OPTION
        from lerobot.utils.random_utils import set_seed

        if seed is not None:
            set_seed(seed)  # same global seeding as lerobot-eval (affects SmolVLA sampling)
        self.env.init_state_id = int(episode_index)
        self.policy.reset()
        self._last_gripper = None
        seeds = None if seed is None else [int(seed)]
        self._obs, _ = self.vec.reset(seed=seeds, options={NEW_ROLLOUT_OPTION: True})

    def state(self) -> RobotState:
        s = self.env.saver_state()
        return RobotState(
            step=s["step"], time=s["time"], eef_xyz=s["eef"], object_xyz=s["object"], target_xyz=s["target"],
            gripper_command=getattr(self, "_last_gripper", None),
        )

    def policy_action(self) -> np.ndarray:
        from lerobot.envs import preprocess_observation
        from lerobot.utils.constants import ACTION

        torch = self._torch
        observation = preprocess_observation(self._obs)
        observation["task"] = [self.task_description]
        observation = self.env_preprocessor(observation)
        observation = self.preprocessor(observation)
        with torch.inference_mode():
            action = self.policy.select_action(observation)
        action = self.postprocessor(action)
        action = self.env_postprocessor({ACTION: action})[ACTION]
        return action.to("cpu").numpy()[0].astype(float)

    def reset_policy(self) -> None:
        self.policy.reset()

    # ------------------------------------------------------------------ video

    def start_recording(self) -> None:
        self._frames = [self.env.render()]

    def save_recording(self, path: Path) -> None:
        from lerobot.utils.io_utils import write_video

        frames = getattr(self, "_frames", None)
        if frames:
            write_video(str(path), frames, fps=20)  # 20 Hz control = real time
        self._frames = None

    def step(self, action: np.ndarray) -> tuple[bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        self._obs, reward, terminated, truncated, info = self.vec.step(action[None, :])
        if getattr(self, "_frames", None) is not None:
            self._frames.append(self.env.render())
        self._last_gripper = float(action[self.profile.gripper_index])
        success = bool(self.env.saver_success)
        done = bool(terminated[0] or truncated[0]) or self.env.saver_step >= self.env._max_episode_steps
        return done, success, {"reward": float(np.asarray(reward)[0])}

    def inject_object_fault(self, kind: str, direction_xy: np.ndarray, cfg: FaultConfig) -> dict[str, Any]:
        return self.env.saver_inject_object_fault(kind, direction_xy, cfg.displacement_m, cfg.impulse_speed_mps)

    def close(self) -> None:
        self.vec.close()
