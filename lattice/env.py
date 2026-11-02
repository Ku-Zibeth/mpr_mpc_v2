"""
SafeMetaDrive environment wired with a Frenet local planner.

The planner/controller implementation lives in frenet_metadrive.py so this file
stays focused on environment configuration and Gym-compatible entry points.
"""
import copy
import sys
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np

from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.utils.doc_utils import generate_gif

try:
    from frenet_metadrive import (
        FRENET_DEFAULT_CONFIG,
        MetaDriveFrenetController,
        draw_frenet_overlay,
        topdown_to_image,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from frenet_metadrive import (
        FRENET_DEFAULT_CONFIG,
        MetaDriveFrenetController,
        draw_frenet_overlay,
        topdown_to_image,
    )


DEFAULT_CONFIG = {
    # The below are default configs copied from SafeMetaDriveEnv.
    "accident_prob": 0.8,
    "traffic_density": 0.05,
    # Termination conditions.
    "crash_vehicle_done": True,
    "crash_object_done": False,
    # Reward.
    "success_reward": 10.0,
    "driving_reward": 1.0,
    "speed_reward": 0.1,
    # Penalty will be negated and added to reward.
    "out_of_road_penalty": 5.0,
    "crash_vehicle_penalty": 1.0,
    "crash_object_penalty": 1.0,
    # Cost will be returned in info["cost"].
    "crash_vehicle_cost": 1.0,
    "crash_object_cost": 1.0,
    "out_of_road_cost": 1.0,
    # Episode GIF recording. Kept off by default so training/evaluation runs do not fill the disk.
    "save_episode_gifs": False,
    "episode_gif_dir": "episode_gifs",
    "episode_gif_fps": 12,
    "episode_gif_frame_interval": 2,
    "episode_gif_max_frames": 400,
    "episode_gif_capture_topdown": True,
    "episode_gif_capture_main": True,
}
DEFAULT_CONFIG.update(FRENET_DEFAULT_CONFIG)


class FrenetSafeMetaDriveEnv(SafeMetaDriveEnv):
    def default_config(self):
        config = super(FrenetSafeMetaDriveEnv, self).default_config()
        config.update(DEFAULT_CONFIG, allow_add_new_key=True)
        return config

    @property
    def observation_space(self):
        space = super(FrenetSafeMetaDriveEnv, self).observation_space
        if not self.config["frenet_append_reference_obs"]:
            return space
        if not isinstance(space, gym.spaces.Box) or len(space.shape) != 1:
            return space

        obs_dim = int(self.config["frenet_reference_obs_dim"])
        low = np.concatenate([space.low.astype(np.float32), -np.ones(obs_dim, dtype=np.float32)])
        high = np.concatenate([space.high.astype(np.float32), np.ones(obs_dim, dtype=np.float32)])
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def __init__(self, config=None):
        super(FrenetSafeMetaDriveEnv, self).__init__(config)
        self.frenet_controller: Optional[MetaDriveFrenetController] = None
        self._last_frenet_info = {}
        self._episode_topdown_frames = []
        self._episode_main_frames = []
        self._episode_gif_pending_save = False
        self._episode_gif_index = 0
        self._episode_gif_run_id = time.strftime("%Y%m%d_%H%M%S")
        self._last_episode_gif_paths = {}
        self._use_frenet_first_action = True

    def reset(self, *args, **kwargs):
        kwargs.pop("options", None)
        self._save_pending_episode_gifs()
        ret = super(FrenetSafeMetaDriveEnv, self).reset(*args, **kwargs)
        self._ensure_frenet_controller()
        self.frenet_controller.reset()
        self._last_frenet_info = {}
        self._clear_episode_gif_frames()
        self._use_frenet_first_action = True
        return self._augment_reset_return(ret)

    def step(self, action=None):
        if self.config["frenet_planner"]:
            self._ensure_frenet_controller()
            try:
                rl_action = self._prepare_frenet_rl_action(action)
                action = self.frenet_controller.act(self.agent, rl_action=rl_action)
                self._last_frenet_info = dict(self.frenet_controller.last_info)
            except Exception as exc:
                if self.config["frenet_strict"]:
                    raise
                action = [0.0, 0.0]
                self._last_frenet_info = {"frenet_error": repr(exc)}
        elif action is None:
            action = [0.0, 0.0]

        obs, reward, terminated, truncated, info = super(FrenetSafeMetaDriveEnv, self).step(action)
        info.update(self._last_frenet_info)
        reference_reward = self._reference_reward(info)
        if reference_reward != 0.0:
            reward += reference_reward
            info["frenet_reference_reward"] = float(reference_reward)
        obs = self._augment_observation(obs)
        if self.config["save_episode_gifs"] and (terminated or truncated):
            self._episode_gif_pending_save = True
        return obs, reward, terminated, truncated, info

    def _prepare_frenet_rl_action(self, action):
        if not self.config["frenet_rl_action"]:
            return None
        if self._use_frenet_first_action and self.config["frenet_rl_first_action"] is not None:
            self._use_frenet_first_action = False
            return self.config["frenet_rl_first_action"]
        self._use_frenet_first_action = False
        return action

    def render(self, text=None, mode=None, *args, **kwargs):
        is_topdown = mode in ["top_down", "topdown", "bev", "birdview"]
        to_image = kwargs.get("to_image", True)

        # Avoid MetaDrive's pygame SysFont branch on Windows; Frenet data is drawn as lines instead.
        ret = super(FrenetSafeMetaDriveEnv, self).render(
            text=None if is_topdown else text,
            mode=mode,
            *args,
            **kwargs
        )
        if is_topdown and self.config["frenet_show_in_window"]:
            renderer = self.top_down_renderer
            if renderer is not None and self.frenet_controller is not None:
                draw_frenet_overlay(renderer, self.frenet_controller.planner, self.config)
                renderer.blit()
                ret = topdown_to_image(renderer) if to_image else renderer.screen_canvas
        self._record_episode_gif_frame(is_topdown, ret)
        self._save_pending_episode_gifs()
        return ret

    def _ensure_frenet_controller(self):
        if self.frenet_controller is None:
            self.frenet_controller = MetaDriveFrenetController(self.config)

    def _augment_reset_return(self, ret):
        if isinstance(ret, tuple) and len(ret) == 2:
            obs, info = ret
            return self._augment_observation(obs), info
        return self._augment_observation(ret)

    def _augment_observation(self, obs):
        if not self.config["frenet_append_reference_obs"] or isinstance(obs, dict):
            return obs
        self._ensure_frenet_controller()
        base_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        try:
            reference_obs = self.frenet_controller.reference_observation(self.agent)
        except Exception:
            reference_obs = np.zeros(int(self.config["frenet_reference_obs_dim"]), dtype=np.float32)
        return np.concatenate([base_obs, reference_obs]).astype(np.float32)

    def _reference_reward(self, info):
        weight = float(self.config["frenet_reference_reward_weight"])
        if weight <= 0.0 or self.frenet_controller is None:
            return 0.0

        try:
            reference_obs = self.frenet_controller.reference_observation(self.agent)
        except Exception:
            return 0.0
        actual_d_norm = abs(float(reference_obs[0])) if len(reference_obs) else 0.0
        target_d = abs(float(info.get("frenet_selected_target_d", 0.0)))
        try:
            lane_width = max(0.1, float(self.agent.navigation.get_current_lane_width()))
        except Exception:
            lane_width = 3.5
        target_d_norm = min(target_d / lane_width, 1.0)
        return -weight * (0.35 * actual_d_norm + 0.65 * target_d_norm)

    def _record_episode_gif_frame(self, is_topdown, topdown_ret):
        if not self.config["save_episode_gifs"]:
            return
        interval = max(1, int(self.config["episode_gif_frame_interval"]))
        if self.episode_step % interval != 0:
            return

        max_frames = max(1, int(self.config["episode_gif_max_frames"]))
        if is_topdown and self.config["episode_gif_capture_topdown"]:
            topdown_frame = self._topdown_frame_to_image(topdown_ret)
            self._append_gif_frame(self._episode_topdown_frames, topdown_frame, max_frames)

        if self.config["episode_gif_capture_main"]:
            main_frame = self._capture_main_camera_frame()
            self._append_gif_frame(self._episode_main_frames, main_frame, max_frames)

    @staticmethod
    def _append_gif_frame(frames, frame, max_frames):
        if frame is None or len(frames) >= max_frames:
            return
        frames.append(frame.copy() if hasattr(frame, "copy") else frame)

    def _topdown_frame_to_image(self, topdown_ret):
        if topdown_ret is None:
            return None
        if hasattr(topdown_ret, "get_size"):
            return topdown_to_image(type("RendererProxy", (), {"screen_canvas": topdown_ret})())
        return topdown_ret

    def _capture_main_camera_frame(self):
        if self.main_camera is None or not self.config["use_render"]:
            return None
        try:
            return self.main_camera.perceive(to_float=False)
        except Exception:
            return None

    def _save_pending_episode_gifs(self):
        if not self._episode_gif_pending_save:
            return
        self._save_episode_gifs()

    def _save_episode_gifs(self):
        gif_dir = Path(self.config["episode_gif_dir"])
        if not gif_dir.is_absolute():
            gif_dir = Path(__file__).resolve().parent / gif_dir
        gif_dir.mkdir(parents=True, exist_ok=True)

        seed = self.current_seed if self.engine is not None else "none"
        stem = "{}_ep{:04d}_seed{}".format(self._episode_gif_run_id, self._episode_gif_index, seed)
        duration = max(1, int(1000 / max(1, int(self.config["episode_gif_fps"]))))
        saved_paths = {}

        if self._episode_topdown_frames:
            topdown_path = gif_dir / "{}_topdown.gif".format(stem)
            generate_gif(self._episode_topdown_frames, str(topdown_path), duration=duration)
            saved_paths["topdown"] = str(topdown_path)
        if self._episode_main_frames:
            main_path = gif_dir / "{}_main.gif".format(stem)
            generate_gif(self._episode_main_frames, str(main_path), duration=duration)
            saved_paths["main"] = str(main_path)

        if saved_paths:
            print("[GIF] saved episode {}: {}".format(self._episode_gif_index, saved_paths))
        self._last_episode_gif_paths = saved_paths
        self._episode_gif_index += 1
        self._episode_gif_pending_save = False
        self._clear_episode_gif_frames()

    def _clear_episode_gif_frames(self):
        self._episode_topdown_frames = []
        self._episode_main_frames = []


TRAINING_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
TRAINING_CONFIG.update(
    {
        "num_scenarios": 500,
        "start_seed": 100,
    }
)


def get_training_env(extra_config=None):
    config = copy.deepcopy(TRAINING_CONFIG)
    if extra_config:
        config.update(extra_config)
    return FrenetSafeMetaDriveEnv(config)


VALIDATION_CONFIG = copy.deepcopy(DEFAULT_CONFIG)
VALIDATION_CONFIG.update(
    {
        "num_scenarios": 500,
        "start_seed": 1000,
    }
)


def get_validation_env(extra_config=None):
    config = copy.deepcopy(VALIDATION_CONFIG)
    if extra_config:
        config.update(extra_config)
    return FrenetSafeMetaDriveEnv(config)


class SafeMetaDriveEnv_mini(FrenetSafeMetaDriveEnv):
    def default_config(self):
        config = super(SafeMetaDriveEnv_mini, self).default_config()
        config.update(DEFAULT_CONFIG, allow_add_new_key=True)
        return config


if __name__ == "__main__":
    env = get_training_env(
        {
            "manual_control": False,
            "use_render": True,
            "traffic_density": 0.05,
            "accident_prob": 0.8,
            "frenet_planner": True,
            "frenet_debug_info": True,
            "save_episode_gifs": True,
            "episode_gif_dir": "episode_gifs",
            "vehicle_config": {
                "show_navi_mark": True,
                "show_line_to_navi_mark": True,
            },
        }
    )
    try:
        env.reset()
        step_count = 0
        while True:
            _, reward, terminated, truncated, info = env.step()
            env.render(mode="topdown", target_agent_heading_up=True)
            if step_count % 20 == 0:
                print(
                    "step={}, reward={:.3f}, action={}, candidates={}/{}, obstacles={}, selected_path={}, target_d={:.2f}".format(
                        step_count,
                        float(reward),
                        info.get("frenet_action"),
                        info.get("frenet_valid_candidate_count"),
                        info.get("frenet_candidate_count"),
                        info.get("frenet_obstacle_count"),
                        info.get("frenet_selected_path_index"),
                        float(info.get("frenet_selected_target_d", 0.0)),
                    )
                )
            step_count += 1
            if terminated or truncated:
                env.reset()
                step_count = 0
    finally:
        env.close()
