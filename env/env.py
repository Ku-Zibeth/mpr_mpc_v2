"""MetaDrive environment adapter for the full TD-MPC2 training stack.

The upstream TD-MPC2 trainer uses a compact pre-Gymnasium interface:

* ``reset() -> torch.Tensor``
* ``step(action) -> obs, reward, done, info``
* ``info["terminated"]`` distinguishes a real terminal state from a timeout.

This adapter keeps that contract while running MetaDrive with its native
continuous steering/throttle action space.  The scalar reward learned by the
TD-MPC2 world model is the MetaDrive task reward minus a bounded risk-field
penalty.  Keeping the progress and destination terms is important: a pure
negative-risk objective has the trivial solution of never driving.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import torch


try:
    from metadrive.constants import TerminationState
    from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
except ImportError as exc:  # pragma: no cover - produces a clearer setup error
    raise ImportError(
        "MetaDrive is required. Activate the environment with `conda activate tdmpc2`."
    ) from exc

try:
    from .risk_field import RiskFieldCalculator
except ImportError as exc:  # pragma: no cover - produces a clearer setup error
    raise ImportError(
        "RiskFieldCalculator was not found in tdmpc2/env. "
        "Create/copy the risk_field module before training."
    ) from exc


DEFAULT_SIMULATOR_CONFIG: dict[str, Any] = {
    "use_render": False,
    "image_observation": False,
    "manual_control": False,
    "discrete_action": False,
    "action_check": False,
    "start_seed": 100,
    "num_scenarios": 50,
    "map": 3,
    "traffic_density": 0.1,
    "random_traffic": False,
    "accident_prob": 0.8,
    "remove_crashed_traffic_vehicle": False,
    "horizon": 1000,
    "truncate_as_terminate": False,
    "decision_repeat": 5,
    "out_of_road_done": True,
    "crash_vehicle_done": False,
    "crash_object_done": False,
    "cost_to_reward": False,
    "success_reward": 10.0,
    "out_of_road_penalty": 8.0,
    "crash_vehicle_penalty": 1.0,
    "crash_object_penalty": 1.0,
    "crash_vehicle_cost": 1.0,
    "crash_object_cost": 1.0,
    "out_of_road_cost": 1.0,
    "driving_reward": 1.0,
    "speed_reward": 0.12,
    "use_risk_field_reward": True,
    "risk_field_reward_scale": 25.0,
    "enable_idm_lane_change": True,
    "disable_idm_deceleration": False,
    "out_of_road_mode": "legacy",
    "out_of_road_warning_limit": 5,
    "out_of_road_warning_penalty": 1.0,
    "out_of_road_warning_cost": 1.0,
    "out_of_road_recovery_steps": 15,
    "out_of_road_terminate_after_budget": True,
    "log_level": 50,
    "vehicle_config": {
        "lidar": {
            "num_lasers": 240,
            "distance": 50,
            "num_others": 0,
            "gaussian_noise": 0.0,
            "dropout_prob": 0.0,
            "add_others_navi": False,
        }
    },
}

DEFAULT_REWARD_CONFIG: dict[str, float] = {
    "base_reward_weight": 1.0,
    "risk_field_reward_scale": 25.0,
    "reward_min": -10.0,
    "reward_max": 10.0,
}

DEFAULT_RISK_CONFIG: dict[str, float] = {
    "risk_field_max_distance": 50.0,
    "risk_field_boundary_weight": 2.0,
    "risk_field_lane_weight": 0.1,
    "risk_field_offroad_weight": 2.0,
    "risk_field_vehicle_weight": 2.0,
    "risk_field_object_weight": 1.0,
    "risk_field_headway_weight": 0.0,
    "risk_field_ttc_weight": 0.0,
    "risk_field_boundary_sigma": 0.75,
    "risk_field_lane_edge_sigma": 0.75,
    "risk_field_broken_line_sigma": 0.1,
    "risk_field_lane_core_sigma_scale": 2.2,
    "risk_field_lane_shoulder_sigma_scale": 1.5,
    "risk_field_lane_shoulder_weight": 0.9,
    "risk_field_broken_line_factor": 0.1,
    "risk_field_solid_line_factor": 1.0,
    "risk_field_boundary_line_factor": 1.0,
    "risk_field_oncoming_line_factor": 1.5,
    "risk_field_offroad_cost": 1.0,
    "risk_field_offroad_sigma": 1.0,
    "risk_field_on_lane_margin": 0.05,
    "risk_field_vehicle_longitudinal_sigma": 6.8,
    "risk_field_vehicle_lateral_sigma": 2.0,
    "risk_field_vehicle_beta": 2.0,
    "risk_field_vehicle_dynamic_sigma_scale": 1.2,
    "risk_field_vehicle_dynamic_alpha": 0.35,
    "risk_field_vehicle_min_dynamic_sigma": 0.5,
    "risk_field_object_longitudinal_sigma": 5.5,
    "risk_field_object_lateral_sigma": 2.0,
    "risk_field_object_beta": 2.0,
    "risk_field_lane_beta": 2.0,
    "risk_field_headway_time_threshold": 1.2,
    "risk_field_ttc_threshold": 3.0,
    "risk_field_min_speed": 0.5,
    "risk_field_headway_cost_clip": 1.0,
    "risk_field_ttc_cost_clip": 1.0,
    "risk_field_raw_clip": 10.0,
}

DEFAULT_COST_CONFIG: dict[str, Any] = {
    "enabled": True,
    "risk_field_weight": 1.0,
    "event_weights": {
        "cost": 1.0,
        "crash": 1.0,
        "crash_vehicle": 1.0,
        "crash_object": 1.0,
        "out_of_road": 1.0,
        "out_of_road_warning": 1.0,
    },
}

SAFE_METADRIVE_EXTRA_CONFIG: dict[str, Any] = {
    "state_observation_with_offscreen_render": False,
    "remove_crashed_traffic_vehicle": False,
    "use_risk_field_reward": True,
    "risk_field_reward_scale": 25.0,
    "enable_idm_lane_change": True,
    "disable_idm_deceleration": False,
    "out_of_road_mode": "legacy",
    "out_of_road_warning_limit": 5,
    "out_of_road_warning_penalty": 1.0,
    "out_of_road_warning_cost": 1.0,
    "out_of_road_recovery_steps": 15,
    "out_of_road_terminate_after_budget": True,
}


class TDMPC2SafeMetaDriveEnv(SafeMetaDriveEnv):
    """SafeMetaDriveEnv variant that accepts envs_0831 extension keys."""

    def default_config(self):
        config = super(TDMPC2SafeMetaDriveEnv, self).default_config()
        config.update(SAFE_METADRIVE_EXTRA_CONFIG, allow_add_new_key=True)
        return config

    def get_single_observation(self):
        if self.config.get("state_observation_with_offscreen_render", False):
            from metadrive.obs.state_obs import LidarStateObservation

            return LidarStateObservation(self.config)
        return super().get_single_observation()


def _plain_dict(value: Any) -> dict[str, Any]:
    """Convert DictConfig/mapping values without requiring OmegaConf here."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {key: _plain_value(item) for key, item in value.items()}
    raise TypeError(f"Expected a mapping, got {type(value)!r}.")


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: _plain_value(value) for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = _plain_value(value)
    return result


def _flag(info: Mapping[str, Any], key: str) -> float:
    return float(bool(info.get(key, False)))


def _end_reason(info: Mapping[str, Any], terminated: bool, truncated: bool) -> str:
    for key in (
        TerminationState.SUCCESS,
        TerminationState.CRASH_VEHICLE,
        TerminationState.CRASH_OBJECT,
        TerminationState.CRASH_BUILDING,
        TerminationState.CRASH_HUMAN,
        TerminationState.CRASH_SIDEWALK,
        TerminationState.OUT_OF_ROAD,
        TerminationState.MAX_STEP,
    ):
        if info.get(key, False):
            return str(key)
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return "not_done"


class MetaDriveTDMPC2Env(gym.Env):
    """State-based MetaDrive task with risk-field reward shaping for TD-MPC2."""

    metadata = {"render_modes": ["topdown"], "render_fps": 10}

    def __init__(self, config: Mapping[str, Any] | None = None, seed: int = 1):
        super().__init__()
        user_config = _plain_dict(config)
        self.algorithm_version = str(
            user_config.get("algorithm_version", "lattice_tdmpc2_v3")
        ).lower()
        self.simulator_config = _deep_merge(
            DEFAULT_SIMULATOR_CONFIG, user_config.get("simulator", {})
        )
        if "window_size" in self.simulator_config:
            self.simulator_config["window_size"] = tuple(
                int(value) for value in self.simulator_config["window_size"]
            )
        self.reward_config = _deep_merge(
            DEFAULT_REWARD_CONFIG, user_config.get("reward", {})
        )
        self.risk_config = _deep_merge(
            DEFAULT_RISK_CONFIG, user_config.get("risk_field", {})
        )
        self.cost_config = _deep_merge(
            DEFAULT_COST_CONFIG, user_config.get("cost", {})
        )

        if self.simulator_config.get("image_observation", False) and not self.simulator_config.get(
            "state_observation_with_offscreen_render", False
        ):
            raise ValueError("This TD-MPC2 adapter currently supports state observations only.")
        if self.simulator_config.get("discrete_action", False):
            raise ValueError("TD-MPC2 requires MetaDrive's continuous two-dimensional actions.")
        self.reward_config.setdefault(
            "risk_field_reward_scale",
            self.reward_config.get("risk_penalty_scale", DEFAULT_REWARD_CONFIG["risk_field_reward_scale"]),
        )
        if float(self.reward_config["risk_field_reward_scale"]) < 0:
            raise ValueError("reward.risk_field_reward_scale must be non-negative.")
        if float(self.risk_config["risk_field_raw_clip"]) <= 0:
            raise ValueError("risk_field.risk_field_raw_clip must be positive.")

        self._env = TDMPC2SafeMetaDriveEnv(self.simulator_config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.max_episode_steps = int(self.simulator_config["horizon"])
        self._risk_calculator = RiskFieldCalculator(self.risk_config)
        self._rng = np.random.RandomState(seed)
        self._last_info: dict[str, Any] = {}
        self._active_cost_events: set[str] = set()
        self._safety_cost_calculator = self._make_safety_cost_calculator()
        self.action_space.seed(seed)
        self._validate_spaces()

    @property
    def unwrapped_metadrive(self) -> TDMPC2SafeMetaDriveEnv:
        return self._env

    @property
    def last_info(self) -> Mapping[str, Any]:
        return self._last_info

    def _validate_spaces(self) -> None:
        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError(f"Expected a Box observation space, got {self.observation_space!r}.")
        if len(self.observation_space.shape) != 1:
            raise ValueError(
                f"Expected a flat state observation, got {self.observation_space.shape}."
            )
        if not isinstance(self.action_space, gym.spaces.Box) or self.action_space.shape != (2,):
            raise ValueError(
                "Expected MetaDrive action space Box(shape=(2,)) for steering and "
                f"throttle/brake, got {self.action_space!r}."
            )

    def _observation(self, observation: Any) -> torch.Tensor:
        array = np.asarray(observation, dtype=np.float32).reshape(-1)
        if array.shape != self.observation_space.shape:
            raise ValueError(
                f"Observation shape changed from {self.observation_space.shape} to {array.shape}."
            )
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        array = np.clip(array, self.observation_space.low, self.observation_space.high)
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    def _scenario_seed(self, seed: int | None) -> int:
        start = int(self.simulator_config["start_seed"])
        count = int(self.simulator_config["num_scenarios"])
        if count <= 0:
            raise ValueError("simulator.num_scenarios must be positive.")
        if seed is not None:
            seed = int(seed)
            if start <= seed < start + count:
                return seed
            return int(start + seed % count)
        return int(self._rng.randint(start, start + count))

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> torch.Tensor:
        del options
        scenario_seed = self._scenario_seed(seed)
        self.action_space.seed(scenario_seed)
        observation, info = self._env.reset(seed=scenario_seed)
        self._active_cost_events.clear()
        self._safety_cost_calculator.reset()
        self._last_info = dict(info)
        return self._observation(observation)

    def rand_act(self) -> torch.Tensor:
        action = self.action_space.sample().astype(np.float32)
        return torch.from_numpy(action)

    def _risk_penalty(self, raw_risk: float) -> tuple[float, float]:
        if not bool(self.simulator_config.get("use_risk_field_reward", True)):
            return 0.0, 0.0
        raw_clip = max(float(self.risk_config["risk_field_raw_clip"]), 1e-6)
        normalized = float(np.clip(raw_risk / raw_clip, 0.0, 1.0))
        penalty = float(self.reward_config["risk_field_reward_scale"]) * normalized
        return normalized, penalty

    def _event_costs(self, info: Mapping[str, Any]) -> dict[str, float]:
        weights = self.cost_config.get("event_weights", {})
        raw_cost = max(float(weights.get("cost", 0.0)) * float(info.get("cost", 0.0)), 0.0)
        crash_weight = float(weights.get("crash", 0.0))
        events: dict[str, float] = {}

        if _flag(info, "crash_vehicle"):
            events["crash_vehicle"] = max(
                raw_cost, crash_weight, float(weights.get("crash_vehicle", 0.0))
            )
        if _flag(info, "crash_object"):
            events["crash_object"] = max(
                raw_cost, crash_weight, float(weights.get("crash_object", 0.0))
            )
        if _flag(info, "out_of_road"):
            events["out_of_road"] = max(raw_cost, float(weights.get("out_of_road", 0.0)))
        if _flag(info, "out_of_road_warning"):
            events["out_of_road_warning"] = float(weights.get("out_of_road_warning", 0.0))
        if _flag(info, "crash") and not ({"crash_vehicle", "crash_object"} & events.keys()):
            events["crash_other"] = max(raw_cost, crash_weight)
        if raw_cost > 0.0 and not events:
            events["raw_cost"] = raw_cost
        return events

    def _safety_cost(
        self, info: Mapping[str, Any], normalized_risk: float
    ) -> tuple[float, float, float]:
        calculator = getattr(self, "_safety_cost_calculator", None)
        if calculator is None:
            calculator = self._make_safety_cost_calculator()
            self._safety_cost_calculator = calculator
        return calculator.compute(info, normalized_risk)

    def _make_safety_cost_calculator(self):
        version = getattr(self, "algorithm_version", "lattice_tdmpc2_v3")
        if version in {"v2", "lattice_tdmpc2_v2"}:
            from lattice_tdmpc2.versions.v2.cost_v2 import SafetyCostV2

            return SafetyCostV2(self.cost_config, _flag)
        if version not in {
            "v3",
            "lattice_tdmpc2_v3",
            "v4",
            "lattice_tdmpc2_v4",
            "mpr_mpc_residual_sac_mppi_v2",
        }:
            raise ValueError(f"Unsupported algorithm_version={version!r} for safety cost.")
        from lattice_tdmpc2.versions.v3.cost_v3 import SafetyCostV3

        return SafetyCostV3(self.cost_config, _flag)

    def step(self, action: torch.Tensor | np.ndarray):
        if torch.is_tensor(action):
            action = action.detach().cpu().numpy()
        action_array = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action_array = np.clip(action_array, self.action_space.low, self.action_space.high)

        observation, base_reward, terminated, truncated, raw_info = self._env.step(action_array)
        risk_cost, risk_info = self._risk_calculator.calculate(self._env, self._env.agent)
        normalized_risk, risk_penalty = self._risk_penalty(float(risk_cost))

        reward = (
            float(self.reward_config["base_reward_weight"]) * float(base_reward)
            - risk_penalty
        )
        reward = float(
            np.clip(
                reward,
                float(self.reward_config["reward_min"]),
                float(self.reward_config["reward_max"]),
            )
        )
        done = bool(terminated or truncated)

        info = dict(raw_info)
        info.update(risk_info)
        info.update(
            success=_flag(info, TerminationState.SUCCESS),
            terminated=torch.tensor(float(bool(terminated)), dtype=torch.float32),
            truncated=float(bool(truncated)),
            end_reason=_end_reason(info, bool(terminated), bool(truncated)),
            crash=_flag(info, TerminationState.CRASH),
            crash_vehicle=_flag(info, TerminationState.CRASH_VEHICLE),
            crash_object=_flag(info, TerminationState.CRASH_OBJECT),
            out_of_road=_flag(info, TerminationState.OUT_OF_ROAD),
            metadrive_reward=float(base_reward),
            risk_field_normalized_cost=normalized_risk,
            risk_field_reward_penalty=risk_penalty,
            tdmpc2_reward=reward,
        )
        safety_cost, safety_risk_cost, safety_event_cost = self._safety_cost(info, normalized_risk)
        info.update(
            cost=safety_cost,
            safety_risk_cost=safety_risk_cost,
            safety_event_cost=safety_event_cost,
        )
        self._last_info = info
        return (
            self._observation(observation),
            torch.tensor(reward, dtype=torch.float32),
            done,
            info,
        )

    def render(self):
        return self._env.render(mode="topdown")

    def close(self) -> None:
        self._env.close()


def make_env(cfg: Any) -> MetaDriveTDMPC2Env:
    """Create the environment and fill dimensions expected by TD-MPC2."""
    metadrive_config = dict(cfg.metadrive)
    metadrive_config["algorithm_version"] = getattr(
        cfg, "algorithm_version", "lattice_tdmpc2_v3"
    )
    env = MetaDriveTDMPC2Env(config=metadrive_config, seed=int(cfg.seed))
    cfg.obs_shape = {"state": env.observation_space.shape}
    cfg.action_dim = int(env.action_space.shape[0])
    cfg.episode_length = int(env.max_episode_steps)
    cfg.seed_steps = max(int(cfg.seed_steps), 5 * cfg.episode_length)
    return env


__all__ = ["MetaDriveTDMPC2Env", "make_env"]
