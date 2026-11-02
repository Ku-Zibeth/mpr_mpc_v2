from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from lattice.frenet_metadrive import FRENET_DEFAULT_CONFIG, FrenetPath, MetaDriveFrenetController
from lattice_tdmpc2.action_adapter import LatticeActionAdapter
from lattice_tdmpc2.evaluator import WorldModelEvaluation, WorldModelEvaluator
from lattice_tdmpc2.features import FEATURE_NAMES, extract_coarse_path_features
from lattice_tdmpc2.planner import ResidualSACLatticePlanner
from lattice_tdmpc2.residual_action import (
    LANE_WIDTH_ACTION_MAPPING_VERSION,
    ResidualActionAdapter,
)
from lattice_tdmpc2.state_builder import ResidualStateBuilder
from sac.model import SACAgent
from tdmpc2 import TDMPC2


def make_path(target_d=0.0, target_speed=10.0, collision=False, horizon=1.0):
    times = np.linspace(0.0, horizon, 11)
    x = target_speed * times
    d = np.linspace(0.0, target_d, len(times))
    zeros = np.zeros_like(times)
    return FrenetPath(
        t=times.tolist(), d=d.tolist(), d_d=zeros.tolist(), d_dd=zeros.tolist(),
        d_ddd=zeros.tolist(), s=x.tolist(), s_d=np.full_like(times, target_speed).tolist(),
        s_dd=zeros.tolist(), s_ddd=zeros.tolist(), x=x.tolist(), y=d.tolist(),
        yaw=zeros.tolist(), target_d=target_d, target_speed=target_speed,
        horizon=horizon, collision=collision, obstacle_cost=float(collision),
        min_obstacle_clearance=-0.1 if collision else float("inf"),
    )


class FakeVehicle:
    position = np.array([0.0, 0.0])
    velocity = np.array([5.0, 0.0])
    heading_theta = 0.0
    speed = 5.0


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self._termination = None
        self.cost_q_enabled = True

    def encode(self, obs, task):
        return obs[:, :2] * self.weight

    def reward(self, z, action, task):
        return (action[:, :1] + 0.1 * z[:, :1]) * self.weight

    def cost(self, z, action, task):
        return action[:, 1:].abs() * self.weight.abs()

    def next(self, z, action, task):
        return z + action[:, :2] * self.weight

    def pi(self, z, task):
        return torch.zeros((len(z), 2), device=z.device), {}

    def Q(self, z, action, task, return_type="avg"):
        values = (z[:, :1] + action[:, :1]) * self.weight
        if return_type == "all":
            return torch.stack([values - 0.1, values, values + 0.1])
        return values

    def cost_Q(self, z, action, task, return_type="avg"):
        values = torch.full((len(z), 1), 2.0, device=z.device)
        if return_type == "all":
            return torch.stack([values, values])
        return values


def tiny_agent():
    cfg = SimpleNamespace(
        action_dim=2, latent_dim=2, episodic=False, num_bins=1, num_q=3,
    )
    return SimpleNamespace(
        model=TinyModel(), device=torch.device("cpu"), cfg=cfg, discount=0.95,
    )


def residual_config():
    return {
        "action_mapping": "symmetric_coarse_speed_v2",
        "delta_d_max": 0.2, "sigma_d": 0.2, "sigma_v": 1.0,
        "path_feature_scales": None,
        "path_feature_clip": 5.0, "include_wm_return_in_state": True,
        "include_wm_cost_in_state": False, "include_wm_uncertainty_in_state": False,
        "wm_return_state_scale": 10.0, "wm_cost_state_scale": 10.0,
        "wm_uncertainty_state_scale": 10.0, "use_world_model_cost": False,
        "use_uncertainty": True, "wm_cost_weight": 1.0,
        "wm_uncertainty_weight": 0.0, "wm_delta_weight": 0.1,
        "freeze_tdmpc2": True,
    }


def planner_cfg(mode):
    return SimpleNamespace(
        planner={"type": mode}, residual_rl=residual_config(), horizon=3, action_dim=2,
        metadrive={"simulator": {"decision_repeat": 5, "physics_world_step_size": 0.02}},
    )


class FakeLattice:
    def __init__(self, invalidate_nonzero=False):
        self.config = dict(FRENET_DEFAULT_CONFIG)
        self.last_lateral_bounds = (-1.0, 1.0)
        self.last_lane_centers = [0.0]
        self.last_lane_safe_half_width = 1.0
        self.last_candidates = []
        self.last_selected_index = 0
        self.invalidate_nonzero = invalidate_nonzero

    def generate_candidates(self, vehicle):
        self.last_candidates = [
            make_path(-0.5, 8.0), make_path(0.0, 10.0),
            make_path(0.5, 12.0, collision=True),
        ]
        return self.last_candidates

    def select_nominal_path(self, candidates):
        return min(range(len(candidates)), key=lambda index: abs(candidates[index].target_d))

    def filter_feasible_paths(self, paths):
        return [path for path in paths if not path.collision]

    def generate_parameterized_paths(self, parameters, horizon):
        paths = []
        for target_d, speed in parameters:
            changed = abs(target_d) > 1e-8 or abs(speed - 10.0) > 1e-8
            paths.append(
                make_path(
                    target_d, speed,
                    collision=self.invalidate_nonzero and changed, horizon=horizon,
                )
            )
        return paths


class FakeController:
    def __init__(self, invalidate_nonzero=False):
        self.planner = FakeLattice(invalidate_nonzero)

    def reset(self):
        pass

    def _track_path(self, vehicle, path):
        action = [np.clip(path.target_d, -1, 1), np.clip(path.target_speed / 20, -1, 1)]
        return action, 1


class FakeAdapter:
    def paths_to_actions(self, paths, vehicle, horizon, control_dt, device, dtype=torch.float32):
        rows = [[[path.target_d, path.target_speed / 20.0]] * horizon for path in paths]
        return torch.tensor(rows, dtype=dtype, device=device)


class FakeEvaluator:
    def encode(self, observation):
        return torch.zeros((1, 2))

    def evaluate_action_sequences(self, z_t, actions):
        score = actions[..., 1].sum(1) - 0.05 * actions[..., 0].abs().sum(1)
        zeros = torch.zeros_like(score)
        return WorldModelEvaluation(score, zeros, zeros, zeros, score)


def make_planner(mode, invalidate_nonzero=False):
    agent = tiny_agent() if "tdmpc" in mode else None
    planner = ResidualSACLatticePlanner(
        planner_cfg(mode), FakeController(invalidate_nonzero), tdmpc_agent=agent,
        action_space=gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
    )
    if agent is not None:
        planner.evaluator = FakeEvaluator()
        planner.path_action_adapter = FakeAdapter()
    return planner, agent


class IntegrationUnitTests(unittest.TestCase):
    def test_original_components_import(self):
        self.assertTrue(callable(TDMPC2))
        self.assertTrue(callable(MetaDriveFrenetController))
        self.assertTrue(callable(SACAgent))

    def test_path_features_are_fixed_and_finite(self):
        features = extract_coarse_path_features(make_path())
        self.assertEqual(features.shape, (len(FEATURE_NAMES),))
        self.assertTrue(torch.isfinite(features).all())

    def test_action_adapter_uses_virtual_future_states(self):
        adapter = LatticeActionAdapter(MetaDriveFrenetController(dict(FRENET_DEFAULT_CONFIG)))
        actions = adapter.paths_to_actions(
            [make_path(-0.2), make_path(0.2)], FakeVehicle(), horizon=3,
            control_dt=0.1, device="cpu",
        )
        self.assertEqual(actions.shape, (2, 3, 2))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.all((actions >= -1.0) & (actions <= 1.0)))

    def test_tracker_accepts_zero_target_speed_as_full_stop(self):
        controller = MetaDriveFrenetController(dict(FRENET_DEFAULT_CONFIG))
        action, _ = controller._track_path(FakeVehicle(), make_path(target_speed=0.0))
        self.assertAlmostEqual(action[1], -1.0)

    def test_world_model_batch_evaluation_is_finite_and_has_no_grad(self):
        agent = tiny_agent()
        evaluator = WorldModelEvaluator(agent, use_cost=True, cost_weight=0.2)
        latent = evaluator.encode(torch.tensor([1.0, 2.0]))
        output = evaluator.evaluate_action_sequences(latent, torch.zeros((4, 3, 2)))
        self.assertEqual(output.score.shape, (4,))
        self.assertTrue(torch.isfinite(output.predicted_return).all())
        self.assertTrue(all(parameter.grad is None for parameter in agent.model.parameters()))

    def test_world_model_appends_discounted_terminal_cost_q(self):
        agent = tiny_agent()
        evaluator = WorldModelEvaluator(
            agent, use_cost=True, use_terminal_cost_q=True, gamma_cost=0.9,
        )
        latent = evaluator.encode(torch.tensor([1.0, 2.0]))
        output = evaluator.evaluate_action_sequences(latent, torch.zeros((1, 3, 2)))
        self.assertAlmostEqual(float(output.predicted_cost[0]), 0.9**3 * 2.0, places=6)

    def test_residual_state_shape(self):
        builder = ResidualStateBuilder(
            include_latent=True, include_wm_return=True, include_wm_cost=True,
            include_wm_uncertainty=True,
        )
        evaluation = WorldModelEvaluation(*(torch.ones(1) for _ in range(5)))
        state = builder.build(
            torch.ones((1, 2)), extract_coarse_path_features(make_path()), evaluation,
        )
        self.assertEqual(state.shape, (2 + len(FEATURE_NAMES) + 3,))
        self.assertTrue(torch.isfinite(state).all())

    def test_sac_actor_and_linear_residual_mapping(self):
        planner, _ = make_planner("lattice_sac")
        context = planner.prepare(torch.zeros(4), FakeVehicle())
        actor = SACAgent(context.residual_state.numel(), 2, device="cpu", hidden_sizes=(16, 16))
        normalized = actor.act(context.residual_state.numpy())
        delta = ResidualActionAdapter.to_physical(normalized, context.bounds)
        self.assertEqual(normalized.shape, (2,))
        self.assertTrue(np.all((-1.0 <= normalized) & (normalized <= 1.0)))
        result = planner.refine(context, normalized, FakeVehicle())
        self.assertAlmostEqual(result.refined_d, result.coarse_d + result.delta_d)
        self.assertAlmostEqual(result.refined_v, result.coarse_v + result.delta_v)
        self.assertTrue(np.isfinite(delta).all())

    def test_residual_mapping_uses_lateral_width_and_coarse_speed(self):
        adapter = ResidualActionAdapter({"delta_d_max": 0.4})
        bounds = adapter.bounds(make_path(target_speed=10.0), None)
        np.testing.assert_allclose(bounds.low, [-0.4, -10.0])
        np.testing.assert_allclose(bounds.high, [0.4, 10.0])
        np.testing.assert_allclose(adapter.to_physical([-1.0, -1.0], bounds), [-0.4, -10.0])
        np.testing.assert_allclose(adapter.to_physical([0.0, 0.0], bounds), [0.0, 0.0])
        np.testing.assert_allclose(adapter.to_physical([1.0, 1.0], bounds), [0.4, 10.0])
        np.testing.assert_allclose(adapter.to_physical([0.25, -0.5], bounds), [0.1, -5.0])

    def test_v4_residual_mapping_uses_one_lane_width(self):
        adapter = ResidualActionAdapter({
            "action_mapping": LANE_WIDTH_ACTION_MAPPING_VERSION,
            "delta_d_max": 0.4,
        })
        lattice = SimpleNamespace(last_lane_width=3.5)
        bounds = adapter.bounds(make_path(target_speed=10.0), lattice)
        np.testing.assert_allclose(bounds.low, [-3.5, -10.0])
        np.testing.assert_allclose(bounds.high, [3.5, 10.0])
        np.testing.assert_allclose(adapter.to_physical([0.5, -1.0], bounds), [1.75, -10.0])

    def test_all_ablation_modes_complete_one_plan(self):
        modes = ("lattice", "lattice_sac", "lattice_tdmpc_sac")
        for mode in modes:
            with self.subTest(mode=mode):
                planner, _ = make_planner(mode)
                context = planner.prepare(torch.zeros(4), FakeVehicle())
                result = planner.refine(context, np.zeros(2, np.float32), FakeVehicle())
                self.assertFalse(result.refined_path.collision)
                self.assertEqual(result.low_level_action.shape, (2,))
                self.assertEqual(context.coarse_candidate_count, 3)

    def test_unsafe_refinement_falls_back_to_zero(self):
        planner, _ = make_planner("lattice_sac", invalidate_nonzero=True)
        context = planner.prepare(torch.zeros(4), FakeVehicle())
        result = planner.refine(context, np.ones(2, np.float32), FakeVehicle())
        self.assertEqual((result.delta_d, result.delta_v), (0.0, 0.0))
        self.assertTrue(result.unsafe_residual)
        self.assertFalse(result.refined_path.collision)

    def test_disabled_safe_fallback_executes_unchecked_refinement(self):
        cfg = planner_cfg("lattice_sac")
        cfg.residual_rl["safe_fallback"] = False
        controller = FakeController(invalidate_nonzero=True)
        planner = ResidualSACLatticePlanner(cfg, controller)
        context = planner.prepare(torch.zeros(4), FakeVehicle())
        result = planner.refine(context, np.ones(2, np.float32), FakeVehicle())
        self.assertNotEqual((result.delta_d, result.delta_v), (0.0, 0.0))
        self.assertTrue(result.refined_path.collision)
        self.assertTrue(result.residual_valid)
        self.assertFalse(result.unsafe_residual)
        self.assertEqual(result.fallback_reason, "")

class MetaDriveSmokeTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("RUN_METADRIVE_SMOKE") == "1", "set RUN_METADRIVE_SMOKE=1")
    def test_lattice_sac_closed_loop(self):
        from omegaconf import OmegaConf
        from common.parser import cfg_to_dataclass
        from env import make_env

        raw = OmegaConf.load("lattice_tdmpc2/config.yaml")
        raw.planner.type = "lattice_sac"
        raw.enable_wandb = False
        raw.metadrive.simulator.use_render = False
        raw.metadrive.simulator.accident_prob = 0.0
        raw.metadrive.simulator.num_scenarios = 1
        raw.metadrive.simulator.horizon = 5
        cfg = cfg_to_dataclass(raw)
        env = make_env(cfg)
        try:
            observation = env.reset(seed=100)
            planner = ResidualSACLatticePlanner(
                cfg, MetaDriveFrenetController(dict(FRENET_DEFAULT_CONFIG)),
                action_space=env.action_space,
            )
            context = planner.prepare(observation, env.unwrapped_metadrive.agent)
            sac = SACAgent(context.residual_state.numel(), 2, device="cpu", hidden_sizes=(16, 16))
            for _ in range(3):
                residual = sac.act(context.residual_state.numpy())
                result = planner.refine(context, residual, env.unwrapped_metadrive.agent)
                observation, reward, done, _ = env.step(result.low_level_action)
                self.assertTrue(np.isfinite(float(reward)))
                self.assertFalse(result.refined_path.collision)
                if done:
                    break
                context = planner.prepare(observation, env.unwrapped_metadrive.agent)
        finally:
            env.close()

    @unittest.skipUnless(
        os.environ.get("RUN_TDMPC2_SMOKE") == "1" and torch.cuda.is_available(),
        "set RUN_TDMPC2_SMOKE=1 on a CUDA machine",
    )
    def test_full_tdmpc_residual_closed_loop(self):
        from omegaconf import OmegaConf
        from common.parser import cfg_to_dataclass
        from env import make_env

        checkpoint = os.environ.get(
            "TDMPC2_CHECKPOINT",
            "logs/metadrive-risk/1/tdmpc2_metadrive_risk/models/final.pt",
        )
        self.assertTrue(os.path.isfile(checkpoint), checkpoint)
        raw = OmegaConf.load("lattice_tdmpc2/config.yaml")
        raw.planner.type = "lattice_tdmpc_sac"
        raw.residual_rl.use_world_model_cost = False
        raw.compile = False
        raw.multitask = False
        raw.tasks = [raw.task]
        raw.bin_size = (raw.vmax - raw.vmin) / (raw.num_bins - 1)
        raw.metadrive.simulator.use_render = False
        raw.metadrive.simulator.accident_prob = 0.0
        raw.metadrive.simulator.num_scenarios = 1
        raw.metadrive.simulator.horizon = 5
        cfg = cfg_to_dataclass(raw)
        env = make_env(cfg)
        try:
            agent = TDMPC2(cfg)
            agent.load(checkpoint)
            planner = ResidualSACLatticePlanner(
                cfg, MetaDriveFrenetController(dict(FRENET_DEFAULT_CONFIG)),
                tdmpc_agent=agent, action_space=env.action_space,
            )
            observation = env.reset(seed=100)
            context = planner.prepare(observation, env.unwrapped_metadrive.agent)
            sac = SACAgent(context.residual_state.numel(), 2, device="cuda")
            result = planner.refine(
                context, sac.act(context.residual_state.numpy()), env.unwrapped_metadrive.agent,
            )
            _, reward, _, _ = env.step(result.low_level_action)
            self.assertTrue(np.isfinite(float(reward)))
            self.assertTrue(np.isfinite(result.refined_return))
            self.assertTrue(all(parameter.grad is None for parameter in agent.model.parameters()))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
