from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from lattice.frenet_metadrive import FRENET_DEFAULT_CONFIG, FrenetPath, MetaDriveFrenetController
from mpr_mpc.planning.coordinator import MPRMPCPlanner
from mpr_mpc.planning.kinematic_filter import BatchedKinematicCorridor, CorridorContext
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.time_alignment import align_frenet_path
from mpr_mpc.planning.trajectory_adapter import LatticeActionAdapter
from mpr_mpc.residual_rl import ResidualReplayBuffer, ResidualSACAgent, ResidualStateBuilder


def make_path(horizon=2.0, dt=0.1, target_d=0.2, target_speed=10.0):
    times = np.arange(0.0, horizon + 0.5 * dt, dt)
    s = target_speed * times
    d = target_d * (times / horizon) ** 2
    d_d = np.gradient(d, dt)
    d_dd = np.gradient(d_d, dt)
    d_ddd = np.gradient(d_dd, dt)
    zeros = np.zeros_like(times)
    return FrenetPath(
        t=times.tolist(), d=d.tolist(), d_d=d_d.tolist(), d_dd=d_dd.tolist(),
        d_ddd=d_ddd.tolist(), s=s.tolist(), s_d=np.full_like(times, target_speed).tolist(),
        s_dd=zeros.tolist(), s_ddd=zeros.tolist(), x=s.tolist(), y=d.tolist(),
        yaw=np.arctan2(d_d, np.full_like(d_d, target_speed)).tolist(),
        target_d=target_d, target_speed=target_speed, horizon=horizon,
    )


class FakeVehicle:
    position = np.asarray([0.0, 0.0])
    heading_theta = 0.0
    speed = 5.0


class RecordingEvaluator:
    def __init__(self, constant=False):
        self.calls = []
        self.constant = constant

    def evaluate_mppi_candidates(self, _z, actions, task=None):
        del task
        self.calls.append(actions.detach().clone())
        if self.constant:
            values = torch.zeros(len(actions), device=actions.device)
        else:
            values = actions[:, :, 0].sum(dim=1)
        return SimpleNamespace(total_value=values, terminal_q_std=torch.zeros_like(values))


class FixedValueEvaluator:
    def __init__(self, values):
        self.values = torch.as_tensor(values, dtype=torch.float32)

    def evaluate_mppi_candidates(self, _z, actions, task=None):
        del task
        values = self.values.to(actions.device)
        return SimpleNamespace(total_value=values, terminal_q_std=torch.zeros_like(values))


def mppi_cfg(**updates):
    cfg = {
        "num_samples": 24,
        "num_elites": 6,
        "iterations": 3,
        "initial_std": [0.025, 0.10],
        "min_std": [0.005, 0.02],
        "max_std": [0.05, 0.20],
        "max_delta": [0.05, 0.20],
        "temperature": 0.5,
        "proposal_init": "structured_baseline",
        "deviation_coef": 0.0,
        "smoothness_coef": 0.0,
        "min_improvement_abs": 0.0,
        "min_improvement_ratio": 0.0,
        "eval_seed": 7,
    }
    cfg.update(updates)
    return cfg


class MPRMPCUnitTests(unittest.TestCase):
    def test_action_adapter_is_h_plus_one(self):
        controller = MetaDriveFrenetController(dict(FRENET_DEFAULT_CONFIG))
        actions = LatticeActionAdapter(controller).path_to_actions(
            make_path(), FakeVehicle(), required_steps=11, control_dt=0.1, device="cpu"
        )
        self.assertEqual(actions.shape, (11, 2))

    def test_time_alignment_with_different_dt(self):
        aligned = align_frenet_path(make_path(dt=0.1), required_steps=6, control_dt=0.2)
        np.testing.assert_allclose(aligned.times, np.arange(6) * 0.2)

    def test_1_residual_sac_action_is_two_dimensional_and_normalized(self):
        agent = ResidualSACAgent(518, {"hidden_dims": [16], "lr": 1e-3}, device="cpu")
        output = agent.select_action(torch.zeros((4, 518)), deterministic=True)
        self.assertEqual(output.action.shape, (4, 2))
        self.assertEqual(output.log_std.shape, (4, 2))
        self.assertTrue(torch.all(output.action.abs() <= 1.0 + 1e-6))

    def test_2_residual_never_changes_horizon(self):
        fake = SimpleNamespace(clamp_parameters=lambda values: np.asarray(values).copy())
        coarse = np.asarray([0.1, 10.0, 2.75], dtype=np.float32)
        refined = MPRMPCPlanner.apply_residual_parameters(fake, coarse, [0.4, -2.0])
        np.testing.assert_allclose(refined, [0.5, 8.0, 2.75])

    def test_dynamic_bounds_are_lane_width_and_half_speed(self):
        fake = SimpleNamespace(
            structured=SimpleNamespace(planner=SimpleNamespace(last_lane_width=3.6)),
            device=torch.device("cpu"),
        )
        bounds = MPRMPCPlanner.residual_bounds(fake, make_path(target_speed=10.0))
        torch.testing.assert_close(bounds, torch.tensor([[3.6, 5.0]]))

    def test_3_vector_std_broadcast(self):
        planner = LocalMPPI(RecordingEvaluator(), mppi_cfg(), horizon=3, action_dim=2, device="cpu")
        result = planner.plan(torch.zeros((1, 2)), torch.zeros((4, 2)), eval_mode=True)
        expected = torch.tensor([0.025, 0.10]).expand(4, -1)
        torch.testing.assert_close(result.initial_std, expected)

    def test_4_trust_region_holds_across_all_iterations(self):
        evaluator = RecordingEvaluator()
        planner = LocalMPPI(evaluator, mppi_cfg(initial_std=[0.05, 0.20]), horizon=3, action_dim=2, device="cpu")
        baseline = torch.linspace(-0.4, 0.4, 8).reshape(4, 2)
        planner.plan(torch.zeros((1, 2)), baseline, eval_mode=True)
        for candidates in evaluator.calls:
            delta = (candidates - baseline).abs()
            self.assertLessEqual(float(delta[..., 0].max()), 0.050001)
            self.assertLessEqual(float(delta[..., 1].max()), 0.200001)

    def test_5_baseline_is_candidate_zero_every_iteration(self):
        evaluator = RecordingEvaluator()
        planner = LocalMPPI(evaluator, mppi_cfg(), horizon=3, action_dim=2, device="cpu")
        baseline = torch.linspace(-0.2, 0.2, 8).reshape(4, 2)
        planner.plan(torch.zeros((1, 2)), baseline, eval_mode=True)
        self.assertEqual(len(evaluator.calls), 3)
        for candidates in evaluator.calls:
            torch.testing.assert_close(candidates[0], baseline)

    def test_6_training_has_no_post_selection_noise(self):
        planner = LocalMPPI(RecordingEvaluator(), mppi_cfg(), horizon=3, action_dim=2, device="cpu")
        result = planner.plan(torch.zeros((1, 2)), torch.zeros((4, 2)), eval_mode=False)
        torch.testing.assert_close(result.action, result.selected_sequence[0])

    def test_7_evaluation_is_deterministic(self):
        planner = LocalMPPI(RecordingEvaluator(), mppi_cfg(), horizon=3, action_dim=2, device="cpu")
        baseline = torch.zeros((4, 2))
        first = planner.plan(torch.zeros((1, 2)), baseline, eval_mode=True).action
        second = planner.plan(torch.zeros((1, 2)), baseline, eval_mode=True).action
        torch.testing.assert_close(first, second)

    def test_8_residual_state_builder_uses_latent_path_return_and_feasibility(self):
        builder = ResidualStateBuilder(
            {"include_wm_cost_in_state": False, "wm_return_scale": 50.0},
            latent_dim=4,
            device="cpu",
        )
        state = builder.build(
            latent=torch.ones((1, 4)),
            coarse_path=make_path(),
            consequence=SimpleNamespace(total_value=torch.tensor([25.0])),
            feasible_ratio=0.5,
            used_fallback=True,
        )
        self.assertEqual(state.shape, (builder.state_dim,))
        self.assertTrue(torch.isfinite(state).all())
        torch.testing.assert_close(state[-2:], torch.tensor([0.5, 1.0]))

    def test_8b_residual_replay_stores_terminal_without_bootstrap(self):
        replay = ResidualReplayBuffer(8, state_dim=3, action_dim=2)
        replay.add([1, 2, 3], [0.1, -0.2], 1.0, 0.5, [9, 9, 9], True)
        batch = replay.sample(1, device="cpu", gamma=0.99)
        self.assertEqual(float(batch["done"][0, 0]), 1.0)
        torch.testing.assert_close(batch["discount"], torch.full((1, 1), 0.99))

    def test_9_mppi_improvement_gate_falls_back_to_baseline(self):
        planner = LocalMPPI(
            RecordingEvaluator(constant=True),
            mppi_cfg(min_improvement_abs=1.0),
            horizon=3, action_dim=2, device="cpu",
        )
        baseline = torch.linspace(-0.2, 0.2, 8).reshape(4, 2)
        result = planner.plan(torch.zeros((1, 2)), baseline, eval_mode=True)
        torch.testing.assert_close(result.selected_sequence, baseline)
        self.assertTrue(result.baseline_selected)

    def test_10_corridor_rejects_large_lateral_rollout(self):
        corridor = BatchedKinematicCorridor(
            {"lateral_corridor_ratio": 0.1, "lateral_safety_margin": 0.0},
            control_dt=0.1,
            device="cpu",
        )
        baseline = torch.zeros((11, 2))
        candidates = torch.stack([baseline, torch.tensor([[1.0, 1.0]]).expand(11, -1)])
        context = CorridorContext(
            torch.zeros(10), 1.0, -5.0, 5.0, 15.0, 1.8, 2.5, 0.7
        )
        result = corridor.check(candidates, baseline, context)
        self.assertTrue(bool(result.valid[0]))
        self.assertFalse(bool(result.valid[1]))

    def test_11_residual_sac_update_trains_reward_and_cost_critics(self):
        agent = ResidualSACAgent(8, {"hidden_dims": [16], "lr": 1e-3}, device="cpu")
        batch = {
            "state": torch.randn((6, 8)),
            "action": torch.randn((6, 2)).clamp(-1.0, 1.0),
            "reward": torch.randn((6, 1)),
            "cost": torch.rand((6, 1)),
            "next_state": torch.randn((6, 8)),
            "done": torch.zeros((6, 1)),
            "discount": torch.full((6, 1), 0.99),
        }
        info = agent.update(batch)
        self.assertIn("sac/loss_reward_critic", info)
        self.assertIn("sac/loss_cost_critic", info)
        self.assertEqual(agent.update_steps, 1)

    def test_12_stage_switches_at_global_steps(self):
        fake = SimpleNamespace(
            residual_start_step=20000,
            mppi_start_step=50000,
            allow_mppi=True,
            allow_residual=True,
        )
        stage_a = MPRMPCPlanner.stage_for_step(fake, 10000)
        stage_b = MPRMPCPlanner.stage_for_step(fake, 30000)
        stage_c = MPRMPCPlanner.stage_for_step(fake, 60000)
        self.assertEqual((stage_a.use_residual, stage_a.use_mppi), (False, False))
        self.assertEqual((stage_b.use_residual, stage_b.use_mppi), (True, False))
        self.assertEqual((stage_c.use_residual, stage_c.use_mppi), (True, True))

    def test_a_vehicle_width_is_applied_to_raw_road_bounds(self):
        corridor = BatchedKinematicCorridor(
            {"lateral_corridor_ratio": 10.0, "lateral_safety_margin": 0.2},
            control_dt=0.1,
            device="cpu",
        )
        actions = torch.zeros((2, 3, 2))
        baseline = torch.zeros((3, 2))
        positive = CorridorContext(
            torch.full((2,), 2.31), 3.5, -3.5, 3.5, 0.0, 2.0, 2.5, 0.7
        )
        negative = CorridorContext(
            torch.full((2,), -2.31), 3.5, -3.5, 3.5, 0.0, 2.0, 2.5, 0.7
        )
        self.assertFalse(bool(corridor.check(actions, baseline, positive).valid[0]))
        self.assertFalse(bool(corridor.check(actions, baseline, negative).valid[0]))

    def test_b_vehicle_geometry_prefers_live_metadrive_parameters(self):
        planner = SimpleNamespace(mppi_cfg={})
        vehicle = SimpleNamespace(
            FRONT_WHEELBASE=1.0,
            REAR_WHEELBASE=1.5,
            max_steering=40.0,
            WIDTH=1.8,
        )
        width, wheelbase, steering = MPRMPCPlanner._vehicle_geometry(planner, vehicle)
        self.assertAlmostEqual(width, 1.8)
        self.assertAlmostEqual(wheelbase, 2.5)
        self.assertAlmostEqual(steering, 0.6981317, places=6)

    def test_c_mppi_separates_raw_wm_value_and_planner_score(self):
        planner = LocalMPPI(
            FixedValueEvaluator([10.0, 12.0, 11.0]),
            mppi_cfg(
                num_samples=3,
                num_elites=3,
                iterations=1,
                deviation_coef=5.0,
            ),
            horizon=1,
            action_dim=2,
            device="cpu",
        )
        baseline = torch.zeros((2, 2))
        fixed = torch.stack(
            [baseline, torch.tensor([[0.05, 0.0], [0.05, 0.0]]),
             torch.tensor([[0.01, 0.0], [0.01, 0.0]])]
        )
        planner._candidate_batch = lambda mean, std, base, generator=None: fixed.clone()
        result = planner.plan(torch.zeros((1, 2)), baseline, eval_mode=True)
        self.assertAlmostEqual(float(result.selected_wm_value), 11.0, places=5)
        self.assertAlmostEqual(float(result.selected_score), 10.8, places=5)
        self.assertAlmostEqual(float(result.wm_value_gain), 1.0, places=5)
        self.assertAlmostEqual(float(result.planner_score_gain), 0.8, places=5)

    def test_d_improvement_gate_compares_score_with_score(self):
        def run(selected_score):
            planner = LocalMPPI(
                FixedValueEvaluator([10.0, selected_score]),
                mppi_cfg(
                    num_samples=2,
                    num_elites=2,
                    iterations=1,
                    min_improvement_abs=1.0,
                ),
                horizon=1,
                action_dim=2,
                device="cpu",
            )
            baseline = torch.zeros((2, 2))
            fixed = torch.stack([baseline, torch.full((2, 2), 0.01)])
            planner._candidate_batch = lambda mean, std, base, generator=None: fixed.clone()
            return planner.plan(torch.zeros((1, 2)), baseline, eval_mode=True)

        self.assertTrue(run(10.5).baseline_selected)
        accepted = run(12.0)
        self.assertFalse(accepted.baseline_selected)
        self.assertAlmostEqual(float(accepted.planner_score_gain), 2.0, places=5)

    def test_e_terminal_q_action_is_not_integrated_by_corridor(self):
        corridor = BatchedKinematicCorridor({}, control_dt=0.1, device="cpu")
        baseline = torch.zeros((4, 2))  # H=3 explicit transitions + one terminal-Q action
        candidates = torch.stack([baseline, baseline.clone()])
        candidates[1, -1] = torch.tensor([1.0, 1.0])
        context = CorridorContext(
            torch.zeros(3), 3.5, -10.0, 10.0, 10.0, 1.8, 2.5, 0.7
        )
        result = corridor.check(candidates, baseline, context)
        torch.testing.assert_close(result.lateral[0], result.lateral[1])
        self.assertEqual(result.lateral.shape, (2, 3))

    def test_f_selected_mppi_visualization_ignores_terminal_q_action(self):
        corridor = BatchedKinematicCorridor({}, control_dt=0.1, device="cpu")
        planner = SimpleNamespace(
            horizon=3,
            control_dt=0.1,
            local_mppi=SimpleNamespace(corridor=corridor),
        )
        baseline = torch.zeros((4, 2))
        selected = baseline.clone()
        selected[-1] = torch.tensor([1.0, 1.0])
        context = CorridorContext(
            torch.zeros(3), 3.5, -10.0, 10.0, 5.0, 1.8, 2.5, 0.7
        )
        path = make_path(target_speed=5.0)
        final_xy = MPRMPCPlanner._selected_sequence_xy(
            planner, path, baseline, selected, context
        )
        aligned = align_frenet_path(path, required_steps=4, control_dt=0.1)
        np.testing.assert_allclose(final_xy, aligned.positions)


if __name__ == "__main__":
    unittest.main()
