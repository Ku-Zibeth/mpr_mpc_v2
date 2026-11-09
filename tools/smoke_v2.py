"""Small real-simulator check; never writes or loads trained checkpoints."""

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap

bootstrap()
from common.parser import parse_cfg
from common.seed import set_seed
from env import make_env
from mpr_mpc.evaluate_model_suite import _make_agent


def main():
    with tempfile.TemporaryDirectory(prefix="mpr_v2_smoke_") as temporary:
        raw = OmegaConf.load(ROOT / "config.yaml")
        raw.log_root = temporary
        raw.enable_wandb = raw.save_agent = raw.save_csv = raw.save_video = False
        raw.batch_size = 4
        raw.residual_rl.buffer_size = 16
        raw.residual_rl.risk_features = ["cumulative_cost", "min_ttc", "min_clearance", "boundary_margin"]
        raw.metadrive.simulator.map = "S"
        raw.metadrive.simulator.start_seed = 100
        raw.metadrive.simulator.num_scenarios = 2
        raw.metadrive.simulator.traffic_density = 0.1
        raw.metadrive.simulator.accident_prob = 0.0
        cfg = parse_cfg(raw)
        set_seed(1)
        env = make_env(cfg)
        try:
            agent = _make_agent(cfg, env)
            observation = env.reset(seed=100)
            observations, actions, rewards, costs = [observation.clone()], [], [], []
            for _ in range(20):
                action = torch.tensor([0., 1.])
                observation, reward, done, info = env.step(action)
                assert not done
                observations.append(observation.clone())
                actions.append(action)
                rewards.append(float(reward))
                costs.append(float(info["cost"]))
            vehicle = env.unwrapped_metadrive.agent
            width, wheelbase, steering = agent.planner._vehicle_geometry(vehicle)
            context = agent.planner.safety.context(agent.planner.structured.planner, vehicle,
                    width=width, wheelbase=wheelbase, max_steering=steering)
            heading = float(vehicle.heading_theta)
            test_action = torch.tensor([.2, 0.])
            sequence = test_action.expand(1, cfg.horizon + 1, 2).to(agent.device)
            predicted, _, _ = agent.planner.safety.rollout(sequence, context)
            observation, _, done, _ = env.step(test_action)
            yaw_change = float(np.arctan2(np.sin(vehicle.heading_theta-heading), np.cos(vehicle.heading_theta-heading)))
            assert yaw_change > 0, "Simulator steering sign differs from bicycle model."
            first_prediction = predicted[0, agent.planner.safety.substeps].cpu().numpy()
            position_error = float(np.linalg.norm(first_prediction - vehicle.position))
            assert position_error < 1, "One-step kinematic position error exceeds smoke-test tolerance."
            horizon_errors = [position_error]
            for t in range(2, cfg.horizon + 1):
                observation, _, done, _ = env.step(test_action)
                assert not done
                point = predicted[0, t * agent.planner.safety.substeps].cpu().numpy()
                horizon_errors.append(float(np.linalg.norm(point - vehicle.position)))
            assert max(horizon_errors) < 1, "Gentle-turn one-second rollout error exceeds tolerance."
            h, b, device = cfg.horizon, cfg.batch_size, agent.device
            obs = torch.stack(observations[:h+1])[:, None].expand(-1, b, -1).to(device)
            act = torch.stack(actions[:h])[:, None].expand(-1, b, -1).to(device)
            rew = torch.tensor(rewards[:h], device=device)[:, None, None].expand(-1, b, 1)
            cost = torch.tensor(costs[:h], device=device)[:, None, None].expand(-1, b, 1)
            training = agent.tdmpc_agent._update(obs, act, rew, torch.zeros_like(cost), cost=cost)
            assert training["cost_grad_norm"] > 0 and training["cost_supervised_samples"] == h*b
            # Even a step beyond the deployment threshold must not enable training MPPI.
            training_plan = agent.planner.plan(observation, vehicle, global_step=50000, t0=True, eval_mode=False)
            assert not training_plan.stage.use_mppi
            plan = agent.planner.plan(observation, vehicle, global_step=50000, t0=True, eval_mode=True)
            assert plan.action.shape == (2,) and torch.isfinite(plan.action).all()
            assert plan.metrics["mppi/cost_enabled"] == 1
            assert np.isfinite(plan.metrics["mppi/predicted_cost"])
            if not (plan.metrics.get("mppi/lattice_fallback") or plan.metrics.get("mppi/emergency_fallback")):
                assert torch.all((plan.debug["selected_mppi_actions"]-plan.refined_actions.cpu()).abs()
                                 <= agent.planner.local_mppi.max_delta.cpu()+1e-6)
            for name, kind in (("wm", "world_model"), ("sac", "world_model_residual_rl"), ("full", "full_mpr_mpc")):
                path = Path(temporary) / f"{name}.pt"
                agent.save_milestone(path, milestone_type=kind)
                agent.load_milestone(path)
            print("V2_SMOKE_RESULT", json.dumps({
                "runtime": str(sys.modules["tdmpc2"].__file__),
                "residual_state_dim": agent.planner.residual_state_dim,
                "positive_steering_yaw_change": yaw_change,
                "one_step_position_error_m": position_error,
                "one_second_max_position_error_m": max(horizon_errors),
                "cost_training_loss": float(training["cost_loss"]),
                "cost_grad_norm": float(training["cost_grad_norm"]),
                "cost_supervised_samples": float(training["cost_supervised_samples"]),
                "risk_feature_status": agent.planner.residual_state_builder.last_risk_status,
                "cost_ready_after_one_update": agent.model.cost_ready,
                "training_mppi_enabled": training_plan.stage.use_mppi,
                "plan_action": plan.action.tolist(),
                "native_cost_diagnostics": {key: value for key, value in plan.metrics.items() if key.startswith("mppi/")},
                "safety_diagnostics": {key: value for key, value in plan.metrics.items() if
                    any(word in key for word in ("reject", "boundary_margin", "no_safe", "_safe", "min_clearance", "min_ttc"))},
                "geometry": {"position": vehicle.position.tolist(), "road_bounds": context.road.bounds,
                    "lane_coordinates": vehicle.lane.local_coordinates(vehicle.position),
                    "lane_polygon_bounds": vehicle.lane.shapely_polygon.bounds,
                    "heading": vehicle.heading_theta, "length": context.length, "width": context.width},
                "planning_ms": plan.metrics["mpr/planning_ms"],
                "milestone_roundtrips": 3,
            }))
        finally:
            env.close()


if __name__ == "__main__":
    main()
