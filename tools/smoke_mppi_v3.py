"""Read-only CUDA/MetaDrive inference smoke, not an episode success benchmark."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
from mpr_mpc.evaluate_model_suite import _make_cfg, _make_agent, _apply_evaluation_stack
from mpr_mpc.evaluation_diagnostics import prediction_outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--scenarios", nargs="+", type=int, default=[1714, 539])
    parser.add_argument("--recovery-probe", action="store_true",
                        help="One artificial unsafe-baseline probe in a real live state, not a success benchmark")
    args = parser.parse_args()
    if args.steps <= 0 or args.output.exists():
        parser.error("Require positive steps and a fresh output file.")
    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT)):
        raw = compose(config_name="config", overrides=["+experiment=E5"])
    raw.residual_rl.buffer_size = 8
    raw.enable_wandb = raw.save_agent = raw.save_video = False
    effective, cfg = _make_cfg(raw, args.checkpoint.parent.parent.name, args.checkpoint, args.scenarios)
    cfg.residual_rl["buffer_size"] = 8
    env = make_env(cfg)
    records, recovery_probe = [], None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agent = _make_agent(cfg, env)
            loaded = agent.load(args.checkpoint)
        agent.eval()
        agent.set_global_step(max(agent.global_env_step, agent.planner.mppi_start_step))
        for scenario in args.scenarios:
            for stack in ("world_model_residual_rl", "lattice_mppi", "full_mpr_mpc"):
                _apply_evaluation_stack(agent.planner, stack, loaded)
                seed = (int(cfg.seed)+scenario) % 2**32
                set_seed(seed)
                observation = env.reset(seed=scenario)
                agent.planner.reset(seed=seed)
                for step in range(args.steps):
                    action = agent.act(observation, t0=step == 0, eval_mode=True)
                    result = agent.last_plan
                    baseline = result.refined_actions[0].cpu()
                    delta = action-baseline
                    assert torch.isfinite(action).all() and (action.abs() <= 1).all()
                    assert result.stage.use_mppi == (stack != "world_model_residual_rl")
                    assert result.stage.use_residual == (stack != "lattice_mppi")
                    if result.stage.use_mppi:
                        assert np.isclose(result.metrics["mppi/first_action_delta_norm"], float(delta.norm()), rtol=1e-5, atol=1e-7)
                    row = dict(scenario=scenario, stack=stack, step=step+1,
                               executed_action=action.tolist(), baseline_action=baseline.tolist(),
                               actual_first_delta=delta.tolist(), **result.metrics)
                    records.append(row)
                    observation, _, done, info = env.step(action)
                    row.update(prediction_outcomes(info, result))
                    if done:
                        break
        if args.recovery_probe:
            scene = args.scenarios[-1]
            _apply_evaluation_stack(agent.planner, "world_model", loaded)
            observation = env.reset(seed=scene)
            for step in range(20):
                action = agent.act(observation, t0=step == 0, eval_mode=True)
                observation, _, done, _ = env.step(action)
                if done:
                    raise RuntimeError("Recovery probe prefix ended before the shared live state.")
            planner, vehicle = agent.planner, env.unwrapped_metadrive.agent
            context = planner.prepare(observation, vehicle, need_world_model_features=True,
                                      need_mppi_safety=True, need_residual_state=False)
            unsafe = torch.ones_like(context.coarse_actions)
            result = planner.local_mppi.plan(context.latent, unsafe, eval_mode=True,
                safety_context=context.mppi_safety_context, lattice_actions=context.coarse_actions,
                lattice_corridor_context=planner._corridor_context(context.selection.coarse_path, vehicle))
            assert result.metrics["baseline_safety_valid"] == 0 and result.metrics["recovery_search"] == 1
            assert result.metrics["lattice_baseline_safety_valid"] == 1
            assert result.metrics["selected_safety_valid"] == 1 and result.metrics["emergency_fallback"] == 0
            assert result.planning_mode == "recovery_lattice"
            _, _, _, info = env.step(result.action.cpu())
            recovery_probe = dict(scenario=scene, injected_baseline="all steering=1/throttle=1; not a SAC output",
                planning_mode_name=result.planning_mode, executed_action=result.action.cpu().tolist(),
                actual_collision=any(bool(info.get(k, False)) for k in ("crash", "crash_vehicle", "crash_object")),
                actual_offroad=bool(info.get("out_of_road", False)),
                vehicle_parameters={k: vehicle.config.get(k) for k in
                    ("max_engine_force", "max_brake_force", "max_steering", "mass", "wheel_friction")},
                **result.metrics)
        assert agent.planner.residual_state_dim == 535
    finally:
        env.close()
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checksum
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checksum,
        note="Partial inference/physics smoke only; no training and no episode success claim.",
        records=records, recovery_probe=recovery_probe), indent=2))
    active = [row for row in records if row["stack"] != "world_model_residual_rl"]
    print(json.dumps(dict(steps=len(records), mppi_steps=len(active),
        nonzero_first_corrections=sum(row["mppi/first_action_delta_norm"] > 1e-6 for row in active),
        safety_fallbacks=sum(row["mppi/emergency_fallback"] for row in active),
        unverified_fallbacks=sum(row["mppi/safety_fallback_unverified"] for row in active),
        mean_full_planning_ms=float(np.mean([row["mpr/total_planning_ms"] for row in active])),
        mean_safety_ms=float(np.mean([row["mppi/safety_check_ms"] for row in active])),
        residual_state_dim=535, checkpoint_unchanged=True,
        recovery_probe_mode=recovery_probe["planning_mode_name"] if recovery_probe else None), indent=2), flush=True)


if __name__ == "__main__":
    main()
