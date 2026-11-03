"""Collect real counterfactual MPPI modes by deterministic reset/history replay."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

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
from mpr_mpc.evaluate_model_suite import _make_agent, _apply_checkpoint_runtime_config
from mpr_mpc.planning.multimodal import MultiModalProposals, MODE_COMPONENT_NAMES
from mpr_mpc.planning.calibration_anchor import make_anchor


def json_finite(value):
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_finite(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def fingerprint(planner, vehicle):
    width, wheelbase, steering = planner._vehicle_geometry(vehicle)
    ctx = planner.safety.context(planner.structured.planner, vehicle,
                               width=width, wheelbase=wheelbase, max_steering=steering)
    actors = sorted(tuple(np.r_[o.position, o.velocity, o.heading, o.length, o.width]) for o in ctx.obstacles)
    return np.r_[ctx.position, np.asarray(vehicle.velocity)[:2], ctx.heading, ctx.speed,
                 ctx.length, ctx.width, ctx.wheelbase, ctx.max_steering, ctx.previous_action,
                 np.asarray(actors).flatten()]


def collect_anchor(agent, env, scene, prefix_steps, args):
    set_seed(args.seed + scene)
    obs = env.reset(seed=scene)
    history = []
    for step in range(prefix_steps):
        action = agent.act(obs, t0=step == 0, eval_mode=True, global_step=agent.planner.residual_start_step)
        history.append(action.cpu().numpy())
        obs, _, done, _ = env.step(action)
        if done:
            raise RuntimeError(f"History terminated in {scene}; shorten anchor prefix.")
    vehicle = env.unwrapped_metadrive.agent
    state = fingerprint(agent.planner, vehicle)
    anchor_obs = obs.cpu().numpy().copy()
    anchor_id, payload = make_anchor(scene, args.seed+scene, history, anchor_obs, state, agent.planner.control_dt)
    plan = agent.planner.plan(obs, vehicle, eval_mode=True, global_step=agent.planner.residual_start_step)
    if plan.stage.use_mppi:
        raise RuntimeError("Collector must not invoke MPPI optimization.")
    baseline, local = plan.refined_actions, agent.planner.local_mppi
    bank = MultiModalProposals(dict(local._sampling_cfg, mode_counts=None), baseline, baseline,
        local.initial_std_vector.expand_as(baseline), num_samples=max(64, 12*args.samples_per_mode),
        num_elites=local.num_elites, temperature=local.temperature, min_std=local.min_std_vector,
        max_std=local.max_std_vector, max_delta=local.max_delta)
    generator = torch.Generator(device=agent.device).manual_seed(args.seed+scene+prefix_steps)
    candidates = bank.sample(generator)
    selected = [int(i) for mode in range(6) for i in torch.nonzero(bank.ids == mode).flatten()[:args.samples_per_mode]]
    if any(int((bank.ids[selected] == mode).sum()) != args.samples_per_mode for mode in range(6)):
        raise RuntimeError("Mode budget insufficient; no labels were synthesized.")
    interventions = [(int(bank.ids[i]), candidates[i]) for i in selected]
    if args.include_coarse:
        interventions.append((-1, plan.coarse_actions))
    records = []
    for mode_id, sequence in interventions:
        set_seed(args.seed+scene)
        replay_obs = env.reset(seed=scene)
        for action in history:
            replay_obs, _, done, _ = env.step(torch.as_tensor(action))
            if done:
                raise RuntimeError("History replay terminated unexpectedly.")
        restored = fingerprint(agent.planner, env.unwrapped_metadrive.agent)
        if (restored.shape != state.shape or not np.allclose(restored, state, atol=1e-5, rtol=0)
                or not np.allclose(replay_obs.cpu().numpy(), anchor_obs, atol=1e-5, rtol=0)):
            raise RuntimeError("Simulator state replay mismatch; refusing incomparable counterfactual labels.")
        actions, costs = [], []
        for action in sequence[:args.horizon]:
            _, _, done, info = env.step(action)
            if "cost" not in info or not np.isfinite(float(info["cost"])):
                raise RuntimeError("Real environment cost is missing/nonfinite.")
            actions.append(action.cpu().numpy())
            costs.append(float(info["cost"]))
            if done:
                break
        records.append(dict(source="metadrive_mode_replay_v2", replay_verified=True, scene_id=scene,
            mode_id=mode_id, anchor_id=anchor_id, trajectory_kind="coarse_lattice" if mode_id == -1 else "mppi_mode",
            observation=anchor_obs, actions=np.asarray(actions), costs=np.asarray(costs)))
    print(f"[calibration] scene={scene} anchor_step={prefix_steps} id={anchor_id[:12]} real_candidates={len(records)}", flush=True)
    return records, anchor_id, payload


def assert_heldout(scenes, metadata, start=None, count=None):
    training = metadata.get("training_scene_range", {})
    if start is None:
        start = training.get("start_seed")
        count = training.get("num_scenarios")
    if start is None or count is None:
        raise ValueError("Old checkpoint lacks training provenance: supply --training-start and --training-count.")
    reserved = set(metadata.get("validation_scene_ids", []))
    overlap = set(scenes) & (set(range(int(start), int(start)+int(count))) - reserved)
    if overlap:
        raise ValueError(f"Calibration scenes overlap the training pool: {sorted(overlap)}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scenarios", nargs="+", type=int, required=True)
    p.add_argument("--training-start", type=int)
    p.add_argument("--training-count", type=int)
    p.add_argument("--prefix-steps", type=int, default=20)
    p.add_argument("--anchor-steps", type=int, nargs="+", help="Several prediction origins per scene; overrides prefix-steps")
    p.add_argument("--include-coarse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--samples-per-mode", type=int, default=4)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(set(args.scenarios)) != len(args.scenarios) or args.samples_per_mode < 1 or args.prefix_steps < 0 or not 0 < args.horizon < 63:
        p.error("Require distinct scenes, positive samples and horizon in [1,62].")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert_heldout(args.scenarios, state.get("metadata", {}), args.training_start, args.training_count)
    raw = OmegaConf.load(ROOT / "config.yaml")
    raw.enable_wandb = raw.save_agent = raw.save_csv = raw.save_video = False
    raw.compile = False
    raw.log_root = str(args.output)
    raw.horizon = args.horizon
    raw.residual_rl.buffer_size = 8
    raw.metadrive.simulator.start_seed = min(args.scenarios)
    raw.metadrive.simulator.num_scenarios = max(args.scenarios)-min(args.scenarios)+1
    _apply_checkpoint_runtime_config(raw, args.checkpoint)
    cfg = parse_cfg(raw)
    set_seed(args.seed)
    env = make_env(cfg)
    records, anchors = [], {}
    try:
        agent = _make_agent(cfg, env)
        agent.load(args.checkpoint)
        agent.eval()
        steps = args.anchor_steps or [args.prefix_steps]
        if len(set(steps)) != len(steps) or min(steps) < 0:
            raise ValueError("Anchor steps must be distinct nonnegative prediction origins.")
        for scene in args.scenarios:
            for step in steps:
                rows, anchor_id, payload = collect_anchor(agent, env, scene, step, args)
                records.extend(rows)
                anchors[anchor_id] = payload
        agent.cost_validator.set_mode_episodes(records)
        agent.validate_cost_model()
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "mode_replays.npz"
        actions = np.zeros((len(records), args.horizon, 2), dtype=np.float32)
        costs = np.zeros((len(records), args.horizon), dtype=np.float32)
        for i, r in enumerate(records):
            actions[i, :len(r["actions"])] = r["actions"]
            costs[i, :len(r["costs"])] = r["costs"]
        np.savez_compressed(path, source="metadrive_mode_replay_v2", anchor_ids=[r["anchor_id"] for r in records],
            observations=np.asarray([r["observation"] for r in records]),
            actions=actions, costs=costs, lengths=[len(r["actions"]) for r in records], modes=[r["mode_id"] for r in records],
            scenes=[r["scene_id"] for r in records], verified=np.ones(len(records), dtype=bool))
        manifest = dict(source_checkpoint=str(args.checkpoint.resolve()), seed=args.seed, scenes=args.scenarios,
            prefix_steps=args.prefix_steps, horizon=args.horizon, samples_per_mode=args.samples_per_mode,
            mode_names=MODE_COMPONENT_NAMES, cost_definition=agent._cost_definition(), anchors=anchors,
            dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), report=agent.cost_validator.mode_report,
            paired_report=agent.cost_validator.paired_report, calibration_schema=2)
        training = state.get("metadata", {}).get("training_scene_range", {})
        manifest["training_provenance"] = dict(start_seed=args.training_start if args.training_start is not None else training.get("start_seed"),
            num_scenarios=args.training_count if args.training_start is not None else training.get("num_scenarios"),
            reserved=state.get("metadata", {}).get("validation_scene_ids", []))
        (args.output / "calibration.json").write_text(json.dumps(json_finite(manifest), indent=2, allow_nan=False))
        # Copy the checkpoint container, changing only calibrated model buffers/metadata.
        state["model"] = agent.model.state_dict()
        state.setdefault("metadata", {})["cost_definition"] = agent._cost_definition()
        state["metadata"]["mode_calibration"] = {k: v for k, v in manifest.items() if k != "report"}
        torch.save(state, args.output / "calibrated.pt")
        print(f"Results: {args.output}; ready={agent.model.cost_ready}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
