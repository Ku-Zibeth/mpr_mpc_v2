"""Read-only bounded recovery smoke from recorded V3.1.1 stopped states."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import cProfile
import pstats
import gzip

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite
from mpr_mpc.planning.safety import _corners
from mpr_mpc.planning.mppi_safety import motion_response_errors
from shapely.geometry import Polygon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "logs/benchmark/mpr_mpc_v311_sac_vs_sac_mppi_fixed100_20261109_202823")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--snapshots", nargs="+", default=["1714:250", "559:600"])
    parser.add_argument("--profile-first", action="store_true")
    parser.add_argument("--disable-anti-stall", action="store_true", help="Bounded recovery ablation only")
    parser.add_argument("--save-candidates", action="store_true", help="Scalar candidate ledger, gzip JSONL; no trajectory tensors")
    args = parser.parse_args()
    if args.output.exists() or not 0 < args.steps <= 200:
        parser.error("Require a fresh output path and 1..200 smoke steps.")
    if args.save_candidates and args.output.with_suffix(".candidates.jsonl.gz").exists():
        parser.error("Candidate evidence path already exists.")
    snapshots = dict(tuple(map(int, s.split(":"))) for s in args.snapshots)
    records = {seed: [] for seed in snapshots}
    for line in next((args.run / "steps").glob("*_full_mpr_mpc_seed1.jsonl")).open():
        row = json.loads(line); seed = row["scenario"]
        if seed in snapshots and row["episode_step"] <= snapshots[seed]:
            records[seed].append(row)
        if all(len(records[s]) >= step for s, step in snapshots.items()):
            break
    raw = OmegaConf.load(args.run / "config_D_lattice_sac_mppi_seed1.yaml")
    raw.mpr_mpc.mppi = OmegaConf.load(ROOT / "config.yaml").mpr_mpc.mppi
    raw.restore_mppi_runtime = False
    if args.disable_anti_stall:
        raw.mpr_mpc.mppi.recovery.anti_stall = False
    raw.enable_wandb = raw.save_agent = raw.save_video = False
    raw.residual_rl.buffer_size = 8
    checkpoint = Path(json.loads((args.run / "manifest.json").read_text())["checkpoint"])
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    scenarios = json.loads((args.run / "scenarios.json").read_text())["scenarios"]
    _, cfg = suite._make_cfg(raw, checkpoint.parent.parent.name, checkpoint, scenarios)
    cfg.residual_rl["buffer_size"] = 8
    torch.set_float32_matmul_precision("high"); set_seed(cfg.seed)
    env = make_env(cfg)
    results = []
    candidate_stream = None
    if args.save_candidates:
        path = args.output.with_suffix(".candidates.jsonl.gz")
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate_stream = gzip.open(path, "wt", encoding="utf-8")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agent = suite._make_agent(cfg, env); loaded = agent.load(checkpoint)
        agent.eval(); suite._apply_evaluation_stack(agent.planner, "full_mpr_mpc", loaded)
        agent.set_global_step(max(agent.global_env_step, agent.planner.mppi_start_step))
        for seed, step in snapshots.items():
            episode_seed = (int(cfg.seed)+seed) % 2**32
            set_seed(episode_seed); agent.planner.reset(seed=episode_seed)
            obs = env.reset(seed=seed); replay_error = 0.; info = {}
            for row in records[seed][:-1]:
                speed = float(np.linalg.norm(env.unwrapped_metadrive.agent.velocity))
                replay_error = max(replay_error, abs(speed-row["mpr/actual_speed_mps"]))
                obs, _, done, info = env.step(np.asarray(row["executed_action"], dtype=np.float32))
                if done:
                    raise RuntimeError("Recorded-action replay terminated early.")
            if replay_error > 1e-4:
                raise RuntimeError(f"Replay state mismatch in {seed}: speed error={replay_error}")
            start_route = float(info.get("route_completion", 0.))
            initial_contacts = {key: bool(info.get(key, False)) for key in
                ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk", "out_of_road")}
            smoke = []
            lattice_audit = None
            geometry_audit = None
            for i in range(args.steps):
                profile = cProfile.Profile() if args.profile_first and i == 0 else None
                if profile: profile.enable()
                action = agent.act(obs, t0=False, eval_mode=True)
                if profile:
                    profile.disable()
                    stream = io.StringIO()
                    pstats.Stats(profile, stream=stream).sort_stats("cumtime").print_stats(30)
                    profile_path = args.output.with_suffix(f".{seed}.profile.txt")
                    if profile_path.exists():
                        raise FileExistsError(profile_path)
                    profile_path.parent.mkdir(parents=True, exist_ok=True)
                    profile_path.write_text(stream.getvalue())
                    print(stream.getvalue(), flush=True)
                result, context = agent.last_plan, agent.planner.last_context.mppi_safety_context
                if candidate_stream is not None:
                    candidate_stream.write(json.dumps(dict(scenario=seed, step=i+1,
                        candidates=result.debug.get("mppi_candidates", [])), allow_nan=False)+"\n")
                if i == 0:
                    f = agent.planner.local_mppi.candidate_safety
                    r = f.check(result.coarse_actions[None].cpu(), context)
                    lattice_audit = dict(near_valid=bool(r.near_valid[0]), full_valid=bool(r.full_valid[0]),
                        admissible=bool(r.valid[0]), unrecoverable=bool(r.unrecoverable[0]))
                    footprint = _corners(torch.tensor(context.position)[None, None], torch.tensor([[context.heading]]),
                                         context.length, context.width)[0, 0].numpy()
                    vehicle = env.unwrapped_metadrive.agent
                    geometry_audit = dict(length=context.length, width=context.width, wheelbase=context.wheelbase,
                        max_steering_rad=context.max_steering, control_dt=agent.planner.control_dt,
                        footprint_hausdorff_m=Polygon(footprint).hausdorff_distance(Polygon(vehicle.bounding_box)))
                    geometry_audit.update(longitudinal_parameter_source=context.longitudinal_parameter_source,
                        dynamics_parameters=context.dynamics_parameters)
                prediction, predicted_heading, predicted_speed = agent.planner.local_mppi.candidate_safety._rollout(
                    result.debug["selected_mppi_actions"][None, :2], context)
                obs, _, done, info = env.step(action)
                executed = bool(info.get("action_executed", True))
                result.metrics.update({f"mppi/{key}": value for key, value in motion_response_errors(
                    result.debug.get("motion_prediction"), env.unwrapped_metadrive.agent, info).items()})
                crash = any(bool(info.get(k, False)) for k in
                    ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk"))
                offroad = bool(info.get("out_of_road", False))
                smoke.append(dict(step=i+1, mode=result.planning_mode, action=action.tolist(),
                    action_executed=executed, safety_failure=bool(info.get("safety_failure", False)),
                    safety_failure_reason=info.get("safety_failure_reason", ""),
                    safety_certificate=result.debug.get("safety_certificate"),
                    heading_before=context.heading, predicted_heading=float(predicted_heading[0, -1]),
                    actual_heading=float(env.unwrapped_metadrive.agent.heading_theta),
                    lateral_before=float(context.road_reference.project(context.position)[1]),
                    predicted_lateral=float(context.road_reference.project(prediction[0, -1].numpy())[1]),
                    actual_lateral=float(context.road_reference.project(env.unwrapped_metadrive.agent.position)[1]),
                    contacts={key: bool(info.get(key, False)) for key in initial_contacts},
                    speed=float(np.linalg.norm(env.unwrapped_metadrive.agent.velocity)),
                    route=float(info.get("route_completion", 0.)), crash=crash, offroad=offroad,
                    emergency_streak=result.metrics["mppi/emergency_streak"],
                    baseline_near=result.metrics["mppi/baseline_near_valid"], baseline_full=result.metrics["mppi/baseline_full_valid"],
                    baseline_near_offroad=result.metrics["mppi/baseline_near_offroad"],
                    baseline_future_hard=result.metrics["mppi/baseline_future_hard_reject"],
                    emergency_reason=result.metrics["mppi/emergency_reason"],
                    one_step_position_error_m=float(np.linalg.norm(prediction[0, -1].numpy()-env.unwrapped_metadrive.agent.position)) if executed else float("nan"),
                    one_step_speed_error_mps=abs(float(predicted_speed[0, -1])-float(np.linalg.norm(env.unwrapped_metadrive.agent.velocity))) if executed else float("nan"),
                    risk=result.metrics["mppi/long_risk_score"], planning_ms=result.metrics["mpr/total_planning_ms"]))
                smoke[-1]["diagnostics"] = {k: float(v) for k, v in result.metrics.items() if k.startswith("mppi/")
                    and not k.endswith("cpu_wall_ms")}
                if i == 0 or (i+1) % 20 == 0:
                    print(f"[smoke] seed={seed} step={i+1} speed={smoke[-1]['speed']:.3f} route={smoke[-1]['route']:.4f} mode={result.planning_mode}", flush=True)
                if done or crash or offroad:
                    break
            results.append(dict(scenario=seed, snapshot_before_step=step, replay_max_speed_error=replay_error,
                initial_contacts=initial_contacts,
                start_route=start_route, end_route=smoke[-1]["route"], initial_lattice=lattice_audit, records=smoke))
            results[-1]["geometry_audit"] = geometry_audit
            def distribution(values):
                values = [float(v) for v in values if np.isfinite(v)]
                return dict(mean=float(np.mean(values)), **{f"p{q}": float(np.percentile(values, q)) for q in (50, 95, 99)},
                    max=float(max(values))) if values else None
            physical = [s for s in smoke if s["action_executed"]]
            results[-1]["summary"] = dict(steps=len(physical), requested_steps=args.steps, planning_attempts=len(smoke),
                safety_failure=any(s["safety_failure"] for s in smoke),
                unverified_action_executions=sum(s["action_executed"] and not s["safety_certificate"]["valid"] for s in smoke),
                emergency_steps=sum(s["mode"].startswith("emergency") for s in physical),
                emergency_unverified_attempts=sum(s["mode"] == "emergency_model_check_failed" for s in smoke),
                route_gain=smoke[-1]["route"]-start_route,
                mean_speed=float(np.mean([s["speed"] for s in physical])) if physical else 0.,
                max_speed=max((s["speed"] for s in physical), default=0.),
                stall_steps=sum(s["speed"] < .3 for s in physical), crash=any(s["crash"] for s in smoke),
                offroad=any(s["offroad"] for s in smoke), planning=distribution([s["planning_ms"] for s in smoke]),
                modes={mode: sum(s["mode"] == mode for s in smoke) for mode in sorted(set(s["mode"] for s in smoke))},
                distributions={key: distribution([s["diagnostics"].get(key, float("nan")) for s in smoke])
                    for key in sorted(set().union(*(s["diagnostics"].keys() for s in smoke)))})
        assert agent.planner.residual_state_dim == 535
    finally:
        env.close()
        if candidate_stream is not None:
            candidate_stream.close()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(checkpoint=str(checkpoint), checkpoint_sha256=checksum,
        anti_stall=not args.disable_anti_stall,
        note="Bounded live-physics smoke from recorded states; not a success-rate benchmark.", results=results), indent=2))
    print(json.dumps([dict(scenario=r["scenario"], steps=len(r["records"]), route_gain=r["end_route"]-r["start_route"],
        max_speed=max(s["speed"] for s in r["records"]), max_emergency_streak=max(s["emergency_streak"] for s in r["records"]),
        crash=any(s["crash"] for s in r["records"]), offroad=any(s["offroad"] for s in r["records"]),
        initial_lattice=r["initial_lattice"]) for r in results], indent=2), flush=True)


if __name__ == "__main__":
    main()
