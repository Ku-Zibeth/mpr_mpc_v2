"""CUDA-synchronized, fixed-live-state benchmark; no training or weight writes."""

import argparse
import contextlib
import csv
import io
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time


def statistics(values):
    import numpy as np
    return {"mean_ms": float(np.mean(values)), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(np.max(values)), "planning_calls": len(values)}


def source_fingerprint(root):
    tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"])
    digest = hashlib.sha256()
    for name in sorted(set(tracked.decode().split("\0"))):
        path = Path(name)
        if (not name or path.parts[0] in {"logs", "reports", "tests"}
                or path.suffix not in {".py", ".yaml", ".yml"}):
            continue
        file = root / path
        if file.is_file():
            digest.update(name.encode())
            digest.update(file.read_bytes())
    return digest.hexdigest()


def phase_summary(profiles):
    import numpy as np
    keys = set().union(*(row.keys() for row in profiles))
    return {key: {metric: float(np.mean([row[key][metric] for row in profiles if key in row]))
                  for metric in ("cpu_wall_ms", "cuda_stream_elapsed_ms")} for key in sorted(keys)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenarios", type=int, nargs="+", default=[1714, 539, 1525, 1084, 694])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup calls per scene")
    parser.add_argument("--prefix-steps", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--profile-calls", type=int, default=10)
    parser.add_argument("--kernel-profile", action="store_true", help="Separate one-call Kineto CPU/CUDA kernel trace")
    parser.add_argument("--cost-coef", type=float, default=1.)
    parser.add_argument("--allow-dirty-source", action="store_true", help="Fingerprint the exact runtime source instead of requiring a clean commit")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.calls < 100 or args.warmup < 1 or len(set(args.scenarios)) < 2:
        parser.error("Require >=100 measured calls, warmup >=1 and >=2 distinct scenarios.")
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.source_root.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if not args.allow_dirty_source and subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Benchmark requires a clean committed source tree; use an independent clean worktree.")
    fingerprint = source_fingerprint(root)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from mpr_mpc._bootstrap import bootstrap
    bootstrap()
    from common.parser import parse_cfg
    from common.seed import set_seed
    from env import make_env
    from mpr_mpc.evaluate_model_suite import _make_agent, _apply_checkpoint_runtime_config

    if not torch.cuda.is_available():
        raise RuntimeError("Benchmark requires CUDA.")
    raw = OmegaConf.load(root / "config.yaml")
    raw.enable_wandb = raw.save_agent = raw.save_csv = raw.save_video = False
    raw.compile = False
    raw.restore_checkpoint_runtime = True
    raw.log_root = str(args.output.parent)
    raw.residual_rl.buffer_size = 8
    raw.horizon = args.horizon
    raw.mpr_mpc.mppi.num_samples = args.samples
    raw.mpr_mpc.mppi.iterations = args.iterations
    raw.mpr_mpc.mppi.eval_seed = args.seed
    raw.mpr_mpc.mppi.cost_enabled = True
    raw.mpr_mpc.mppi.cost_coef = args.cost_coef
    raw.metadrive.simulator.start_seed = min(args.scenarios)
    raw.metadrive.simulator.num_scenarios = max(args.scenarios) - min(args.scenarios) + 1
    _apply_checkpoint_runtime_config(raw, args.checkpoint)
    # CLI workload is authoritative; checkpoint metadata only restores architecture.
    raw.horizon = args.horizon
    raw.mpr_mpc.mppi.cost_enabled = True
    raw.mpr_mpc.mppi.cost_coef = args.cost_coef
    raw.residual_rl.buffer_size = 8
    raw.mpr_mpc.mppi.num_samples = args.samples
    raw.mpr_mpc.mppi.iterations = args.iterations
    raw.mpr_mpc.mppi.eval_seed = args.seed
    if int(raw.mpr_mpc.mppi.num_elites) > args.samples:
        raise ValueError("num_elites exceeds benchmark sample budget.")
    cfg = parse_cfg(raw)
    set_seed(args.seed)
    env = make_env(cfg)
    durations, records, profiles, states, kernel_profile = [], [], [], [], {}
    pending_trace, trace_phases = None, set()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agent = _make_agent(cfg, env)
            agent.load(args.checkpoint)
        agent.eval()
        for index, scene in enumerate(args.scenarios):
            set_seed(args.seed + scene)
            observation = env.reset(seed=scene)
            for step in range(args.prefix_steps):
                action = agent.act(observation, t0=step == 0, eval_mode=True, global_step=0)
                observation, _, done, _ = env.step(action)
                if done:
                    raise RuntimeError(f"Prefix terminated in scene {scene}; use a shorter shared prefix.")
            vehicle = env.unwrapped_metadrive.agent
            states.append({"scenario": scene, "position": vehicle.position.tolist(),
                           "velocity": vehicle.velocity.tolist(), "heading": float(vehicle.heading_theta)})
            planner = agent.planner
            planner.reset()
            for _ in range(args.warmup):
                agent.act(observation, eval_mode=True, global_step=planner.mppi_start_step)
            count = args.calls // len(args.scenarios) + int(index < args.calls % len(args.scenarios))
            for call in range(count):
                torch.cuda.synchronize()
                started = time.perf_counter()
                result = planner.plan(observation, vehicle, global_step=planner.mppi_start_step, eval_mode=True)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                if not torch.isfinite(result.action).all() or not result.stage.use_mppi:
                    raise RuntimeError("Benchmark received an invalid or MPPI-disabled plan.")
                durations.append(elapsed)
                records.append({"scenario": scene, "call": call, "total_planning_ms": elapsed,
                    "predicted_cost": result.metrics.get("mppi/predicted_cost"),
                    "cost_penalty": result.metrics.get("mppi/cost_penalty"),
                    "cost_validated": result.metrics.get("mppi/cost_validated")})
                for name in ("safety_check_ms", "first_action_delta_norm", "baseline_safety_valid", "safety_valid_ratio",
                             "safety_reject_count", "wm_evaluated_candidates", "unique_candidate_count",
                             "emergency_fallback", "lattice_fallback", "safety_fallback_unverified",
                             "world_model_rollout_ms", "cost_prediction_ms", "elite_update_ms"):
                    records[-1][name] = result.metrics.get(f"mppi/{name}")
            if args.profile_calls:
                from planning_profiler import PlanningProfiler
                with PlanningProfiler(planner) as profiler:
                    for _ in range(max(1, args.profile_calls // len(args.scenarios))):
                        planner.plan(observation, vehicle, global_step=planner.mppi_start_step, eval_mode=True)
                        profiles.append(profiler.report())
            if args.kernel_profile and index == len(args.scenarios)-1:
                from planning_profiler import PlanningProfiler
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as trace:
                    with PlanningProfiler(planner, record_ranges=True) as profiler:
                        planner.plan(observation, vehicle, global_step=planner.mppi_start_step, eval_mode=True)
                        profiler.report()
                trace_phases = profiler.phases
                trace_path = args.output.with_suffix(".trace.json")
                if trace_path.exists():
                    raise FileExistsError(trace_path)
                pending_trace = trace
    finally:
        env.close()
    if subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip() != revision:
        raise RuntimeError("Source revision changed during the benchmark.")
    result = {**statistics(durations), "revision": revision, "source_root": str(root),
              "checkpoint": str(args.checkpoint.resolve()), "device": torch.cuda.get_device_name(),
              "horizon": args.horizon, "samples": args.samples, "iterations": args.iterations,
              "seed": args.seed, "scenarios": args.scenarios, "warmup_per_scene": args.warmup,
              "prefix_steps": args.prefix_steps, "states": states, "profiles": profiles,
              "measurement": "Synchronized wall ms; repeated live states; selected controls are NOT executed.",
              "cost_readiness": {"updates": int(agent.model._cost_updates), "ready": agent.model.cost_ready}}
    result["effective_mppi_config"] = dict(cfg.mpr_mpc["mppi"])
    result["source_dirty"] = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    result["source_snapshot_sha256"] = fingerprint
    if source_fingerprint(root) != fingerprint:
        raise RuntimeError("Runtime source changed during benchmark.")
    if result["source_dirty"] and not args.allow_dirty_source:
        raise RuntimeError("Source tree was modified during the benchmark.")
    result["phase_summary_ms"] = phase_summary(profiles)
    result["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    result["kernel_profile_last_scene"] = kernel_profile
    result["control_dt"] = planner.control_dt
    result["cost_readiness"].update(schema=int(getattr(agent.model, "_cost_validation_schema", 0)),
        validated_horizons=int(agent.model._cost_validated_horizons))
    result["profile_notes"] = ("Separate instrumented pass; nested phase times are not additive. CPU wall includes waits. "
        "CUDA events measure stream elapsed including idle CPU gaps, NOT pure GPU kernel time. "
        "Context includes coarse planning/WM; WM includes cost. Geometry is legacy SAC input only. "
        "Candidate scoring contains deviation/smoothness/WM cost, without geometric or uncertainty penalties.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if pending_trace is not None:
        from planning_profiler import kernel_trace_summary
        trace_path = args.output.with_suffix(".trace.json")
        pending_trace.export_chrome_trace(str(trace_path))
        kernel_profile, audit = kernel_trace_summary(json.loads(trace_path.read_text()), trace_phases)
        result["kernel_profile_last_scene"] = kernel_profile
        result["kernel_trace_audit"] = audit
    args.output.write_text(json.dumps(result, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    if args.compare:
        before = json.loads(args.compare.read_text())
        if before.get("source_dirty", True) and not (
            args.allow_dirty_source and before.get("source_snapshot_sha256") == fingerprint
        ):
            raise ValueError("Baseline was not measured from a clean commit.")
        for key in ("device", "checkpoint", "horizon", "samples", "iterations", "seed", "scenarios", "prefix_steps", "states"):
            if before[key] != result[key]:
                raise ValueError(f"Unmatched comparison contract: {key}")
        for key in ("num_elites", "initial_std", "min_std", "max_std", "max_delta", "temperature"):
            if before["effective_mppi_config"].get(key) != result["effective_mppi_config"].get(key):
                raise ValueError(f"Unmatched effective workload: {key}")
        with (args.output.parent / "benchmark_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["metric", "before_ms", "after_ms"], lineterminator="\n")
            writer.writeheader()
            for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
                writer.writerow({"metric": key, "before_ms": before[key], "after_ms": result[key]})
            phases_before = before.get("phase_summary_ms", phase_summary(before["profiles"]))
            for phase in sorted(set(phases_before) & set(result["phase_summary_ms"])):
                for measure in ("cpu_wall_ms", "cuda_stream_elapsed_ms"):
                    writer.writerow(dict(metric=f"{phase}_{measure}", before_ms=phases_before[phase][measure],
                                         after_ms=result["phase_summary_ms"][phase][measure]))
    print(json.dumps({k: result[k] for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "planning_calls")}))


if __name__ == "__main__":
    main()
