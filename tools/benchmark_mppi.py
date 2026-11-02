"""CUDA-synchronized, fixed-live-state benchmark; no training or weight writes."""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import time


def statistics(values):
    import numpy as np
    return {"mean_ms": float(np.mean(values)), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(np.max(values)), "planning_calls": len(values)}


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
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    if args.calls < 100 or args.warmup < 1 or len(set(args.scenarios)) < 2:
        parser.error("Require >=100 measured calls, warmup >=1 and >=2 distinct scenarios.")
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.source_root.resolve()
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
    raw.log_root = str(args.output.parent)
    raw.residual_rl.buffer_size = 8
    raw.horizon = args.horizon
    raw.mpr_mpc.mppi.num_samples = args.samples
    raw.mpr_mpc.mppi.iterations = args.iterations
    raw.mpr_mpc.mppi.eval_seed = args.seed
    raw.metadrive.simulator.start_seed = min(args.scenarios)
    raw.metadrive.simulator.num_scenarios = max(args.scenarios) - min(args.scenarios) + 1
    _apply_checkpoint_runtime_config(raw, args.checkpoint)
    # CLI workload is authoritative; checkpoint metadata only restores architecture.
    raw.horizon = args.horizon
    raw.mpr_mpc.mppi.num_samples = args.samples
    raw.mpr_mpc.mppi.iterations = args.iterations
    raw.mpr_mpc.mppi.eval_seed = args.seed
    if int(raw.mpr_mpc.mppi.num_elites) > args.samples:
        raise ValueError("num_elites exceeds benchmark sample budget.")
    cfg = parse_cfg(raw)
    set_seed(args.seed)
    env = make_env(cfg)
    durations, records, profiles, states = [], [], [], []
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
                records.append({"scenario": scene, "call": call, "total_planning_ms": elapsed})
            if args.profile_calls:
                from planning_profiler import PlanningProfiler
                with PlanningProfiler(planner) as profiler:
                    for _ in range(max(1, args.profile_calls // len(args.scenarios))):
                        planner.plan(observation, vehicle, global_step=planner.mppi_start_step, eval_mode=True)
                        profiles.append(profiler.report())
    finally:
        env.close()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    result = {**statistics(durations), "revision": revision, "source_root": str(root),
              "checkpoint": str(args.checkpoint.resolve()), "device": torch.cuda.get_device_name(),
              "horizon": args.horizon, "samples": args.samples, "iterations": args.iterations,
              "seed": args.seed, "scenarios": args.scenarios, "warmup_per_scene": args.warmup,
              "prefix_steps": args.prefix_steps, "states": states, "profiles": profiles,
              "measurement": "Synchronized wall ms; repeated live states; selected controls are NOT executed.",
              "cost_readiness": {"updates": int(agent.model._cost_updates), "ready": agent.model.cost_ready}}
    result["effective_mppi_config"] = dict(planner.local_mppi._sampling_cfg)
    result["source_dirty"] = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip())
    result["profile_notes"] = ("Separate instrumented pass; nested phase times are not additive. CPU wall includes waits. "
        "CUDA events measure stream elapsed including idle CPU gaps, NOT pure GPU kernel time. "
        "Context includes coarse planning/WM; WM includes cost. Distribution is per-mode update only. "
        "Baseline scoring covers penalties only when _score_candidates is absent. No approved cost => cost time zero.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    if args.compare:
        before = json.loads(args.compare.read_text())
        for key in ("device", "checkpoint", "horizon", "samples", "iterations", "seed", "scenarios", "prefix_steps", "states"):
            if before[key] != result[key]:
                raise ValueError(f"Unmatched comparison contract: {key}")
        with (args.output.parent / "benchmark_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["metric", "before_ms", "after_ms"], lineterminator="\n")
            writer.writeheader()
            for key in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"):
                writer.writerow({"metric": key, "before_ms": before[key], "after_ms": result[key]})
    print(json.dumps({k: result[k] for k in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "planning_calls")}))


if __name__ == "__main__":
    main()
