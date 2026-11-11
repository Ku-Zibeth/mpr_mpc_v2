"""Read-only action replay; empirical pose tubes with disjoint scene validation.

No WM/SAC fitting, checkpoint writes, candidate search or safety relaxation.
An unsuccessful validation is saved as such and cannot change authorization.
"""
import argparse
import contextlib
from dataclasses import replace
import gzip
import hashlib
import io
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
from common.seed import set_seed
from env import make_env
import evaluate_model_suite as suite


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {**{f"p{q}": float(np.percentile(a, q)) for q in (50, 95, 99)}, "max": float(a.max())} if len(a) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT/"logs/benchmark/mpr_mpc_v311_sac_vs_sac_mppi_fixed100_20261109_202823")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-stack", choices=("world_model_residual_rl", "full_mpr_mpc"),
                   default="full_mpr_mpc", help="Executed source policy; never mix B and D labels")
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a fresh output directory; existing calibration evidence is never overwritten.")
    args.output.mkdir(parents=True)
    fit_scenes, heldout_scenes = [1525, 539], [1714, 559]
    limits = {1525: 300, 539: 300, 1714: 310, 559: 660}
    records = {s: [] for s in limits}
    source_log = next((args.run/"steps").glob(f"*_{args.source_stack}_seed1.jsonl"))
    for line in source_log.open():
        row = json.loads(line)
        s = row["scenario"]
        if s in limits and row["episode_step"] <= limits[s]:
            records[s].append(row)
        if all(len(records[s]) >= n for s, n in limits.items()):
            break
    raw = OmegaConf.load(args.run/"config_D_lattice_sac_mppi_seed1.yaml")
    raw.mpr_mpc.mppi = OmegaConf.load(ROOT/"config.yaml").mpr_mpc.mppi
    # Measure the CURRENT nominal model; never consume the report being built.
    raw.mpr_mpc.mppi.safety.motion_validation_path = None
    raw.enable_wandb = raw.save_agent = raw.save_video = False
    raw.restore_mppi_runtime = False
    raw.residual_rl.buffer_size = 8
    checkpoint = Path(json.loads((args.run/"manifest.json").read_text())["checkpoint"])
    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    scenarios = json.loads((args.run/"scenarios.json").read_text())["scenarios"]
    _, cfg = suite._make_cfg(raw, checkpoint.parent.parent.name, checkpoint, scenarios)
    set_seed(cfg.seed)
    env = make_env(cfg)
    dataset, errors, signatures = [], {"fit": {}, "heldout": {}}, []
    domain_speeds, domain_actions = [], []
    windows_count = {"fit": 0, "heldout": 0}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agent = suite._make_agent(cfg, env)
            agent.load(checkpoint)
        agent.eval()
        planner = agent.planner
        f = planner.local_mppi.candidate_safety
        nominal = f.motion_nominal_config()
        for seed in limits:
            set_seed((int(cfg.seed)+seed) % 2**32)
            env.reset(seed=seed)
            f.reset()
            trace, contexts = [], {}
            vehicle = env.unwrapped_metadrive.agent
            def state():
                return [*map(float, vehicle.position), float(vehicle.heading_theta), float(np.linalg.norm(vehicle.velocity))]
            for i, row in enumerate(records[seed]):
                if i % 10 == 0:
                    width, wheelbase, steering = planner._vehicle_geometry(vehicle)
                    ctx = f.context(planner.structured.planner, vehicle, width=width, wheelbase=wheelbase, max_steering=steering)
                    ctx.position = ctx.position.copy()
                    ctx.previous_action = ctx.previous_action.copy()
                    contexts[i] = ctx
                action = np.asarray(row["executed_action"], dtype=np.float32)
                before = state()
                ticks = []
                engine = vehicle.engine
                original = engine.step_physics_world
                def observe_physics():
                    original()
                    ticks.append(state())
                engine.step_physics_world = observe_physics
                try:
                    _, _, done, info = env.step(action)
                finally:
                    engine.step_physics_world = original
                trace.append(dict(before=before, action=action.tolist(), ticks=ticks,
                    crash=any(bool(info.get(k, False)) for k in ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_human", "crash_sidewalk")),
                    offroad=bool(info.get("out_of_road", False))))
                if done:
                    break
            split = "fit" if seed in fit_scenes else "heldout"
            for index, ctx in contexts.items():
                if index+10 > len(trace):
                    continue
                windows_count[split] += 1
                if ctx.motion_signature not in signatures:
                    signatures.append(ctx.motion_signature)
                if split == "fit":
                    domain_speeds.append(ctx.speed)
                    domain_actions.extend([r["action"] for r in trace[index:index+10]])
                window = dict(scenario=seed, index=index, split=split,
                    road_wkb=ctx.road.wkb_hex, vehicle_signature=ctx.motion_signature,
                    obstacles=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(o).items()}
                        for o in ctx.obstacles], states=trace[index:index+10])
                window["errors"] = {}
                for horizon in range(1, 11):
                    controls = torch.tensor([r["action"] for r in trace[index:index+horizon]]+[trace[index+horizon-1]["action"]])[None]
                    actual = np.asarray([trace[index]["before"]]+[tick for r in trace[index:index+horizon] for tick in r["ticks"]])
                    if len(actual) != horizon*5+1:
                        raise RuntimeError("Expected MetaDrive five physics ticks per control; no interpolated actual data permitted.")
                    for model, prediction in (("center", f._rollout(controls, ctx)), ("legacy", f.geometry.rollout(controls, ctx))):
                        xy, yaw, speed = (a[0].numpy() for a in prediction)
                        source_times = np.linspace(0, horizon, len(xy))
                        actual_times = np.linspace(0, horizon, len(actual))
                        xy = np.column_stack([np.interp(actual_times, source_times, xy[:, j]) for j in range(2)])
                        yaw = np.interp(actual_times, source_times, np.unwrap(yaw))
                        speed = np.interp(actual_times, source_times, speed)
                        delta = xy-actual[:, :2]
                        # Reference normal at the measured INITIAL heading;
                        # independent of either model's predicted heading.
                        normal = np.array([-np.sin(ctx.heading), np.cos(ctx.heading)])
                        angle = yaw-actual[:, 2]
                        values = dict(position_m=float(np.linalg.norm(delta, axis=1).max()),
                            heading_rad=float(np.abs(np.arctan2(np.sin(angle), np.cos(angle))).max()),
                            lateral_m=float(np.abs(delta@normal).max()), speed_mps=float(np.abs(speed-actual[:, 3]).max()))
                        key = f"{model}_h{horizon}"
                        errors[split].setdefault(key, []).append(values)
                        window["errors"][key] = values
                    # Near-event labels use the SAME measured future outcome.
                    if horizon == 1:
                        for model, c in (("center", ctx), ("legacy", replace(ctx, rear_axle_to_center=None))):
                            check = f._strict_check(controls, c)
                            window[model+"_events"] = {event: dict(predicted=bool(getattr(check, name)[0]), actual=bool(trace[index][event]))
                                for event, name in (("crash", "collision"), ("offroad", "offroad"))}
                dataset.append(window)
            print(f"[motion calibration] scene={seed} replay_controls={len(trace)} windows={sum(w['scenario']==seed for w in dataset)}", flush=True)
    finally:
        env.close()
    path = (args.output/"measured_windows.jsonl.gz").resolve()
    with gzip.open(path, "wt") as stream:
        for window in dataset:
            stream.write(json.dumps(window)+"\n")
    bounds, valid_horizons, comparisons = {}, [], {}
    for h in range(1, 11):
        key = f"center_h{h}"
        fit, held = errors["fit"].get(key, []), errors["heldout"].get(key, [])
        if not fit or not held:
            continue
        # Measured upper residual plus the existing geometric numeric tolerance.
        bound = {name: max(v[name] for v in fit)+(1e-4 if name != "speed_mps" else np.finfo(np.float32).eps)
            for name in fit[0]}
        bounds[str(h)] = bound
        passes = all(v[name] <= bound[name] for v in held for name in bound)
        comparisons[str(h)] = dict(fit_bound=bound, heldout_max={name: max(v[name] for v in held) for name in bound}, passed=passes)
        if passes and min(len(fit), len(held)) >= 20:
            valid_horizons.append(h)
    confusion = {}
    for model in ("center", "legacy"):
        for event in ("crash", "offroad"):
            pairs = [w[model+"_events"][event] for w in dataset]
            confusion[f"{model}_{event}"] = {"samples": len(pairs),
                "false_positive": sum(v["predicted"] and not v["actual"] for v in pairs),
                "false_negative": sum(not v["predicted"] and v["actual"] for v in pairs),
                "actual_positive": sum(v["actual"] for v in pairs)}
    # No observed positive event cannot establish collision/road-event recall.
    events_supported = all(confusion[f"center_{e}"]["actual_positive"] > 0 and
        confusion[f"center_{e}"]["false_negative"] == 0 for e in ("crash", "offroad"))
    evidence = dict(schema=1, predictor_schema="center_bicycle_v1", protocol="disjoint_scene_physics_tick_v1",
        source_stack=args.source_stack, source_log=str(source_log),
        source_log_sha256=hashlib.sha256(source_log.read_bytes()).hexdigest(),
        checkpoint_sha256=checksum, dataset=str(path), dataset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        fit_scenes=fit_scenes, heldout_scenes=heldout_scenes, fit_windows=windows_count["fit"], heldout_windows=windows_count["heldout"],
        control_dt=f.geometry.dt, nominal_config=nominal, predictor_sha256=f.motion_predictor_sha(),
        parameters={}, vehicle_signatures=signatures,
        domain=dict(speed=[min(domain_speeds), max(domain_speeds)], action_min=np.min(domain_actions, axis=0).tolist(),
            action_max=np.max(domain_actions, axis=0).tolist()), bounds=bounds, valid_horizons=valid_horizons,
        validated=bool(10 in valid_horizons and events_supported), event_confusion=confusion,
        validation=comparisons, uncertainty_scope="empirical operating domain only; not a real-world safety guarantee",
        event_validation_supported=events_supported,
        error_distributions={split: {k: {name: distribution([v[name] for v in rows]) for name in rows[0]}
            for k, rows in section.items()} for split, section in errors.items()})
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checksum
    (args.output/"validation.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(dict(validated=evidence["validated"], valid_horizons=valid_horizons,
        event_validation_supported=events_supported, evidence=str(args.output/"validation.json"))), flush=True)


if __name__ == "__main__":
    main()
