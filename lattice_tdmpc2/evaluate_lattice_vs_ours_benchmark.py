"""Paired, non-rendered Lattice versus V4 benchmark with W&B logging."""

from __future__ import annotations

import copy
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
UPSTREAM_ROOT = REPO_ROOT / "tdmpc2"
for path in (REPO_ROOT, UPSTREAM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from env import make_env
from lattice.frenet_metadrive import FRENET_DEFAULT_CONFIG, MetaDriveFrenetController
from lattice_tdmpc2.evaluate_visual import _resolve_checkpoints, _visual_config
from lattice_tdmpc2.planner import ResidualSACLatticePlanner
from lattice_tdmpc2.trainer import load_tdmpc2, make_sac, seed_everything
from lattice_tdmpc2.versions import training_semantics


@dataclass
class EpisodeResult:
    density: float
    policy: str
    episode_index: int
    scenario: int
    episode_return: float
    episode_cost: float
    risk_cost: float
    event_cost: float
    success: float
    route_completion: float
    episode_steps: int
    collision: float
    out_of_road: float
    fallback_rate: float
    mean_planning_ms: float
    end_reason: str


def _density_tag(density: float) -> str:
    return f"{density:g}".replace(".", "_")


def _sample_scenarios(cfg, count: int, seed: int, limit: int) -> list[int]:
    simulator = cfg.metadrive["simulator"]
    training_start = int(simulator["start_seed"])
    training_end = training_start + int(simulator["num_scenarios"])
    candidates = np.arange(limit, dtype=np.int64)
    candidates = candidates[(candidates < training_start) | (candidates >= training_end)]
    if count <= 0 or count > len(candidates):
        raise ValueError(f"benchmark.episodes must be in [1, {len(candidates)}].")
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.choice(candidates, size=count, replace=False)]


def _controller(cfg):
    lattice_config = dict(FRENET_DEFAULT_CONFIG)
    lattice_config.update(dict(cfg.lattice))
    lattice_config["_tracking_target_speed"] = training_semantics(cfg).tracking_target_speed
    return MetaDriveFrenetController(lattice_config)


def _prepare_cfg(base_cfg, density: float, scenario_limit: int, policy: str):
    cfg = copy.deepcopy(base_cfg)
    cfg.metadrive["simulator"]["start_seed"] = 0
    cfg.metadrive["simulator"]["num_scenarios"] = int(scenario_limit)
    cfg.metadrive["simulator"]["traffic_density"] = float(density)
    cfg.metadrive["simulator"]["use_render"] = False
    cfg.metadrive["simulator"]["image_observation"] = False
    cfg.metadrive["simulator"]["state_observation_with_offscreen_render"] = False
    cfg.metadrive["simulator"]["show_interface"] = False
    cfg.metadrive["simulator"]["force_render_fps"] = None
    if policy == "lattice":
        cfg.planner["type"] = "lattice"
    return cfg


def _run_episode(
    env, planner, sac_agent, scenario: int, density: float, policy: str, index: int,
    max_steps: int | None,
):
    observation = env.reset(seed=scenario)
    planner.reset()
    context = planner.prepare(observation, env.unwrapped_metadrive.agent)
    done = False
    episode_return = 0.0
    episode_cost = 0.0
    risk_cost = 0.0
    event_cost = 0.0
    collision = False
    out_of_road = False
    fallback_count = 0
    planning_seconds = 0.0
    steps = 0
    info = {}
    while not done and (max_steps is None or steps < max_steps):
        normalized_action = (
            sac_agent.act(context.residual_state.numpy(), deterministic=True)
            if sac_agent is not None else np.zeros(2, dtype=np.float32)
        )
        started = time.perf_counter()
        result = planner.refine(context, normalized_action, env.unwrapped_metadrive.agent)
        planning_seconds += time.perf_counter() - started
        observation, reward, done, info = env.step(result.low_level_action)
        episode_return += float(reward)
        episode_cost += float(info.get("cost", 0.0))
        risk_cost += float(info.get("safety_risk_cost", 0.0))
        event_cost += float(info.get("safety_event_cost", 0.0))
        collision = collision or any(
            bool(info.get(key, False))
            for key in ("crash", "crash_vehicle", "crash_object")
        )
        out_of_road = out_of_road or bool(info.get("out_of_road", False))
        fallback_count += int(not result.residual_valid)
        steps += 1
        if not done:
            context = planner.prepare(observation, env.unwrapped_metadrive.agent)
    return EpisodeResult(
        density=float(density),
        policy=policy,
        episode_index=int(index),
        scenario=int(scenario),
        episode_return=episode_return,
        episode_cost=episode_cost,
        risk_cost=risk_cost,
        event_cost=event_cost,
        success=float(info.get("success", 0.0)),
        route_completion=float(info.get("route_completion", 0.0)),
        episode_steps=steps,
        collision=float(collision),
        out_of_road=float(out_of_road),
        fallback_rate=fallback_count / max(steps, 1),
        mean_planning_ms=1000.0 * planning_seconds / max(steps, 1),
        end_reason=str(info.get("end_reason", "max_steps" if not done else "unknown")),
    )


def _make_policy(cfg, policy: str, sac_checkpoint: Path, sac_state: dict):
    env = make_env(cfg)
    tdmpc_agent = load_tdmpc2(cfg, str(cfg.planner["type"]))
    planner = ResidualSACLatticePlanner(
        cfg, _controller(cfg), tdmpc_agent=tdmpc_agent, action_space=env.action_space
    )
    sac_agent = None
    if policy == "ours":
        observation = env.reset(seed=0)
        planner.reset()
        context = planner.prepare(observation, env.unwrapped_metadrive.agent)
        saved_residual = (sac_state.get("config") or {}).get("residual_rl", {})
        sac_agent = make_sac(
            cfg,
            int(context.residual_state.numel()),
            torch.device(str(cfg.residual_rl["device"])),
            str(cfg.planner["type"]),
            hidden_sizes=saved_residual.get("hidden_sizes", cfg.residual_rl["hidden_sizes"]),
        )
        sac_agent.load(sac_checkpoint, load_optimizers=False)
    return env, planner, sac_agent


def _wandb_log_episode(run, row: EpisodeResult) -> None:
    if run is None:
        return
    prefix = f"density_{_density_tag(row.density)}/{row.policy}"
    run.log({
        f"{prefix}/episode_index": row.episode_index,
        f"{prefix}/scenario_seed": row.scenario,
        f"{prefix}/return": row.episode_return,
        f"{prefix}/cost": row.episode_cost,
        f"{prefix}/risk_cost": row.risk_cost,
        f"{prefix}/event_cost": row.event_cost,
        f"{prefix}/success": row.success,
        f"{prefix}/route_completion": row.route_completion,
        f"{prefix}/episode_steps": row.episode_steps,
        f"{prefix}/collision": row.collision,
        f"{prefix}/out_of_road": row.out_of_road,
        f"{prefix}/fallback_rate": row.fallback_rate,
        f"{prefix}/mean_planning_ms": row.mean_planning_ms,
    })


def _mean(rows: list[EpisodeResult], attribute: str) -> float:
    return float(np.mean([float(getattr(row, attribute)) for row in rows]))


def _log_summary(run, rows: list[EpisodeResult], densities: list[float]) -> dict[str, float]:
    metrics = (
        "episode_return", "episode_cost", "risk_cost", "event_cost", "success",
        "route_completion", "episode_steps", "collision", "out_of_road",
        "fallback_rate", "mean_planning_ms",
    )
    summary: dict[str, float] = {}
    for density in densities:
        tag = _density_tag(density)
        by_policy = {}
        for policy in ("lattice", "ours"):
            selected = [
                row for row in rows
                if row.policy == policy and np.isclose(row.density, density)
            ]
            by_policy[policy] = selected
            for metric in metrics:
                summary[f"summary/density_{tag}/{policy}/mean_{metric}"] = _mean(selected, metric)
        lattice_by_seed = {row.scenario: row for row in by_policy["lattice"]}
        ours_by_seed = {row.scenario: row for row in by_policy["ours"]}
        common = sorted(lattice_by_seed.keys() & ours_by_seed.keys())
        for metric in metrics:
            deltas = [
                float(getattr(ours_by_seed[seed], metric))
                - float(getattr(lattice_by_seed[seed], metric))
                for seed in common
            ]
            summary[f"paired/density_{tag}/ours_minus_lattice/{metric}"] = float(np.mean(deltas))
    if run is not None:
        for key, value in summary.items():
            run.summary[key] = value
        run.log(summary)
    return summary


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    sac_checkpoint, wm_checkpoint, sac_state, saved_cfg = _resolve_checkpoints(raw_cfg)
    cfg = _visual_config(raw_cfg, saved_cfg, sac_checkpoint, wm_checkpoint)
    benchmark = dict(raw_cfg.benchmark)
    episodes = int(benchmark["episodes"])
    scenario_seed = int(benchmark["scenario_seed"])
    scenario_limit = int(benchmark["scenario_seed_limit"])
    densities = [float(value) for value in benchmark["traffic_densities"]]
    max_steps_value = benchmark.get("max_steps")
    max_steps = None if max_steps_value is None else int(max_steps_value)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("benchmark.max_steps must be positive or null.")
    if densities != [0.1, 0.2]:
        raise ValueError("This benchmark requires traffic_densities=[0.1, 0.2].")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the V4 world-model benchmark.")

    seed_everything(int(cfg.seed))
    scenarios = _sample_scenarios(cfg, episodes, scenario_seed, scenario_limit)
    output_dir = (
        Path(hydra.utils.get_original_cwd()) / str(benchmark["output_dir"])
        / str(benchmark["wandb_run_name"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "scenarios.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "scenarios": scenarios,
            "scenario_seed": scenario_seed,
            "densities": densities,
            "sac_checkpoint": str(sac_checkpoint),
            "world_model_checkpoint": str(wm_checkpoint),
        }, stream, indent=2)

    run = None
    if bool(raw_cfg.enable_wandb):
        import wandb

        run = wandb.init(
            project=str(raw_cfg.wandb_project),
            entity=raw_cfg.wandb_entity,
            name=str(benchmark["wandb_run_name"]),
            config={
                **OmegaConf.to_container(raw_cfg, resolve=True),
                "benchmark_scenarios": scenarios,
                "resolved_sac_checkpoint": str(sac_checkpoint),
                "resolved_world_model_checkpoint": str(wm_checkpoint),
            },
        )
        for density in densities:
            for policy in ("lattice", "ours"):
                prefix = f"density_{_density_tag(density)}/{policy}"
                run.define_metric(f"{prefix}/episode_index")
                run.define_metric(f"{prefix}/*", step_metric=f"{prefix}/episode_index")

    rows: list[EpisodeResult] = []
    csv_path = output_dir / "episodes.csv"
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(EpisodeResult.__dataclass_fields__))
            writer.writeheader()
            for density in densities:
                for policy in ("lattice", "ours"):
                    policy_cfg = _prepare_cfg(cfg, density, scenario_limit, policy)
                    env, planner, sac_agent = _make_policy(
                        policy_cfg, policy, sac_checkpoint, sac_state
                    )
                    try:
                        for index, scenario in enumerate(scenarios, start=1):
                            row = _run_episode(
                                env, planner, sac_agent, scenario, density, policy, index,
                                max_steps,
                            )
                            rows.append(row)
                            writer.writerow(asdict(row))
                            stream.flush()
                            _wandb_log_episode(run, row)
                            print(
                                f"density={density:.1f} policy={policy:7s} "
                                f"[{index:03d}/{episodes:03d}] seed={scenario:04d} "
                                f"R={row.episode_return:.2f} C={row.episode_cost:.2f} "
                                f"S={row.success:.0f} RC={row.route_completion:.3f} "
                                f"steps={row.episode_steps}"
                            )
                    finally:
                        env.close()

        summary = _log_summary(run, rows, densities)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        if run is not None:
            import wandb

            table = wandb.Table(columns=list(EpisodeResult.__dataclass_fields__))
            for row in rows:
                table.add_data(*(asdict(row)[key] for key in EpisodeResult.__dataclass_fields__))
            run.log({"benchmark/per_scene_results": table})
        print(f"Benchmark complete: {output_dir}")
    finally:
        if run is not None:
            run.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
