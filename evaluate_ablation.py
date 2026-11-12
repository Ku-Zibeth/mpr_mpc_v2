"""Paired Lattice/SAC/MPPI ablations using one frozen checkpoint."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import evaluate_model_suite as suite
import hydra
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from evaluation_statistics import paired_comparisons, summarize_runs


VARIANTS = (
    ("A_lattice", "world_model"),
    ("B_lattice_sac", "world_model_residual_rl"),
    ("C_lattice_mppi", "lattice_mppi"),
    ("D_lattice_sac_mppi", "full_mpr_mpc"),
)
DEFAULT_CHECKPOINT = (
    suite.DEFAULT_MODELS_ROOT
    / "mpr_mpc_frozen_wm200k_rl500k_seed1_v2"
    / "models/milestone_500k_world_model_residual_rl.pt"
)


def _select_variants(benchmark):
    if bool(benchmark.get("cost_ablation", False)):
        raise ValueError("Cost scoring ablation is retired: simplified MPPI optimises WM return only.")
    available = VARIANTS
    requested = benchmark.get("variants")
    if requested is None:
        return available
    if isinstance(requested, str) or not requested:
        raise ValueError("benchmark.variants must be a nonempty list of configuration names.")
    names = list(requested)
    by_name = dict(available)
    if len(set(names)) != len(names) or any(name not in by_name for name in names):
        raise ValueError(f"Unknown or duplicate variants; available: {list(by_name)}")
    return tuple((name, by_name[name]) for name in names)


class _EpisodeCSVWriter:
    """Flush each episode and expand the schema when optional metrics appear."""

    def __init__(self, stream):
        self.stream = stream
        self.rows = []
        self.fieldnames = []
        self.writer = None

    def writerow(self, row: dict) -> None:
        row = dict(row)
        new_fields = [key for key in row if key not in self.fieldnames]
        if new_fields:
            self.fieldnames.extend(new_fields)
            # Prior rows need blank cells for newly observed SAC/MPPI metrics.
            self.stream.seek(0)
            self.stream.truncate()
            self.writer = csv.DictWriter(self.stream, fieldnames=self.fieldnames)
            self.writer.writeheader()
            self.writer.writerows(self.rows)
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, fieldnames=self.fieldnames)
            self.writer.writeheader()
        self.writer.writerow(row)
        self.rows.append(row)
        self.stream.flush()


def _repeat_seeds(benchmark: DictConfig, default: int) -> list[int]:
    values = list(benchmark.get("repeat_seeds", [default]))
    if not values or any(type(value) is not int or not 0 <= value < 2**32 for value in values):
        raise ValueError("repeat_seeds must contain nonnegative 32-bit integers.")
    if len(set(values)) != len(values):
        raise ValueError("repeat_seeds contains duplicates.")
    return values


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA for TD-MPC2.")
    checkpoint = Path(str(raw_cfg.checkpoint or DEFAULT_CHECKPOINT)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    version = checkpoint.parent.parent.name
    with open_dict(raw_cfg):
        if "benchmark" not in raw_cfg:
            raw_cfg.benchmark = {}
        benchmark = raw_cfg.benchmark
        raw_cfg.eval_episodes = int(benchmark.get("episodes", 100))
        benchmark.paired_episode_seeds = True
        benchmark.scenario_seed = int(benchmark.get("scenario_seed", raw_cfg.seed))
        simulator = raw_cfg.metadrive.simulator
        simulator.start_seed = int(benchmark.get("scenario_start", simulator.start_seed))
        simulator.num_scenarios = int(benchmark.get(
            "scenario_pool_count", max(int(simulator.num_scenarios), raw_cfg.eval_episodes)
        ))
    scenarios = suite._scenarios(raw_cfg)
    repeat_seeds = _repeat_seeds(benchmark, int(raw_cfg.seed))
    cost_ablation = bool(benchmark.get("cost_ablation", False))
    variants = _select_variants(benchmark)
    bootstrap_samples = int(benchmark.get("bootstrap_samples", 2000))
    bootstrap_seed = int(benchmark.get("bootstrap_seed", benchmark.scenario_seed))
    if bootstrap_samples <= 0 or bootstrap_seed < 0:
        raise ValueError("bootstrap_samples must be positive and bootstrap_seed nonnegative.")
    with open_dict(raw_cfg):
        simulator = raw_cfg.metadrive.simulator
        last_seed = max(int(simulator.start_seed) + int(simulator.num_scenarios) - 1, max(scenarios))
        simulator.start_seed = min(int(simulator.start_seed), min(scenarios))
        simulator.num_scenarios = last_seed - int(simulator.start_seed) + 1
    output_dir = Path(str(raw_cfg.benchmark.get(
        "output_dir", suite.SCRIPT_ROOT / "logs/benchmark/ablation_seed1"
    ))).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_cfg, _ = suite._make_cfg(raw_cfg, version, checkpoint, scenarios)
    OmegaConf.save(effective_cfg, output_dir / "config.yaml")
    (output_dir / "scenarios.json").write_text(
        json.dumps({"scenarios": scenarios}, indent=2), encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "seed": int(raw_cfg.seed),
        "repeat_seeds": repeat_seeds,
        "unique_scenarios": len(scenarios),
        "episodes_per_configuration": len(scenarios) * len(repeat_seeds),
        "episode_seed_rule": "(test_seed + scenario) modulo 2**32",
        "mppi_eval_seed_rule": "(test_seed + scenario) modulo 2**32; persistent generator within each episode",
        "horizon": int(effective_cfg.horizon),
        "variants": dict(variants),
        "cost_ablation": cost_ablation,
        "event_metrics": "fraction of episodes with an event at any step",
        "joint_event_denominator": "all evaluated episodes",
        "ttc_definition": "environment risk_field_min_ttc: longitudinal gap / closing speed to a front vehicle in the same corridor",
        "mean_ttc": "episode mean of finite nonnegative step TTC values, then mean across valid episodes",
        "minimum_ttc": "minimum finite TTC across all evaluated steps; no CI for this extreme statistic",
        "missing_ttc": "no finite closing-front-vehicle TTC; excluded, never converted to zero",
        "mppi_metrics": "only steps with actual MPPI execution; null for A/B",
        "ci95_method": "scenario-cluster percentile bootstrap; conservative exact boundary bounds for all-zero/all-one binary rates using unique scenario count",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }, indent=2), encoding="utf-8")
    print(f"Checkpoint: {checkpoint}\nOutput dir: {output_dir}\nScenarios: {scenarios}")

    all_rows, comparisons, summaries, seed_summaries = [], [], [], []
    with (output_dir / "episodes.csv").open("w", newline="") as all_file:
        all_writer = _EpisodeCSVWriter(all_file)
        for label, stack in variants:
            for test_seed in repeat_seeds:
                print(f"\n=== {label}, test_seed={test_seed} ===")
                variant_cfg = OmegaConf.create(OmegaConf.to_container(effective_cfg, resolve=True))
                variant_cfg.evaluation_stack = stack
                variant_cfg.seed = test_seed
                variant_cfg.benchmark.mppi_eval_seed = test_seed
                OmegaConf.save(variant_cfg, output_dir / f"config_{label}_seed{test_seed}.yaml")
                with (output_dir / f"episodes_{label}_seed{test_seed}.csv").open("w", newline="") as file:
                    writer = _EpisodeCSVWriter(file)

                    def save_episode(row: dict) -> None:
                        row["configuration"] = label
                        row["test_seed"] = test_seed
                        writer.writerow(row)
                        all_writer.writerow(row)
                        all_rows.append(row)

                    rows, summary = suite._evaluate_checkpoint(
                        variant_cfg, version, checkpoint, scenarios, on_episode=save_episode
                    )
                summary.update(configuration=label, test_seed=test_seed)
                summaries.append(summary)
                seed_summary = summarize_runs(label, rows, samples=bootstrap_samples, seed=bootstrap_seed)
                seed_summary["test_seed"] = test_seed
                seed_summaries.append(seed_summary)
                comparisons = []
                for configuration, _ in variants:
                    group = [r for r in all_rows if r["configuration"] == configuration]
                    if group:
                        comparisons.append(summarize_runs(
                            configuration, group, samples=bootstrap_samples, seed=bootstrap_seed
                        ))
                        suite._write_csv(output_dir / f"episodes_{configuration}.csv", group)
                suite._write_csv(output_dir / "summary.csv", comparisons)
                suite._write_csv(output_dir / "seed_summary.csv", seed_summaries)
                suite._write_csv(output_dir / "paired_comparisons.csv", paired_comparisons(
                    all_rows, samples=bootstrap_samples, seed=bootstrap_seed
                ))
                (output_dir / "summary.json").write_text(
                    json.dumps(comparisons, indent=2, allow_nan=False), encoding="utf-8"
                )
                (output_dir / "detailed_summary.json").write_text(
                    json.dumps(summaries, indent=2, allow_nan=True), encoding="utf-8"
                )
                print(f"{label}, seed={test_seed}: success={seed_summary['success']:.2%}, "
                      f"crash={seed_summary['crash']:.2%}, out_of_road={seed_summary['out_of_road']:.2%}", flush=True)

    print("\nconfiguration              out_of_road  crash  route_completion  success")
    for row in comparisons:
        print(
            f"{row['configuration']:<27} {row['out_of_road']:>10.2%} "
            f"{row['crash']:>6.2%} {row['route_completion']:>17.2%} {row['success']:>8.2%}"
        )
    print(f"\nResults: {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
