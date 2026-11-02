"""Paired E0--E5 evaluation with distinct, contract-checked trained checkpoints."""

import json
import math
import csv
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
import torch

import evaluate_model_suite as suite
from evaluate_ablation import _repeat_seeds
from mpr_mpc.evaluation_statistics import paired_comparisons, summarize_runs


PROFILES = tuple(f"E{i}" for i in range(6))


def profile_config(profile, overrides):
    if profile not in PROFILES:
        raise ValueError(f"Unknown experiment profile: {profile}")
    shared = [value for value in overrides if value.split("=", 1)[0].lstrip("+") != "experiment"]
    return hydra.compose(config_name="config", overrides=shared + [f"+experiment={profile}"])


def checkpoint_contract(path, profile, cfg, *, allow_unlabelled=False):
    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = state.get("metadata", {})
    saved_profile = metadata.get("experiment_profile")
    if saved_profile != profile:
        if saved_profile is not None or not allow_unlabelled:
            raise ValueError(f"{path}: requested {profile}, checkpoint profile={saved_profile!r}; train the matching profile first.")
        print(f"[warning] {profile}: historical checkpoint has no profile label; training provenance is unverified.")
    sac = state.get("mpr_mpc", {}).get("residual_sac", {})
    if not sac or int(sac.get("update_steps", 0)) <= 0:
        raise ValueError(f"{path}: no trained residual SAC; refusing to benchmark an initialized actor.")
    if bool(sac.get("constraint_alignment", False)) != bool(cfg.residual_rl.constraint_alignment):
        raise ValueError(f"{path}: SAC constraint semantics do not match {profile}.")
    features = list(metadata.get("risk_features", []))
    expected = list(cfg.residual_rl.risk_features)
    if features != expected:
        raise ValueError(f"{path}: risk feature schema does not match {profile}.")
    model = state.get("model", {})
    encoder = [(key, tuple(value.shape)) for key, value in model.items()
               if key.startswith("_encoder.") and torch.is_tensor(value)]
    return {"saved_profile": saved_profile, "sac_updates": int(sac["update_steps"]),
            "cost_updates": int(model.get("_cost_updates", 0)),
            "cost_validated": bool(model.get("_cost_validated", False)),
            "encoder_signature": sorted(encoder)}


def prepare_jobs(raw_cfg, overrides):
    options = raw_cfg.improvements
    profiles = list(options.profiles)
    if not profiles or len(set(profiles)) != len(profiles) or set(profiles) - set(PROFILES):
        raise ValueError("improvements.profiles must contain distinct E0--E5 labels.")
    root = Path(str(options.models_root or (suite.LOG_ROOT / str(raw_cfg.task) / str(options.training_seed)))).expanduser().resolve()
    jobs = []
    for profile in profiles:
        cfg = profile_config(profile, overrides)
        configured = options.checkpoints.get(profile)
        path = Path(str(configured)).expanduser().resolve() if configured else root / cfg.exp_name / "models" / str(options.checkpoint_name)
        if not path.is_file():
            raise FileNotFoundError(f"{profile} checkpoint missing: {path}. Train that profile or set improvements.checkpoints.{profile}.")
        jobs.append((profile, cfg, path))
    if len({path for _, _, path in jobs}) != len(jobs):
        raise ValueError("E0--E5 require distinct trained checkpoints, not one actor relabelled six times.")
    contracts = {}
    for profile, cfg, path in jobs:
        contracts[profile] = checkpoint_contract(path, profile, cfg, allow_unlabelled=bool(options.allow_unlabelled_checkpoints))
    signatures = [record["encoder_signature"] for record in contracts.values()]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("Encoder architectures differ; this would confound the E0--E5 ablation.")
    return jobs, contracts


def _clean(value):
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(raw_cfg, overrides):
    jobs, contracts = prepare_jobs(raw_cfg, overrides)
    options = raw_cfg.improvements
    seeds = _repeat_seeds(options, int(raw_cfg.seed))
    samples = int(options.bootstrap_samples)
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    with open_dict(raw_cfg):
        raw_cfg.eval_episodes = int(options.episodes)
        if "benchmark" not in raw_cfg:
            raw_cfg.benchmark = {}
        raw_cfg.benchmark.episodes = int(options.episodes)
        raw_cfg.benchmark.paired_episode_seeds = True
    scenarios = suite._scenarios(raw_cfg)
    output = Path(str(options.output_dir or (suite.SCRIPT_ROOT / "logs/benchmark/improvements"))).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "episodes.csv").exists():
        raise FileExistsError(f"Refusing to overwrite previous results: {output}; choose a new output_dir.")
    (output / "scenarios.json").write_text(json.dumps({"scenarios": scenarios}, indent=2))
    (output / "manifest.json").write_text(json.dumps({
        "profiles": [label for label, _, _ in jobs], "repeat_seeds": seeds,
        "unique_scenarios": len(scenarios), "checkpoints": {label: str(path) for label, _, path in jobs},
        "checkpoint_contracts": contracts, "paired_episode_seed": "(test_seed + scenario) modulo 2**32",
        "mppi_eval_seed": "test_seed", "calibration": "post-run diagnostics never alter that run's actions",
    }, indent=2))
    all_rows, detailed = [], []
    comparisons = [(f"E{i}", f"E{i+1}") for i in range(5)] + [("E0", "E5")]
    with (output / "episodes.csv").open("w", newline="") as combined_file:
        combined_writer = None
        for label, base, checkpoint in jobs:
            for seed in seeds:
                cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
                with open_dict(cfg):
                    cfg.seed = seed
                    cfg.evaluation_stack = "full_mpr_mpc"
                    cfg.restore_checkpoint_runtime = False
                    cfg.benchmark = OmegaConf.to_container(raw_cfg.benchmark, resolve=True)
                    cfg.benchmark.mppi_eval_seed = seed
                effective, _ = suite._make_cfg(cfg, label, checkpoint, scenarios)
                OmegaConf.save(effective, output / f"config_{label}_seed{seed}.yaml")
                with (output / f"episodes_{label}_seed{seed}.csv").open("w", newline="") as seed_file:
                    writer = None

                    def save_episode(row):
                        nonlocal writer, combined_writer
                        row.update(configuration=label, test_seed=seed)
                        all_rows.append(row)
                        if writer is None:
                            writer = csv.DictWriter(seed_file, fieldnames=list(row))
                            writer.writeheader()
                        if combined_writer is None:
                            combined_writer = csv.DictWriter(combined_file, fieldnames=list(row))
                            combined_writer.writeheader()
                        writer.writerow(row)
                        combined_writer.writerow(row)
                        seed_file.flush()
                        combined_file.flush()

                    _, details = suite._evaluate_checkpoint(cfg, label, checkpoint, scenarios, on_episode=save_episode)
                detailed.append({"configuration": label, "test_seed": seed, **details})
                summaries = [summarize_runs(profile, [r for r in all_rows if r["configuration"] == profile],
                             samples=samples, seed=int(raw_cfg.seed)) for profile, _, _ in jobs
                             if any(r["configuration"] == profile for r in all_rows)]
                suite._write_csv(output / "summary.csv", summaries)
                suite._write_csv(output / "paired_comparisons.csv", paired_comparisons(
                    all_rows, samples=samples, seed=int(raw_cfg.seed), pairs=comparisons))
                (output / "summary.json").write_text(json.dumps(_clean(summaries), indent=2, allow_nan=False))
                (output / "detailed_summary.json").write_text(json.dumps(_clean(detailed), indent=2, allow_nan=False))
                print(f"[{label}] seed={seed} complete; results: {output}", flush=True)
    return output


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(raw_cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA for TD-MPC2.")
    run(raw_cfg, list(HydraConfig.get().overrides.task))


if __name__ == "__main__":
    main()
