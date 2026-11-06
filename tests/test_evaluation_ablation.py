from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from omegaconf import OmegaConf

import evaluate_ablation as ablation
import evaluate_model_suite as suite
from evaluation_statistics import paired_comparisons, summarize_runs


class FakePlanner:
    mppi_start_step = 500001
    allow_residual = True
    allow_mppi = True

    def reset(self):
        pass

    def stage_for_step(self, step):
        return SimpleNamespace(
            use_residual=step >= 200000 and self.allow_residual,
            use_mppi=step >= self.mppi_start_step and self.allow_mppi,
        )


class FakeAgent:
    def validate_cost_model(self, episodes):
        return {}

    def __init__(self):
        self.planner = FakePlanner()
        self.planner.local_mppi = SimpleNamespace(eval_seed=0)
        self.global_env_step = 500000
        self.last_plan = None
        self.executed = []

    def load(self, checkpoint):
        return {"milestone_type": "world_model_residual_rl"}

    def set_global_step(self, step):
        self.global_env_step = step

    def eval(self):
        pass

    def act(self, obs, **kwargs):
        stage = self.planner.stage_for_step(self.global_env_step)
        self.executed.append((stage.use_residual, stage.use_mppi))
        self.last_plan = SimpleNamespace(stage=stage, metrics={
            "mpr/baseline_selected": float(len(self.executed) % 2),
            "mpr/planner_score_gain": 0.0 if len(self.executed) % 2 else 2.0,
        })
        return np.zeros(2)


class FakeEnv:
    def __init__(self, remap=False):
        self.unwrapped_metadrive = self
        self.remap = remap
        self.closed = False

    def reset(self, *, seed):
        self.current_seed = seed + int(self.remap)
        self.steps = 0
        return np.zeros(1)

    def step(self, action):
        self.steps += 1
        done = self.steps == 2
        return np.zeros(1), 1.0, done, {
            "crash_vehicle": not done,
            "out_of_road": not done,
            "success": done,
            "route_completion": 0.98 if done else 0.5,
            "risk_field_min_ttc": 2.0 if done else 4.0,
            "cost": 3.0,
        }

    def close(self):
        self.closed = True


@pytest.mark.parametrize("label,stack,expected", [
    ("A_lattice", "world_model", (False, False)),
    ("B_lattice_sac", "world_model_residual_rl", (True, False)),
    ("C_lattice_mppi", "lattice_mppi", (False, True)),
    ("D_lattice_sac_mppi", "full_mpr_mpc", (True, True)),
])
def test_ablation_executes_requested_stack_and_counts_mid_episode_events(label, stack, expected):
    cfg = OmegaConf.create({
        "seed": 1, "evaluation_stack": stack,
        "benchmark": {"paired_episode_seeds": True, "mppi_eval_seed": 7},
        "residual_rl": {"cost_limit": 5.0},
    })
    env, agent, saved, random_seeds = FakeEnv(), FakeAgent(), [], []
    with (
        patch.object(suite, "_make_cfg", return_value=(cfg, cfg)),
        patch.object(suite, "make_env", return_value=env),
        patch.object(suite, "_make_agent", return_value=agent),
        patch.object(suite, "set_seed", side_effect=random_seeds.append),
    ):
        rows, summary = suite._evaluate_checkpoint(
            cfg, "old_model", Path("checkpoint.pt"), [1714, 1525], on_episode=saved.append
        )
    assert agent.executed == [expected] * 4
    assert agent.global_env_step == 500001
    assert agent.planner.local_mppi.eval_seed == 7
    assert random_seeds == [1, 1715, 1526]
    assert [r["episode_seed"] for r in rows] == [1715, 1526]
    assert [r["scenario"] for r in rows] == [1714, 1525]
    assert saved == rows
    assert all(r["final_crash_vehicle"] == 0.0 for r in rows)
    assert all(r["final_out_of_road"] == 0.0 for r in rows)
    assert summary["mean_episode_crash"] == 1.0
    assert summary["mean_episode_out_of_road"] == 1.0
    assert "mean_episode_seed" not in summary
    for row in rows:
        assert row["collision_and_success"] == row["collision_and_offroad"] == 1.0
        assert row["mean_ttc"] == 3.0
        assert row["minimum_ttc"] == 2.0
        assert row["ttc_valid_steps"] == 2
        assert row["episode_cost_over_limit"] == 1.0
        assert row["first_crash_step"] == row["first_out_of_road_step"] == 1
        if expected[1]:
            assert row["mppi_steps"] == 2
            assert row["mppi_baseline_selected"] == 0.5
            assert row["mppi_score_gain"] == 1.0
        else:
            assert row["mppi_steps"] == 0
            assert math.isnan(row["mppi_baseline_selected"])
            assert math.isnan(row["mppi_score_gain"])
    assert env.closed


def test_paired_evaluation_rejects_remapped_scenarios():
    cfg = OmegaConf.create({
        "seed": 1, "evaluation_stack": "lattice_mppi",
        "benchmark": {"paired_episode_seeds": True},
    })
    env, agent = FakeEnv(remap=True), FakeAgent()
    with (
        patch.object(suite, "_make_cfg", return_value=(cfg, cfg)),
        patch.object(suite, "make_env", return_value=env),
        patch.object(suite, "_make_agent", return_value=agent),
        patch.object(suite, "set_seed"),
        pytest.raises(RuntimeError, match="remapped scenario seed"),
    ):
        suite._evaluate_checkpoint(cfg, "old_model", Path("checkpoint.pt"), [1714])
    assert env.closed


def test_scenario_order_matches_previous_random100_run():
    cfg = OmegaConf.create({
        "seed": 1, "eval_episodes": 100,
        "metadrive": {"simulator": {"start_seed": 300, "num_scenarios": 2000}},
        "benchmark": {"episodes": 100, "scenario_start": 300, "scenario_pool_count": 2000},
    })
    scenarios = suite._scenarios(cfg)
    assert len(set(scenarios)) == 100
    assert scenarios[:5] == [1714, 1525, 539, 559, 1084]
    assert scenarios[-5:] == [1991, 2024, 1719, 1828, 1869]


def test_fixed_scenarios_extend_without_changing_prefix(tmp_path):
    fixed = [1714, 1525, 539, 559]
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": fixed}))
    cfg = OmegaConf.create({
        "seed": 3, "eval_episodes": 20,
        "metadrive": {"simulator": {"start_seed": 300, "num_scenarios": 2000}},
        "benchmark": {"episodes": 20, "scenario_start": 300, "scenario_pool_count": 2000,
                      "scenarios_file": str(path), "extend_scenarios": True, "scenario_seed": 1},
    })
    ids = suite._scenarios(cfg)
    assert ids[:len(fixed)] == fixed
    assert len(ids) == len(set(ids)) == 20
    cfg.seed = 9
    assert suite._scenarios(cfg) == ids
    cfg.benchmark.extend_scenarios = False
    with pytest.raises(ValueError, match="length differs"):
        suite._scenarios(cfg)
    path.write_text(json.dumps([1714, 1714]))
    with pytest.raises(ValueError, match="duplicate"):
        suite._scenarios(cfg)


def test_missing_ttc_is_not_zero():
    class NoTTCEnv(FakeEnv):
        def step(self, action):
            obs, reward, done, info = super().step(action)
            info["risk_field_min_ttc"] = float("nan") if done else float("inf")
            return obs, reward, done, info

    cfg = OmegaConf.create({"seed": 1, "evaluation_stack": "world_model"})
    with (
        patch.object(suite, "_make_cfg", return_value=(cfg, cfg)),
        patch.object(suite, "make_env", return_value=NoTTCEnv()),
        patch.object(suite, "_make_agent", return_value=FakeAgent()),
        patch.object(suite, "set_seed"),
    ):
        rows, _ = suite._evaluate_checkpoint(cfg, "old", Path("checkpoint.pt"), [1714])
    assert rows[0]["ttc_valid_steps"] == 0
    assert math.isnan(rows[0]["mean_ttc"])
    assert math.isnan(rows[0]["minimum_ttc"])


def diagnostic_row(scenario, test_seed, success, *, configuration="A_lattice"):
    return {
        "scenario": scenario, "test_seed": test_seed, "configuration": configuration,
        "episode_out_of_road": 1.0 - success, "episode_crash": 0.0,
        "final_route_completion": success, "final_success": success,
        "collision_and_success": 0.0, "collision_and_offroad": 0.0,
        "mean_ttc": float("nan"), "minimum_ttc": float("nan"), "ttc_valid_steps": 0,
        "length": 10, "sum_cost": 2.0, "cost_limit": 5.0, "episode_cost_over_limit": 0.0,
        "mppi_baseline_selected": float("nan"), "mppi_score_gain": float("nan"),
    }


def test_episode_csv_expands_schema_without_losing_values():
    stream = io.StringIO(newline="")
    writer = ablation._EpisodeCSVWriter(stream)
    rows = [
        {"configuration": "A", "scenario": 1714, "end_reason": "arrive_dest"},
        {"configuration": "B", "scenario": 1714, "mean_mpr_sac_inference_cpu_wall_ms": 1.25},
        {"configuration": "C", "scenario": 1714, "mean_mpr_mppi_cpu_wall_ms": 75.0},
        {"configuration": "D", "scenario": 1714, "mean_mpr_sac_inference_cpu_wall_ms": 2.0,
         "mean_mpr_mppi_cpu_wall_ms": 80.0},
        {"configuration": "D", "scenario": 1525, "late_metric": 3.0},
        {"configuration": "D", "scenario": 539},
    ]
    for index, row in enumerate(rows, 1):
        writer.writerow(row)
        saved = list(csv.DictReader(io.StringIO(stream.getvalue())))
        assert len(saved) == index
        assert all(None not in entry for entry in saved)
        for original, entry in zip(rows[:index], saved):
            assert all(entry[key] == str(value) for key, value in original.items())
    assert saved[0]["mean_mpr_sac_inference_cpu_wall_ms"] == ""
    assert saved[0]["end_reason"] == "arrive_dest"
    assert saved[-1]["late_metric"] == ""


def test_summary_includes_late_metrics_and_ignores_missing_values():
    summary = suite._summarize("model", Path("checkpoint.pt"), [
        {"episode": 1, "value": 2.0, "optional": 4.0},
        {"episode": 2, "value": 4.0, "late": 8.0},
        {"episode": 3, "value": 6.0, "optional": 6.0, "late": 12.0},
    ])
    assert summary["mean_value"] == 4.0
    assert summary["mean_optional"] == 5.0
    assert summary["mean_late"] == 10.0
    assert summary["std_late"] == 2.0


def test_cluster_intervals_do_not_treat_repeats_as_new_scenarios():
    rows = [diagnostic_row(i, 1, float(i % 2)) for i in range(10)]
    repeated = [{**row, "test_seed": seed} for seed in (1, 2, 3) for row in rows]
    a = summarize_runs("A", rows, samples=300, seed=1)
    b = summarize_runs("A", repeated, samples=300, seed=1)
    assert a["unique_scenarios"] == b["unique_scenarios"] == 10
    assert b["episodes"] == 30
    for key in ("success", "success_ci95_low", "success_ci95_high"):
        assert a[key] == pytest.approx(b[key])
    assert b["mean_ttc"] is b["minimum_ttc"] is b["mppi_score_gain"] is None
    assert b["offroad_given_collision"] is None
    assert b["collision_given_success"] == 0.0
    json.dumps(b, allow_nan=False)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_runs("A", rows + rows, samples=30, seed=1)


def test_paired_differences_match_by_scenario_and_seed():
    a = [diagnostic_row(i, seed, float(i % 2)) for seed in (1, 2) for i in range(2)]
    b = [{**r, "configuration": "B_lattice_sac", "final_success": 1.0} for r in a]
    comparisons = paired_comparisons(a + list(reversed(b)), samples=100, seed=1)
    assert len(comparisons) == 1
    assert comparisons[0]["paired_episodes"] == 4
    assert comparisons[0]["unique_scenarios"] == 2
    assert comparisons[0]["success_delta"] == 0.5
    assert comparisons[0]["success_rescued_count"] == 2
    assert comparisons[0]["success_regressed_count"] == 0


def test_unobserved_collision_rate_has_nonzero_upper_bound():
    rows = [diagnostic_row(i, seed, 1.0) for i in range(100) for seed in (1, 2, 3)]
    result = summarize_runs("A", rows, samples=50, seed=1)
    assert result["crash"] == 0.0
    assert result["crash_ci95_low"] == 0.0
    assert result["crash_ci95_high"] == pytest.approx(1 - 0.025**(1/100))
    assert result["success_ci95_low"] == pytest.approx(0.025**(1/100))
    assert result["success_ci95_high"] == 1.0


def test_multiseed_entry_writes_diagnostics_and_reuses_fixed_ids(tmp_path):
    checkpoint = tmp_path / "old" / "models" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    scenario_file = tmp_path / "fixed.json"
    scenario_file.write_text(json.dumps({"scenarios": [1714, 1525]}))
    output = tmp_path / "results"
    cfg = OmegaConf.create({
        "seed": 1, "checkpoint": str(checkpoint), "horizon": 10,
        "metadrive": {"simulator": {"start_seed": 100, "num_scenarios": 50}},
        "benchmark": {"episodes": 2, "scenarios_file": str(scenario_file),
                      "repeat_seeds": [1, 2], "bootstrap_samples": 30, "output_dir": str(output)},
    })
    calls = []
    labels = {stack: label for label, stack in ablation.VARIANTS}

    def evaluate(raw, version, path, scenarios, *, on_episode):
        calls.append((raw.evaluation_stack, raw.seed, list(scenarios), raw.benchmark.mppi_eval_seed))
        assert raw.metadrive.simulator.start_seed <= min(scenarios)
        assert raw.metadrive.simulator.start_seed + raw.metadrive.simulator.num_scenarios > max(scenarios)
        rows = [diagnostic_row(scenario, raw.seed, float(i % 2),
                               configuration=labels[raw.evaluation_stack])
                for i, scenario in enumerate(scenarios)]
        if raw.evaluation_stack in {"world_model_residual_rl", "full_mpr_mpc"}:
            for row in rows:
                row["mean_mpr_sac_inference_cpu_wall_ms"] = 1.25
        if raw.evaluation_stack in {"lattice_mppi", "full_mpr_mpc"}:
            rows[1]["mean_mpr_mppi_cpu_wall_ms"] = 75.0
        for row in rows:
            on_episode(row)
        return rows, suite._summarize(version, path, rows)

    with (
        patch.object(ablation.torch.cuda, "is_available", return_value=True),
        patch.object(suite, "_make_cfg", side_effect=lambda raw, *_: (raw, raw)),
        patch.object(suite, "_evaluate_checkpoint", side_effect=evaluate),
    ):
        ablation.main.__wrapped__(cfg)
    assert len(calls) == 8
    assert all(scenarios == [1714, 1525] and seed == mppi_seed
               for _, seed, scenarios, mppi_seed in calls)
    summaries = list(csv.DictReader((output / "summary.csv").open()))
    assert len(summaries) == 4
    assert all(r["unique_scenarios"] == "2" and r["episodes"] == "4" for r in summaries)
    assert len(list(csv.DictReader((output / "seed_summary.csv").open()))) == 8
    episodes = list(csv.DictReader((output / "episodes.csv").open()))
    assert len(episodes) == 16
    assert all(None not in row for row in episodes)
    assert episodes[0]["mean_mpr_sac_inference_cpu_wall_ms"] == ""
    assert episodes[4]["mean_mpr_sac_inference_cpu_wall_ms"] == "1.25"
    assert episodes[9]["mean_mpr_mppi_cpu_wall_ms"] == "75.0"
    mppi_rows = list(csv.DictReader((output / "episodes_C_lattice_mppi_seed1.csv").open()))
    assert mppi_rows[0]["mean_mpr_mppi_cpu_wall_ms"] == ""
    assert mppi_rows[1]["mean_mpr_mppi_cpu_wall_ms"] == "75.0"
    assert len(list(csv.DictReader((output / "paired_comparisons.csv").open()))) == 3
    assert json.loads((output / "scenarios.json").read_text())["scenarios"] == [1714, 1525]


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1], [True]])
def test_invalid_repeat_seeds(seeds):
    with pytest.raises(ValueError):
        ablation._repeat_seeds(OmegaConf.create({"repeat_seeds": seeds}), 1)
