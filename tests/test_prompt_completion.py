import json
from pathlib import Path
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

import evaluate_improvements as improvements
from mpr_mpc._bootstrap import PACKAGE_ROOT
from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics
from mpr_mpc.lattice_tdmpc2.versions.v3.cost_v3 import SafetyCostV3
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.safety import TrajectorySafetyEvaluator
from tests.test_v2_safety import context
from tests.test_mpr_mpc import RecordingEvaluator, mppi_cfg
from tests.test_evaluation_ablation import FakeAgent, FakeEnv








@pytest.mark.parametrize("horizon", [10, 15, 20])
def test_native_planning_horizon_contract(horizon):
    planner = LocalMPPI(RecordingEvaluator(), mppi_cfg(num_samples=64),
                        horizon=horizon, action_dim=2, device="cpu")
    result = planner.plan(torch.zeros(1, 8), torch.zeros(horizon + 1, 2), eval_mode=True)
    assert result.selected_sequence.shape == (horizon + 1, 2)
    assert not hasattr(planner, "_mode_bank") and not hasattr(planner, "safety_evaluator")
    assert torch.isfinite(result.action).all()


def test_native_mppi_rejects_nonfinite_baseline():
    planner = LocalMPPI(RecordingEvaluator(), mppi_cfg(), horizon=10, action_dim=2, device="cpu")
    with pytest.raises(ValueError, match="baseline must be finite"):
        planner.plan(torch.zeros(1, 8), torch.full((11, 2), float("nan")), eval_mode=True)


def test_lateral_acceleration_limit_includes_final_substep():
    evaluator = TrajectorySafetyEvaluator({"max_lateral_acceleration": .2}, control_dt=.1, device="cpu")
    ctx = context(speed=1.)
    ctx.previous_action = np.array([.5, 1.])
    actions = torch.tensor([[[.5, 1.], [.5, 1.]]])
    assert evaluator.check(actions, ctx).kinematic_violation.item()
    ctx.speed = float("nan")
    with pytest.raises(ValueError, match="physically valid"):
        evaluator.check(actions, ctx)


def test_event_attribution_preserves_total_without_relabelling_soft_risk_as_collision():
    cost = SafetyCostV3({"enabled": True, "event_weights": {"crash": 1., "out_of_road": 1.}},
                        lambda info, key: float(bool(info.get(key, False))))
    total, risk, event = cost.compute({"crash_vehicle": True}, .3)
    assert (total, risk, event) == (1.3, .3, 1.)
    assert cost.last_components["dynamic_collision"] == 1
    assert sum(cost.last_components.values()) == event
    assert cost.compute({"crash_vehicle": True}, .3)[2] == 0
    assert sum(cost.last_components.values()) == 0
    cost.reset()
    assert cost.compute({"crash_vehicle": True, "out_of_road": True}, .3)[2] == 1
    assert cost.last_components["unallocated_event"] == 1
    assert cost.last_components["dynamic_collision"] == 0
    cost.reset()
    assert cost.compute({}, .3) == (.3, .3, 0)


def test_prompt_metric_names_and_mppi_off_values():
    off = EpisodeDiagnostics()
    result = SimpleNamespace(stage=SimpleNamespace(use_mppi=False, use_residual=False), metrics={
        "mpr/baseline_selected": 1., "mpr/no_safe_candidate": 0., "mpr/sac_predicted_cost": .1,
        "mpr/baseline_value": float("nan")})
    off.observe(np.zeros(3), np.zeros(2), {"risk_field_vehicle_cost": 3.}, result)
    summary = off.summary({})
    assert np.isnan(summary["baseline_selected_rate"])
    assert np.isnan(summary["no_safe_candidate_rate"])
    assert np.isnan(summary["sac_predicted_cost"])
    assert np.isnan(summary["mean_mpr_baseline_value"])
    assert summary["dynamic_risk_field_cost"] == 3
    assert summary["dynamic_collision_cost"] == 0


def checkpoint(path, profile, *, labelled=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    features = ["cumulative_cost", "min_ttc", "min_clearance", "boundary_margin"] if profile == "E5" else []
    torch.save({"metadata": {"experiment_profile": profile if labelled else None, "risk_features": features},
        "model": {"_encoder.state.0.weight": torch.zeros(2, 2), "_cost_updates": torch.tensor(2)},
        "mpr_mpc": {"residual_sac": {"update_steps": 4, "constraint_alignment": profile != "E0"}}}, path)


def test_profile_contract_rejects_unlabelled_and_wrongly_labelled_models(tmp_path):
    with initialize_config_dir(version_base="1.3", config_dir=str(PACKAGE_ROOT)):
        cfg = compose(config_name="config", overrides=["+experiment=E1"])
        path = tmp_path / "model.pt"
        checkpoint(path, "E0")
        with pytest.raises(ValueError, match="checkpoint profile"):
            improvements.checkpoint_contract(path, "E1", cfg)
        checkpoint(path, "E1", labelled=False)
        with pytest.raises(ValueError, match="checkpoint profile"):
            improvements.checkpoint_contract(path, "E1", cfg)
        assert improvements.checkpoint_contract(path, "E1", cfg, allow_unlabelled=True)["sac_updates"] == 4


def test_all_six_profiles_share_scenarios_and_repeat_seeds(tmp_path, monkeypatch):
    for profile in improvements.PROFILES:
        checkpoint(tmp_path / f"mpr_mpc_v2_{profile}/models/model.pt", profile)
    seen = []

    def make_agent(cfg, env):
        seen.append((cfg.seed, cfg.benchmark["paired_episode_seeds"], cfg.benchmark["mppi_eval_seed"]))
        return FakeAgent()

    monkeypatch.setattr(improvements.suite, "make_env", lambda cfg: FakeEnv())
    monkeypatch.setattr(improvements.suite, "_make_agent", make_agent)
    overrides = ["improvements.episodes=2", "improvements.repeat_seeds=[1,7]",
        "improvements.bootstrap_samples=20", "improvements.checkpoint_name=model.pt",
        f"improvements.models_root={tmp_path}", f"improvements.output_dir={tmp_path}/results",
        "evaluation_scenarios=[1714,539]"]
    with initialize_config_dir(version_base="1.3", config_dir=str(PACKAGE_ROOT)):
        raw = compose(config_name="config", overrides=overrides)
        output = improvements.run(raw, overrides)
    assert seen == [(1, True, 1), (7, True, 7)] * 6
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(set(manifest["checkpoints"].values())) == 6
    assert json.loads((output / "scenarios.json").read_text())["scenarios"] == [1714, 539]
    summary = json.loads((output / "summary.json").read_text())
    assert [row["configuration"] for row in summary] == list(improvements.PROFILES)
    assert all(row["episodes"] == 4 and row["unique_scenarios"] == 2 for row in summary)
    assert (output / "paired_comparisons.csv").is_file()
