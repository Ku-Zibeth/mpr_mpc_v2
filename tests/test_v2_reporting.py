import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics, fixed_scenarios, write_report


def test_no_sampling_with_replacement_and_fixed_order(tmp_path):
    raw = OmegaConf.create({"seed": 1, "eval_episodes": 100,
        "metadrive": {"simulator": {"start_seed": 300, "num_scenarios": 50}}})
    with pytest.raises(ValueError, match="sampling with replacement is disabled"):
        fixed_scenarios(raw)
    raw.eval_episodes = 2
    raw.metadrive.simulator.num_scenarios = 2000
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps({"scenarios": [1714, 539]}))
    raw.evaluation_scenarios_file = str(path)
    assert fixed_scenarios(raw).tolist() == [1714, 539]
    raw.metadrive.simulator.num_scenarios = 2
    with pytest.raises(ValueError, match="configured simulator seed range"):
        fixed_scenarios(raw)


def test_episode_events_and_strict_json_summary(tmp_path):
    diagnostics = EpisodeDiagnostics()
    diagnostics.observe(np.zeros(3), np.zeros(2), {"cost": .2, "crash_vehicle": True, "risk_field_min_ttc": 2}, None)
    final = {"success": True, "crash": False, "cost": .1, "risk_field_min_ttc": 4}
    diagnostics.observe(np.zeros(3), np.zeros(2), final, None)
    first = {"episode": 1, "scenario": 1714, "success": 1., "cost": .3, "length": 2, "route_completion": 1.,
             **diagnostics.summary(final)}
    assert first["collision"] == first["collision_and_success_rate"] == 1
    second = {**first, "episode": 2, "scenario": 539, "collision": 0., "min_ttc": 5., "mean_ttc": 5., "ttc_valid_steps": 1}
    raw = OmegaConf.create({"seed": 1, "evaluation_output_dir": str(tmp_path)})
    write_report(raw, [1714, 539], [first, second], {"wm/cost_ready": 0, "cost_prediction_mae": float("nan")})
    summary = json.loads((tmp_path / "summary.json").read_text(), parse_constant=lambda x: pytest.fail(f"Nonstandard JSON: {x}"))
    assert summary["collision_rate"] == .5
    assert summary["min_ttc"] == 2
    assert summary["mean_ttc"] == pytest.approx(11/3)
    assert summary["cost_prediction_mae"] is None
    assert summary["actor_safety_loss"] is None
    assert (tmp_path / "episodes.csv").is_file()
