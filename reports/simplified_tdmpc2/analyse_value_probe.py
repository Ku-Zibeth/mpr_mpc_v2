"""Compare two bounded offline probes without installing a planner qualification gate."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
results = []
for scenario, before in ((1714, 250), (559, 600)):
    sac = json.loads((root / f"probe_{scenario}_sac.json").read_text())
    mean = json.loads((root / f"probe_{scenario}_mean.json").read_text())
    assert sac["before"] == mean["before"] == before
    assert sac["observed_state_id"] == mean["observed_state_id"]
    assert sac["checkpoint_sha256"] == mean["checkpoint_sha256"]
    for field in ("observation", "baseline", "mean", "predicted_values"):
        assert np.array_equal(sac[field], mean[field]), (scenario, field)
    assert sac["episode_complete"] and mean["episode_complete"]
    actual_gain = mean["actual_discounted_return"] - sac["actual_discounted_return"]
    results.append({
        "scenario": scenario,
        "before_step": before,
        "recorded_state_and_candidates_identical": True,
        "predicted_gain": sac["predicted_gain"],
        "actual_discounted_gain": actual_gain,
        "ordering_agrees": bool(sac["predicted_gain"] * actual_gain > 0),
        "SAC_return": sac["actual_discounted_return"],
        "mean_return": mean["actual_discounted_return"],
        "SAC_crash": sac["crash"],
        "mean_crash": mean["crash"],
        "SAC_success": sac["success"],
        "mean_success": mean["success"],
        "scope": sac["scope"],
    })
output = {
    "origins": results,
    "scope": "Two prespecified observations, not a global ranking validation or latent-dynamics causal attribution. H+1 real candidate commands then a common SAC continuation; not receding-MPPI return. Matching recorded observations/ego/obstacles does not prove all hidden simulator states identical. No runtime gate or controller was added.",
}
(root / "value_ranking_probe.json").write_text(json.dumps(output, indent=2))
print(json.dumps(results, indent=2))
