"""One-factor value-definition ablations; no new planning modes or scores."""
from copy import deepcopy

VALUE_ABLATIONS = {
    "D0": dict(temperature_definition="divisor", terminal_action="candidate", q_aggregation="random_pair"),
    "D1": dict(temperature_definition="inverse", terminal_action="candidate", q_aggregation="random_pair"),
    "D2": dict(temperature_definition="divisor", terminal_action="policy_prior", q_aggregation="random_pair"),
    "D3": dict(temperature_definition="divisor", terminal_action="candidate", q_aggregation="fixed_pair"),
    "D4": dict(temperature_definition="divisor", terminal_action="candidate", q_aggregation="ensemble_mean"),
}


def value_ablation_config(config, name):
    if name not in VALUE_ABLATIONS:
        raise ValueError(f"Unknown ablation {name}; choose {list(VALUE_ABLATIONS)}")
    result = deepcopy(dict(config))
    result.update(VALUE_ABLATIONS[name])
    return result
