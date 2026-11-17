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


TERMINAL_WEIGHT_ABLATIONS = {"Q1": 1.0, "Q05": 0.5, "Q025": 0.25, "Q0": 0.0}


def terminal_weight_ablation_config(config, name):
    """D4 at effective beta=2; change ONLY lambda_Q, never sampling parameters."""
    if name not in TERMINAL_WEIGHT_ABLATIONS:
        raise ValueError(f"Unknown terminal-weight ablation {name}")
    result = value_ablation_config(config, "D4")
    # D4's old tau=.5 equals beta=2 exactly, NOT the default beta=.5.
    result.update(temperature_definition="inverse", temperature=2.0,
                  terminal_q_weight=TERMINAL_WEIGHT_ABLATIONS[name])
    return result


def iterations_ablation_config(config, iterations):
    """Independent experiment only; preserve every option except iterations.

    This does not select or change the online default. In particular, an H-step
    intervention experiment cannot establish receding-horizon driving benefit.
    """
    if iterations not in (1, 2, 4):
        raise ValueError("Iteration ablations require 1, 2 or 4 iterations.")
    result = deepcopy(dict(config))
    result["iterations"] = int(iterations)
    return result
