"""V2 environment cost: dense risk plus every active raw event flag each step."""

from __future__ import annotations


class SafetyCostV2:
    def __init__(self, config, flag_fn):
        self.config = config
        self.flag_fn = flag_fn

    def reset(self) -> None:
        pass

    def compute(self, info, normalized_risk: float) -> tuple[float, float, float]:
        if not bool(self.config.get("enabled", True)):
            return 0.0, 0.0, 0.0
        risk_cost = max(
            float(self.config.get("risk_field_weight", 1.0)) * float(normalized_risk),
            0.0,
        )
        event_cost = 0.0
        for key, weight in self.config.get("event_weights", {}).items():
            if key == "cost":
                event_cost += float(weight) * float(info.get("cost", 0.0))
            else:
                event_cost += float(weight) * self.flag_fn(info, key)
        event_cost = max(event_cost, 0.0)
        return risk_cost + event_cost, risk_cost, event_cost
