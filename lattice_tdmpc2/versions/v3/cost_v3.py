"""V3 environment cost with alias deduplication and one-shot contact events."""

from __future__ import annotations


class SafetyCostV3:
    def __init__(self, config, flag_fn):
        self.config = config
        self.flag_fn = flag_fn
        self.active_events: set[str] = set()

    def reset(self) -> None:
        self.active_events.clear()

    def _events(self, info) -> dict[str, float]:
        weights = self.config.get("event_weights", {})
        raw_cost = max(float(weights.get("cost", 0.0)) * float(info.get("cost", 0.0)), 0.0)
        crash_weight = float(weights.get("crash", 0.0))
        events: dict[str, float] = {}
        if self.flag_fn(info, "crash_vehicle"):
            events["crash_vehicle"] = max(
                raw_cost, crash_weight, float(weights.get("crash_vehicle", 0.0))
            )
        if self.flag_fn(info, "crash_object"):
            events["crash_object"] = max(
                raw_cost, crash_weight, float(weights.get("crash_object", 0.0))
            )
        if self.flag_fn(info, "out_of_road"):
            events["out_of_road"] = max(raw_cost, float(weights.get("out_of_road", 0.0)))
        if self.flag_fn(info, "out_of_road_warning"):
            events["out_of_road_warning"] = float(weights.get("out_of_road_warning", 0.0))
        if self.flag_fn(info, "crash") and not ({"crash_vehicle", "crash_object"} & events.keys()):
            events["crash_other"] = max(raw_cost, crash_weight)
        if raw_cost > 0.0 and not events:
            events["raw_cost"] = raw_cost
        return events

    def compute(self, info, normalized_risk: float) -> tuple[float, float, float]:
        if not bool(self.config.get("enabled", True)):
            self.reset()
            return 0.0, 0.0, 0.0
        risk_cost = max(
            float(self.config.get("risk_field_weight", 1.0)) * float(normalized_risk),
            0.0,
        )
        current_events = self._events(info)
        new_events = current_events.keys() - self.active_events
        event_cost = max((current_events[key] for key in new_events), default=0.0)
        self.active_events = set(current_events)
        return risk_cost + event_cost, risk_cost, event_cost
