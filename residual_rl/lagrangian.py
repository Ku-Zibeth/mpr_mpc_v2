"""Projected dual-gradient Lagrangian multiplier."""

from __future__ import annotations


class ProjectedLagrangian:
    """Non-negative multiplier updated once per completed episode."""

    def __init__(self, learning_rate: float = 0.01, initial_value: float = 0.0):
        self.learning_rate = float(learning_rate)
        if self.learning_rate <= 0.0:
            raise ValueError("lagrangian_lr must be positive.")
        self.value = max(float(initial_value), 0.0)

    def step(self, episode_cost: float, cost_limit: float) -> float:
        self.value = max(
            0.0,
            self.value + self.learning_rate * (float(episode_cost) - float(cost_limit)),
        )
        return self.value

    def state_dict(self) -> dict[str, float]:
        return {"learning_rate": self.learning_rate, "value": self.value}

    def load_state_dict(self, state: dict) -> None:
        if "learning_rate" in state:
            learning_rate = float(state["learning_rate"])
            if learning_rate <= 0.0:
                raise ValueError("Checkpoint lagrangian learning rate must be positive.")
            self.learning_rate = learning_rate
        self.value = max(float(state.get("value", 0.0)), 0.0)
