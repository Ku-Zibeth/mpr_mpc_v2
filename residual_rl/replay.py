"""Fixed-size replay for one-step residual SAC transitions."""

from __future__ import annotations

import numpy as np
import torch


class ResidualReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int = 2):
        if int(capacity) <= 0:
            raise ValueError("Residual replay capacity must be positive.")
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.empty((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.costs = np.empty((self.capacity, 1), dtype=np.float32)
        self.next_states = np.empty((self.capacity, self.state_dim), dtype=np.float32)
        self.dones = np.empty((self.capacity, 1), dtype=np.float32)
        self.diagnostics = {key: np.full((self.capacity, size), np.nan, dtype=np.float32)
                            for key, size in {"lattice_predicted_cost": 1, "sac_predicted_cost": 1,
                                              "mppi_predicted_cost": 1, "mppi_correction_magnitude": 1,
                                              "executed_action": 2, "sac_risk_valid": 1}.items()}
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, state, action, reward: float, cost: float, next_state, done: bool, *, diagnostics=None) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1)
        if state.shape != (self.state_dim,) or next_state.shape != (self.state_dim,):
            raise ValueError("Residual replay state shape mismatch.")
        if action.shape != (self.action_dim,):
            raise ValueError("Residual replay action shape mismatch.")
        if not np.isfinite(state).all() or not np.isfinite(next_state).all():
            raise RuntimeError("Residual replay received non-finite state.")
        index = self.position
        self.states[index] = state
        self.actions[index] = np.clip(action, -1.0, 1.0)
        self.rewards[index, 0] = float(reward)
        self.costs[index, 0] = float(cost)
        self.next_states[index] = next_state
        self.dones[index, 0] = float(bool(done))
        for key, values in self.diagnostics.items():
            values[index] = np.asarray((diagnostics or {}).get(key, 0.0 if key == "sac_risk_valid" else np.nan))
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        *,
        device: str | torch.device,
        gamma: float,
    ) -> dict[str, torch.Tensor]:
        if self.size < int(batch_size):
            raise ValueError(f"Cannot sample {batch_size} from residual replay size {self.size}.")
        indices = np.random.randint(0, self.size, size=int(batch_size))
        arrays = {
            "state": self.states[indices],
            "action": self.actions[indices],
            "reward": self.rewards[indices],
            "cost": self.costs[indices],
            "next_state": self.next_states[indices],
            "done": self.dones[indices],
            "discount": np.full((int(batch_size), 1), float(gamma), dtype=np.float32),
        }
        arrays.update({key: values[indices] for key, values in self.diagnostics.items()})
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in arrays.items()
        }
