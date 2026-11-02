"""SAC actor and critic networks for normalized residual actions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = int(input_dim)
    for hidden in hidden_dims:
        layers.extend([nn.Linear(previous, int(hidden)), nn.ReLU()])
        previous = int(hidden)
    layers.append(nn.Linear(previous, int(output_dim)))
    network = nn.Sequential(*layers)
    for module in network.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)
    return network


class GaussianActor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        *,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.network = mlp(state_dim, hidden_dims, 2 * self.action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be less than log_std_max.")

    def distribution(self, state: torch.Tensor) -> tuple[Normal, torch.Tensor, torch.Tensor]:
        mean, raw_log_std = self.network(state).chunk(2, dim=-1)
        log_std = raw_log_std.clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp()), mean, log_std

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        distribution, mean, log_std = self.distribution(state)
        if deterministic:
            return torch.tanh(mean), None, mean, log_std
        raw_action = distribution.rsample()
        action = torch.tanh(raw_action)
        correction = torch.log(1.0 - action.square() + 1e-6)
        log_prob = (distribution.log_prob(raw_action) - correction).sum(-1, keepdim=True)
        return action, log_prob, mean, log_std


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.network = mlp(state_dim + action_dim, hidden_dims, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, action], dim=-1))
