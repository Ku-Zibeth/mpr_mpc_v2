"""Offline shared scalar candidate ranking. Never imported by action selection.

Inputs are current encoded observation, imagined terminal latent and planned
physical commands only. Actual future observations/rewards are labels, not inputs.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CandidateValue(nn.Module):
    def __init__(self, latent_dim=512, horizon=10, action_dim=2,
                 input_kind="context_terminal_actions", hidden=(66, 64)):
        super().__init__()
        if input_kind not in {"terminal", "context_terminal", "context_terminal_actions"}:
            raise ValueError("Unknown candidate input definition.")
        self.latent_dim, self.horizon, self.action_dim = latent_dim, horizon, action_dim
        self.input_kind, self.hidden = input_kind, tuple(hidden)
        width = latent_dim if input_kind == "terminal" else 2 * latent_dim
        if input_kind == "context_terminal_actions":
            width += horizon * action_dim
        self.register_buffer("input_mean", torch.zeros(width))
        self.register_buffer("input_std", torch.ones(width))
        # Pairwise supervision cannot identify a common absolute-return offset.
        self.register_buffer("difference_scale", torch.ones(()))
        layers = []
        for i, (left, right) in enumerate(zip((width, *hidden), (*hidden, 1))):
            layers.append(nn.Linear(left, right))
            if i < len(hidden):
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def features(self, z0, terminal, actions):
        if terminal.ndim != 2 or terminal.shape[-1] != self.latent_dim:
            raise ValueError("Expected a batch of imagined terminal latents.")
        if self.input_kind == "terminal":
            return terminal
        z0 = z0.reshape(-1, self.latent_dim).expand_as(terminal)
        pieces = [z0, terminal]
        if self.input_kind == "context_terminal_actions":
            if actions.ndim != 3 or actions.shape[0] != len(terminal) or actions.shape[1] < self.horizon or actions.shape[2] != self.action_dim:
                raise ValueError("Expected matching H-command candidate sequences.")
            pieces.append(actions[:, :self.horizon].flatten(1))
        return torch.cat(pieces, -1)

    def forward(self, z0, terminal, actions):
        x = self.features(z0, terminal, actions)
        return self.net((x - self.input_mean) / self.input_std.clamp_min(1e-3)).squeeze(-1) * self.difference_scale


def pairwise_objective(scores, returns, pairs, *, weights=None, tolerance=1.,
                       difference_scale=1., difference_weight=.25):
    """Rank non-ties, regress all differences; balance scenes/origins upstream.

    The tolerance is in native discounted-return units. Near-ties are retained
    for difference fitting, never turned into noisy binary preferences.
    """
    if tolerance < 0 or difference_scale <= 0 or difference_weight < 0:
        raise ValueError("Invalid fixed pairwise loss scales.")
    pairs = pairs.reshape(-1, 2)
    prediction = (scores[pairs[:, 0]] - scores[pairs[:, 1]]) / difference_scale
    target = (returns[pairs[:, 0]] - returns[pairs[:, 1]]) / difference_scale
    weight = torch.ones_like(target) if weights is None else weights.to(target)
    active = target.abs() > tolerance / difference_scale
    rank_terms = F.softplus(-target.sign() * prediction)
    rank = (weight[active] * rank_terms[active]).sum() / weight[active].sum().clamp_min(1e-12)
    difference = (weight * F.smooth_l1_loss(prediction, target, reduction="none")).sum() / weight.sum().clamp_min(1e-12)
    return rank + difference_weight * difference, dict(rank=rank, difference=difference, effective_pairs=int(active.sum()))


def aligned_returns(rewards, starts, *, horizon, gamma, done):
    """Native terminal RTG; no invented H-endpoint for an early failure.

    Returns total/prefix/continuation labels and a valid-H mask. A prefix that
    ends early still has a real total label, but no terminal latent is supplied.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    done = np.asarray(done, dtype=bool)
    if not 0 <= gamma <= 1 or horizon <= 0 or len(done) != len(rewards):
        raise ValueError("Invalid native reward/done alignment.")
    if not len(done) or not done[-1] or done[:-1].any() or (starts < 0).any() or (starts >= len(rewards)).any():
        raise ValueError("Require one native episode, no post-terminal samples.")
    rtg = np.zeros(len(rewards) + 1)
    for i in range(len(rewards) - 1, -1, -1):
        rtg[i] = rewards[i] + gamma * rtg[i + 1]
    valid = starts + horizon <= len(rewards)
    continuation = np.full(len(starts), np.nan)
    continuation[valid] = rtg[starts[valid] + horizon]
    prefix = np.array([sum(gamma**j * r for j, r in enumerate(rewards[s:s + horizon])) for s in starts])
    return dict(total=rtg[starts], prefix=prefix, continuation=continuation,
                valid_terminal=valid, rtg=rtg)
