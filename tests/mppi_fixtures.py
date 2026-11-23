"""Shared exact fixtures retained from the original tests, without obsolete test-version imports."""
import torch

def mppi_cfg(**updates):
    cfg = {
        "cost_enabled": False,
        "num_samples": 24,
        "num_elites": 6,
        "iterations": 3,
        "initial_std": [0.025, 0.10],
        "min_std": [0.005, 0.02],
        "max_std": [0.05, 0.20],
        "max_delta": [0.05, 0.20],
        "temperature": 0.5,
        "proposal_init": "structured_baseline",
        "deviation_coef": 0.0,
        "smoothness_coef": 0.0,
        "min_improvement_abs": 0.0,
        "min_improvement_ratio": 0.0,
        "eval_seed": 7,
        # Legacy algorithm-only fixtures opt out explicitly; V3 integration
        # tests supply live geometry and keep safety enabled.
        "safety": {"enabled": False},
        # Normal-search unit fixtures; V3.1 regressions enable recovery explicitly.
        "recovery": {"enabled": False},
    }
    cfg.update(updates)
    return cfg

class ExactCostModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_cost_updates", torch.tensor(2000))
        self.register_buffer("_cost_validated", torch.tensor(False))
        self.register_buffer("_cost_validated_horizons", torch.tensor(0))
        self.register_buffer("_cost_validated_at", torch.tensor(0))
        self.register_buffer("_cost_validation_schema", torch.tensor(0))

    @property
    def cost_ready(self):
        return bool(self._cost_validated and self._cost_validated_at == self._cost_updates)

    def invalidate_cost_validation(self):
        for name, value in self.named_buffers():
            if name != "_cost_updates":
                value.zero_()

    def encode(self, x, task):
        return x

    def next(self, z, action, task):
        return z + 1

    def cost(self, z, action, task):
        return action[:, :1].abs()

import numpy as np
from lattice.frenet_metadrive import FrenetPath

def make_path(horizon=2.0, dt=0.1, target_d=0.2, target_speed=10.0):
    times = np.arange(0.0, horizon + 0.5 * dt, dt)
    s = target_speed * times
    d = target_d * (times / horizon) ** 2
    d_d = np.gradient(d, dt)
    d_dd = np.gradient(d_d, dt)
    d_ddd = np.gradient(d_dd, dt)
    zeros = np.zeros_like(times)
    return FrenetPath(
        t=times.tolist(), d=d.tolist(), d_d=d_d.tolist(), d_dd=d_dd.tolist(),
        d_ddd=d_ddd.tolist(), s=s.tolist(), s_d=np.full_like(times, target_speed).tolist(),
        s_dd=zeros.tolist(), s_ddd=zeros.tolist(), x=s.tolist(), y=d.tolist(),
        yaw=np.arctan2(d_d, np.full_like(d_d, target_speed)).tolist(),
        target_d=target_d, target_speed=target_speed, horizon=horizon,
    )
