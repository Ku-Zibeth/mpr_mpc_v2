"""Current-world-model targets and replay for two-dimensional residual supervision."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ResidualExample:
    observation: torch.Tensor
    coarse_parameters: torch.Tensor
    coarse_actions: torch.Tensor
    residual_bounds: torch.Tensor
    target_residual: torch.Tensor


class ResidualSupervisionBuffer:
    """Store raw planner inputs so evolving encoder/WM features never become stale."""

    def __init__(self, capacity: int):
        self._items: deque[ResidualExample] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self._items)

    def add(self, context, target_residual: torch.Tensor) -> None:
        observation = context.observation
        if not torch.is_tensor(observation):
            observation = torch.as_tensor(observation, dtype=torch.float32)
        self._items.append(
            ResidualExample(
                observation=observation.detach().cpu().clone(),
                coarse_parameters=context.coarse_parameters.squeeze(0).detach().cpu(),
                coarse_actions=context.coarse_actions.detach().cpu(),
                residual_bounds=context.residual_bounds.squeeze(0).detach().cpu(),
                target_residual=target_residual.reshape(2).detach().cpu(),
            )
        )

    def sample(self, batch_size: int, device) -> ResidualExample:
        if len(self) < int(batch_size):
            raise RuntimeError("Residual supervision buffer does not contain a full batch.")
        indices = np.random.choice(len(self), size=int(batch_size), replace=False)
        examples = [self._items[int(index)] for index in indices]
        stack = lambda name: torch.stack([getattr(item, name) for item in examples]).to(device)
        return ResidualExample(
            observation=stack("observation"),
            coarse_parameters=stack("coarse_parameters"),
            coarse_actions=stack("coarse_actions"),
            residual_bounds=stack("residual_bounds"),
            target_residual=stack("target_residual"),
        )


class ResidualTargetGenerator:
    """Search zero-centred local corrections around the selected coarse path."""

    def __init__(
        self,
        structured,
        action_adapter,
        evaluator,
        *,
        horizon: int,
        control_dt: float,
        sample_count: int = 32,
        elite_count: int = 4,
        target_mode: str = "softmax_elite",
        temperature: float = 0.5,
        distribution: str = "truncated_gaussian",
        std_scale: float = 0.25,
        min_improvement_abs: float = 1.0,
        min_improvement_ratio: float = 0.02,
        parameter_clamper=None,
    ):
        self.structured = structured
        self.action_adapter = action_adapter
        self.evaluator = evaluator
        self.horizon = int(horizon)
        self.control_dt = float(control_dt)
        self.sample_count = int(sample_count)
        self.elite_count = int(elite_count)
        self.target_mode = str(target_mode)
        self.temperature = float(temperature)
        self.distribution = str(distribution)
        self.std_scale = float(std_scale)
        self.min_improvement_abs = float(min_improvement_abs)
        self.min_improvement_ratio = float(min_improvement_ratio)
        self.parameter_clamper = parameter_clamper
        if self.sample_count < 1 or not 1 <= self.elite_count <= self.sample_count:
            raise ValueError("Require 1 <= elite_count <= sample_count.")
        if self.target_mode not in {"best", "softmax_elite"}:
            raise ValueError("target_mode must be best or softmax_elite.")
        if self.distribution != "truncated_gaussian":
            raise ValueError("Only target_distribution=truncated_gaussian is supported.")
        if self.std_scale <= 0:
            raise ValueError("target_std_scale must be positive.")

    def sample_residuals(self, delta_bounds: np.ndarray) -> np.ndarray:
        bounds = np.asarray(delta_bounds, dtype=np.float32).reshape(2)
        samples = np.random.normal(
            loc=0.0,
            scale=self.std_scale * bounds,
            size=(self.sample_count, 2),
        ).astype(np.float32)
        samples = np.clip(samples, -bounds, bounds)
        samples[0] = 0.0
        return samples

    @torch.no_grad()
    def generate(self, context, vehicle) -> tuple[torch.Tensor, dict[str, float]]:
        coarse = context.selection.coarse_path
        coarse_parameters = np.asarray(
            [coarse.target_d, coarse.target_speed, coarse.horizon], dtype=np.float32
        )
        samples = self.sample_residuals(
            context.residual_bounds.squeeze(0).detach().cpu().numpy()
        )
        paths, residuals = [], []
        for index, residual in enumerate(samples):
            parameters = coarse_parameters.copy()
            parameters[:2] += residual
            if self.parameter_clamper is not None:
                parameters = self.parameter_clamper(parameters)
            actual_residual = np.asarray(parameters[:2], dtype=np.float32) - coarse_parameters[:2]
            if index == 0 or np.all(np.abs(actual_residual) < 1e-8):
                path = coarse
            else:
                path, _ = self.structured.regenerate(parameters)
            if path is not None:
                paths.append(path)
                residuals.append(actual_residual)

        zero = torch.zeros(2, device=context.latent.device)
        baseline_value = float(context.coarse_consequence.total_value[0].cpu())
        if not paths:
            return zero, {
                "target_candidates": 0.0,
                "target_best_value": baseline_value,
                "target_improvement": 0.0,
                "target_gate_passed": 0.0,
                "target_d": 0.0,
                "target_v": 0.0,
            }
        actions = self.action_adapter.paths_to_actions(
            paths,
            vehicle,
            required_steps=self.horizon + 1,
            control_dt=self.control_dt,
            device=context.latent.device,
        )
        values = self.evaluator.evaluate_trajectory_consequence(
            context.latent, actions
        ).total_value
        residual_tensor = torch.as_tensor(
            np.asarray(residuals), device=values.device, dtype=values.dtype
        )
        elite_count = min(self.elite_count, len(paths))
        elite_indices = torch.topk(values, elite_count).indices
        best_value = values[elite_indices[0]]
        improvement = best_value - values[0]
        threshold = max(
            self.min_improvement_abs,
            self.min_improvement_ratio * abs(float(values[0].cpu())),
        )
        gate_passed = bool(float(improvement.cpu()) >= threshold)
        if not gate_passed:
            target = zero
        elif self.target_mode == "best":
            target = residual_tensor[elite_indices[0]]
        else:
            elite_values = values[elite_indices]
            weights = torch.softmax(
                self.temperature * (elite_values - elite_values.max()), dim=0
            )
            target = (weights.unsqueeze(-1) * residual_tensor[elite_indices]).sum(dim=0)
        return target.detach(), {
            "target_candidates": float(len(paths)),
            "target_best_value": float(best_value.cpu()),
            "target_zero_value": float(values[0].cpu()),
            "target_improvement": float(improvement.cpu()),
            "target_gate_passed": float(gate_passed),
            "target_d": float(target[0].cpu()),
            "target_v": float(target[1].cpu()),
        }


class ResidualLearner:
    def __init__(self, planner, cfg):
        self.planner = planner
        self.batch_size = int(cfg.get("batch_size", 64))
        self.reg_coef = float(cfg.get("residual_reg_coef", 0.05))
        self.grad_clip = float(cfg.get("grad_clip_norm", 10.0))
        self.buffer = ResidualSupervisionBuffer(int(cfg.get("buffer_size", 10000)))
        self.optimizer = torch.optim.Adam(
            list(planner.residual_prior.parameters()), lr=float(cfg.get("lr", 3e-4))
        )
        self.update_steps = 0

    def add(self, context, target_residual: torch.Tensor) -> None:
        self.buffer.add(context, target_residual)

    def update(self) -> dict[str, torch.Tensor]:
        if len(self.buffer) < self.batch_size:
            return {"residual_buffer_size": torch.tensor(float(len(self.buffer)))}
        batch = self.buffer.sample(self.batch_size, self.planner.device)
        with torch.no_grad():
            latent = self.planner.evaluator.encode(batch.observation)
            consequence = self.planner.evaluator.evaluate_trajectory_consequence(
                latent, batch.coarse_actions
            )
            wm_features = consequence.features / self.planner.wm_feature_scales
            residual_input = torch.cat(
                [latent, batch.coarse_parameters, wm_features], dim=-1
            )
        loss, info = self.planner.residual_prior.supervised_loss(
            residual_input,
            batch.target_residual,
            delta_bounds=batch.residual_bounds,
            residual_reg_coef=self.reg_coef,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.planner.residual_prior.parameters(), self.grad_clip
        )
        self.optimizer.step()
        self.update_steps += 1
        return {
            "residual_loss": loss.detach(),
            "residual_grad_norm": torch.as_tensor(grad_norm).detach(),
            "residual_nll": info["residual_nll"],
            "residual_reg": info["residual_reg"],
            "residual_buffer_size": torch.tensor(float(len(self.buffer))),
        }


__all__ = [
    "ResidualExample",
    "ResidualLearner",
    "ResidualSupervisionBuffer",
    "ResidualTargetGenerator",
]
