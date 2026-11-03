"""Residual SAC state assembly from TD-MPC2 and coarse Lattice outputs."""

from __future__ import annotations

import torch

from .features import FEATURE_NAMES, extract_coarse_path_features, feature_scales_from_config


class ResidualStateBuilder:
    def __init__(
        self,
        cfg: dict,
        *,
        latent_dim: int,
        device: str | torch.device,
    ):
        self.device = torch.device(device)
        self.latent_dim = int(latent_dim)
        self.include_wm_cost = bool(cfg.get("include_wm_cost_in_state", False))
        self.risk_features = list(cfg.get("risk_features", []))
        if self.include_wm_cost and "cumulative_cost" not in self.risk_features:
            self.risk_features.append("cumulative_cost")
        allowed = {"cumulative_cost", "min_ttc", "min_clearance", "boundary_margin"}
        if len(set(self.risk_features)) != len(self.risk_features) or set(self.risk_features) - allowed:
            raise ValueError("Unknown or repeated residual risk feature.")
        self.risk_scales = dict(cfg.get("risk_feature_scales", {}))
        self.wm_return_scale = max(abs(float(cfg.get("wm_return_scale", 50.0))), 1e-6)
        self.wm_cost_scale = max(abs(float(cfg.get("wm_cost_scale", 10.0))), 1e-6)
        self.feature_scales = feature_scales_from_config(cfg)
        self.feature_clip = float(cfg.get("path_feature_clip", 5.0))
        self.state_dim = self.latent_dim + len(FEATURE_NAMES) + 1 + 2 + 2 * len(self.risk_features)
        self.wm_cost_dim = int("cumulative_cost" in self.risk_features)
        self.last_risk_status = {}

    def path_features(self, path) -> torch.Tensor:
        return extract_coarse_path_features(
            path,
            scales=self.feature_scales,
            clip=self.feature_clip,
            device=self.device,
        )

    def build(
        self,
        *,
        latent: torch.Tensor,
        coarse_path,
        consequence,
        feasible_ratio: float,
        used_fallback: bool,
    ) -> torch.Tensor:
        if latent is None:
            raise ValueError("Residual SAC state requires TD-MPC2 latent.")
        z = latent.detach().to(self.device, dtype=torch.float32)
        if z.ndim == 2:
            if z.shape[0] != 1:
                raise ValueError(f"Residual state expects one latent, got {tuple(z.shape)}.")
            z = z.squeeze(0)
        if z.shape != (self.latent_dim,):
            raise ValueError(f"Latent shape {tuple(z.shape)} != {(self.latent_dim,)}.")
        features = self.path_features(coarse_path).reshape(-1)
        if consequence is None:
            raise ValueError("Residual SAC state requires coarse world-model evaluation.")
        wm_return = consequence.total_value.detach().reshape(-1).to(self.device, dtype=torch.float32)
        if wm_return.numel() != 1:
            raise ValueError("Coarse WM return must be scalar for one selected trajectory.")
        pieces = [
            z,
            features,
            wm_return / self.wm_return_scale,
        ]
        pieces.append(
            torch.tensor(
                [
                    float(feasible_ratio),
                    1.0 if used_fallback else 0.0,
                ],
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.last_risk_status = {}
        for name in self.risk_features:
            value = consequence.risk_summary.get(name) if hasattr(consequence, "risk_summary") else None
            if name == "cumulative_cost":
                value = consequence.cumulative_cost if getattr(consequence, "cost_validated", False) else None
            number = float(torch.as_tensor(value).reshape(-1)[0]) if value is not None else float("nan")
            valid = bool(torch.isfinite(torch.tensor(number)))
            scale = float(self.risk_scales.get(name, {"cumulative_cost": 10, "min_ttc": 5,
                                                     "min_clearance": 20, "boundary_margin": 4}[name]))
            if scale <= 0:
                raise ValueError("Risk feature scales must be positive.")
            if number == float("inf") and name in {"min_ttc", "min_clearance"}:
                number, valid = 5 * scale, True
            reason = "available" if valid else (getattr(consequence, "cost_unavailable_reason", "unvalidated")
                                                if name == "cumulative_cost" else "missing_geometry")
            self.last_risk_status[name] = {"valid": valid, "reason": reason}
            pieces.append(torch.tensor([max(-5.0, min(5.0, number / scale)) if valid else 0.0,
                                        float(valid)], device=self.device))
        state = torch.cat(pieces, dim=0)
        if state.shape != (self.state_dim,):
            raise RuntimeError(f"Residual state shape {tuple(state.shape)} != {(self.state_dim,)}.")
        if not torch.isfinite(state).all():
            raise RuntimeError("Residual SAC state contains non-finite values.")
        return state
