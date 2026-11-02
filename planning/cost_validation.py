"""Validation on held-out executed action sequences, never on imaginary labels."""

from __future__ import annotations

import numpy as np
import torch


class CostModelValidator:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.episodes = []
        self.metrics = {}

    def set_episodes(self, episodes):
        self.episodes = episodes

    @torch.no_grad()
    def validate(self, model, device, *, horizon: int) -> dict:
        if not self.episodes:
            return {"wm/cost_ready": float(getattr(model, "cost_ready", False))}
        was_training = model.training
        model.eval()
        gamma = float(self.cfg.get("gamma", 0.99))
        metrics = {}
        rows_by_horizon = {}
        horizons = sorted(set([1, horizon] + list(self.cfg.get("horizons", [1, 5, 10, 15, 20]))))
        for h in horizons:
            windows = []
            for obs, actions, costs in self.episodes:
                stride = max(1, int(self.cfg.get("stride", 5)))
                for start in range(0, len(actions) - h + 1, stride):
                    windows.append((obs[start], actions[start:start+h], costs[start:start+h]))
            limit = int(self.cfg.get("max_windows", 512))
            if len(windows) > limit:
                indices = np.linspace(0, len(windows)-1, limit, dtype=int)
                windows = [windows[i] for i in indices]
            predictions, targets = [], []
            for start in range(0, len(windows), 128):
                batch = windows[start:start+128]
                obs = torch.stack([torch.as_tensor(w[0]) for w in batch]).to(device).float()
                actions = torch.stack([torch.as_tensor(w[1]) for w in batch]).to(device).float()
                actual = np.asarray([w[2] for w in batch], dtype=float)
                z = model.encode(obs, None)
                predicted = torch.zeros((len(batch), 1), device=device)
                for t in range(h):
                    predicted += gamma**t * model.cost(z, actions[:, t], None)
                    z = model.next(z, actions[:, t], None)
                predictions.extend(predicted.squeeze(-1).cpu().tolist())
                targets.extend((actual * gamma**np.arange(h)).sum(axis=1).tolist())
            if not predictions:
                continue
            p, y = np.asarray(predictions), np.asarray(targets)
            error = p - y
            correlation = float(np.corrcoef(p, y)[0, 1]) if min(p.std(), y.std()) > 1e-8 else float("nan")
            difference = y[:, None] - y[None, :]
            eligible = difference > float(self.cfg.get("ranking_min_gap", 0.05))
            rank = float(np.mean((p[:, None] > p[None, :])[eligible])) if eligible.any() else float("nan")
            current = {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
                       "correlation": correlation, "ranking_accuracy": rank, "samples": len(y)}
            rows_by_horizon[h] = current
            for key, value in current.items():
                metrics[f"wm/cost_{key}_h{h}"] = value
        one = rows_by_horizon.get(1, {})
        long = rows_by_horizon.get(horizon, {})
        one_ready = (
            int(model._cost_updates.item()) >= int(self.cfg.get("min_updates", 1000))
            and one.get("samples", 0) >= int(self.cfg.get("min_samples", 128))
            and one.get("mae", float("inf")) <= float(self.cfg.get("max_mae", 0.1))
            and one.get("rmse", float("inf")) <= float(self.cfg.get("max_rmse", 0.5))
        )
        mask = 0
        for h, stats in rows_by_horizon.items():
            if (one_ready and 0 < h < 63
                    and stats["samples"] >= int(self.cfg.get("min_samples", 128))
                    and stats["mae"] <= h * float(self.cfg.get("max_mae", 0.1))
                    and stats["rmse"] <= h * float(self.cfg.get("max_rmse", 0.5))
                    and stats["ranking_accuracy"] >= float(self.cfg.get("min_ranking_accuracy", 0.55))):
                mask |= 1 << h
        ready = bool(mask & (1 << horizon))
        model._cost_validated_horizons.fill_(mask)
        model._cost_validated.fill_(ready)
        metrics.update({"wm/cost_ready": float(ready), "cost_prediction_mae": one.get("mae", float("nan")),
                        "cost_prediction_rmse": one.get("rmse", float("nan")),
                        "predicted_vs_actual_cost": one.get("correlation", float("nan")),
                        "risk_ranking_accuracy": long.get("ranking_accuracy", float("nan"))})
        self.metrics = metrics
        model.train(was_training)
        return metrics
