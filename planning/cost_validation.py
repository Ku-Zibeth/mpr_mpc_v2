"""Independent executed-trajectory Cost Head diagnostics, not safety certification."""

from __future__ import annotations

import numpy as np
import torch


def cost_statistics(prediction, actual, *, min_gap=.05, groups=None):
    p, y = np.asarray(prediction, float), np.asarray(actual, float)
    if p.shape != y.shape or not len(y) or not np.isfinite([p, y]).all():
        raise ValueError("Cost validation requires finite, equally sized predictions and labels.")
    if groups is not None and len(groups) != len(y):
        raise ValueError("Candidate origin groups must align with the predictions.")
    error = p-y
    correlation = float(np.corrcoef(p, y)[0, 1]) if min(p.std(), y.std()) > 1e-8 else float("nan")
    ranking = float("nan")
    pairs = 0
    if groups is not None:
        wins = 0
        for group in set(groups):
            indices = np.flatnonzero(np.asarray(groups) == group)
            i, j = np.where(y[indices, None]-y[indices][None, :] > min_gap)
            pairs += len(i)
            wins += int((p[indices[i]] > p[indices[j]]).sum())
        if pairs:
            ranking = wins / pairs
    return dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
                correlation=correlation, ranking_accuracy=ranking, ranking_pairs=pairs,
                samples=len(y), prediction_mean=float(p.mean()), actual_mean=float(y.mean()))


class CostModelValidator:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.episodes = []
        self.candidates = []
        self.metrics = {}

    def set_episodes(self, episodes):
        self.episodes = list(episodes)

    def set_candidates(self, candidates):
        """Optional real replays sharing an explicit origin_id, not mode labels."""
        origins = {}
        for row in candidates:
            if not row.get("executed", False):
                raise ValueError("Candidate cost labels must come from executed environment trajectories.")
            obs = np.asarray(row["observation"])
            origin = row["origin_id"]
            if origin in origins and not np.array_equal(obs, origins[origin]):
                raise ValueError("Candidate ranking requires the same observation at each origin_id.")
            origins[origin] = obs
            if len(row["actions"]) != len(row["costs"]):
                raise ValueError("Candidate actions and real cost labels must have equal lengths.")
        self.candidates = list(candidates)

    @torch.no_grad()
    def validate(self, model, device, *, horizon):
        if not self.episodes and not self.candidates:
            return {"wm/cost_ready": float(getattr(model, "cost_ready", False))}
        if not bool(self.cfg.get("enabled", True)):
            return {"wm/cost_ready": 0.0}
        was_training = model.training
        model.eval()
        metrics, stats_by_horizon = {}, {}
        gamma = float(self.cfg.get("gamma", .99))
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("Cost validation gamma must be in [0,1].")
        horizons = sorted(set([1, horizon] + list(self.cfg.get("horizons", [1, 5, 10, 15, 20]))))
        try:
            if hasattr(model, "invalidate_cost_validation"):
                model.invalidate_cost_validation()
            mask = 0
            for h in horizons:
                if not 0 < h < 63:
                    raise ValueError("Cost validation horizons must be in [1,62].")
                windows = []
                for observations, actions, costs in self.episodes:
                    if len(observations) < len(actions) or len(costs) != len(actions):
                        raise ValueError("Held-out observations, actions and costs are not aligned.")
                    for start in range(0, len(actions)-h+1, max(1, int(self.cfg.get("stride", 5)))):
                        windows.append((observations[start], actions[start:start+h], costs[start:start+h]))
                limit = max(1, int(self.cfg.get("max_windows", 512)))
                if len(windows) > limit:
                    windows = [windows[i] for i in np.linspace(0, len(windows)-1, limit, dtype=int)]
                # Candidate pairs are used only with genuinely executed labels
                # from a common initial observation. Unpaired windows cannot
                # establish counterfactual trajectory-ranking accuracy.
                candidate_rows = [r for r in self.candidates if len(r["actions"]) >= h]
                all_windows = windows + [(r["observation"], r["actions"][:h], r["costs"][:h])
                                         for r in candidate_rows]
                if not all_windows:
                    continue
                prediction, actual = [], []
                for start in range(0, len(all_windows), 128):
                    batch = all_windows[start:start+128]
                    obs = torch.as_tensor(np.asarray([w[0] for w in batch]), device=device).float()
                    actions = torch.as_tensor(np.asarray([w[1] for w in batch]), device=device).float()
                    labels = np.asarray([w[2] for w in batch], float)
                    if not torch.isfinite(obs).all() or not torch.isfinite(actions).all() or not np.isfinite(labels).all() or (labels < 0).any():
                        raise ValueError("Cost validation requires real finite, nonnegative labels and finite inputs.")
                    z, value = model.encode(obs, None), torch.zeros((len(batch), 1), device=device)
                    for t in range(h):
                        value += gamma**t * model.cost(z, actions[:, t], None)
                        z = model.next(z, actions[:, t], None)
                    prediction.extend(value.flatten().cpu().tolist())
                    actual.extend((labels * gamma**np.arange(h)).sum(1).tolist())
                stats = cost_statistics(prediction, actual, min_gap=float(self.cfg.get("ranking_min_gap", .05)))
                if candidate_rows:
                    rank = cost_statistics(prediction[len(windows):], actual[len(windows):],
                        min_gap=float(self.cfg.get("ranking_min_gap", .05)),
                        groups=[r["origin_id"] for r in candidate_rows])
                    stats["ranking_accuracy"], stats["ranking_pairs"] = rank["ranking_accuracy"], rank["ranking_pairs"]
                stats_by_horizon[h] = stats
                for key, value in stats.items():
                    metrics[f"wm/cost_{key}_h{h}"] = float(value)
                passed = (int(getattr(model, "_cost_updates", 0)) >= int(self.cfg.get("min_updates", 1000))
                          and stats["samples"] >= int(self.cfg.get("min_samples", 128))
                          and stats["mae"] <= h*float(self.cfg.get("max_mae", .1))
                          and stats["rmse"] <= h*float(self.cfg.get("max_rmse", .5))
                          and (h == 1 or mask & (1 << 1)))
                if passed:
                    mask |= 1 << h
            if hasattr(model, "_cost_validated"):
                model._cost_validated_horizons.fill_(mask)
                model._cost_validated.fill_(bool(mask & (1 << horizon)))
                model._cost_validated_at.copy_(model._cost_updates)
                model._cost_validation_schema.fill_(3)
            one, long = stats_by_horizon.get(1, {}), stats_by_horizon.get(horizon, {})
            metrics.update({
                "wm/cost_ready": float(getattr(model, "cost_ready", False)),
                "cost_prediction_mae": one.get("mae", float("nan")),
                "cost_prediction_rmse": one.get("rmse", float("nan")),
                "predicted_vs_actual_cost": one.get("correlation", float("nan")),
                "risk_ranking_accuracy": long.get("ranking_accuracy", float("nan")),
                "wm/cost_ranking_pairs": long.get("ranking_pairs", 0),
            })
            self.metrics = metrics
            return metrics
        finally:
            model.train(was_training)
