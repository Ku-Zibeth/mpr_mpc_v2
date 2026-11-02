"""Validation on held-out executed action sequences, never on imaginary labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .multimodal import MODE_COMPONENT_NAMES


def calibration_statistics(prediction, actual, scenes, cfg):
    """Cluster bootstrap: resample scenes, never individual overlapping windows."""
    p, y = np.asarray(prediction, float), np.asarray(actual, float)
    unique, group = np.unique(scenes, return_inverse=True)
    error = p - y
    n = len(unique)
    counts = np.bincount(group, minlength=n)
    absolute = np.bincount(group, weights=np.abs(error), minlength=n)
    squared = np.bincount(group, weights=error**2, minlength=n)
    difference = y[:, None] - y[None, :]
    eligible = ((difference > float(cfg.get("ranking_min_gap", .05)))
                & (group[:, None] != group[None, :]))
    wins = eligible * (p[:, None] > p[None, :])
    pair_group = group[:, None] * n + group[None, :]
    pair_count = np.bincount(pair_group.ravel(), weights=eligible.ravel(), minlength=n*n).reshape(n, n)
    pair_wins = np.bincount(pair_group.ravel(), weights=wins.ravel(), minlength=n*n).reshape(n, n)
    identifiable = min(p.std(), y.std()) > 1e-8 and eligible.any()
    rank = float(wins.sum() / eligible.sum()) if identifiable else float("nan")
    rng = np.random.default_rng(int(cfg.get("bootstrap_seed", 1)))
    weights = rng.multinomial(n, np.full(n, 1/n), size=int(cfg.get("bootstrap_samples", 1000)))
    denominator = np.einsum("bi,ij,bj->b", weights, pair_count, weights)
    numerator = np.einsum("bi,ij,bj->b", weights, pair_wins, weights)
    bootstrap_rank = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    total = weights @ counts
    bootstrap_mae = (weights @ absolute) / total
    bootstrap_rmse = np.sqrt((weights @ squared) / total)
    # Degenerate bootstrap draws do not silently disappear from the readiness gate.
    rank_ci = (np.quantile(bootstrap_rank[np.isfinite(bootstrap_rank)], [.025, .975])
               if identifiable and np.isfinite(bootstrap_rank).mean() >= .95 else [float("nan")]*2)
    return dict(samples=len(y), scenes=n, prediction_mean=float(p.mean()), actual_mean=float(y.mean()),
                mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
                correlation=float(np.corrcoef(p, y)[0, 1]) if identifiable else float("nan"),
                ranking_accuracy=rank, ranking_ci_low=float(rank_ci[0]), ranking_ci_high=float(rank_ci[1]),
                mae_ci=np.quantile(bootstrap_mae, [.025, .975]).tolist(),
                rmse_ci=np.quantile(bootstrap_rmse, [.025, .975]).tolist(), identifiable=bool(identifiable))


class CostModelValidator:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.episodes = []
        self.metrics = {}
        self.mode_episodes = []
        self.mode_report = {}

    def set_episodes(self, episodes):
        self.episodes = episodes

    def set_mode_episodes(self, records):
        for row in records:
            if (row.get("source") != "metadrive_mode_replay_v1" or not row.get("replay_verified")
                    or not 0 <= int(row["mode_id"]) < len(MODE_COMPONENT_NAMES)):
                raise ValueError("Mode calibration requires verified real-environment candidate replays.")
            if len(row["actions"]) != len(row["costs"]) or not np.isfinite(row["costs"]).all():
                raise ValueError("Invalid real cost/action sequence.")
        self.mode_episodes = list(records)

    def load_dataset(self, path, *, cost_definition=None):
        path = Path(path)
        manifest = json.loads((path.parent / "calibration.json").read_text())
        if manifest.get("dataset_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Calibration dataset checksum mismatch.")
        if cost_definition is not None and manifest.get("cost_definition") != cost_definition:
            raise ValueError("Calibration dataset has a different cost definition.")
        provenance = manifest.get("training_provenance", {})
        start, count = provenance.get("start_seed"), provenance.get("num_scenarios")
        if start is None or count is None:
            raise ValueError("Calibration dataset lacks held-out scene provenance.")
        trained = set(range(int(start), int(start)+int(count))) - set(provenance.get("reserved", []))
        with np.load(path, allow_pickle=False) as data:
            if str(data["source"]) != "metadrive_mode_replay_v1":
                raise ValueError("Unsupported mode calibration dataset.")
            if trained & set(data["scenes"].tolist()) or set(data["scenes"].tolist()) != set(manifest["scenes"]):
                raise ValueError("Calibration dataset scene provenance mismatch.")
            self.set_mode_episodes([dict(source=str(data["source"]), replay_verified=bool(data["verified"][i]),
                scene_id=int(data["scenes"][i]), mode_id=int(data["modes"][i]), observation=data["observations"][i],
                actions=data["actions"][i, :int(length)], costs=data["costs"][i, :int(length)])
                for i, length in enumerate(data["lengths"])])

    @torch.no_grad()
    def validate(self, model, device, *, horizon):
        metrics = self._validate_legacy(model, device, horizon=horizon)
        if not self.mode_episodes:
            metrics["wm/cost_ready"] = float(getattr(model, "cost_ready", False))
            return metrics
        was_training = model.training
        model.eval()
        report = {}
        try:
            model.invalidate_cost_validation()
            horizons = sorted(set([1, horizon] + list(self.cfg.get("horizons", [1, 5, 10, 15, 20]))))
            for mode, name in enumerate(MODE_COMPONENT_NAMES):
                report[name] = {}
                for h in horizons:
                    if not 0 < h < 63:
                        continue
                    rows = [r for r in self.mode_episodes if r["mode_id"] == mode and len(r["actions"]) >= h]
                    if not rows:
                        continue
                    prediction, actual = [], []
                    gamma = float(self.cfg.get("gamma", .99))
                    for start in range(0, len(rows), 128):
                        batch = rows[start:start+128]
                        obs = torch.as_tensor(np.asarray([r["observation"] for r in batch]), device=device).float()
                        actions = torch.as_tensor(np.asarray([r["actions"][:h] for r in batch]), device=device).float()
                        z = model.encode(obs, None)
                        value = torch.zeros((len(batch), 1), device=device)
                        for t in range(h):
                            value += gamma**t * model.cost(z, actions[:, t], None)
                            z = model.next(z, actions[:, t], None)
                        prediction.extend(value.flatten().cpu().tolist())
                        actual.extend([float(np.dot(r["costs"][:h], gamma**np.arange(h))) for r in batch])
                    if not np.isfinite(prediction).all():
                        report[name][h] = {"ready": False, "reason": "nonfinite prediction"}
                        continue
                    stats = calibration_statistics(prediction, actual, [r["scene_id"] for r in rows], self.cfg)
                    ready = (int(model._cost_updates) >= int(self.cfg.get("min_updates", 1000))
                        and stats["samples"] >= int(self.cfg.get("min_samples", 128))
                        and stats["scenes"] >= int(self.cfg.get("min_scenes", 10))
                        and stats["identifiable"]
                        and stats["mae"] <= h * float(self.cfg.get("max_mae", .1))
                        and stats["rmse"] <= h * float(self.cfg.get("max_rmse", .5))
                        and stats["ranking_accuracy"] >= float(self.cfg.get("min_ranking_accuracy", .60))
                        and stats["ranking_ci_low"] >= float(self.cfg.get("min_ranking_ci_low", .50)))
                    # Longer rollouts cannot hide poor one-step calibration.
                    ready = ready and (h == 1 or bool(model._cost_validated_by_mode[mode, 1]))
                    stats["ready"] = ready
                    stats["predictions"] = prediction
                    stats["actuals"] = actual
                    stats["scene_ids"] = [int(r["scene_id"]) for r in rows]
                    report[name][h] = stats
                    model._cost_validated_by_mode[mode, h] = ready
                    model._cost_validation_samples_by_mode[mode, h] = stats["samples"]
                    model._cost_validation_scenes_by_mode[mode, h] = stats["scenes"]
                    for key in ("samples", "scenes", "mae", "rmse", "correlation", "ranking_accuracy", "ranking_ci_low", "ready"):
                        metrics[f"wm/cost_{name}_{key}_h{h}"] = float(stats[key])
            model._cost_validated_by_horizon.copy_(model._cost_validated_by_mode.any(0))
            model._cost_mode_validated_at.copy_(model._cost_updates)
            model._cost_validated.fill_(bool(model._cost_validated_by_mode[:, horizon].any()))
            model._cost_validated_horizons.fill_(sum(1 << h for h in horizons if 0 < h < 63 and model._cost_validated_by_horizon[h]))
            metrics["wm/cost_ready"] = float(model.cost_ready)
            self.mode_report = report
            self.metrics = metrics
            return metrics
        finally:
            model.train(was_training)

    @torch.no_grad()
    def _validate_legacy(self, model, device, *, horizon: int) -> dict:
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
        # Unlabelled executed trajectories are useful diagnostics, not mode approvals.
        metrics.update({"wm/cost_ready": float(getattr(model, "cost_ready", False)), "cost_prediction_mae": one.get("mae", float("nan")),
                        "cost_prediction_rmse": one.get("rmse", float("nan")),
                        "predicted_vs_actual_cost": one.get("correlation", float("nan")),
                        "risk_ranking_accuracy": long.get("ranking_accuracy", float("nan"))})
        self.metrics = metrics
        model.train(was_training)
        return metrics
