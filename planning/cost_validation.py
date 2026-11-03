"""Validation on held-out executed action sequences, never on imaginary labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .multimodal import MODE_COMPONENT_NAMES
from .calibration_anchor import verify_anchor


def calibration_statistics(prediction, actual, scenes, cfg, *, anchors=None, modes=None, focus_mode=None):
    """Rank ONLY matched prediction origins; bootstrap whole scene clusters."""
    p, y = np.asarray(prediction, float), np.asarray(actual, float)
    unique, group = np.unique(scenes, return_inverse=True)
    n = len(unique)
    if not len(y):
        raise ValueError("Calibration statistics require at least one record.")
    modes = np.asarray(modes if modes is not None else np.zeros(len(y)), int)
    anchors = np.asarray(anchors if anchors is not None else [""] * len(y), str)
    selected = np.ones(len(y), bool) if focus_mode is None else modes == focus_mode
    error = p[selected] - y[selected]
    selected_group = group[selected]
    counts = np.bincount(selected_group, minlength=n)
    absolute = np.bincount(selected_group, weights=np.abs(error), minlength=n)
    squared = np.bincount(selected_group, weights=error**2, minlength=n)
    origins = {}
    for index, (scene, anchor) in enumerate(zip(group, anchors)):
        if anchor:
            origins.setdefault((int(scene), anchor), []).append(index)
    pair_i, pair_j = [], []
    for indices in origins.values():
        indices = np.asarray(indices)
        difference = y[indices, None] - y[indices][None, :]
        eligible = ((difference > float(cfg.get("ranking_min_gap", .05)))
                    & np.isfinite(p[indices, None]) & np.isfinite(p[indices][None, :]))
        if focus_mode is not None:
            eligible &= selected[indices, None] | selected[indices][None, :]
        # Coarse Lattice is its own source, never relabelled as an MPPI mode.
        if focus_mode != -1:
            eligible &= (modes[indices, None] >= 0) & (modes[indices][None, :] >= 0)
        i, j = eligible.nonzero()
        pair_i.extend(indices[i])
        pair_j.extend(indices[j])
    pair_i, pair_j = np.asarray(pair_i, int), np.asarray(pair_j, int)
    cross = modes[pair_i] != modes[pair_j]
    rng = np.random.default_rng(int(cfg.get("bootstrap_seed", 1)))
    weights = rng.multinomial(n, np.full(n, 1/n), size=int(cfg.get("bootstrap_samples", 1000)))

    def ranking(mask):
        i, j = pair_i[mask], pair_j[mask]
        count = np.bincount(group[i], minlength=n)
        wins = np.bincount(group[i], weights=(p[i] > p[j]), minlength=n).astype(float)
        denominator, numerator = weights @ count, weights @ wins
        draws = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
        identified = bool(len(i) and np.isfinite(p[selected]).all()
                          and np.ptp(p[selected]) > 1e-8 and np.ptp(y[selected]) > 1e-8)
        # A mode may have constant risk within one anchor yet separate actions
        # across anchors. No between-anchor pairs are ever used to prove this.
        ci = (np.quantile(draws[np.isfinite(draws)], [.025, .975])
              if identified and np.isfinite(draws).mean() >= .95 else [float("nan")]*2)
        active = np.unique(np.r_[i, j])
        return dict(accuracy=float(wins.sum()/count.sum()) if identified else float("nan"),
                    ci_low=float(ci[0]), ci_high=float(ci[1]), pairs=len(i),
                    anchors=len(set(anchors[active])), scenes=len(set(np.asarray(scenes)[active])), identifiable=identified)

    all_rank, within_rank, cross_rank = ranking(np.ones(len(pair_i), bool)), ranking(~cross), ranking(cross)
    total = weights @ counts
    bootstrap_mae = np.divide(weights @ absolute, total, out=np.full(len(weights), np.nan), where=total > 0)
    bootstrap_rmse = np.sqrt(np.divide(weights @ squared, total, out=np.full(len(weights), np.nan), where=total > 0))
    ps, ys = p[selected], y[selected]
    return dict(samples=int(selected.sum()), scenes=len(set(np.asarray(scenes)[selected])),
                prediction_mean=float(ps.mean()), actual_mean=float(ys.mean()),
                mae=float(np.abs(error).mean()), rmse=float(np.sqrt(np.mean(error**2))),
                correlation=float(np.corrcoef(ps, ys)[0, 1]) if min(ps.std(), ys.std()) > 1e-8 else float("nan"),
                ranking_accuracy=all_rank["accuracy"], ranking_ci_low=all_rank["ci_low"], ranking_ci_high=all_rank["ci_high"],
                within_mode_ranking=within_rank, cross_mode_ranking=cross_rank, overall_ranking=all_rank,
                mae_ci=np.nanquantile(bootstrap_mae, [.025, .975]).tolist(),
                rmse_ci=np.nanquantile(bootstrap_rmse, [.025, .975]).tolist(), identifiable=all_rank["identifiable"])


class CostModelValidator:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.episodes = []
        self.metrics = {}
        self.mode_episodes = []
        self.mode_report = {}
        self.paired_report = {}

    def set_episodes(self, episodes):
        self.episodes = episodes

    def set_mode_episodes(self, records):
        anchor_states = {}
        for row in records:
            mode = int(row["mode_id"])
            coarse = mode == -1 and row.get("trajectory_kind") == "coarse_lattice"
            if (row.get("source") not in {"metadrive_mode_replay_v1", "metadrive_mode_replay_v2"}
                    or not row.get("replay_verified") or not (coarse or 0 <= mode < len(MODE_COMPONENT_NAMES))):
                raise ValueError("Mode calibration requires verified real-environment candidate replays.")
            if (len(row["actions"]) != len(row["costs"]) or not np.isfinite(row["costs"]).all()
                    or not np.isfinite(row["actions"]).all() or not np.isfinite(row["observation"]).all()):
                raise ValueError("Invalid real cost/action sequence.")
            anchor = row.get("anchor_id", "")
            if anchor:
                signature = (int(row["scene_id"]), np.asarray(row["observation"], dtype=float).tobytes())
                if anchor in anchor_states and anchor_states[anchor] != signature:
                    raise ValueError("One anchor_id cannot describe different scenes or observations.")
                anchor_states[anchor] = signature
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
            if str(data["source"]) not in {"metadrive_mode_replay_v1", "metadrive_mode_replay_v2"}:
                raise ValueError("Unsupported mode calibration dataset.")
            if trained & set(data["scenes"].tolist()) or set(data["scenes"].tolist()) != set(manifest["scenes"]):
                raise ValueError("Calibration dataset scene provenance mismatch.")
            anchors = data["anchor_ids"] if "anchor_ids" in data else [""] * len(data["lengths"])
            for anchor, payload in manifest.get("anchors", {}).items():
                verify_anchor(anchor, payload)
            for i, anchor in enumerate(anchors):
                if anchor:
                    payload = manifest.get("anchors", {}).get(str(anchor))
                    if (payload is None or payload["scene_id"] != int(data["scenes"][i])
                            or not np.array_equal(payload["observation"], data["observations"][i])):
                        raise ValueError("Replay observation/scene differs from its anchor.")
            self.set_mode_episodes([dict(source=str(data["source"]), replay_verified=bool(data["verified"][i]),
                anchor_id=str(anchors[i]), trajectory_kind="coarse_lattice" if int(data["modes"][i]) == -1 else "mppi_mode",
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
        self.paired_report = {}
        try:
            model.invalidate_cost_validation()
            horizons = sorted(set([1, horizon] + list(self.cfg.get("horizons", [1, 5, 10, 15, 20]))))
            if not 0 < horizon < 63:
                raise ValueError("Cost calibration horizon must be in [1,62].")
            sources = list(enumerate(MODE_COMPONENT_NAMES)) + [(-1, "coarse_lattice")]
            report = {name: {} for _, name in sources}
            for h in horizons:
                if not 0 < h < 63:
                    continue
                rows = [r for r in self.mode_episodes if len(r["actions"]) >= h]
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
                scenes, modes = [r["scene_id"] for r in rows], np.asarray([r["mode_id"] for r in rows])
                anchors = [r.get("anchor_id", "") for r in rows]
                self.paired_report[h] = calibration_statistics(prediction, actual, scenes, self.cfg, anchors=anchors, modes=modes)
                for mode, name in sources:
                    selected = modes == mode
                    if not selected.any():
                        continue
                    stats = calibration_statistics(prediction, actual, scenes, self.cfg,
                        anchors=anchors, modes=modes, focus_mode=mode)
                    cross = stats["cross_mode_ranking"]
                    ready = (int(model._cost_updates) >= int(self.cfg.get("min_updates", 1000))
                        and stats["samples"] >= int(self.cfg.get("min_samples", 128))
                        and stats["scenes"] >= int(self.cfg.get("min_scenes", 10))
                        and stats["identifiable"]
                        and cross["identifiable"]
                        and cross["pairs"] >= int(self.cfg.get("min_pairwise_pairs", 128))
                        and cross["anchors"] >= int(self.cfg.get("min_pairwise_anchors", 16))
                        and cross["scenes"] >= int(self.cfg.get("min_scenes", 10))
                        and stats["mae"] <= h * float(self.cfg.get("max_mae", .1))
                        and stats["rmse"] <= h * float(self.cfg.get("max_rmse", .5))
                        and stats["ranking_accuracy"] >= float(self.cfg.get("min_ranking_accuracy", .60))
                        and stats["ranking_ci_low"] >= float(self.cfg.get("min_ranking_ci_low", .50))
                        and cross["accuracy"] >= float(self.cfg.get("min_ranking_accuracy", .60))
                        and cross["ci_low"] >= float(self.cfg.get("min_ranking_ci_low", .50)))
                    # Longer rollouts cannot hide poor one-step calibration.
                    approval = model._cost_validated_by_mode[mode] if mode >= 0 else model._cost_validated_coarse
                    ready = ready and (h == 1 or bool(approval[1]))
                    stats["ready"] = ready
                    stats["reason"] = "passed" if ready else "insufficient supervised updates, errors, or identifiable matched-anchor pairs"
                    stats["predictions"] = np.asarray(prediction)[selected].tolist()
                    stats["actuals"] = np.asarray(actual)[selected].tolist()
                    stats["scene_ids"] = np.asarray(scenes)[selected].tolist()
                    stats["anchor_ids"] = np.asarray(anchors)[selected].tolist()
                    report[name][h] = stats
                    approval[h] = ready
                    if mode >= 0:
                        model._cost_validation_samples_by_mode[mode, h] = stats["samples"]
                        model._cost_validation_scenes_by_mode[mode, h] = stats["scenes"]
                    for key in ("samples", "scenes", "mae", "rmse", "correlation", "ranking_accuracy", "ranking_ci_low", "ready"):
                        metrics[f"wm/cost_{name}_{key}_h{h}"] = float(stats[key])
                    for category in ("within_mode_ranking", "cross_mode_ranking", "overall_ranking"):
                        for key in ("accuracy", "ci_low", "pairs", "anchors", "scenes"):
                            metrics[f"wm/cost_{name}_{category}_{key}_h{h}"] = stats[category][key]
            model._cost_validated_by_horizon.copy_(model._cost_validated_by_mode.any(0) | model._cost_validated_coarse)
            model._cost_mode_validated_at.copy_(model._cost_updates)
            model._cost_validation_schema.fill_(2)
            model._cost_validated.fill_(bool(model._cost_validated_by_horizon[horizon]))
            model._cost_validated_horizons.fill_(sum(1 << h for h in horizons if 0 < h < 63 and model._cost_validated_by_horizon[h]))
            metrics["wm/cost_ready"] = float(model.cost_ready)
            metrics["risk_ranking_accuracy"] = self.paired_report.get(horizon, {}).get("ranking_accuracy", float("nan"))
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
                        "wm/legacy_unpaired_ranking_accuracy": long.get("ranking_accuracy", float("nan")),
                        "risk_ranking_accuracy": float("nan")})
        self.metrics = metrics
        model.train(was_training)
        return metrics
