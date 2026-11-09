"""Independent executed-trajectory Cost Head diagnostics, not safety certification."""

from __future__ import annotations

import numpy as np
import torch
import copy
import json
import time

from .config import section


def cost_definition(cfg):
    """Label semantics, independent of planner penalty weights/validation thresholds."""
    meta, cost = section(cfg, "metadrive"), section(cfg, "cost")
    simulator = dict(meta.get("simulator", {}))
    return json.dumps({"gamma": cost.get("gamma", .99), "cost": meta.get("cost", {}),
        "risk_field": meta.get("risk_field", {}),
        "simulator": {k: v for k, v in simulator.items() if "cost" in k or "out_of_road" in k}},
        sort_keys=True, default=str)


def _weight_version(model):
    # Optimizer steps and load_state_dict increment Tensor._version. No tensor
    # copies, checksums, extra buffers, or changes to saved model state.
    return tuple((id(p), p._version) for p in model.parameters())


def candidate_ranking_ready(model, horizon, definition):
    evidence = getattr(model, "_cost_ranking_evidence", None)
    if not evidence:
        return False
    valid = (evidence["definition"] == definition
             and evidence["generation"] == evidence["owner"]._generation
             and evidence["expires_at"] >= time.monotonic()
             and evidence["weights"] == _weight_version(model)
             and evidence["updates"] == int(getattr(model, "_cost_updates", -1))
             and bool(getattr(model, "cost_ready", False)))
    if not valid:
        model._cost_ranking_evidence = None
        return False
    return int(horizon) in evidence["horizons"]


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
    def __init__(self, cfg, *, label_definition=None):
        self.cfg = dict(cfg)
        self.label_definition = label_definition or cost_definition({"cost": self.cfg})
        self._generation = 0
        self._candidate_data_time = time.monotonic()
        self.episodes = []
        self.candidates = []
        self.metrics = {}

    def set_episodes(self, episodes):
        self._generation += 1
        self.episodes = copy.deepcopy(list(episodes))

    def set_candidates(self, candidates):
        """Optional real replays sharing an explicit origin_id, not mode labels."""
        self._generation += 1
        self._candidate_data_time = time.monotonic()
        origins, states = {}, {}
        candidates = list(candidates)
        for row in candidates:
            if not row.get("executed", False):
                raise ValueError("Candidate cost labels must come from executed environment trajectories.")
            obs = np.asarray(row["observation"])
            origin = row["origin_id"]
            if origin in origins and not np.array_equal(obs, origins[origin]):
                raise ValueError("Candidate ranking requires the same observation at each origin_id.")
            origins[origin] = obs
            state = row.get("initial_state_id")
            if origin in states and states[origin] != state:
                raise ValueError("Candidate ranking requires the same simulator initial_state_id per origin.")
            states[origin] = state
            if len(row["actions"]) != len(row["costs"]):
                raise ValueError("Candidate actions and real cost labels must have equal lengths.")
        self.candidates = copy.deepcopy(candidates)

    def invalidate_data(self):
        """Revoke all issued ranking evidence, including after dataset expiry."""
        self._generation += 1
        self.episodes, self.candidates = [], []

    def _ranking_passed(self, predictions, labels, rows, stats):
        # Pairwise order within restored simulator states, plus the actual
        # cost regret of choosing the predicted safest candidate. Correlation
        # and unpaired MAE cannot open this gate.
        groups, accuracies, regrets = [r["origin_id"] for r in rows], [], []
        seen_states, qualified_pairs = set(), 0
        for origin in set(groups):
            indices = [i for i, g in enumerate(groups) if g == origin]
            if (len(indices) < 2 or any(not rows[i].get("initial_state_id") for i in indices)
                    or any(rows[i].get("cost_definition") != self.label_definition for i in indices)):
                continue
            state_id = str(rows[indices[0]]["initial_state_id"])
            if state_id in seen_states:
                continue  # duplicated restored states are not independent origins
            seen_states.add(state_id)
            p, y = np.asarray(predictions)[indices], np.asarray(labels)[indices]
            group = cost_statistics(p, y, groups=[origin]*len(indices),
                                    min_gap=float(self.cfg.get("ranking_min_gap", .05)))
            if group["ranking_pairs"]:
                accuracies.append(group["ranking_accuracy"])
                regrets.append(float(y[np.argmin(p)]-y.min()))
                qualified_pairs += group["ranking_pairs"]
        n = len(accuracies)
        target = float(self.cfg.get("ranking_min_accuracy", .8))
        fraction = np.mean(np.asarray(accuracies) >= target) if n else 0.
        z = 1.96
        lower = ((fraction+z*z/(2*n)-z*np.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n)))
                 /(1+z*z/n)) if n else 0.
        stats.update(ranking_origins=n, ranking_qualified_pairs=qualified_pairs, ranking_origin_ci95_low=float(lower),
                     ranking_top1_regret=float(np.mean(regrets)) if n else float("nan"))
        return (n >= int(self.cfg.get("ranking_min_origins", 20))
                and qualified_pairs >= int(self.cfg.get("ranking_min_pairs", 100))
                and np.mean(accuracies) >= target
                and lower >= float(self.cfg.get("ranking_min_origin_ci95", .6))
                and np.mean(regrets) <= float(self.cfg.get("ranking_max_top1_regret", .05)))

    @torch.no_grad()
    def validate(self, model, device, *, horizon):
        model._cost_ranking_evidence = None
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
            mask, ranking_horizons = 0, set()
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
                    ranking_passed = self._ranking_passed(prediction[len(windows):], actual[len(windows):],
                                                          candidate_rows, stats)
                else:
                    ranking_passed = False
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
                    if ranking_passed:
                        ranking_horizons.add(h)
                metrics[f"wm/cost_horizon_validated_h{h}"] = float(passed)
                metrics[f"wm/cost_ranking_validated_h{h}"] = float(passed and ranking_passed)
            if hasattr(model, "_cost_validated"):
                model._cost_validated_horizons.fill_(mask)
                model._cost_validated.fill_(bool(mask & (1 << horizon)))
                model._cost_validated_at.copy_(model._cost_updates)
                model._cost_validation_schema.fill_(3)
            if ranking_horizons:
                max_age = float(self.cfg.get("ranking_max_age_seconds", 86400.))
                if not np.isfinite(max_age) or max_age <= 0:
                    raise ValueError("Ranking validation data lifetime must be finite and positive.")
                model._cost_ranking_evidence = dict(owner=self, generation=self._generation,
                    definition=self.label_definition, weights=_weight_version(model),
                    updates=int(getattr(model, "_cost_updates", -1)), horizons=ranking_horizons,
                    expires_at=self._candidate_data_time+max_age)
            one, long = stats_by_horizon.get(1, {}), stats_by_horizon.get(horizon, {})
            metrics.update({
                "wm/cost_ready": float(getattr(model, "cost_ready", False)),
                "cost_prediction_mae": one.get("mae", float("nan")),
                "cost_prediction_rmse": one.get("rmse", float("nan")),
                "predicted_vs_actual_cost": one.get("correlation", float("nan")),
                "risk_ranking_accuracy": long.get("ranking_accuracy", float("nan")),
                "wm/cost_ranking_pairs": long.get("ranking_pairs", 0),
                "wm/cost_ranking_ready": float(candidate_ranking_ready(model, horizon, self.label_definition)),
            })
            self.metrics = metrics
            return metrics
        finally:
            model.train(was_training)
