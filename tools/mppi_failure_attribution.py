"""Offline, scene-paired MPPI attribution statistics. Never selects online actions."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr

RETURN_TOLERANCE = 1.0  # pre-fixed physical-return indifference band, not score margin


def ranking_metrics(predicted, actual, baseline=0, tolerance=RETURN_TOLERANCE):
    p, y = np.asarray(predicted, float), np.asarray(actual, float)
    if p.shape != y.shape or p.ndim != 1 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Require finite, matching one-dimensional candidate scores.')
    i, j = np.triu_indices(len(y), 1)
    eligible = np.abs(y[i] - y[j]) > tolerance
    product = (p[i] - p[j]) * (y[i] - y[j])
    pair = np.where(product > 0, 1., np.where(product < 0, 0., .5))
    selected = int(np.argmax(p))  # fixed original-order tie rule, never future labels
    q = np.arange(len(y)) != baseline
    effective = q & (np.abs(y - y[baseline]) > tolerance)
    baseprod = (p - p[baseline]) * (y - y[baseline])
    baseacc = np.where(baseprod > 0, 1., np.where(baseprod < 0, 0., .5))
    varying = len(y) > 1 and np.ptp(p) > 0 and np.ptp(y) > 0
    false_positive = (p[q] > p[baseline]) & (y[q] < y[baseline] - tolerance)
    predicted_positive = p[q] > p[baseline]
    return dict(candidates=len(y), eligible_pairs=int(eligible.sum()),
        pair_accuracy=float(pair[eligible].mean()) if eligible.any() else None,
        spearman=float(spearmanr(p, y).statistic) if varying else None,
        kendall=float(kendalltau(p, y).statistic) if varying else None,
        top1_regret=float(y.max() - y[selected]), selected_index=selected,
        selected_gain=float(y[selected] - y[baseline]),
        oracle_best_gain=float(y.max() - y[baseline]),
        positive_fraction=float(np.mean(y[q] > y[baseline] + tolerance)) if q.any() else None,
        negative_fraction=float(np.mean(y[q] < y[baseline] - tolerance)) if q.any() else None,
        equal_fraction=float(np.mean(np.abs(y[q] - y[baseline]) <= tolerance)) if q.any() else None,
        baseline_accuracy=float(baseacc[effective].mean()) if effective.any() else None,
        false_improvement_rate=float(false_positive.sum()/predicted_positive.sum()) if predicted_positive.any() else None)


def scene_bootstrap(rows, keys, samples=2000, seed=1):
    """Average within map first; seeds/origins/candidate pairs aren't independent maps."""
    rng = np.random.default_rng(seed)
    result = {}
    for key in keys:
        scene_values = []
        for scene in sorted({r['scenario'] for r in rows}):
            vals = [r[key] for r in rows if r['scenario'] == scene and r.get(key) is not None and np.isfinite(r[key])]
            if vals:
                scene_values.append(float(np.mean(vals)))
        v = np.asarray(scene_values)
        if not len(v):
            result[key] = dict(mean=None, ci95=[None, None], scenes=0)
            continue
        draws = v[rng.integers(0, len(v), (samples, len(v)))].mean(1)
        result[key] = dict(mean=float(v.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(), scenes=len(v))
    return result


def distribution_from_indices(actions, values, baseline, previous_mean, minimum, maximum,
                              indices, beta=.5, active=11):
    a, v, idx = np.asarray(actions), np.asarray(values), np.asarray(indices)
    w = np.exp(beta * (v[idx] - v[idx].max())); w /= w.sum()
    mean = np.sum(w[:, None, None] * a[idx], axis=0)
    std = np.clip(np.sqrt(np.sum(w[:, None, None] * (a[idx] - mean)**2, axis=0)), minimum, maximum)
    delta, first = mean[:active] - baseline[:active], mean[0] - baseline[0]
    pairs = a[idx, :active].reshape(len(idx), -1)
    distances = np.linalg.norm(pairs[:, None] - pairs[None, :], axis=-1)
    upper = distances[np.triu_indices(len(idx), 1)]
    stats = dict(elite_value_gap=float(v[idx[0]]-v[idx[-1]]), weight_entropy=float(-np.sum(w*np.log(w.clip(1e-30)))),
        effective_sample_size=float(1/np.sum(w*w)), baseline_in_elite=float(0 in idx),
        first_action_delta=float(np.linalg.norm(first)), first_steer_delta=float(first[0]),
        first_throttle_delta=float(first[1]), sequence_delta=float(np.sqrt(np.mean(delta**2))),
        steer_std=float(std[:active, 0].mean()), throttle_std=float(std[:active, 1].mean()),
        boundary_saturation_fraction=float(np.mean(np.abs(a[1:, :active]) >= 1-1e-7)),
        elite_pairwise_distance=float(upper.mean()),
        mean_change=float(np.sqrt(np.mean((mean[:active]-previous_mean[:active])**2))))
    return mean, std, idx, stats


def replanning_error(plans, sources, executed, h):
    """Sequence open-loop commitment vs actual future executed action, no reward claim."""
    n = len(executed) - h
    if n <= 0:
        return np.empty((0, 2))
    rows = np.arange(n)
    return plans[rows, np.asarray(sources[:n], int), h] - executed[h:]
