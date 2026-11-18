"""Offline, map-clustered diagnostics. Never imported by an online planner."""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

TOLERANCE = 1.0  # unchanged Phase 1 raw-return indifference band


def correlation(a, b, method='spearman'):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    fn = {'spearman': spearmanr, 'kendall': kendalltau, 'pearson': pearsonr}[method]
    return float(fn(a, b).statistic)


def discounted(rewards, done, terminated, truncated, horizon, gamma=.995):
    r = np.asarray(rewards, float)
    d, term, trunc = map(lambda x: np.asarray(x, bool), (done, terminated, truncated))
    if not len(r) or not np.isfinite(r).all() or not (r.shape == d.shape == term.shape == trunc.shape):
        raise ValueError('Finite reward and matching termination vectors required.')
    if not np.array_equal(d, term | trunc) or not d[-1] or d[:-1].any():
        raise ValueError('Require exactly one native terminal step and done=terminated|truncated.')
    if not 0 <= gamma <= 1 or horizon < 1:
        raise ValueError('Invalid original gamma or prefix length.')
    g = np.zeros(len(r) + 1)
    for i in range(len(r) - 1, -1, -1):
        g[i] = r[i] + gamma * g[i + 1]
    h = min(horizon, len(r))
    prefix = float(r[:h] @ gamma ** np.arange(h))
    return dict(total=float(g[0]), prefix=prefix, after_first=float(g[1]),
                after_prefix=float(g[h]), prefix_steps=h, early_terminal=len(r) < horizon)


def assert_submitted_actions(sequence, recorded, horizon):
    s, a = np.asarray(sequence), np.asarray(recorded)
    if s.ndim != 2 or a.ndim != 2 or s.shape[1] != 2 or a.shape[1] != 2:
        raise ValueError('Expected steer/throttle in that order.')
    if not np.isfinite(a).all() or (np.abs(a) > 1).any():
        raise ValueError('Invalid submitted physical-control command.')
    h = min(horizon, len(a))
    if not np.array_equal(s[:h], a[:h]):
        raise ValueError('Submitted prefix differs from stored candidate.')


def pair_rows(predicted, actual):
    p, y = np.asarray(predicted, float), np.asarray(actual, float)
    if p.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Finite aligned scores and labels required.')
    i, j = np.triu_indices(len(y), 1)
    dy, dp = y[i] - y[j], p[i] - p[j]
    return i, j, dy, dp, np.abs(dy) > TOLERANCE


def ranking(predicted, actual, baseline=0):
    p, y = np.asarray(predicted, float), np.asarray(actual, float)
    i, j, dy, dp, valid = pair_rows(p, y)
    credit = np.sign(dp * dy) * .5 + .5
    nonbase = np.arange(len(y)) != baseline
    pg, yg = p - p[baseline], y - y[baseline]
    baseline_valid = nonbase & (np.abs(yg) > TOLERANCE)
    top = int(np.argmax(p))
    positives, negatives = yg > TOLERANCE, yg < -TOLERANCE
    predicted_positive = pg > 0
    return dict(pair_accuracy=float(credit[valid].mean()) if valid.any() else None,
                raw_pairs=len(i), effective_pairs=int(valid.sum()), tied_pairs=int((~valid).sum()),
                spearman=correlation(p, y), kendall=correlation(p, y, 'kendall'),
                top1_regret=float(y.max() - y[top]), selected_index=top,
                baseline_gain_mae=float(np.abs(pg[nonbase] - yg[nonbase]).mean()) if nonbase.any() else None,
                gain_pearson=correlation(pg[nonbase], yg[nonbase], 'pearson'),
                baseline_accuracy=float((np.sign(pg[baseline_valid] * yg[baseline_valid]) * .5 + .5).mean()) if baseline_valid.any() else None,
                positive_identification_rate=float((predicted_positive & positives).sum() / (positives & nonbase).sum()) if (positives & nonbase).any() else None,
                positive_fraction=float(positives[nonbase].mean()) if nonbase.any() else None,
                selected_gain=float(yg[top]), selected_positive=float(positives[top]),
                selected_negative=float(negatives[top]), oracle_gain=float(y.max() - y[baseline]),
                predicted_range=float(np.ptp(p)), actual_range=float(np.ptp(y)))


def cluster_bootstrap(rows, keys, samples=2000, seed=20261118):
    """Resample entire maps; preserve all origins/candidates within each map."""
    scenes = sorted({r['scene_id'] for r in rows})
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scenes), (samples, len(scenes))) if scenes else None
    result = {}
    for key in keys:
        values = []
        for scene in scenes:
            local = [r[key] for r in rows if r['scene_id'] == scene and r.get(key) is not None and np.isfinite(r[key])]
            values.append(float(np.mean(local)) if local else np.nan)
        v = np.asarray(values)
        if not len(v) or not np.isfinite(v).any():
            result[key] = dict(mean=None, ci95=[None, None], maps=len(scenes), effective_maps=0)
            continue
        resampled = v[draws]
        counts = np.isfinite(resampled).sum(1)
        estimates = np.nansum(resampled, 1)[counts > 0] / counts[counts > 0]
        result[key] = dict(mean=float(np.nanmean(v)), ci95=np.quantile(estimates, [.025, .975]).tolist(),
                           maps=len(scenes), effective_maps=int(np.isfinite(v).sum()),
                           undefined_resamples=int((counts == 0).sum()))
    return result


def paired_origins(before, after):
    def identity(r):
        return r['scene_id'], r['origin_id'], r['horizon']
    a, b = {identity(r): r for r in before}, {identity(r): r for r in after}
    if a.keys() != b.keys() or len(a) != len(before) or len(b) != len(after):
        raise ValueError('Oracle comparison must match scene/origin/horizon exactly.')
    out = []
    for key in sorted(a):
        x, y = a[key], b[key]
        row = dict(scene_id=key[0], origin_id=key[1], horizon=key[2])
        for name in ['pair_accuracy', 'spearman', 'kendall', 'baseline_accuracy', 'positive_identification_rate']:
            row[name + '_improvement'] = y.get(name) - x.get(name) if y.get(name) is not None and x.get(name) is not None else None
        row['regret_improvement'] = x['top1_regret'] - y['top1_regret']
        out.append(row)
    return out


def oracle_values(actual, predicted, random_index, baseline=0):
    p, y = np.asarray(predicted, float), np.asarray(actual, float)
    ranking(p, y, baseline)  # validate before selecting indices
    q = np.arange(len(y)) != baseline
    gain = y - y[baseline]
    selected = int(np.argmax(p))
    return dict(baseline_return=float(y[baseline]), random_gain=float(gain[random_index]),
                uniform_nonbaseline_expected_gain=float(gain[q].mean()), oracle_gain=float(gain.max()),
                model_gain=float(gain[selected]), model_worst_gain=float(gain[int(np.argmin(p))]),
                top1_regret=float(y.max() - y[selected]), positive_fraction=float((gain[q] > TOLERANCE).mean()),
                any_positive=float((gain[q] > TOLERANCE).any()), selected_positive=float(gain[selected] > TOLERANCE),
                selected_negative=float(gain[selected] < -TOLERANCE))


def isolated_state_inputs(current, predicted_terminal, actual_terminal, scorer):
    """Q_B(s_t,a0) has no predicted successor: never make mixed-time 535 inputs."""
    if scorer == 'current_qb':
        return np.array(current, copy=True), np.array(current, copy=True)
    if scorer == 'terminal_td_q':
        if np.shape(predicted_terminal) != np.shape(actual_terminal):
            raise ValueError('Predicted and real endpoint latents must have the same horizon.')
        return np.array(predicted_terminal, copy=True), np.array(actual_terminal, copy=True)
    raise ValueError('Unknown scorer/input contract.')
