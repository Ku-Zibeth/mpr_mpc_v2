"""Frozen selected Q_B diagnosis on existing labels; no fitting or map selection."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'reports/mppi_first_action_value_20261117'))
import numpy as np
import torch
from tools.first_action_study import file_sha, clean
from tools.first_action_failure import (ranking, pair_rows, cluster_bootstrap,
                                       oracle_values, discounted, assert_submitted_actions)
from offline_model import load_model

R = Path(__file__).parent
OLD = ROOT / 'reports/mppi_first_action_value_20261117'
KEYS = ['pair_accuracy', 'spearman', 'kendall', 'top1_regret', 'baseline_gain_mae',
        'gain_pearson', 'baseline_accuracy', 'positive_identification_rate', 'selected_gain',
        'oracle_gain', 'positive_fraction', 'selected_positive', 'selected_negative',
        'predicted_range', 'actual_range']


def load(path):
    return json.loads(Path(path).read_text())


def dump(name, value):
    with (R / name).open('x') as stream:
        json.dump(clean(value), stream, indent=2, allow_nan=False)


def table(name, rows):
    with (R / name).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(clean(rows))


def split_of(scene, protocol):
    return 'fit' if scene in protocol['fit_scenes'] else 'inner_stop' if scene in protocol['inner_stop_scenes'] else 'seen_validation'


def band(value, edges, labels):
    return labels[int(np.searchsorted(edges, value, side='right'))]


def main():
    torch.set_num_threads(1)
    p = load(OLD / 'protocol.json')
    chosen = load(OLD / 'training_decision.json')['selected']
    assert chosen['schema'] == 'F2' and file_sha(ROOT / chosen['checkpoint']) == chosen['checkpoint_sha']
    model, artifact = load_model(ROOT / chosen['checkpoint'], chosen['context'])
    dump('protocol.json', dict(
        date='2026-11-18', analysis_only=True, new_environment_runs=0, new_training=0,
        bootstrap_samples=2000, bootstrap_seed=20261118, return_tolerance=1.0,
        previously_exposed_validation=p['validation_scenes'], new_independent_scenes=[],
        selected_model=chosen['checkpoint'], checkpoint_sha=chosen['checkpoint_sha'],
        scorer_direction='larger raw Q_B predicts larger H1 total raw discounted return',
        current_qb_input='current535 + candidate physical command2; no future latent or WM next()',
        horizon_transfer='The same H1-trained Q_B is evaluated against H2/4/10 labels only as an out-of-target diagnostic.',
        comparisons='All-map H1; same six pilot-map origins for paired H1/2/4/10. No data-driven scene selection.',
        strata=dict(action_pair_max_l2_edges=[.01, .05, .1], actual_pair_gap_edges=[1., 5., 20.],
                    initial_speed_edges=[3., 8., 15.], coarse_path_curvature_proxy_edges=[.02, .1],
                    traffic_density='fixed .1; do not invent variable density',
                    outcomes='episode-level crash/OOR, not accident timing'),
        state_oracle='Pure terminal-latent swap applies to original TD terminal Q, NOT to current-state Q_B; early-terminal missing z_H remains missing.'))
    data, groups, pairs, oracle, predictions, features = [], [], [], [], {}, {}
    integrity = []
    origins = []
    # All saved current features are read before any labels are used for analysis.
    for folder in sorted((OLD / 'origins').iterdir()):
        if not (folder / 'catalog.json').exists():
            continue
        c = {k: a.copy() for k, a in np.load(folder / 'catalog.npz').items()}
        meta, state = load(folder / 'catalog.json'), load(folder / 'origin.json')
        with torch.no_grad():
            pred = model(torch.tensor(c['F2'])).numpy()
        assert np.array_equal(c['F2'][:, :535], np.broadcast_to(c['state535'], (len(pred), 535)))
        assert np.array_equal(c['F2'][:, -2:], c['actions'][:, 0])
        predictions[folder.name], features[folder.name] = pred, c
        velocity = np.asarray(state['vehicle']['velocity'], float)
        curvature = float(c['state535'][522] * .5)  # original default path scale, proxy only
        origins.append(dict(scene_id=meta['scene'], origin_id=folder.name,
                            split=split_of(meta['scene'], p), candidates=meta['unique'],
                            speed=float(np.linalg.norm(velocity)), coarse_curvature_proxy=curvature,
                            curvature_source='normalized coarse-path curvature_max * .5; not measured road curvature',
                            traffic_density=meta['simulator']['traffic_density'],
                            traffic_objects=len(state['traffic']), source=str(folder)))
        for h in p['horizons']:
            records, arrays = [], []
            for i, rep in enumerate(meta['representatives']):
                file = OLD / 'branches' / folder.name / f"H{h}_{rep['candidate_id']}_baseline.json"
                if not file.exists():
                    break
                rec = load(file)
                with np.load(ROOT / rec['source']) as archive:
                    values = {k: a.copy() for k, a in archive.items()}
                label = discounted(values['rewards'], values['done'], values['terminated'], values['truncated'], h, p['gamma'])
                assert_submitted_actions(c['actions'][i], values['actions'], h)
                assert abs(label['total'] - rec['actual_return']) < 1e-8
                records.append(rec)
                arrays.append(values)
            if len(records) != len(pred):
                continue
            y = np.array([r['actual_return'] for r in records])
            common = dict(scene_id=meta['scene'], origin_id=folder.name, horizon=h, split=split_of(meta['scene'], p))
            for name, score in [('qb_original', pred), ('qb_reversed_diagnostic_only', -pred),
                                ('original_h10_score', c['scores'][c['pool_indices']])]:
                groups.append(common | dict(scorer=name) | ranking(score, y))
            random_idx = next(i for i, rep in enumerate(meta['representatives']) if 'fixed_gaussian' in rep['roles'])
            original_selected = next(i for i, rep in enumerate(meta['representatives']) if 'mppi_selected' in rep['roles'])
            oracle.append(common | oracle_values(y, pred, random_idx) | dict(
                original_mppi_gain=float(y[original_selected] - y[0]), random_index=random_idx,
                selected_index=int(pred.argmax()), oracle_index=int(y.argmax()),
                selected_success=records[int(pred.argmax())]['success'], selected_crash=records[int(pred.argmax())]['crash'],
                selected_oor=records[int(pred.argmax())]['out_of_road']))
            ii, jj, dy, dp, eligible = pair_rows(pred, y)
            deltas = np.linalg.norm(c['actions'][:, 0] - c['baseline'][0], axis=1)
            for k, (i, j) in enumerate(zip(ii, jj)):
                pairs.append(common | dict(i=int(i), j=int(j), actual_difference=float(dy[k]),
                    predicted_difference=float(dp[k]), eligible=bool(eligible[k]),
                    original_credit=float(.5 + .5 * np.sign(dp[k] * dy[k])),
                    reversed_credit=float(.5 - .5 * np.sign(dp[k] * dy[k])),
                    action_band=band(max(deltas[i], deltas[j]), [.01, .05, .1], ['<.01', '.01-.05', '.05-.1', '>=.1']),
                    return_gap_band=band(abs(dy[k]), [1., 5., 20.], ['tie<=1', '1-5', '5-20', '>=20']),
                    speed_band=band(origins[-1]['speed'], [3., 8., 15.], ['<3', '3-8', '8-15', '>=15']),
                    coarse_curvature_band=band(curvature, [.02, .1], ['<.02', '.02-.1', '>=.1']),
                    pair_has_crash=bool(records[i]['crash'] or records[j]['crash']),
                    pair_has_oor=bool(records[i]['out_of_road'] or records[j]['out_of_road']), traffic_density=origins[-1]['traffic_density']))
            for i, (rec, archive) in enumerate(zip(records, arrays)):
                data.append(common | dict(candidate_id=rec['candidate_id'], predicted_qb=float(pred[i]),
                    true_return=float(y[i]), predicted_gain=float(pred[i] - pred[0]), actual_gain=float(y[i] - y[0]),
                    action_delta_l2=float(deltas[i]), steer_delta=float(c['actions'][i, 0, 0] - c['baseline'][0, 0]),
                    throttle_delta=float(c['actions'][i, 0, 1] - c['baseline'][0, 1]),
                    success=rec['success'], crash=rec['crash'], out_of_road=rec['out_of_road'],
                    route_completion=rec['route_completion'], source=rec['source']))
    # All-branch integrity index includes replay and non-SAC takeovers, separately labelled.
    with (R / 'branch_index.csv').open() as stream:
        all_rows = list(csv.DictReader(stream))
    for row in all_rows:
        row = row | dict(recorded_action_scope='submitted to env.step; no measured postprocessing wheel forces',
                         repeat_seed_scope='same scenario policy seed, not an independent traffic replicate')
        integrity.append(row)
    table('branch_integrity.csv', integrity)
    table('origin_coverage.csv', origins)
    table('candidate_predictions.csv', data)
    table('pair_diagnostics.csv', pairs)
    table('origin_level_ranking.csv', groups)
    table('oracle_candidate_analysis.csv', oracle)
    np.savez_compressed(R / 'frozen_qb_predictions.npz', **predictions)
    summaries, horizon_rows, scene_rows = {}, [], []
    for split in ['fit', 'inner_stop', 'seen_validation', 'all_seen']:
        for h in p['horizons']:
            for cohort in ['all_available', 'common_pilot']:
                for scorer in ['qb_original', 'qb_reversed_diagnostic_only', 'original_h10_score']:
                    rows = [r for r in groups if r['horizon'] == h and r['scorer'] == scorer
                            and (split == 'all_seen' or r['split'] == split)
                            and (cohort != 'common_pilot' or r['scene_id'] in p['pilot_scenes'])]
                    if not rows:
                        continue
                    stat = cluster_bootstrap(rows, KEYS)
                    key = f'{split}/H{h}/{cohort}/{scorer}'
                    summaries[key] = stat
                    for metric, value in stat.items():
                        horizon_rows.append(dict(split=split, horizon=h, cohort=cohort, scorer=scorer,
                                                 metric=metric, **value))
    for (scene, h, scorer) in sorted({(r['scene_id'], r['horizon'], r['scorer']) for r in groups}):
        rows = [r for r in groups if (r['scene_id'], r['horizon'], r['scorer']) == (scene, h, scorer)]
        stat = cluster_bootstrap(rows, KEYS)
        scene_rows.append(dict(scene_id=scene, horizon=h, scorer=scorer, split=rows[0]['split'],
                               origins=len(rows), raw_pairs=sum(r['raw_pairs'] for r in rows),
                               effective_pairs=sum(r['effective_pairs'] for r in rows), tied_pairs=sum(r['tied_pairs'] for r in rows),
                               **{k: v['mean'] for k, v in stat.items()}))
    strata = {}
    for key in ['action_band', 'return_gap_band', 'speed_band', 'coarse_curvature_band', 'pair_has_crash', 'pair_has_oor', 'traffic_density']:
        for value in sorted({r[key] for r in pairs}, key=str):
            for h in p['horizons']:
                rows = [r for r in pairs if r[key] == value and r['horizon'] == h]
                if not rows:
                    continue
                # Aggregate pair credits inside origin, then maps, never pair bootstrap.
                rr = []
                for origin in sorted({r['origin_id'] for r in rows}):
                    local = [r for r in rows if r['origin_id'] == origin and r['eligible']]
                    if local:
                        rr.append(dict(scene_id=local[0]['scene_id'], pair_accuracy=float(np.mean([r['original_credit'] for r in local]))))
                strata[f'{key}/{value}/H{h}'] = dict(raw_pairs=len(rows), effective_pairs=sum(r['eligible'] for r in rows),
                    tied_pairs=sum(not r['eligible'] for r in rows), statistics=cluster_bootstrap(rr, ['pair_accuracy']))
    oracle_summary = {}
    for split in ['fit', 'inner_stop', 'seen_validation', 'all_seen']:
        for h in p['horizons']:
            rows = [r for r in oracle if r['horizon'] == h and (split == 'all_seen' or r['split'] == split)]
            if rows:
                oracle_summary[f'{split}/H{h}'] = cluster_bootstrap(rows, [
                    'oracle_gain', 'model_gain', 'random_gain', 'uniform_nonbaseline_expected_gain',
                    'top1_regret', 'positive_fraction', 'any_positive', 'selected_positive', 'selected_negative', 'original_mppi_gain'])
    original = summaries['seen_validation/H1/all_available/qb_original']['pair_accuracy']['mean']
    assert abs(original - load(OLD / 'validation.json')['evaluation']['summary']['pair_accuracy']['mean']) < 1e-12
    table('horizon_ranking_comparison.csv', horizon_rows)
    table('scene_level_ranking.csv', scene_rows)
    dump('ranking_analysis.json', dict(summaries=summaries, strata=strata,
        pair_counts={f'H{h}':dict(raw=sum(r['horizon'] == h for r in pairs), effective=sum(r['horizon'] == h and r['eligible'] for r in pairs),
                                 tied=sum(r['horizon'] == h and not r['eligible'] for r in pairs)) for h in p['horizons']},
        old_validation_accuracy_reproduced=original))
    dump('oracle_candidate_report.json', dict(summaries=oracle_summary,
        random='Pre-existing fixed_gaussian role, no new draw or outcome-dependent choice; uniform expectation separately reported.',
        oracle_scope='Actually measured representatives only; not a deployable policy; no accident-conditioned oracle selection.'))
    audit = load(R / 'integrity_audit.json')
    audit.update(prompt_complete_received=True, original_branch_count=576,
                 normalized_action_mapping_scope=load(R / 'action_mapping_scope.json'),
                 hierarchy=dict(independent_maps=18, origins=36, exact_replays=1,
                                stochastic_environment_replicates=0, h2_h4_h10_maps=6),
                 no_new_training_or_rollout=True)
    audit['limitations'] = [x for x in audit['limitations'] if 'prompt is truncated' not in x]
    dump('data_integrity_audit.json', audit)
    dump('paired_bootstrap_statistics.json', dict(ranking=summaries, oracle=oracle_summary,
        units='Whole-map cluster bootstrap,2000 draws,seed20261118; means first within origin then map. All data retrospective, no new validation.'))
    # Frozen-model normalization and direction audit, no training or sign change.
    fit_x = np.concatenate([features[k]['F2'] for k in features if int(k.split('_')[0]) in p['fit_scenes']])
    gradients, shifts = [], []
    normalized_fit = (fit_x - model.input_mean.numpy()) / model.input_scale.numpy()
    for origin, c in features.items():
        x = torch.tensor(c['F2'], requires_grad=True)
        gradient = torch.autograd.grad(model(x).sum(), x)[0][:, -2:].detach().numpy()
        normalized = (c['F2'] - model.input_mean.numpy()) / model.input_scale.numpy()
        distances = np.sqrt(np.mean((normalized[:1, None, :535] - normalized_fit[None, :, :535]) ** 2, axis=2))
        shifts.append(dict(scene_id=int(origin.split('_')[0]), origin_id=origin, split=split_of(int(origin.split('_')[0]), p),
            nearest_fit_context_standardized_rms=float(distances.min()),
            context_fraction_beyond_fit3sigma=float((np.abs(normalized[0, :535]) > 3).mean()),
            predicted_action_score_range=float(np.ptp(predictions[origin])),
            mean_steering_gradient=float(gradient[:, 0].mean()), mean_throttle_gradient=float(gradient[:, 1].mean())))
    table('feature_shift_and_sensitivity.csv', shifts)
    dump('model_implementation_audit.json', dict(selected_checkpoint=chosen['checkpoint'],
        checkpoint_sha=chosen['checkpoint_sha'], input_dimension=artifact['dimension'],
        input='original CURRENT535 + candidate physical steer/throttle2; no predicted future latent',
        output='scalar raw H1 total return; positive stored target_scale', target_scale=float(model.target_scale),
        input_scales_positive=bool((model.input_scale > 0).all()),
        label='true r0 + .995 original deterministic LatticeSAC continuation to native done',
        loss_sign='softplus(-sign(y_i-y_j)*(pred_i-pred_j)/10): minimizing rewards matching directions',
        online_selection='not deployed; offline argmax(score), larger is better',
        reverse_only_diagnostic=True, training_input='only present-state features, normalization buffers fit-only',
        input_buffer_sha=file_sha(ROOT / chosen['checkpoint']),
        source_code={str(f.relative_to(ROOT)):file_sha(f) for f in [OLD/'train_offline.py', OLD/'offline_model.py', OLD/'validate_offline.py']},
        future_dynamics_direct_path_to_QB=False, no_future_535_hybrid_constructed=True))
    print('Reproduced seen validation PA:', original, 'all pairs', len(pairs), flush=True)


if __name__ == '__main__':
    main()
