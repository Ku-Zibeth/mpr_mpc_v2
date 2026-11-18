"""Balanced coverage, signal scale, matched-horizon and frozen-normalizer audit."""
import csv
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'reports/mppi_first_action_value_20261117'))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from torch.nn import functional as F
from analyze import R, OLD, load, dump, table
from tools.first_action_failure import cluster_bootstrap, correlation, paired_origins
from offline_model import load_model


def rows(name):
    with (R/name).open() as f:
        return list(csv.DictReader(f))


def main():
    torch.set_num_threads(1)
    p = load(OLD/'protocol.json')
    selected = load(OLD/'training_decision.json')['selected']
    model, _ = load_model(ROOT/selected['checkpoint'], selected['context'])
    candidates = rows('candidate_predictions.csv')
    groups = rows('origin_level_ranking.csv')
    origin = rows('origin_coverage.csv')
    pairs = rows('pair_diagnostics.csv')
    coverage = rows('oracle_candidate_analysis.csv')
    shift = rows('feature_shift_and_sensitivity.csv')
    def numeric(row):
        out = {}
        for k, v in row.items():
            if not v:
                out[k] = None
            else:
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
        return out
    groups = [numeric(r) for r in groups]
    original = [r for r in groups if r['scorer'] == 'qb_original']
    reverse = [r for r in groups if r['scorer'] == 'qb_reversed_diagnostic_only']
    reverse_difference = {}
    for split in ['fit', 'inner_stop', 'seen_validation']:
        a = [r for r in original if r['horizon'] == 1 and r['split'] == split]
        b = [r for r in reverse if r['horizon'] == 1 and r['split'] == split]
        paired = paired_origins(a, b)
        reverse_difference[split] = cluster_bootstrap(paired, ['pair_accuracy_improvement', 'regret_improvement'])
    # Same physical candidates, not disjoint maps, for every horizon target comparison.
    target_agreement = {}
    for h in [2,4,10]:
        result = []
        for name in sorted({r['origin_id'] for r in candidates}):
            a = {r['candidate_id']:float(r['true_return']) for r in candidates if r['origin_id']==name and r['horizon']=='1'}
            b = {r['candidate_id']:float(r['true_return']) for r in candidates if r['origin_id']==name and int(r['horizon'])==h}
            if not b:
                continue
            assert a.keys() == b.keys()
            ids = sorted(a)
            yy, zz = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
            result.append(dict(scene_id=int(name.split('_')[0]),
                spearman=correlation(yy,zz), kendall=correlation(yy,zz,'kendall'),
                same_top1=float(np.argmax(yy)==np.argmax(zz))))
        target_agreement[f'H{h}_vs_H1'] = cluster_bootstrap(result,['spearman','kendall','same_top1'])
    # Recreate fit-only weights and normalization from genuine CURRENT features.
    xx, yy, ww, contexts = [], [], [], []
    for scene in p['fit_scenes']:
        for before in p['before_steps']:
            name = f'{scene}_{before}'
            c = np.load(OLD/'origins'/name/'catalog.npz')
            local = sorted([r for r in candidates if r['origin_id']==name and r['horizon']=='1'], key=lambda r:int(r['candidate_id'][1:]))
            xx.extend(c['F2'])
            yy.extend(float(r['true_return']) for r in local)
            ww.extend([1/len(p['fit_scenes'])/len(p['before_steps'])/len(local)]*len(local))
            contexts.append(name)
    x, y, w = torch.tensor(np.array(xx)), torch.tensor(yy), torch.tensor(ww)
    mu = (w[:,None]*x).sum(0)
    sd = (w[:,None]*(x-mu).square()).sum(0).sqrt().clamp_min(.001)
    ym = (w*y).sum()
    ys = (w*(y-ym).square()).sum().sqrt().clamp_min(1.)
    assert torch.allclose(mu, model.input_mean, atol=1e-4, rtol=1e-5)
    assert torch.allclose(sd, model.input_scale, atol=1e-4, rtol=1e-5)
    assert torch.allclose(ym, model.target_mean, atol=1e-4, rtol=1e-5)
    assert torch.allclose(ys, model.target_scale, atol=1e-4, rtol=1e-5)
    with torch.no_grad():
        pred = model(x)
        huber = float((w*F.smooth_l1_loss((pred-ym)/ys,(y-ym)/ys,reduction='none')).sum())
    logit_rows = []
    for name in contexts:
        local = [r for r in pairs if r['origin_id']==name and r['horizon']=='1' and r['eligible']=='True']
        if local:
            logits = np.array([np.sign(float(r['actual_difference']))*float(r['predicted_difference'])/10 for r in local])
            logit_rows.append(dict(scene_id=int(name.split('_')[0]), mean_rank_logit=float(logits.mean()),
                pair_probability=float((1/(1+np.exp(-logits))).mean()), ranking_loss=float(np.logaddexp(0,-logits).mean())))
    loss = cluster_bootstrap(logit_rows,['mean_rank_logit','pair_probability','ranking_loss'])
    range_summary, distribution = {}, {}
    for split in ['fit','inner_stop','seen_validation']:
        rr = [r for r in original if r['horizon']==1 and r['split']==split]
        range_summary[split] = cluster_bootstrap(rr,['pair_accuracy','predicted_range','actual_range','baseline_gain_mae','selected_gain'])
        features = [numeric(r) for r in shift if r['split']==split]
        distribution[split] = cluster_bootstrap(features,['nearest_fit_context_standardized_rms','context_fraction_beyond_fit3sigma'])
    oracle_stability = {}
    for split in ['fit','inner_stop','seen_validation']:
        rr = [numeric(r) for r in coverage if r['split']==split and r['horizon']=='1']
        for r in rr:
            r['oracle_gain_above_fixed_tolerance'] = r['oracle_gain']-1.
        map_values = {}
        for scene in sorted({r['scene_id'] for r in rr}):
            local = [r for r in rr if r['scene_id']==scene]
            map_values[int(scene)] = dict(oracle_gain=float(np.mean([r['oracle_gain'] for r in local])),
                has_meaningful_positive=any(r['any_positive']>0 for r in local))
        largest = max(map_values,key=lambda s:map_values[s]['oracle_gain'])
        without = [r for r in rr if int(r['scene_id'])!=largest]
        oracle_stability[split] = dict(
            summaries=cluster_bootstrap(rr,['oracle_gain','oracle_gain_above_fixed_tolerance','model_gain','random_gain','any_positive','selected_positive','selected_negative']),
            per_map=map_values, meaningful_positive_maps=sum(x['has_meaningful_positive'] for x in map_values.values()),
            total_maps=len(map_values), largest_gain_map=largest,
            leave_largest_map_out=cluster_bootstrap(without,['oracle_gain','model_gain','any_positive']),
            note='Excluding largest map is an influence diagnostic, never a new favorable test cohort. Oracle includes Baseline, so raw oracle gain is necessarily nonnegative.')
    counts = {}
    for split in ['fit','inner_stop','seen_validation']:
        for h in [1,2,4,10]:
            pp = [r for r in pairs if r['split']==split and int(r['horizon'])==h]
            if pp:
                eligible = [r for r in pp if r['eligible']=='True']
                counts[f'{split}/H{h}'] = dict(raw=len(pp), effective=len(eligible), ties=len(pp)-len(eligible),
                    pooled_pair_accuracy=float(np.mean([float(r['original_credit']) for r in eligible])) if eligible else None,
                    macro_accuracy=next(v['pair_accuracy']['mean'] for k,v in load(R/'ranking_analysis.json')['summaries'].items() if k==f'{split}/H{h}/all_available/qb_original'),
                    note='Pooled count is descriptive, never used as independent-map accuracy or binomial CI.')
    dump('model_numeric_diagnostics.json', dict(
        normalization_reconstructed_from_fit_only=True, fit_rows=len(x), distinct_fit_origins=len(contexts), fit_maps=len(p['fit_scenes']),
        input_mean_max_abs_error=float((mu-model.input_mean).abs().max()), input_scale_max_abs_error=float((sd-model.input_scale).abs().max()),
        label_mean=float(ym), label_scale=float(ys), final_fit_normalized_huber=huber,
        final_fit_rank_loss=loss, rank_loss_coefficient=.1, rank_logit_divisor=10.,
        signal_scale_by_split=range_summary, context_shift_by_split=distribution,
        reverse_paired_comparison=reverse_difference, pair_counts_and_micro_vs_macro=counts,
        interpretation='Same-origin action score ranges are compressed relative to actual returns. This is observed weak action discrimination, not proof that context, objective, or capacity is uniquely causal.'))
    dump('oracle_stability.json',oracle_stability)
    dump('horizon_target_agreement.json',target_agreement)
    early = [r for r in rows('branch_integrity.csv') if r['terminated_before_requested_prefix_end']=='True']
    dump('early_terminal_samples.json',dict(count=len(early),samples=[{k:r[k] for k in ['scene_id','origin_id','candidate_id','horizon','episode_length','termination_reason','raw_npz']} for r in early],
        retained_in_all_return_and_oracle_statistics=True,missing_fixed_H_latent_not_fabricated=True))
    print('Fit-only normalization reproduced. Fit origins:',len(contexts),'fit labels:',len(x),flush=True)


if __name__ == '__main__':
    main()
