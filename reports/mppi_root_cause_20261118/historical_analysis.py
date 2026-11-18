"""Reanalyse recorded interventions only. No new model, environment or label definition."""
from collections import defaultdict
import csv
import numpy as np
from scipy.stats import spearmanr,kendalltau
from study_io import R,PRIOR,load,dump,table

def read(path):
    with path.open() as f:return list(csv.DictReader(f))

def main():
    source=PRIOR/'reports/mppi_first_action_value_20261117'
    returns=read(source/'first_action_returns.csv')
    candidates={(r['origin'],r['candidate']):r for r in read(source/'first_action_candidates.csv')}
    oracle={(r['origin_id'],r['candidate_id'],int(r['horizon'])):r for r in read(PRIOR/'reports/mppi_first_action_failure_20261118/predicted_vs_actual_state.csv')}
    catalogs={};rows=[];groups=defaultdict(list)
    for r in returns:
        if r['takeover']!='baseline' or r['replay'].lower()=='true':continue
        origin=r['origin_id'];cid=r['candidate_id'];h=int(r['horizon']);c=candidates[origin,cid]
        if origin not in catalogs:catalogs[origin]=np.load(source/'origins'/origin/'catalog.npz')
        z=catalogs[origin];j=int(c['sequence_row']);idx=int(z['pool_indices'][j])
        o=oracle.get((origin,cid,h),{})
        rows.append(dict(scene=int(r['scenario']),origin=origin,candidate=cid,horizon=h,split=r['split'],
            label_protocol=f'execute fixed first {h} candidate actions, then original feedback Lattice+SAC to native done',
            actual_first_reward=float(r['actual_first_reward']),actual_discounted_return=float(r['actual_return']),
            actual_success=float(r['success']),actual_crash=float(r['crash']),actual_out_of_road=float(r['out_of_road']),
            actual_route_completion=float(r['route_completion']),
            predicted_prefix_reward=o.get('predicted_prefix'),predicted_terminal_q=o.get('predicted_terminal_q'),
            predicted_total_return=o.get('J00_predicted'),actual_prefix_reward=o.get('actual_prefix'),
            predicted_common_h10_return=float(z['scores'][idx]),predicted_common_h10_prefix=float(z['prefix'][idx]),
            predicted_common_h10_terminal_contribution=float(z['scores'][idx]-z['prefix'][idx]),
            first_action_distance=float(c['delta_from_baseline']),
            latent_l2=o.get('latent_l2'),latent_cosine=o.get('latent_cosine'),
            q_members=o.get('q_members'),source=r['source']))
        groups[origin,h].append(rows[-1])
    origin_rows=[]
    for (origin,h),g in groups.items():
        base=next(r for r in g if r['candidate']=='c0');y=np.array([r['actual_discounted_return'] for r in g])
        score=np.array([r['predicted_common_h10_return'] for r in g]);gains=y-base['actual_discounted_return']
        selected=int(score.argmax());best=float(gains.max());tie=1.0 # Preserved prior fixed tie threshold, not tuned.
        i,j=np.triu_indices(len(g),1);dy=y[i]-y[j];ds=score[i]-score[j];eligible=np.abs(dy)>tie
        for r in g:
            r.update(actual_gain=r['actual_discounted_return']-base['actual_discounted_return'],
                predicted_gain=r['predicted_common_h10_return']-base['predicted_common_h10_return'],
                h10_prediction_minus_h_label=r['predicted_common_h10_return']-r['actual_discounted_return'],
                top1_regret=best-gains[selected])
        origin_rows.append(dict(scene=base['scene'],origin=origin,horizon=h,split=base['split'],branches=len(g),
            pair_accuracy=float(np.mean(np.sign(dy[eligible])*np.sign(ds[eligible])==1)) if eligible.any() else None,
            spearman=float(spearmanr(score,y).statistic) if np.ptp(y)>0 and np.ptp(score)>0 else None,
            kendall=float(kendalltau(score,y).statistic) if np.ptp(y)>0 and np.ptp(score)>0 else None,
            effective_pairs=int(eligible.sum()),tie_pairs=int((~eligible).sum()),
            oracle_gain=best,model_gain=float(gains[selected]),top1_regret=best-gains[selected],
            positive_candidate_ratio=float(np.mean([r['actual_discounted_return']-base['actual_discounted_return']>tie for r in g if r['candidate']!='c0'])),
            negative_candidate_ratio=float(np.mean([r['actual_discounted_return']-base['actual_discounted_return']<-tie for r in g if r['candidate']!='c0'])),
            baseline_return=base['actual_discounted_return']))
    table('prediction_vs_actual.csv',rows);table('historical_origin_ranking.csv',origin_rows)
    # Same action pool paired across horizons. These are not MPPI closed-loop labels.
    labelsets=defaultdict(dict)
    for r in rows:labelsets[r['origin'],r['candidate']][r['horizon']]=r['actual_discounted_return']
    paired=[x for x in labelsets.values() if 1 in x and 10 in x]
    gainsets=defaultdict(dict)
    for r in rows:
        if r['candidate']!='c0':gainsets[r['origin'],r['candidate']][r['horizon']]=r['actual_gain']
    paired_gain=[x for x in gainsets.values() if 1 in x and 10 in x]
    summary={}
    rng=np.random.default_rng(20261118)
    for h in [1,2,4,10]:
        selected=[r for r in origin_rows if r['horizon']==h];scenes=sorted({r['scene'] for r in selected})
        stats={}
        for k in ['pair_accuracy','oracle_gain','model_gain','top1_regret','positive_candidate_ratio']:
            means=np.array([np.mean([r[k] for r in selected if r['scene']==s and r[k] is not None]) for s in scenes])
            boot=means[rng.integers(len(means),size=(2000,len(means)))].mean(1)
            stats[k]=dict(mean=float(means.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())
        summary[str(h)]=dict(scenes=len(scenes),origins=len(selected),branches=sum(r['branches'] for r in selected),statistics=stats)
    dump('historical_prediction_statistics.json',dict(horizons=summary,
        h1_h10_raw_label_spearman=float(spearmanr([x[1] for x in paired],[x[10] for x in paired]).statistic),
        h1_h10_within_origin_gain_spearman=float(spearmanr([x[1] for x in paired_gain],[x[10] for x in paired_gain]).statistic),
        branch_count=len(rows),raw_record_count=len(returns),excluded_records=[r for r in returns if r['takeover']!='baseline' or r['replay'].lower()=='true'],
        paired_h1_h10=len(paired),tie_threshold=1.0,
        warning='Historical already-used data, not new validation. H10 score tested against different H labels; errors are protocol mismatch as well as model error.'))
    print('HISTORICAL_ROWS',len(rows),'ORIGINS',len(groups),flush=True)

if __name__=='__main__':main()
