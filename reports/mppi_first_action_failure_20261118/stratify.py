"""Full ranking metrics in predeclared strata, with explicit shared Baseline."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
from analyze import R, load, dump, table, band, KEYS
from supplement import rows
from tools.first_action_failure import ranking, cluster_bootstrap


def main():
    candidates=rows('candidate_predictions.csv')
    origins={r['origin_id']:r for r in rows('origin_coverage.csv')}
    output=[]
    for origin in sorted(origins):
        context=origins[origin]
        for h in [1,2,4,10]:
            local=sorted([r for r in candidates if r['origin_id']==origin and int(r['horizon'])==h],key=lambda r:int(r['candidate_id'][1:]))
            if not local:continue
            for factor in ['action_magnitude','absolute_baseline_return_gain','episode_crash','episode_out_of_road',
                           'initial_speed','coarse_path_curvature_proxy','traffic_density']:
                labels=[]
                for r in local:
                    labels.append(
                        band(float(r['action_delta_l2']),[.01,.05,.1],['<.01','.01-.05','.05-.1','>=.1']) if factor=='action_magnitude' else
                        band(abs(float(r['actual_gain'])),[1.,5.,20.],['<=1','1-5','5-20','>=20']) if factor=='absolute_baseline_return_gain' else
                        r['crash'] if factor=='episode_crash' else r['out_of_road'] if factor=='episode_out_of_road' else
                        band(float(context['speed']),[3.,8.,15.],['<3','3-8','8-15','>=15']) if factor=='initial_speed' else
                        band(float(context['coarse_curvature_proxy']),[.02,.1],['<.02','.02-.1','>=.1']) if factor=='coarse_path_curvature_proxy' else context['traffic_density'])
                for label in sorted(set(labels)):
                    matched=[i for i,v in enumerate(labels) if v==label]
                    ids=sorted(set([0]+matched))
                    if len(ids)<2:continue
                    score=np.array([float(local[i]['predicted_qb']) for i in ids])
                    actual=np.array([float(local[i]['true_return']) for i in ids])
                    output.append(dict(scene_id=int(context['scene_id']),origin_id=origin,horizon=h,split=context['split'],
                        factor=factor,stratum=label,candidates_in_stratum=len(matched),baseline_in_stratum=0 in matched,
                        reference='Baseline always retained; subgroup Top1 is a descriptive restricted-pool choice, not original online selection',
                        **ranking(score,actual)))
    summaries={}
    for split in ['fit','inner_stop','seen_validation','all_seen']:
        for h,factor,label in sorted({(r['horizon'],r['factor'],r['stratum']) for r in output}):
            selected=[r for r in output if (r['horizon'],r['factor'],r['stratum'])==(h,factor,label) and (split=='all_seen' or r['split']==split)]
            if selected:
                summaries[f'{split}/H{h}/{factor}/{label}']=dict(
                    raw_pairs=sum(r['raw_pairs'] for r in selected),effective_pairs=sum(r['effective_pairs'] for r in selected),
                    tied_pairs=sum(r['tied_pairs'] for r in selected),statistics=cluster_bootstrap(selected,KEYS))
    table('stratified_ranking.csv',output)
    dump('stratified_ranking_statistics.json',dict(summaries=summaries,
        scope='Retrospective only. Baseline included as shared gain reference even if outside subgroup. Outcome strata must never be an online filter.',
        note='Pair-specific difference strata also remain in pair_diagnostics.csv. Curvature is coarse-path proxy; density constant .1.'))
    print('Full stratified ranking metrics:',len(output),'origin/stratum rows',flush=True)


if __name__=='__main__':main()
