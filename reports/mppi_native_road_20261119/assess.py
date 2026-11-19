"""Fixed paired map-level decision, then once-only validation if eligible."""
from study import *
from run_study import run,verify
import pandas as pd
if __name__=='__main__':
    assert (R/'development_complete.json').exists()
    p=load(R/'protocol.json');f=pd.DataFrame([load(x) for x in (R/'development').glob('*/episode.json')]);assert len(f)==48
    f.drop(columns=['initial_hashes']).to_csv(R/'development_episodes.csv',index=False,mode='x')
    keys=['success','crash','out_of_road','route_completion','discounted_return','episode_return','safety_cost','timeout','replacement_rate','reference_tracking_rms']
    f.groupby('configuration')[keys].mean().to_csv(R/'development_summary.csv',mode='x');base=f[f.configuration=='A'].set_index('scene');candidates=[];paired=[];statistics={}
    for tag in ['Ttrack','Tnative','Tconditional']:
        x=f[f.configuration==tag].set_index('scene');d=x[keys]-base[keys];m=d.mean();positive=int((d.discounted_return>0).sum())
        eligible=bool(m.success>=0 and m.crash<=0 and m.out_of_road<=0 and m.route_completion>=-.005 and m.discounted_return>0 and positive>=6)
        candidates.append(dict(configuration=tag,eligible=eligible,positive_return_maps=positive,**m.to_dict()))
        for row in d.reset_index().to_dict('records'):paired.append(dict(configuration=tag,**row))
        statistics[tag]=clustered(d.reset_index(),keys)
    table(R/'development_comparisons.csv',candidates);table(R/'development_paired.csv',paired);dump(R/'development_statistics.json',statistics)
    eligible=[x for x in candidates if x['eligible']]
    selected=sorted(eligible or candidates,key=lambda x:(x['success'],x['discounted_return'],x['positive_return_maps']),reverse=True)[0]['configuration']
    dump(R/'selection.json',dict(selected=selected,allow_holdout=bool(eligible),candidates=candidates,rule=p['selection_rule'],no_holdout_outcomes_viewed=True))
    print('DEVELOPMENT_DECISION',pd.DataFrame(candidates).to_string(index=False),flush=True)
    if eligible and '--decision-only' in sys.argv:
        print('INDEPENDENT_VALIDATION_DEFERRED_FOR_SEQUENTIAL_LATENCY',flush=True)
        raise SystemExit(0)
    if eligible:
        run([('holdout',s,seed,m) for s in p['holdout_scenes'] for seed in p['holdout_seeds'] for m in ['A','C']]);verify(p['holdout_scenes'],['A','C'],'holdout',p['holdout_seeds'])
    dump(R/'study_complete.json',dict(development_episodes=48,holdout_episodes=96 if eligible else 0,independent_validation_used=bool(eligible)))
