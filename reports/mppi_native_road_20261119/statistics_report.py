"""Whole-map bootstrap and paired outcomes; keep exploratory and confirmatory apart."""
from study import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def summarize(split):
    records=[load(x) for x in (R/split).glob('*/episode.json')]
    if not records:return None
    f=pd.DataFrame(records);keys=['success','crash','out_of_road','route_completion','discounted_return','episode_return','safety_cost','timeout','replacement_rate','mean_action_deviation']
    base=f[f.configuration=='A'].set_index(['scene','seed']);stats={};paired=[]
    f.drop(columns=['initial_hashes']).to_csv(R/f'{split}_all_episodes.csv',index=False,mode='x');f.groupby('configuration')[keys].mean().to_csv(R/f'{split}_all_summary.csv',mode='x')
    for tag,group in f.groupby('configuration'):
        if tag=='A':continue
        delta=group.set_index(['scene','seed'])[keys]-base[keys];delta=delta.reset_index();stats[tag]=clustered(delta,keys)
        for row in delta.to_dict('records'):paired.append(dict(configuration=tag,**row))
    table(R/f'{split}_all_paired.csv',paired);dump(R/f'{split}_map_cluster_statistics.json',dict(resampling_unit='whole scene, all its origins/repeats kept together; not episode or time step',bootstrap=2000,seed=20261119,independent_maps=int(f.scene.nunique()),results=stats))
    if split=='holdout':
        diversity=[]
        for scene in load(R/'protocol.json')['holdout_scenes']:
            initials=[]
            for seed in [1,2]:
                a=load(R/split/f'A_{scene}_r{seed}/initial.json')['hashes'];b=load(R/split/f'C_{scene}_r{seed}/initial.json')['hashes'];assert a==b
                initials.append(a)
            diversity.append(dict(scene=scene,traffic_changed_between_repeats=initials[0]['traffic']!=initials[1]['traffic'],vehicle_spawn_unchanged=initials[0]['vehicle']==initials[1]['vehicle']))
        table(R/'independent_traffic_diversity.csv',diversity)
        byseed={int(seed):group.groupby('configuration')[keys].mean().to_dict('index') for seed,group in f.groupby('seed')};dump(R/'holdout_seed_summaries.json',byseed)
    fig,ax=plt.subplots(figsize=(9,4))
    for i,(tag,s) in enumerate(stats.items()):
        q=s['discounted_return'];ax.errorbar(i,q['mean'],yerr=[[q['mean']-q['ci95'][0]],[q['ci95'][1]-q['mean']]],fmt='o',label=tag)
        x=np.array(list(q['maps'].values()));ax.scatter(i+np.linspace(-.12,.12,len(x)),x,s=14,alpha=.5)
    ax.axhline(0,color='grey');ax.set_xticks(range(len(stats)),list(stats));ax.set_ylabel('Actual discounted return gain vs frozen SAC');ax.set_title(f'{split}: map-cluster 95% CI and every map outcome');fig.tight_layout();fig.savefig(R/f'{split}_paired_returns.png',dpi=160);plt.close(fig)
    return f
if __name__=='__main__':
    assert (R/'study_complete.json').exists()
    if '--holdout-only' in sys.argv:
        summarize('holdout')
        raise SystemExit(0)
    for split in ['development','holdout']:
        if not (R/f'{split}_map_cluster_statistics.json').exists():summarize(split)
    rows=[];activity=[]
    for split in ['development','holdout','latency']:
        for method in sorted(set(load(f)['configuration'] for f in (R/split).glob('*/episode.json'))):
            frames=[pd.read_csv(f.parent/'steps.csv') for f in (R/split).glob('*/episode.json') if load(f)['configuration']==method];s=pd.concat(frames,ignore_index=True)
            if method!='A':activity.append(dict(split=split,configuration=method,steps=len(s),search_active_ratio=float(s['mppi/native_search_active'].mean()),native_baseline_risk_positive=float((s['mppi/native_baseline_risk']>0).mean()),physical_action_replacement=float(s.physical_replacement.mean()),speed_mean=float(s.speed.mean()),speed_maximum=float(s.speed.max())))
            for col in [c for c in s if c.endswith('_ms')]:
                v=s[col].to_numpy(float);v=v[np.isfinite(v)]
                if len(v):rows.append(dict(split=split,configuration=method,phase=col,samples=len(v),p50=float(np.quantile(v,.5)),p95=float(np.quantile(v,.95)),p99=float(np.quantile(v,.99)),mean=float(v.mean()),maximum=float(v.max())))
    table(R/'timing_breakdown.csv',rows);table(R/'search_activity.csv',activity)
    print('STATISTICS_DONE',flush=True)
