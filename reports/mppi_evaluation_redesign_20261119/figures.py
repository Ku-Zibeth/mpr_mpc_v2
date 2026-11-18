"""Scientific export plots; H1 counterfactuals and full episodes remain separate."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from study import *
def save(fig,name):
    (R/'figures').mkdir(exist_ok=True);fig.tight_layout();fig.savefig(R/'figures'/name,dpi=150);plt.close(fig)
def main():
    h=pd.read_csv(R/'h1_role_results.csv');closed=pd.read_csv(R/'paired_closed_loop_episodes.csv');stats=load(R/'h1_statistics.json')
    fig,ax=plt.subplots(figsize=(8,4));names=['B','C','D','F','G'];mean=[stats['selectors'][m]['gain']['mean'] for m in names]
    intervals=np.array([stats['selectors'][m]['gain']['ci95'] for m in names]).T
    ax.bar(names,mean,yerr=np.vstack([np.array(mean)-intervals[0],intervals[1]-np.array(mean)]),capsize=4)
    oracle=h[h.method=='A'].groupby('scene').oracle_gain.mean().mean();ax.axhline(oracle,color='green',label='Oracle over MEASURED method actions only')
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(len(names)),['Native','Random','Reward-only','Frozen margin','Same-pool RQ']);ax.set_ylabel('Actual H1 -> original SAC gain');ax.legend(fontsize=8)
    save(fig,'native_random_measured_oracle.png')
    fig,ax=plt.subplots(figsize=(9,4));maps=h.groupby(['scene','method']).gain.mean().unstack()
    for method,label in [('D','Reward-only all pool'),('G','RQ matched all pool'),('B','Native final2')]:ax.plot(range(len(maps)),maps[method],'-o',label=label)
    ax.set_xticks(range(len(maps)),maps.index,rotation=45);ax.axhline(0,color='black',lw=.6);ax.set_ylabel('Actual H1 gain');ax.legend()
    save(fig,'reward_only_vs_reward_q.png')
    results=pd.read_csv(R/'new_map_candidate_results.csv');xs=[];ys=[]
    for directory in sorted((R/'origins').iterdir()):
        if not (directory/'catalog.json').exists():continue
        meta=load(directory/'catalog.json');base=next(x for x in meta['representatives'] if x['candidate_id']=='A')
        for row in meta['representatives']:
            rec=results[(results.scene==meta['scene'])&(results.seed==meta['seed'])&(results.candidate_id==row['candidate_id'])].iloc[0]
            xs.append(row['raw_score']-base['raw_score']);ys.append(rec.actual_h1_gain)
    fig,ax=plt.subplots(figsize=(7,5));ax.scatter(xs,ys,s=16,alpha=.6);ax.axhline(0,color='black',lw=.6);ax.axvline(0,color='black',lw=.6)
    ax.set_xlabel('H10 + TD-Q predicted relative score');ax.set_ylabel('Actual H1 -> SAC gain');ax.set_title('DESCRIPTIVE ONLY: mismatched objectives, not formal H1 ranking')
    save(fig,'score_vs_actual_h1_gain.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4));rates=h.groupby('method').physical_replacement.mean();axes[0].bar(rates.index,rates.values);axes[0].set_ylabel('H1 physical replacement rate')
    rates=closed.groupby('configuration').replacement_rate.mean();axes[1].bar(rates.index,rates.values);axes[1].set_title('Full closed-loop replacement\nZero replacement is NOT an algorithm improvement');save(fig,'baseline_replacement_rate.png')
    fig,ax=plt.subplots(figsize=(10,4))
    for method in ['B','C','D','F','G']:ax.plot(range(len(maps)),maps[method],'-o',label=method)
    ax.set_xticks(range(len(maps)),maps.index,rotation=45);ax.axhline(0,color='black',lw=.6);ax.set_ylabel('Actual H1 gain');ax.legend();save(fig,'selected_gain_per_map.png')
    regrets=h.groupby(['scene','method']).regret.mean().unstack();fig,ax=plt.subplots(figsize=(10,4))
    for method in ['B','C','D','F','G']:ax.plot(range(len(regrets)),regrets[method],'-o',label=method)
    ax.set_xticks(range(len(regrets)),regrets.index,rotation=45);ax.set_ylabel('Regret vs measured-method oracle');ax.legend();save(fig,'top1_regret_per_map.png')
    fig,axes=plt.subplots(1,3,figsize=(11,4));means=closed.groupby('configuration')[['success','crash','out_of_road']].mean()
    for ax,key in zip(axes,['success','crash','out_of_road']):ax.bar(means.index,means[key]);ax.set_ylim(0,1);ax.set_title('FULL CLOSED: '+key)
    save(fig,'closed_success_crash_oor.png')
    timing=pd.read_csv(R/'latency.csv');probe=timing[(timing.scope=='isolated_frozen_state')&(timing.stage=='planner_ms')];fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(probe.configuration,probe.p95_ms);ax.axhline(100,color='red',ls='--',label='100ms reference');ax.set_ylabel('Full planner P95 ms, isolated fixed states');ax.legend();save(fig,'planner_latency.png')
    old=pd.read_csv(R/'historical_h1_analysis.csv');fig,ax=plt.subplots(figsize=(7,5));positive=old.actual_h1_gain>0
    ax.scatter(old.steering[~positive],old.throttle_brake[~positive],label='nonpositive real H1 gain',alpha=.5)
    ax.scatter(old.steering[positive],old.throttle_brake[positive],label='positive real H1 gain',alpha=.7)
    ax.set_xlabel('steering');ax.set_ylabel('throttle_brake');ax.set_title('Old123 measured controls; not a learned online region');ax.legend();save(fig,'historical_positive_action_regions.png')
if __name__=='__main__':main()
