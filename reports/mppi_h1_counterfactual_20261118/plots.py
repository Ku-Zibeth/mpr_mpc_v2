"""Export figures from actual H1 branches. Oracle never feeds online selection."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from study import R

def save(fig,name):
    fig.tight_layout();fig.savefig(R/name,dpi=160);plt.close(fig)

def main():
    e=pd.read_csv(R/'counterfactual_episodes.csv');a=pd.read_csv(R/'candidate_actions.csv');o=pd.read_csv(R/'oracle_gain_analysis.csv')
    groups=list(o[['scene','step']].itertuples(index=False,name=None))
    fig,axes=plt.subplots(4,4,figsize=(17,12))
    for ax,(scene,step) in zip(axes.flat,groups):
        q=e[(e.scene==scene)&(e.origin_step==step)]
        ax.bar(q.candidate_id,q.actual_gain,color=['#777777' if cid=='A0' else '#2379b8' if cid=='A1' else '#d07821' if cid=='A2' else '#a5c9da' for cid in q.candidate_id])
        ax.axhline(0,color='black',lw=.7);ax.set_title(f'{scene} / before step {step}');ax.set_ylabel('Actual H1 -> SAC RTG gain')
    save(fig,'origin_candidate_returns.png')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,mode in zip(axes,['C','D']):
        for _,r in o.iterrows():ax.plot([0,1],[r.baseline_return,r[f'{mode}_return']],'-o',alpha=.55,lw=1,ms=3)
        ax.set_xticks([0,1],['SAC first action',f'{mode} selected first action']);ax.set_ylabel('Actual H1 -> SAC discounted return');ax.set_title(f'16 paired origins: {mode} vs Baseline')
    save(fig,'paired_selected_vs_sac.png')
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(o))
    ax.bar(x,o.oracle_gain,color='#26867e');ax.axhline(0,color='black',lw=.8)
    ax.set_xticks(x,[f'{r.scene}\n{r.step}' for r in o.itertuples()],rotation=45);ax.set_ylabel('Finite-pool Oracle Gain');ax.set_title('Oracle is an offline upper bound, not a deployable policy')
    save(fig,'oracle_gain_distribution.png')
    fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(x-.18,o.C_top1_regret,width=.36,label='C');ax.bar(x+.18,o.D_top1_regret,width=.36,label='D')
    ax.set_xticks(x,[f'{r.scene}\n{r.step}' for r in o.itertuples()],rotation=45);ax.set_ylabel('Finite-pool Top-1 Regret');ax.legend()
    save(fig,'top1_regret_distribution.png')
    fig,axes=plt.subplots(4,4,figsize=(15,13))
    for ax,(scene,step) in zip(axes.flat,groups):
        q=a[(a.scene==scene)&(a.step==step)]
        ax.scatter(q.normalized_steering,q.normalized_throttle_brake,s=35)
        for r in q.itertuples():ax.annotate('/'.join(json.loads(r.aliases)),(r.normalized_steering,r.normalized_throttle_brake),xytext=(3,3),textcoords='offset points',fontsize=8)
        ax.set_title(f'{scene} / {step}');ax.set_xlabel('steering [-1,1]');ax.set_ylabel('throttle_brake [-1,1]')
    save(fig,'candidate_first_action_space.png')
    fig,ax=plt.subplots(figsize=(8,6))
    for scene,step in groups:
        q=e[(e.scene==scene)&(e.origin_step==step)];base=q[q.candidate_id=='A0'].iloc[0]
        ax.scatter(q.predicted_common_h10_score-base.predicted_common_h10_score,q.actual_gain,alpha=.6,s=20)
    ax.axhline(0,color='black',lw=.7);ax.axvline(0,color='black',lw=.7)
    ax.set_xlabel('H10 + TD-Q predicted gain (common Q pair)');ax.set_ylabel('Actual H1 -> Lattice+SAC gain')
    ax.set_title('DESCRIPTIVE ONLY: different planning and continuation targets\nNot a formal H1 ranking validation')
    save(fig,'descriptive_h10_score_vs_h1_gain.png')
if __name__=='__main__':main()
