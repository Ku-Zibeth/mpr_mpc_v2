"""Standalone scientific figures for the retrospective, map-clustered study."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze import R, OLD, load
from supplement import rows
from tools.first_action_failure import cluster_bootstrap


def main():
    figdir = R/'figures'
    figdir.mkdir(exist_ok=False)
    ranking = load(R/'ranking_analysis.json')['summaries']
    p = load(OLD/'protocol.json')
    hs = [1,2,4,10]
    def save(fig,name):
        fig.tight_layout()
        fig.savefig(figdir/(name+'.png'),dpi=180)
        fig.savefig(figdir/(name+'.pdf'))
        plt.close(fig)
    def errorline(ax,values,label,x=hs):
        mean=np.array([v['mean'] for v in values]);ci=np.array([v['ci95'] for v in values])
        ax.errorbar(x,mean,yerr=np.maximum(0,np.array([mean-ci[:,0],ci[:,1]-mean])),marker='o',capsize=4,label=label)
    for metric,name in [('pair_accuracy','horizon_accuracy'),('top1_regret','horizon_regret')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for mode,label in [('qb_original','Frozen H1-trained Q_B'),('original_h10_score','Original H10 score')]:
            values=[ranking[f'all_seen/H{h}/common_pilot/{mode}'][metric] for h in hs]
            errorline(ax,values,label)
        if metric=='pair_accuracy':ax.axhline(.5,color='gray',ls='--')
        ax.set(xlabel='Forced prefix length H, then feedback SAC',ylabel=metric,
               title='Same six pilot maps; 95% map-cluster CI (retrospective)')
        ax.legend();ax.grid(alpha=.2);save(fig,name)
    # Pure state-only swap: same origin/candidate endpoint subset and same terminal action/mask.
    state=[]
    for row in rows('state_oracle_origin_ranking.csv'):
        if int(row['scene_id']) not in p['pilot_scenes'] or row['scope']!='matched_endpoint_subset':continue
        state.append(row|dict(scene_id=int(row['scene_id']),horizon=int(row['horizon']),
                              pair_accuracy=float(row['pair_accuracy']) if row['pair_accuracy'] else None))
    fig,ax=plt.subplots(figsize=(7,4))
    for condition,label in [('J00_predicted','TD Q: WM endpoint'),('J01_actual_latent_same_predicted_mask','TD Q: actual encoder endpoint')]:
        values=[cluster_bootstrap([r for r in state if r['condition']==condition and r['horizon']==h],['pair_accuracy'])['pair_accuracy'] for h in hs]
        errorline(ax,values,label)
    ax.axhline(.5,color='gray',ls='--');ax.legend();ax.grid(alpha=.2)
    ax.set(xlabel='H',ylabel='Pairwise accuracy',title='Terminal Q oracle; not a future input swap for current-state Q_B')
    save(fig,'predicted_vs_actual_state')
    coverage=[r for r in rows('oracle_candidate_analysis.csv') if r['horizon']=='1']
    scenes=sorted({int(r['scene_id']) for r in coverage})
    oracle,model=[],[]
    for scene in scenes:
        local=[r for r in coverage if int(r['scene_id'])==scene]
        oracle.append(np.mean([float(r['oracle_gain']) for r in local]))
        model.append(np.mean([float(r['model_gain']) for r in local]))
    fig,ax=plt.subplots(figsize=(10,4));xx=np.arange(len(scenes))
    ax.bar(xx-.18,oracle,width=.36,label='Measured-subset oracle gain')
    ax.bar(xx+.18,model,width=.36,label='Frozen Q_B selected gain')
    ax.axhline(0,color='black',lw=.8);ax.legend()
    ax.set(xticks=xx,xticklabels=scenes,ylabel='Actual H1 return gain',title='Per-map paired gain, H1; oracle is not deployable')
    ax.tick_params(axis='x',rotation=60);save(fig,'oracle_vs_model_gain')
    scene_rows=[r for r in rows('scene_level_ranking.csv') if r['horizon']=='1' and r['scorer']=='qb_original']
    fig,ax=plt.subplots(figsize=(10,4));colors={'fit':'#2878B5','inner_stop':'#F8AC8C','seen_validation':'#C82423'}
    ax.bar(np.arange(len(scene_rows)),[float(r['pair_accuracy']) for r in scene_rows],color=[colors[r['split']] for r in scene_rows])
    ax.axhline(.5,color='gray',ls='--');ax.set(xticks=np.arange(len(scene_rows)),xticklabels=[r['scene_id'] for r in scene_rows],
        ylabel='Pairwise accuracy',title='H1 by map: fit (blue), inner stop (orange), seen validation (red)')
    ax.tick_params(axis='x',rotation=60);save(fig,'scene_accuracy')
    errors=rows('dynamics_pair_errors.csv');fig,ax=plt.subplots(figsize=(7,4))
    for h,marker in [(1,'o'),(10,'s')]:
        selected=[r for r in errors if int(r['horizon'])==h]
        points=[]
        for origin in sorted({r['origin_id'] for r in selected}):
            local=[r for r in selected if r['origin_id']==origin]
            points.append((np.mean([float(r['relative_dynamics_error_l2']) for r in local]),np.mean([float(r['ranking_error_credit']) for r in local])))
        a=np.asarray(points);ax.scatter(a[:,0],a[:,1],label=f'H={h}, origin means',marker=marker,alpha=.7)
    ax.set(xlabel='Relative latent dynamics error L2 (not meters)',ylabel='Terminal-Q ranking error rate',
        title='Descriptive origin points; statistical CI clusters whole maps')
    ax.legend();ax.grid(alpha=.2);save(fig,'dynamics_vs_ranking_error')
    print('Six PNG/PDF figure pairs saved.',flush=True)


if __name__=='__main__':main()
