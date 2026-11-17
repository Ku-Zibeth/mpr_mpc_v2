"""Read-only cached-data attribution. Does not instantiate or step MetaDrive."""
import contextlib
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
R = Path(__file__).parent
S = ROOT/'reports/mppi_selection_20261117'
D = ROOT/'reports/mppi_root_cause_20261117'
import numpy as np
import torch
from scipy.stats import spearmanr
from omegaconf import OmegaConf
from tools.mppi_failure_attribution import ranking_metrics, scene_bootstrap, distribution_from_indices, replanning_error
from tools.benchmark_mppi import source_fingerprint
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.parser import parse_cfg
from common.world_model import WorldModel
from common import math as td_math
from planning.evaluator import TrajectoryWorldModelEvaluator
from planning.local_mppi import LocalMPPI


def load(path): return json.loads(Path(path).read_text())
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): digest.update(block)
    return digest.hexdigest()
def dump(name, value): (R/name).write_text(json.dumps(value, indent=2, allow_nan=False))
def table(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (R/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
def clean(value):
    if isinstance(value, dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.floating,float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    return value
def metrics(p, y): return ranking_metrics(p,y)
def stat(rows, keys): return scene_bootstrap(rows, keys)


def manifest():
    if (R/'experiment_manifest.json').exists():
        assert not (R/'analysis_complete.json').exists(), 'Use a new output directory; never overwrite a completed analysis.'
        return load(R/'experiment_manifest.json')
    p = load(S/'protocol.json')
    ck = Path(p['checkpoint']); assert sha(ck) == p['checkpoint_sha256']
    core = ['planning/local_mppi.py','planning/evaluator.py','planning/coordinator.py','planning/types.py','planning/native_config.py']
    patch = subprocess.run(['git','apply','--reverse','--check','--include=planning/*',str(S/'code_changes.patch')],cwd=ROOT,capture_output=True,text=True)
    assert patch.returncode == 0, patch.stderr
    episodes=[]
    for f in sorted((S/'episodes/development').glob('*/episode.json')):
        row=load(f); cfg=f.parent/'config.yaml'; raw=OmegaConf.load(cfg)
        assert Path(raw.checkpoint) == ck
        episodes.append(dict(configuration=row['configuration'],scenario=row['scenario'],seed=row['test_seed'],
            config_path=str(cfg.relative_to(ROOT)),config_sha256=sha(cfg),
            recorded_code_base=p['base_commit'],recorded_source_fingerprint=p['source_snapshot_sha256'],
            checkpoint_sha256=p['checkpoint_sha256'],options=row['options']))
    assert len(episodes)==56
    tracked_delta=subprocess.check_output(['git','diff','HEAD','--',*core],cwd=ROOT,text=True); assert not tracked_delta
    missing=[
        'Real SAC-continuation Return exists for only executed candidate subsets; no labels for all65x4 candidates or all online ticks. Coverage/oracle/regret are measured-subset lower bounds.',
        '56 closed-loop traces lack per-stage planning times, vehicle XY/yaw/speed and full planner contexts. Cannot reconstruct original full phase breakdown or physical WM trajectory from latent distances.',
        'Saved clipping flags/pre-clamp Gaussian innovations are absent. Boundary saturation is observable; exact clipping fraction is not.',
        'No WM physical decoder. Existing bicycle predictions are a separate physical model, never WM XY predictions.',
        'Fixed/ensemble Q offline rescores use same cached candidate pools; they are not new closed-loop runs and cannot establish causal driving improvement.',
        'All historical validation results already seen are retrospective only; new six-map sealed validation remains unopened.'
    ]
    out=dict(current_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        current_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        log5=subprocess.check_output(['git','log','-5','--oneline'],cwd=ROOT,text=True).splitlines(),
        status_at_start=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True).splitlines(),
        historical_56_base_commit=p['base_commit'],historical_source_fingerprint=p['source_snapshot_sha256'],
        source_fingerprint_now=source_fingerprint(ROOT),
        version_note='56 episodes ran ced7688 plus recorded uncommitted patch. Current45073e1 incorporates that patch; runtime fingerprint before this task was identical bdae7742...; reverse-check succeeds. Commit IDs differ, core code is identical.',
        old_patch_sha256=sha(S/'code_changes.patch'),old_patch_reverse_check=True,
        core_sha256={f:sha(ROOT/f) for f in core},checkpoint=str(ck),checkpoint_sha256=sha(ck),
        online_elite_selection_present=True,online_elite_evidence='local_mppi selection_mode=best_final_elite compares cached last-roundTop1 with Baseline using common final Q members; trace selected_source=2 is actual executed action.',
        episodes=episodes,required_reports={f:(ROOT/f).exists() for f in [
            'MPPI_CLOSED_LOOP_SELECTION_REPORT_20261117.md','reports/mppi_selection_20261117/paired_returns.csv',
            'MPPI_RELATIVE_RANKING_REPORT_20261117.md','SAC_CONTINUATION_VALUE_REPORT.md']},
        protocol=dict(gamma=.995,horizon=10,return_tolerance=1.,score_tie_credit=.5,actual_ties='exclude abs(deltaReturn)<=1 pairs; retain for coverage equal_fraction',
            top1_tie='original candidate order',bootstrap='2000 draws seed1; aggregate within scene before scene bootstrap; paired differences inside origin',
            primary_q_pair=[0,1],sensitivity='all10pairs and five-member mean, NO selecting a best pair on observed labels',
            new_environment_runs=0,new_training=0,online_defaults_changed=False),missing_data=missing)
    dump('experiment_manifest.json',out)
    return out


def model_runtime(m):
    torch.set_num_threads(1); torch.set_float32_matmul_precision('high'); torch.manual_seed(1)
    cfg=parse_cfg(OmegaConf.load(ROOT/m['episodes'][0]['config_path']))
    cfg.obs_shape={'state':(259,)}; cfg.action_dim=2; cfg.episode_length=1000
    model=WorldModel(cfg).cuda().eval().requires_grad_(False)
    state=torch.load(m['checkpoint'],map_location='cpu',weights_only=False)['model']
    agent=SimpleNamespace(model=model,cfg=cfg,device=torch.device('cuda'),discount=.995)
    # Identical read-only legacy validation-buffer conversion to the real runtime.
    from tdmpc2 import TDMPC2
    TDMPC2.load(agent, {'model':state})
    return agent,TrajectoryWorldModelEvaluator(agent,10)


@torch.no_grad()
def score_sequences(agent, evaluator, z0, actions):
    a=torch.as_tensor(actions,device='cuda',dtype=torch.float32)
    z=torch.as_tensor(z0,device='cuda',dtype=torch.float32).reshape(1,-1)
    result=evaluator.evaluate_mppi_candidates(z,a,compute_cost=False,q_aggregation='ensemble_mean',capture_ensemble=True)
    # Use exact requested unmasked gamma**H oracle definition. Record native WM termination separately.
    zp=z.expand(len(a),-1); latents=[zp.cpu().numpy()]; rewards=[]
    for t in range(10):
        rewards.append(td_math.two_hot_inv(agent.model.reward(zp,a[:,t],None),agent.cfg).squeeze(-1))
        zp=agent.model.next(zp,a[:,t],None);latents.append(zp.cpu().numpy())
    rr=torch.stack(rewards,1).cpu().numpy()
    return dict(prefix=rr@(.995**np.arange(10)),rewards=rr,
        q=result.q_member_values.cpu().numpy().T,latents=np.stack(latents,1),
        native_prefix=result.discounted_reward.cpu().numpy(),native_discount=result.terminal_discount.cpu().numpy())


def q_scores(q, prefix):
    return {''.join(map(str,pair)):prefix+.995**10*q[:,pair].mean(1) for pair in itertools.combinations(range(5),2)} | {'ensemble':prefix+.995**10*q.mean(1)}


@torch.no_grad()
def counterfactuals(agent,evaluator):
    candidates=[];errors=[];groups=[];qrows=[];qeffects=[];saved=[]
    # These197 exact replay branches contain real latent sequences; no extra environment calls.
    for folder in sorted((D/'states').iterdir()):
        files=sorted(folder.glob('*.npz'))
        if not files: continue
        paths=sorted(files,key=lambda f:(f.stem not in ['B','baseline'],f.name))
        scene,before=map(int,folder.name.split('_')); rows=[];qp=[];qr=[];gp=[];ga=[];ys=[]
        for f in paths:
            data=np.load(f);meta=load(f.with_suffix('.json'));source=np.load(str(data['source']))
            actual=float(data['actual_return']); prefix=float(data['actual_prefix']);h=min(10,len(data['actual'])-1)
            row=dict(cohort='fit_innerstop',scenario=scene,origin=folder.name,protocol='H10_SAC',candidate=f.stem,
                actual_return=actual,actual_prefix=prefix,actual_continuation=float(data['actual_continuation']) if h==10 else None,
                actual_prefix_steps=h,full_horizon=h==10,source=str(f.relative_to(ROOT)),success=meta['end'].get('success'),
                crash=meta['end'].get('crash'),out_of_road=meta['end'].get('out_of_road'),route_completion=meta['end'].get('route_completion'))
            z=torch.tensor(data['predicted'][:h],device='cuda');a=torch.tensor(data['commands'][:h],device='cuda')
            pr=td_math.two_hot_inv(agent.model.reward(z,a,None),agent.cfg).reshape(-1).cpu().numpy()
            row['predicted_prefix']=float(pr@(.995**np.arange(h)))
            for t in range(h+1):
                dp=data['predicted'][t];real=data['actual'][t];den=np.linalg.norm(dp)*np.linalg.norm(real)
                physical=data['actual_physical'][t];kinxy=data['kinematic_xy'][t]
                errors.append(dict(scenario=scene,origin=folder.name,candidate=f.stem,step=t,
                    latent_l2=float(np.linalg.norm(dp-real)),latent_cosine=float(dp@real/den) if den else None,
                    reward_predicted=float(pr[t]) if t<h else None,reward_actual=float(source['rewards'][t]) if t<h else None,
                    reward_error=float(pr[t]-source['rewards'][t]) if t<h else None,
                    bicycle_xy_squared_error=float(np.sum((kinxy-physical[:2])**2)),
                    bicycle_yaw_error=float(abs(np.arctan2(np.sin(data['kinematic_yaw'][t]-physical[2]),np.cos(data['kinematic_yaw'][t]-physical[2])))),
                    bicycle_speed_error=float(abs(data['kinematic_speed'][t]-physical[3])),
                    physical_prediction_source='existing bicycle, NOT WorldModel output'))
            if h==10:
                qpred=data['q_pred_own'][3];qreal=data['q_real_own'][3]
                row.update(predicted_terminal_q=float(qpred[:2].mean()),real_latent_terminal_q=float(qreal[:2].mean()))
                qp.append(qpred);qr.append(qreal);gp.append(row['predicted_prefix']);ga.append(prefix);ys.append(actual);rows.append(row)
            candidates.append(row)
        if rows:
            assert rows[0]['candidate'] in ['B','baseline']
            qpred=np.asarray(qp);qreal=np.asarray(qr);pred=np.asarray(gp);act=np.asarray(ga);y=np.asarray(ys)
            scores={'J00':pred+.995**10*qpred[:,:2].mean(1),'J10':act+.995**10*qpred[:,:2].mean(1),
                    'J01':pred+.995**10*qreal[:,:2].mean(1),'J11':act+.995**10*qreal[:,:2].mean(1),
                    'prefix_reward':pred,'terminal_q_vs_continuation':qpred[:,:2].mean(1)}
            for name,v in scores.items():
                target=np.array([r['actual_continuation'] for r in rows]) if name=='terminal_q_vs_continuation' else act if name=='prefix_reward' else y
                groups.append(dict(cohort='fit_innerstop',scenario=scene,origin=folder.name,score=name,**metrics(v,target)))
            for i,row in enumerate(rows):
                row.update({k:float(v[i]) for k,v in scores.items() if k.startswith('J')})
                row.update(actual_gain=float(y[i]-y[0]),predicted_gain=float(scores['J00'][i]-scores['J00'][0]))
            pool=q_scores(qpred,pred)
            for label,v in pool.items():qrows.append(dict(cohort='fit_innerstop',scenario=scene,origin=folder.name,q_reducer=label,**metrics(v,y)))
            reference=metrics(pool['01'],y)
            for label,v in pool.items():
                r=metrics(v,y);qeffects.append(dict(scenario=scene,origin=folder.name,q_reducer=label,
                    pair_accuracy_change=None if r['pair_accuracy'] is None or reference['pair_accuracy'] is None else r['pair_accuracy']-reference['pair_accuracy'],
                    regret_change=r['top1_regret']-reference['top1_regret']))
            for i,row in enumerate(rows):
                candidatescores=np.array([v[i]-v[0] for v in pool.values()])
                row['q_ensemble_std']=float(qpred[i].std());row['q_gain_std_across_pairs']=float(candidatescores[:-1].std())
            saved.append(dict(scenario=scene,origin=folder.name,actions=None,actual=y,qp=qpred,qr=qreal,gp=pred,ga=act))
    # Four original development origins, all fixed per-round Top1/Means plus random/B, both interventions.
    protocol=load(D/'protocol.json')
    for scene,before in protocol['development_origins']:
        folder=D/'catalog'/f'{scene}_{before}'; c=np.load(folder/'catalog.npz');labels=protocol['choices']
        scored=score_sequences(agent,evaluator,c['z0'],c['actions']);qm=scored['q']
        for intervention in ['A','B']:
            ys=[]; rr=[]
            for i,label in enumerate(labels):
                f=D/'interventions'/folder.name/f'{intervention}_{label}.npz';d=np.load(f);meta=load(f.with_suffix('.json'));ys.append(float(d['rtg'][0]))
                row=dict(cohort='fixed_rounds',scenario=scene,origin=folder.name,protocol='H10_SAC' if intervention=='A' else 'first_action_SAC',
                    candidate=label,actual_return=ys[-1],actual_prefix=float(d['rewards'][:10]@(.995**np.arange(min(10,len(d['rewards']))))),
                    actual_continuation=float(d['rtg'][10]) if len(d['rtg'])>10 else None,
                    predicted_prefix=float(scored['prefix'][i]),predicted_terminal_q=float(qm[i,:2].mean()),
                    J00=float(scored['prefix'][i]+.995**10*qm[i,:2].mean()),
                    predicted_original_common=float(c['predicted_values'][i]),actual_prefix_steps=min(10,len(d['rewards'])),
                    source=str(f.relative_to(ROOT)),**{k:meta[k] for k in ['success','crash','out_of_road','route_completion']})
                candidates.append(row);rr.append(row)
            y=np.asarray(ys); p=c['predicted_values'];pool=q_scores(qm,scored['prefix'])
            for i,row in enumerate(rr):row.update(actual_gain=float(y[i]-y[0]),predicted_gain=float(p[i]-p[0]))
            groups.append(dict(cohort='fixed_rounds_'+intervention,scenario=scene,origin=folder.name,score='original_common_pair',**metrics(p,y)))
            for label,v in pool.items():qrows.append(dict(cohort='fixed_rounds_'+intervention,scenario=scene,origin=folder.name,q_reducer=label,**metrics(v,y)))
    table('candidate_returns.csv',candidates);table('rollout_prediction_error.csv',errors)
    table('candidate_ranking.csv',groups);table('q_candidate_ranking.csv',qrows);table('q_paired_effects.csv',qeffects)
    keys=['pair_accuracy','spearman','kendall','top1_regret','selected_gain','oracle_best_gain','positive_fraction','negative_fraction','equal_fraction','baseline_accuracy','false_improvement_rate']
    summary={}
    for cohort in sorted({r['cohort'] for r in groups}):
        summary[cohort]={name:stat([r for r in groups if r['cohort']==cohort and r['score']==name],keys) for name in sorted({r['score'] for r in groups if r['cohort']==cohort})}
    oracle=[]
    for scene in sorted({r['scenario'] for r in groups if r['cohort']=='fit_innerstop'}):
        origins=sorted({r['origin'] for r in groups if r['scenario']==scene and r['cohort']=='fit_innerstop'})
        for origin in origins:
            a={r['score']:r for r in groups if r['origin']==origin and r['cohort']=='fit_innerstop'}
            for name in ['J10','J01','J11']:
                oracle.append(dict(scenario=scene,origin=origin,replacement=name,
                    pair_accuracy_change=None if a[name]['pair_accuracy'] is None or a['J00']['pair_accuracy'] is None else a[name]['pair_accuracy']-a['J00']['pair_accuracy'],
                    regret_change=a[name]['top1_regret']-a['J00']['top1_regret']))
    table('oracle_paired_effects.csv',oracle)
    out=dict(ranking=summary,oracle_paired={n:stat([r for r in oracle if r['replacement']==n],['pair_accuracy_change','regret_change']) for n in ['J10','J01','J11']},
        q_fixed_vs_ensemble=stat([r for r in qeffects if r['q_reducer']=='ensemble'],['pair_accuracy_change','regret_change']),
        counts=dict(real_branches=len(candidates),state_branches=197,oracle_origins=len(saved),fixed_origins=4),
        note='J00/J10/J01/J11 use gamma**10, fixedQ01 and unchanged candidate a_H. Native early terminals retained in coverage but excluded when real z_H unavailable. B first-action outcomes are NEVER H10 rollout oracle labels. Real continuation is SAC RTG; Q(z_real,a_H) remains an unexecuted-action mismatch, not a value identity.')
    dump('counterfactual_statistics.json',clean(out));print('counterfactual analysis complete',flush=True)
    return out


@torch.no_grad()
def q_stability(agent,evaluator):
    rows=[];meanrows=[];pairs=list(itertools.combinations(range(5),2));qindices=[]
    protocol=load(D/'protocol.json')
    for scene,before in protocol['development_origins']:
        folder=D/'catalog'/f'{scene}_{before}';c=np.load(folder/'catalog.npz');meta=load(folder/'catalog.json')
        audit=np.load(ROOT/'reports/candidate_ranking_20261117/data'/folder.name/'audit.npz')
        baseline=c['actions'][0]
        for k,actions in enumerate(audit['iteration_actions']):
            pool=np.concatenate([baseline[None],actions]); scored=score_sequences(agent,evaluator,c['z0'],pool)
            scores=q_scores(scored['q'],scored['prefix']);selections={};means={}
            for name,v in scores.items():
                idx=torch.topk(torch.tensor(v),8).indices.numpy(); w=np.exp(.5*(v[idx]-v[idx].max()));w/=w.sum();mean=np.sum(w[:,None,None]*pool[idx],0)
                selections[name]=idx;means[name]=mean
                meanrows.append(dict(scenario=scene,origin=folder.name,k=k+1,q_reducer=name,
                    first_steer=float(mean[0,0]),first_throttle=float(mean[0,1]),mean_delta_from_baseline=float(np.sqrt(np.mean((mean-baseline)**2))),
                    top8=' '.join(map(str,idx.tolist())),actual_q_pair=json.dumps(meta['raw_rounds'][k]['raw_pair'])))
            labels=list(scores)
            for i,one in enumerate(labels):
                for two in labels[i+1:]:
                    inter=len(set(selections[one])&set(selections[two]));rows.append(dict(scenario=scene,origin=folder.name,k=k+1,
                        reducer1=one,reducer2=two,ranking_spearman=float(spearmanr(scores[one],scores[two]).statistic),
                        top8_jaccard=inter/(16-inter),mean_sequence_rmse=float(np.sqrt(np.mean((means[one]-means[two])**2))),
                        first_action_delta=float(np.linalg.norm(means[one][0]-means[two][0])),
                        q_std_mean=float(scored['q'].std(1).mean()),
                        relative_gain_std_mean=float(np.std(np.stack([scores[n]-scores[n][0] for n in labels[:-1]]),axis=0).mean())))
            qindices.append(dict(scenario=scene,origin=folder.name,k=k+1,pair=meta['raw_rounds'][k]['raw_pair']))
    table('q_ensemble_rank_stability.csv',rows);table('q_ensemble_means.csv',meanrows)
    dump('q_stability_statistics.json',dict(summary=stat(rows,['ranking_spearman','top8_jaccard','mean_sequence_rmse','first_action_delta','q_std_mean','relative_gain_std_mean']),
        actual_Q_indices=qindices,scope='All10 pairs plus ensemble, same four cached65x4 pools. Conditional one-update Mean, NOT rerunning search under another Q. No actual returns for unexecuted pools.'))
    print('Q stability analysis complete',flush=True)


@torch.no_grad()
def closed_loop_search(agent,evaluator):
    distributions=[];execution=[];commitments=[];qrecord=[];origins=[];tie_checks=[]
    for f in sorted((S/'episodes/development').glob('*/episode.json')):
        ep=load(f);folder=f.parent
        if not ep['options']['mppi']:continue
        with np.load(folder/'trace.npz') as data:
            aa=data['iteration_actions']; vv=data['iteration_values']; actual=data['actions'];selected=data['selected_source'];fa=data['final_candidates'];fq=data['iteration_q_members'];active=10 if ep['options']['terminal_action']=='policy_prior' else 11
            config=OmegaConf.load(folder/'config.yaml').mpr_mpc.mppi
            baseline=aa[:,0,0];previous=baseline.copy();prevstd=np.broadcast_to(config.initial_std,previous.shape).copy();prevtop=None
            # Original plan uses ONE contiguous GPU vector per call. Batched CPU topk
            # has different tie ordering; replay the actual CUDA operation, not a new ranking.
            alltop=np.asarray([[torch.topk(torch.tensor(vv[t,k],device='cuda'),8).indices.cpu().numpy()
                                for k in range(vv.shape[1])] for t in range(len(vv))])
            for k in range(aa.shape[1]):
                idx=alltop[:,k];v=np.take_along_axis(vv[:,k],idx,1);w=np.exp(.5*(v-v.max(1,keepdims=True)));w/=w.sum(1,keepdims=True)
                elites=np.take_along_axis(aa[:,k],idx[:,:,None,None],1)
                mean=(w[:,:,None,None]*elites).sum(1);std=np.clip(np.sqrt((w[:,:,None,None]*(elites-mean[:,None])**2).sum(1)),config.min_std,config.max_std)
                if active==10:mean[:,-1]=baseline[:,-1];std[:,-1]=0
                for t in range(len(actual)):
                    _,_,_,row=distribution_from_indices(aa[t,k],vv[t,k],baseline[t],previous[t],config.min_std,config.max_std,idx[t],active=active)
                    row.update(scenario=ep['scenario'],configuration=ep['configuration'],seed=ep['test_seed'],step=t,k=k+1,
                        q_members=' '.join(map(str,fq[t,k])),std_change=float(np.sqrt(np.mean((std[t,:active]-prevstd[t,:active])**2))),
                        top8_index_jaccard_previous=None if k==0 else len(set(idx[t])&set(prevtop[t]))/len(set(idx[t])|set(prevtop[t])))
                    # Indices across rounds refer to fresh draws, NOT candidate identity.
                    distributions.append(row)
                previous,prevstd,prevtop=mean,std,idx
            maximum=float(np.abs(mean-fa[:,1]).max());assert maximum<2e-6,(folder.name,maximum)
            tie_checks.append(dict(episode=folder.name,max_reconstruction_error=maximum,
                cutoff_ties=int(sum(np.count_nonzero(vv[t,k]==vv[t,k,alltop[t,k,-1]])>1 for t in range(len(vv)) for k in range(vv.shape[1])))))
            for t in range(len(actual)):
                qrecord.append(dict(scenario=ep['scenario'],configuration=ep['configuration'],seed=ep['test_seed'],step=t,
                    iteration_members=json.dumps(fq[t].tolist()),final_members=json.dumps(data['final_q_members'][t].tolist())))
            correction=actual-baseline[:,0]
            execution.append(dict(scenario=ep['scenario'],configuration=ep['configuration'],seed=ep['test_seed'],steps=len(actual),
                correction_l2_mean=float(np.linalg.norm(correction,axis=1).mean()),steer_abs_change=float(np.abs(correction[:,0]).mean()),
                throttle_abs_change=float(np.abs(correction[:,1]).mean()),selected_elite_fraction=float(np.mean(selected==2)),
                selected_mean_fraction=float(np.mean(selected==1)),baseline_fraction=float(np.mean(selected==0)),
                predicted_selected_gain=float(data['selected_gain'].mean())))
            for h in [1,3,5,9]:
                err=replanning_error(fa,selected,actual,h)
                commitments.append(dict(scenario=ep['scenario'],configuration=ep['configuration'],seed=ep['test_seed'],h=h,
                    first_plan_vs_executed_rmse=float(np.sqrt(np.mean(err**2))),
                    steer_abs_error=float(np.abs(err[:,0]).mean()),throttle_abs_error=float(np.abs(err[:,1]).mean()),samples=len(err)))
            # Reconstruct physical-free WM sequence for THE FIRST saved plan only; all real observations are already cached.
            obs=torch.tensor(data['observations'][:11],device='cuda');realz=agent.model.encode(obs,None).cpu().numpy()
            action=fa[0,selected[0]];pred=score_sequences(agent,evaluator,realz[0],action[None])
            for h in [1,3,5,10]:
                origins.append(dict(scenario=ep['scenario'],configuration=ep['configuration'],seed=ep['test_seed'],h=h,
                    closed_loop_latent_l2=float(np.linalg.norm(pred['latents'][0,h]-realz[h])),
                    reward_error=float(pred['rewards'][0,h-1]-data['rewards'][h-1]),
                    note='Sequence uses original first plan; h>1 includes BOTH dynamics error and changed future controls. Not isolated WM dynamics error.'))
        print('trace',folder.name,'analyzed',flush=True)
    table('search_distribution_diagnostics.csv',distributions);table('actual_action_changes.csv',execution)
    dump('distribution_reconstruction_verification.json',dict(rows=tie_checks,
        explanation='Use original per-vector CUDA topk. CPU/batched tie ordering could substitute different equal-score candidates; that was an offline diagnostic mismatch, NOT an online planner bug.'))
    table('replanning_consistency.csv',commitments);table('closed_loop_prediction_error.csv',origins);table('actual_q_member_indices.csv',qrecord)
    keys=['elite_value_gap','weight_entropy','effective_sample_size','baseline_in_elite','first_action_delta','sequence_delta','steer_std','throttle_std','boundary_saturation_fraction','elite_pairwise_distance','mean_change','std_change']
    summary={g:{str(k):stat([r for r in distributions if r['configuration']==g and r['k']==k],keys) for k in sorted({r['k'] for r in distributions if r['configuration']==g})} for g in sorted({r['configuration'] for r in distributions})}
    dump('search_statistics.json',dict(by_configuration_iteration=summary,
        action_changes={g:stat([r for r in execution if r['configuration']==g],['correction_l2_mean','steer_abs_change','throttle_abs_change','baseline_fraction','selected_elite_fraction']) for g in summary},
        replanning={g:{str(h):stat([r for r in commitments if r['configuration']==g and r['h']==h],['first_plan_vs_executed_rmse','steer_abs_error','throttle_abs_error']) for h in [1,3,5,9]} for g in summary},
        top8_note='Cross-round top8 index overlap is NOT identity Jaccard (fresh Gaussian draws); only same-pool Q stability Jaccard compares actual candidate identity.',
        saturation_note='Observed action at physical boundary, not exact pre-clamp clipping count.'))


@torch.no_grad()
def timings(agent,evaluator):
    rows=[];old=load(S/'isolated_latency.json')
    for g in sorted({r['configuration'] for r in old['rows']}):
        v=np.array([r['planning_time_ms'] for r in old['rows'] if r['configuration']==g])
        rows.append(dict(configuration=g,phase='total_planning',source='previous isolated synchronized complete agent.act; same runtime code',
            samples=len(v),p50_ms=float(np.quantile(v,.5)),p95_ms=float(np.quantile(v,.95)),p99_ms=float(np.quantile(v,.99)),mean_ms=float(v.mean())))
    # CPU/CUDA phase timings on exactly cached arrays, no live scene/context reconstruction.
    c=np.load(next((D/'catalog').glob('*/catalog.npz')));op=OmegaConf.to_container(OmegaConf.load(ROOT/load(S/'protocol.json')['source_environment_config']).mpr_mpc.mppi)
    # Current defaults, plus explicit identical candidate/mean instrumentation controls.
    op=OmegaConf.to_container(OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi);op.update(iterations=4,selection_mode='weighted_mean',record_candidates=False,audit_candidates=False)
    planner=LocalMPPI(evaluator,op,horizon=10,action_dim=2,device='cuda');z=torch.tensor(c['z0'],device='cuda');baseline=torch.tensor(c['actions'][0],device='cuda')
    samples={};original_q=evaluator._terminal_prediction
    def measure(key,fn):
        torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize();samples.setdefault(key,[]).append(1000*(time.perf_counter()-t));return out
    def terminal(*args,**kwargs):return measure('q_evaluation',lambda:original_q(*args,**kwargs))
    # These probes instrument only an isolated offline planner instance. RNG/results are checked separately.
    original_sample=planner._candidate_batch;original_update=planner._update_distribution;original_eval=evaluator.evaluate_mppi_candidates
    evaluator._terminal_prediction=terminal
    planner._candidate_batch=lambda *a,**kw:measure('mppi_sampling',lambda:original_sample(*a,**kw))
    planner._update_distribution=lambda *a,**kw:measure('elite_update',lambda:original_update(*a,**kw))
    evaluator.evaluate_mppi_candidates=lambda *a,**kw:measure('candidate_evaluation_including_q',lambda:original_eval(*a,**kw))
    for i in range(35):
        if i==5:samples={}
        measure('local_mppi_total',lambda:planner.plan(z,baseline,eval_mode=True))
    evaluator._terminal_prediction=original_q;evaluator.evaluate_mppi_candidates=original_eval
    for key,v in samples.items():
        rows.append(dict(configuration='M4_cached',phase=key,source='new synchronized instrumented offline cached z/actions; instrumentation overhead; nested phases not additive',
            samples=len(v),p50_ms=float(np.quantile(v,.5)),p95_ms=float(np.quantile(v,.95)),p99_ms=float(np.quantile(v,.99)),mean_ms=float(np.mean(v))))
    for key in ['lattice_generation','sac_inference','trajectory_adaptation','final_selection_exclusive']:
        rows.append(dict(configuration='historical56',phase=key,source='NOT SAVED. Cached obs lacks full planner/traffic geometry; no fabricated attribution.',samples=0,p50_ms=None,p95_ms=None,p99_ms=None,mean_ms=None))
    table('timing_breakdown.csv',rows);dump('timing_statistics.json',dict(gpu=torch.cuda.get_device_name(),rows=rows,
        warning='Historical full planning and new instrumented cached local planner are different scopes/states. Nested times cannot sum; no new MetaDrive calls.'))


def main():
    m=manifest();print('version/checkpoint/config manifest verified',flush=True)
    agent,evaluator=model_runtime(m)
    with torch.no_grad():
        if not (R/'counterfactual_statistics.json').exists():counterfactuals(agent,evaluator)
        if not (R/'q_stability_statistics.json').exists():q_stability(agent,evaluator)
        if not (R/'search_statistics.json').exists():closed_loop_search(agent,evaluator)
        if not (R/'timing_statistics.json').exists():timings(agent,evaluator)
    assert sha(m['checkpoint'])==m['checkpoint_sha256']
    for f,h in m['core_sha256'].items():assert sha(ROOT/f)==h
    dump('analysis_complete.json',dict(environment_runs=0,training=0,core_unchanged=True,checkpoint_unchanged=True))

if __name__=='__main__':main()
