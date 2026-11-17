"""Existing frozen critics on true terminal535; all real labels are native SAC RTG."""
import csv,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from scipy.stats import kendalltau,spearmanr
from omegaconf import OmegaConf
from tools.mppi_failure_attribution import ranking_metrics,scene_bootstrap
from tools.policy_value_diagnosis import discounted_return,calibration,residual_critic_readout
from planning.sac_value import state_sha256
from residual_rl.networks import GaussianActor,Critic

def load(f):return json.loads(Path(f).read_text())
def dump(name,x):(R/name).write_text(json.dumps(x,indent=2,allow_nan=False))
def table(name,rows):
    with (R/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def runtime():
    spec=importlib.util.spec_from_file_location('frozen_attribution_runtime',ROOT/'reports/mppi_failure_attribution_20261117/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    agent,evaluator=module.model_runtime(load(ROOT/'reports/mppi_failure_attribution_20261117/experiment_manifest.json'))
    p=load(R/'protocol.json');ck=torch.load(p['checkpoint'],map_location='cpu',weights_only=False);saved=ck['mpr_mpc']['residual_sac'];cfg=ck['metadata']['residual_config']
    actor=GaussianActor(535,2,cfg['hidden_dims'],log_std_min=cfg['log_std_min'],log_std_max=cfg['log_std_max']).cuda();actor.load_state_dict(saved['actor']);actor.eval().requires_grad_(False)
    critics={}
    for n in ['q1','q2','target_q1','target_q2']:
        m=Critic(535,2,cfg['hidden_dims']).cuda();m.load_state_dict(saved[n]);m.eval().requires_grad_(False);critics[n]=m
    return p,agent,actor,critics,ck


@torch.no_grad()
def values(agent,actor,critics,state,pred,physical,old_candidate_action=None):
    from common import math as td_math
    s=torch.as_tensor(state,dtype=torch.float32,device='cuda').reshape(1,535);z=s[:,:512]
    pred=torch.as_tensor(pred,dtype=torch.float32,device='cuda').reshape(1,512);act=torch.as_tensor(physical,dtype=torch.float32,device='cuda').reshape(1,2)
    read=residual_critic_readout(actor,critics['q1'],critics['q2'],critics['target_q1'],critics['target_q2'],s)
    def q(zz,aa):return td_math.two_hot_inv(agent.model.Q(zz,aa,None,return_type='all'),agent.cfg).reshape(5).cpu().numpy()
    qactual=q(z,act);qpred=q(pred,act)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):_,info=agent.model.pi(z,None)
    prior=q(z,info['mean']);legacy=q(z,torch.tensor(old_candidate_action,device='cuda').reshape(1,2)) if old_candidate_action is not None else None
    scalars={n:float(read[n].reshape(())) for n in ['q1','q2','q_min','q_mean','target_q1','target_q2']}
    scalars.update(td_q_matched=float(qactual[:2].mean()),td_q_matched_ensemble=float(qactual.mean()),
        td_q_predicted_latent=float(qpred[:2].mean()),td_q_actor_mean=float(prior[:2].mean()),
        td_q_unexecuted_candidate=float(legacy[:2].mean()) if legacy is not None else None,
        sac_actor_0=float(read['actor_action'][0,0]),sac_actor_1=float(read['actor_action'][0,1]),
        physical_action_0=float(act[0,0]),physical_action_1=float(act[0,1]))
    return scalars,read['actor_action'].cpu().numpy().reshape(2),qactual,qpred


def audit(p,ck,actor,critics):
    saved=ck['mpr_mpc']['residual_sac'];cfg=ck['metadata']['residual_config'];targetdiff={}
    for n in ['q1','q2']:
        a=saved[n];b=saved['target_'+n];targetdiff[n]=float(torch.sqrt(sum((a[k]-b[k]).square().sum() for k in a)))
    payload=dict(base_commit=p['base_commit'],checkpoint=p['checkpoint'],checkpoint_sha256=p['checkpoint_sha256'],
        modules=list(str(f.relative_to(ROOT)) for f in sorted((ROOT/'residual_rl').glob('*.py'))),
        actor=dict(input_dim=535,hidden_dims=cfg['hidden_dims'],outputs='4 raw values:2 mean,2 logStd; tanh residual action; stochastic training/deterministic evaluation',state_dict_sha256=state_sha256(saved['actor'])),
        reward_critics=dict(count=2,input_dim=537,input='concat(original535,state residual2)',hidden_dims=cfg['hidden_dims'],output_dim=1,
            q1_q2='two independent soft REWARD Q networks, not immediate reward and not costQ',
            trained_policy='current stochastic ResidualSAC actor under actual Lattice+mapping+tracking induced transitions; historical replay, partial535state; deterministic eval continuation not exactly the soft stochastic target',
            target='reward + gamma*(1-done)*(min(targetQ1,nextActorAction,targetQ2)-alpha*nextLogProb)',
            gamma=cfg['gamma'],tau=cfg['tau'],alpha_initial=cfg['alpha'],alpha_checkpoint=float(saved['log_alpha'].exp()),update_steps=saved['update_steps'],
            target_actor='CURRENT stochastic actor, no separate target actor',target_critics='Polyak tau=.05, requires_grad=False, separately saved',target_parameter_l2_from_online=targetdiff),
        state535=dict(latent=[0,512],path=[512,524],wm_return=[524,525],feasibility=[525,527],risk_value_valid_pairs=[527,535],
            coarse_wm_q_aggregation='ensemble_mean: evaluate_trajectory_consequence default, NOT MPPIrandom_pair',
            explicit_lattice_action=False,explicit_full_coarse_sequence=False,lattice_context='12 target/path summary features +WMreturn/feasibility/risk; contains lattice target_d/speed but not all geometry/tracking/controller memory'),
        replay=dict(state='original current535',action='REQUESTED normalized residual [d,v], not executed steer/throttle',
            reward='float(env.step.reward) -invalid_residual_penalty(.2) iff regeneration invalid',
            env_reward='wrapper step reward and info.tdmpc2_reward same scalar before float32 tensor rounding',
            next_state='next actual535 prepared before next action; terminal uses placeholder state with done1',
            done='native combined terminated OR truncated; unlikeTD value bootstrap uses native terminated only',
            discount='batch discount filled from agent.residual_gamma=.99',executed_action='stored only in replay diagnostics, not critic input'),
        execution='Lattice coarsepath -> original535 -> residual policy -> latent_tanh d/v parameter refinement -> regenerate (orcoarse oninvalid) -> original path-to-control -> physical first action. StageB MPPIforcedOFF; eval B same path deterministic actor; D addsMPPI onlyafterSAC.',
        compatibility=dict(wm_gamma=.995,sac_gamma=.99,compatible_without_adjustment=False,
            reasons=['different discount','soft entropy-augmented target vs raw environment RTG','invalid residual penalty onlySAC','stochastic train actor vs deterministic eval policy','different native truncated bootstrap semantics','535summary not provenMarkov'],
            no_rescaling_or_retraining=True),
        provenance_limit='Audit is current source + checkpoint architecture/config. Original training-source SHA and serialized replay unavailable; cannot retrospectively prove every training transition/action/invalid-penalty frequency. Checkpoint explicitly says replay buffers are not serialized.',
        answers=dict(
            critic_policy='ResidualSAC soft Q for residual-induced composed environment, not arbitrary physical-action Q.',
            lattice_baseline_input='Path summaries yes; explicit physical baseline action/full sequence no.',
            complete_composed_value='Can approximate induced compositepolicy return conditional on535+residual if representation sufficient. Cannot assume exact full-state composite V or apply to arbitrary MPPI sequences.',
            action_identical_to_environment=False,action_mismatch_is_automatically_bug=False,
            action_explanation='ResidualMDP action is parameter command; physical action is deterministic/geometry-dependent mapping plus fallback. Semantically valid induced action, numerically different.',
            target_matches_current_residual_policy='Usescurrent stochastic actor in Bellman target; stalePolyakQ and replay, entropy/gamma/invalid costs mean not deterministic deployment raw-return target.'))
    dump('critic_architecture_audit.json',payload)


@torch.no_grad()
def build_historical(p,agent,actor,critics):
    rows=[];arrays=[];excluded=[];D=ROOT/'reports/mppi_root_cause_20261117'
    for f in sorted((D/'states').glob('*/*.npz')):
        d=np.load(f);meta=load(f.with_suffix('.json'));source=np.load(str(d['source']))
        if len(d['state535'])!=535 or len(source['actions'])<=10:
            excluded.append(dict(path=str(f.relative_to(ROOT)),reason='native ended before nonterminalH state / no original535'));continue
        state=d['state535'];pred=d['predicted'][10];observation=source['observations'][10];physical=source['actions'][10]
        assert np.array_equal(state[:512],d['actual'][10]);actual=discounted_return(source['rewards'][10:],.995)
        assert abs(actual-float(d['actual_continuation']))<1e-7
        value,out,qa,qp=values(agent,actor,critics,state,pred,physical,d['commands'][10])
        error=float(np.max(np.abs(out-d['original_actor_action'])));assert error<1e-6
        before=meta['before'];row=dict(split='historical_fit_innerstop',scenario_id=meta['scenario'],episode_seed=meta['scenario'],policy_seed=1+meta['scenario'],origin_id=f.parent.name,candidate_id=f.stem,
            terminal_step=10,absolute_terminal_step=before-1+10,source=str(f.relative_to(ROOT)),observation_source=str(d['source']),
            actual_continuation_return=actual,actual_continuation_sac_gamma=discounted_return(source['rewards'][10:],.99),
            continuation_variance=None,continuation_repeats=1,actor_reproduction_max_error=error,**value)
        row['array_index']=len(arrays);rows.append(row);arrays.append(dict(pred=pred,actual=state[:512],obs=observation,state=state,features=state[512:524],actor=out,qa=qa,qp=qp,physical=physical))
    np.savez_compressed(R/'historical_terminal_dataset.npz',**{k:np.stack([x[k] for x in arrays]) for k in arrays[0]})
    table('historical_value_comparison.csv',rows);dump('historical_coverage.json',dict(available=len(rows),excluded=excluded,scene_count=len(set(r['scenario_id'] for r in rows)),note='RetrospectiveFit+InnerStop; no independentclaim, no training. Lattice features are original535slice; full rawLattice/controllerstates were not cached.'))
    print('historical critics evaluated',len(rows),flush=True)
    return rows


@torch.no_grad()
def build_fresh(p,agent,actor,critics):
    rows=[];arrays=[];rollouts=[];continuations=[];native_value=[]
    for folder in sorted((R/'fresh').iterdir()):
        if not (folder/'terminal.npz').exists():continue
        d=np.load(folder/'terminal.npz');meta=load(folder/'terminal_metadata.json');allruns=[]
        for path in sorted(folder.glob('*.json')):
            row=load(path)
            if 'discounted_return' not in row:continue
            rollouts.append(row|dict(source=str(path.relative_to(ROOT))));allruns.append(row)
            source=np.load(path.with_suffix('.npz'));assert source['done'][-1] and not source['done'][:-1].any()
            assert abs(discounted_return(source['rewards'],.995,source['done'])-row['discounted_return'])<1e-8
            continuations.append(dict(split='fresh',scenario_id=meta['scenario_id'],episode_seed=meta['episode_seed'],origin_id=meta['origin_id'],candidate_id=meta['candidate_id'],terminal_step=10,
                policy=row['policy'],repeat=row['repeat'],replay=row['replay'],actual_continuation_return=row['discounted_return'],actual_continuation_sac_gamma=row['discounted_return_sac_gamma'],
                episode_return=row['episode_return'],success=row['success'],crash=row['crash'],out_of_road=row['out_of_road'],route_completion=row['route_completion'],safety_cost=row['safety_cost'],source=str(path.with_suffix('.npz').relative_to(ROOT))))
            if row['policy']=='td_actor':
                native,_,_,_=values(agent,actor,critics,d['state535'],d['predicted_latent'],row['td_first_action'])
                native_value.append(dict(scenario=meta['scenario_id'],origin=meta['origin_id'],candidate=meta['candidate_id'],repeat=row['repeat'],
                    td_q_at_own_first_action=native['td_q_matched'],actual_td_actor_continuation=row['discounted_return'],
                    absolute_error=abs(native['td_q_matched']-row['discounted_return']),
                    note='Different firstphysicalaction vsSAC takeover; ownpolicy diagnostic, not common-action causal isolation'))
        sac=[r for r in allruns if r['policy']=='lattice_sac' and not r['replay']];assert len(sac)==2
        first=folder/'lattice_sac_r1.npz';assert np.array_equal(np.load(first)['actions'][0],d['sac_physical_action'])
        value,out,qa,qp=values(agent,actor,critics,d['state535'],d['predicted_latent'],d['sac_physical_action'],d['commands'][10]);assert np.array_equal(out,d['actor_output'])
        row=dict(split='fresh_independent_pilot',scenario_id=meta['scenario_id'],episode_seed=meta['episode_seed'],policy_seed=meta['initial_policy_seed'],origin_id=meta['origin_id'],candidate_id=meta['candidate_id'],terminal_step=10,
            absolute_terminal_step=39,source=str((folder/'terminal.npz').relative_to(ROOT)),observation_source=str((folder/'terminal.npz').relative_to(ROOT)),
            actual_continuation_return=float(np.mean([r['discounted_return'] for r in sac])),actual_continuation_sac_gamma=float(np.mean([r['discounted_return_sac_gamma'] for r in sac])),
            continuation_variance=float(np.var([r['discounted_return'] for r in sac],ddof=1)),continuation_repeats=2,actor_reproduction_max_error=0.,**value)
        row['array_index']=len(arrays);rows.append(row);arrays.append(dict(pred=d['predicted_latent'],actual=d['actual_latent'],obs=d['actual_observation'],state=d['state535'],features=d['lattice_features'],actor=out,qa=qa,qp=qp,physical=d['sac_physical_action']))
    np.savez_compressed(R/'fresh_terminal_dataset.npz',**{k:np.stack([x[k] for x in arrays]) for k in arrays[0]})
    table('paired_rollout_results.csv',rollouts);table('fresh_continuation_returns.csv',continuations);table('td_actor_value_consistency.csv',native_value)
    return rows,rollouts,continuations


def rank(rows,label):
    result=[];models=dict(A_tdmpc2_q='td_q_matched',B_sac_critic='q_min',C_actual_oracle=label,D_zero=None,
        diagnostic_sac_mean='q_mean',diagnostic_td_ensemble='td_q_matched_ensemble',diagnostic_td_predicted_z='td_q_predicted_latent',diagnostic_td_actor_mean='td_q_actor_mean')
    for split in sorted({r['split'] for r in rows}):
        for origin in sorted({r['origin_id'] for r in rows if r['split']==split}):
            rr=sorted([r for r in rows if r['split']==split and r['origin_id']==origin],key=lambda r:(r['candidate_id'] not in ['B','baseline'],r['candidate_id']))
            y=np.array([r[label] for r in rr])
            for name,key in models.items():
                v=np.array([r[key] for r in rr]) if key else np.zeros(len(rr))
                result.append(dict(split=split,scenario=rr[0]['scenario_id'],origin=origin,model=name,**ranking_metrics(v,y),**calibration(v,y)))
    summaries={}
    keys=['pair_accuracy','spearman','kendall','top1_regret','mae','bias','calibration_error','rmse']
    for split in sorted({r['split'] for r in result}):summaries[split]={name:scene_bootstrap([r for r in result if r['split']==split and r['model']==name],keys) for name in models}
    differences=[]
    for split in summaries:
        for origin in sorted({r['origin'] for r in result if r['split']==split}):
            a=next(r for r in result if r['split']==split and r['origin']==origin and r['model']=='A_tdmpc2_q');b=next(r for r in result if r['split']==split and r['origin']==origin and r['model']=='B_sac_critic')
            differences.append(dict(split=split,scenario=a['scenario'],origin=origin,
                pair_accuracy_gain=b['pair_accuracy']-a['pair_accuracy'] if b['pair_accuracy'] is not None and a['pair_accuracy'] is not None else None,
                regret_improvement=a['top1_regret']-b['top1_regret'],mae_improvement=a['mae']-b['mae']))
    return dict(label=label,by_split=summaries,paired={split:scene_bootstrap([r for r in differences if r['split']==split],['pair_accuracy_gain','regret_improvement','mae_improvement']) for split in summaries},origin_rows=result,paired_origin_rows=differences)


@torch.no_grad()
def main():
    mode=sys.argv[1];p,agent,actor,critics,ck=runtime();audit(p,ck,actor,critics)
    if mode=='historical':build_historical(p,agent,actor,critics);return
    assert mode=='final';assert (R/'paired_status.json').exists() and all(x==0 for x in load(R/'paired_status.json').values())
    historical=list(csv.DictReader((R/'historical_value_comparison.csv').open()))
    numeric=['scenario_id','episode_seed','policy_seed','terminal_step','absolute_terminal_step','array_index','continuation_repeats']
    strings={'split','origin_id','candidate_id','source','observation_source'}
    for row in historical:
        for k,v in row.items():
            if k in strings:continue
            row[k]=None if v=='' else int(v) if k in numeric else float(v)
    fresh,rollouts,cont=build_fresh(p,agent,actor,critics);rows=historical+fresh;table('value_comparison.csv',rows)
    table('continuation_returns.csv',[dict(split=r['split'],scenario_id=r['scenario_id'],episode_seed=r['episode_seed'],origin_id=r['origin_id'],candidate_id=r['candidate_id'],terminal_step=10,
        policy='lattice_sac',repeat='single_archived',replay=False,actual_continuation_return=r['actual_continuation_return'],actual_continuation_sac_gamma=r['actual_continuation_sac_gamma'],source=r['observation_source']) for r in historical]+cont)
    primary=rank(rows,'actual_continuation_return');secondary=rank(rows,'actual_continuation_sac_gamma')
    dump('ranking_metrics.json',dict(primary=primary,secondary_gamma99=secondary,definition='ALLvaluepredictions left unchanged; gamma995 primary/gamma99secondary labels recomputed from SAME native reward arrays; both critics cannot be made gamma-compatible without changing weights. Oracle is tautological reference, not deployed.'))
    print('value ranking analysis complete',flush=True)

if __name__=='__main__':main()
