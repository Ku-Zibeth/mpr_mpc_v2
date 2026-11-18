"""Offline terminal-action parity on identical stored prefixes, all fiveQs."""
import sys,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *
from tools.mppi_failure_attribution import ranking_metrics,scene_bootstrap
from tools.policy_value_diagnosis import calibration
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.parser import parse_cfg
from common.world_model import WorldModel
from common import math as td_math
from tdmpc2 import TDMPC2
from types import SimpleNamespace
import evaluate_model_suite as suite


@torch.no_grad()
def main():
    p=load(REPORT/'protocol.json');raw=OmegaConf.load(p['source_environment_config']);raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False
    _,cfg=suite._make_cfg(raw,Path(p['checkpoint']).parent.parent.name,Path(p['checkpoint']),[1]);cfg.obs_shape={'state':(259,)};cfg.action_dim=2;cfg.episode_length=1000
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');model=WorldModel(cfg).cuda().eval().requires_grad_(False)
    agent=SimpleNamespace(model=model,cfg=cfg);TDMPC2.load(agent,{'model':torch.load(p['checkpoint'],map_location='cpu',weights_only=False)['model']})
    all_members=[];combinations=[];comparisons=[];random_rows=[];draw_arrays={}
    old=load(SOURCE/'reports/mppi_root_cause_20261117/protocol.json')
    for scene,before in old['development_origins']:
        folder=SOURCE/'reports/mppi_root_cause_20261117/catalog'/f'{scene}_{before}';c=np.load(folder/'catalog.npz');actions=c['actions'];z=torch.tensor(c['predicted_latents'][:,-1],device='cuda');prefix=c['predicted_prefix'];pair=tuple(c['ref_pair']);labels=old['choices']
        ys=np.array([load(SOURCE/'reports/mppi_root_cause_20261117/interventions'/folder.name/f'A_{label}.json')['actual_return'] for label in labels])
        continuations=np.array([np.load(SOURCE/'reports/mppi_root_cause_20261117/interventions'/folder.name/f'A_{label}.npz')['rtg'][10] for label in labels])
        zz=torch.tensor(c['predicted_latents'][:,1:],device='cuda').reshape(-1,512)
        term=model.termination(zz,None).reshape(len(actions),10).cpu().numpy();discount=.995**10*(1-(term>.5).any(1))
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(20261117+scene+before);sample,info=model.pi(z,None)
            draws=[torch.randperm(5,device='cuda')[:2].cpu().numpy() for _ in range(100)]
        modes=dict(candidate=torch.tensor(actions[:,-1],device='cuda'),policy_mean=info['mean'],native_policy_sample=sample)
        for mode,act in modes.items():
            qs=td_math.two_hot_inv(model.Q(z,act,None,return_type='all'),cfg).reshape(5,-1).T.cpu().numpy();reducers=all_reducers(qs)
            sampled=np.stack([qs[:,v].mean(1) for v in draws]);draw_arrays[f'{scene}_{before}_{mode}']=sampled
            for i,label in enumerate(labels):
                all_members.append(dict(scenario=scene,origin=folder.name,candidate=label,terminal_action=mode,
                    **{f'Q{j}':float(qs[i,j]) for j in range(5)},action_steer=float(act[i,0]),action_throttle=float(act[i,1]),
                    predicted_prefix=float(prefix[i]),terminal_discount=float(discount[i]),actual_continuation=float(continuations[i]),actual_total_return=float(ys[i]),split='historical_fit_seen'))
                random_rows.append(dict(scenario=scene,origin=folder.name,candidate=label,terminal_action=mode,draws=100,
                    q_mean=float(sampled[:,i].mean()),q_std=float(sampled[:,i].std()),q_p05=float(np.quantile(sampled[:,i],.05)),q_p95=float(np.quantile(sampled[:,i],.95)),
                    total_mean=float((prefix[i]+discount[i]*sampled[:,i]).mean()),systematic_bias=float(qs[i].mean()-continuations[i])))
            for reducer,q in reducers.items():
                total=prefix+discount*q
                comparisons.append(dict(scenario=scene,origin=folder.name,terminal_action=mode,q_reducer=reducer,
                    **ranking_metrics(total,ys),**calibration(q,continuations),top1=int(total.argmax())))
                for i,label in enumerate(labels):combinations.append(dict(scenario=scene,origin=folder.name,candidate=label,terminal_action=mode,q_reducer=reducer,q=float(q[i]),total=float(total[i]),actual=float(ys[i])))
        draw_arrays[f'{scene}_{before}_member_draws']=np.asarray(draws)
    table(NATIVE/'q_all_members.csv',all_members);table(NATIVE/'q_all_pair_combinations.csv',combinations);table(NATIVE/'terminal_action_comparison.csv',comparisons);table(NATIVE/'q_random_pair_100.csv',random_rows)
    np.savez_compressed(NATIVE/'random_pair_100_distributions.npz',**draw_arrays)
    summary={f'{mode}/{reducer}':scene_bootstrap([r for r in comparisons if r['terminal_action']==mode and r['q_reducer']==reducer],['pair_accuracy','spearman','kendall','top1_regret','mae','bias','calibration_error']) for mode,reducer in sorted({(r['terminal_action'],r['q_reducer']) for r in comparisons})}
    agreement=[]
    for origin in sorted({r['origin'] for r in comparisons}):
        for mode in modes:
            rr=[r for r in comparisons if r['origin']==origin and r['terminal_action']==mode]
            row=dict(scenario=rr[0]['scenario'],origin=origin,mode=mode)
            for family in ['avg','min','all_aggregators']:
                group=rr if family=='all_aggregators' else [r for r in rr if r['q_reducer'].startswith(family) and len(r['q_reducer'])==5]
                row['top1_agreement_'+family]=float(np.mean([a['top1']==b['top1'] for a,b in itertools.combinations(group,2)]))
            agreement.append(row)
    dump(NATIVE/'q_calibration.json',dict(by_terminal_action_and_reducer=summary,top1_agreement=scene_bootstrap(agreement,['top1_agreement_avg','top1_agreement_min','top1_agreement_all_aggregators']),
        native_planning='stochasticpi then Qavg (one randomly drawnshared pair)',native_TD_target='stochasticpi then targetQmin (two randomly drawnsharedmembers)',
        phase1='fixedQ01 mean; not equivalent to targetmin nor anaverage overall10pairs',same_prefix_latent_reward_discount_and_Qpairs=True,
        sampled_mode='oneoriginalpi sampledaction perterminal, fixedseed; mean mode NOT native stochasticpi',
        scope='4 historicalFit maps, actualH10prefixthenSAC labels; descriptive parityonly, NOT independentH1performance or TD-MPC2fullMPC evidence.'))
    # Required old tables are read in full; preserve every historical dataset split.
    inspected={}
    v=load(REPORT/'version_manifest.json')
    for name in v['historical_inputs']:
        f=SOURCE/name
        if f.suffix=='.json':inspected[name]=dict(keys=list(load(f)))
        elif f.suffix=='.csv':
            with f.open() as stream:rows=list(csv.DictReader(stream));inspected[name]=dict(rows=len(rows),columns=list(rows[0]) if rows else [])
    dump(NATIVE/'historical_data_review.json',inspected)
    # Mirror mandatory final-delivery filenames without changing oldresults.
    table(REPORT/'native_terminal_action_parity.csv',comparisons);table(REPORT/'q_all_members.csv',all_members);table(REPORT/'q_pair_comparison.csv',combinations)
    print('native terminal/Q parity complete',flush=True)

if __name__=='__main__':main()
