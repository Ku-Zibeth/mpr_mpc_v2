"""Minimal new regressions; the previous forty tests remain intact."""
import math
import pytest,torch
from tests.test_mppi_value_evaluation import evaluator
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.mppi_ablation import value_ablation_config,terminal_weight_ablation_config
from tools.value_calibration import PrefixValueProbe,terminal_q_audit,assert_origin_match


def test_weight_one_preserves_exact_score_and_weight_only_changes_terminal_contribution():
    e=evaluator();z=torch.zeros(1,3);a=torch.zeros(2,4,2);a[1]=.1
    full=e.evaluate_mppi_candidates(z,a,q_aggregation='ensemble_mean',compute_cost=False)
    for w in (1.,.5,.25,0.):
        r=e.evaluate_mppi_candidates(z,a,q_aggregation='ensemble_mean',compute_cost=False,terminal_q_weight=w)
        assert torch.equal(r.discounted_reward,full.discounted_reward)
        assert torch.equal(r.terminal_latent,full.terminal_latent)
        assert torch.equal(r.terminal_q,full.terminal_q)
        assert torch.equal(r.terminal_discount,full.terminal_discount)
        torch.testing.assert_close(r.total_value,full.discounted_reward+w*full.terminal_discount*full.terminal_q)
        if w==1:assert torch.equal(r.total_value,full.total_value)
    sac=e.evaluate_trajectory_consequence(z,a,compute_cost=False)
    assert torch.equal(sac.features,torch.cat((full.reward_sum[:,None],full.terminal_q[:,None],full.total_value[:,None]),1))
    for w in (-1.,float('nan'),float('inf')):
        with pytest.raises(ValueError):e.evaluate_mppi_candidates(z,a,terminal_q_weight=w)


def test_weight_is_shared_by_every_round_and_final_comparison_without_changing_sampling():
    e=evaluator();original=e.evaluate_mppi_candidates;seen=[]
    def call(z,a,**kw):seen.append((a.clone(),kw.get('terminal_q_weight',1.)));return original(z,a,**kw)
    e.evaluate_mppi_candidates=call
    base=dict(num_samples=64,num_elites=8,iterations=4)
    configs={n:terminal_weight_ablation_config(base,n) for n in ('Q1','Q05','Q025','Q0')}
    assert all({k:v for k,v in c.items() if k!='terminal_q_weight'}=={k:v for k,v in configs['Q1'].items() if k!='terminal_q_weight'} for c in configs.values())
    for n,c in configs.items():
        seen.clear();p=LocalMPPI(e,c,horizon=3,action_dim=2,device='cpu');r=p.plan(torch.zeros(1,3),torch.zeros(4,2),eval_mode=True)
        assert len(seen)==5 and all(w==c['terminal_q_weight'] for _,w in seen)
        torch.testing.assert_close(seen[-1][0],torch.stack((r.initial_mean,r.final_mean)))
        assert r.metrics['wm_evaluated_candidates']==262
        assert r.metrics['predicted_return_gain']==pytest.approx(r.metrics['prefix_reward_gain']+r.metrics['weighted_terminal_q_gain'],abs=1e-5)
    old=LocalMPPI(e,value_ablation_config(base,'D4'),horizon=3,action_dim=2,device='cpu')
    new=LocalMPPI(e,configs['Q1'],horizon=3,action_dim=2,device='cpu')
    assert torch.equal(old._weights(torch.tensor([1.,2.,3.])),new._weights(torch.tensor([1.,2.,3.])))
    assert torch.equal(old.plan(torch.zeros(1,3),torch.zeros(4,2),eval_mode=True).final_mean,
                       new.plan(torch.zeros(1,3),torch.zeros(4,2),eval_mode=True).final_mean)


def test_teacher_open_loop_indices_reward_label_terminal_actions_and_origin_matching():
    e=evaluator();e.encode=lambda o:o.reshape(1,3)
    e.agent.model.reward=lambda z,a,task:torch.log1p(z[:,:1]+a[:,:1])
    p=PrefixValueProbe(e,torch.zeros(1,3));action=torch.tensor([.2,.1])
    p.before(torch.zeros(3),action);r=p.after(torch.ones(3),dict(tdmpc2_reward=7.,metadrive_reward=99.))
    assert r['step']==0 and r['reward']==7. and r['reward_teacher']==pytest.approx(.2)
    p.before(torch.ones(3),action);r=p.after(torch.ones(3)*2,dict(tdmpc2_reward=8.))
    assert r['step']==1 and r['reward_teacher']==pytest.approx(1.2) and r['reward_open_loop']==pytest.approx(.4)
    assert r['latent_l2']==pytest.approx(r['latent_rmse']*math.sqrt(3))
    q=terminal_q_audit(e.agent,p.imagined,p.actual,torch.zeros(1,2),torch.ones(1,2))
    assert q['predicted_z_candidate_q']==pytest.approx(q['actual_z_candidate_q'])
    assert q['actual_z_sac_q'][0]>q['actual_z_candidate_q'][0]
    p.before(torch.zeros(3),action)
    with pytest.raises(KeyError):p.after(torch.zeros(3),dict(metadrive_reward=3.))
    origin=dict(episode_config={'horizon':1000},observation=[1.],state={'speed':2},rng={'seed':3},candidate_actions=[0])
    assert assert_origin_match(origin,dict(origin))
    with pytest.raises(ValueError):assert_origin_match(origin,dict(origin,rng={'seed':4}))
