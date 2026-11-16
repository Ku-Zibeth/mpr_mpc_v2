"""Value-definition regression tests, not another planner or safety test suite."""
from types import SimpleNamespace
from unittest.mock import patch
import math
import torch
import pytest
from mpr_mpc.planning.evaluator import TrajectoryWorldModelEvaluator
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.mppi_ablation import VALUE_ABLATIONS, value_ablation_config
from tests.test_native_mppi_cost import RolloutModel


class Model(RolloutModel):
    def __init__(self):
        super().__init__(); self.next_calls=0;self.reward_calls=0;self.q_modes=[];self.pi_calls=0
    def next(self,z,a,task):self.next_calls+=1;return z+torch.cat((a, a[:,:1]),-1)
    def reward(self,z,a,task):self.reward_calls+=1;return torch.log1p(torch.ones_like(a[:,:1])*2)
    def pi(self,z,task):
        self.pi_calls+=1
        mean=(z[:,:2]*.1).tanh()
        return (mean+torch.randn_like(mean)*.1).clamp(-1,1),{'mean':mean}
    def Q(self,z,a,task,return_type='all'):
        self.q_modes.append(return_type)
        q=torch.stack([4+a[:,:1]+i for i in range(5)])
        if return_type=='all':return torch.log1p(q)
        return q[torch.randperm(5)[:2]].mean(0)


def evaluator():
    cfg=SimpleNamespace(action_dim=2,latent_dim=3,num_q=5,multitask=False,episodic=True,num_bins=1,cost={'enabled':False})
    model=Model()
    return TrajectoryWorldModelEvaluator(SimpleNamespace(model=model,cfg=cfg,discount=.9,device='cpu'),3)


def test_terminal_prior_ignores_padding_and_restores_rng_for_comparable_q_pairs():
    e=evaluator();a=torch.zeros(2,4,2);a[1,:3]=.1;a[:,-1]=.9
    torch.manual_seed(7);state=torch.random.get_rng_state()
    prior=e.evaluate_mppi_candidates(torch.zeros(1,3),a,terminal_action='policy_prior',q_aggregation='fixed_pair',q_indices=(0,1),compute_cost=False)
    assert torch.equal(state,torch.random.get_rng_state())
    assert e.agent.model.next_calls==e.agent.model.reward_calls==3 and e.agent.model.pi_calls==1
    changed=a.clone();changed[:,-1]=-1
    again=e.evaluate_mppi_candidates(torch.zeros(1,3),changed,terminal_action='policy_prior',q_aggregation='fixed_pair',q_indices=(0,1),compute_cost=False)
    assert torch.equal(prior.total_value,again.total_value)
    assert torch.equal(prior.terminal_latent,again.terminal_latent)
    assert not torch.equal(prior.terminal_actions,a[:,-1])
    candidate=e.evaluate_mppi_candidates(torch.zeros(1,3),a,compute_cost=False,q_aggregation='fixed_pair',q_indices=(0,1))
    assert torch.equal(prior.discounted_reward,candidate.discounted_reward)
    assert torch.equal(prior.terminal_latent,candidate.terminal_latent)
    assert torch.equal(prior.terminal_discount,candidate.terminal_discount)
    assert torch.equal(candidate.terminal_actions,a[:,-1])


def test_random_q_pairs_are_not_shifted_by_unused_policy_noise():
    e=evaluator();a=torch.zeros(2,4,2)
    torch.manual_seed(19);e.evaluate_mppi_candidates(torch.zeros(1,3),a,compute_cost=False);candidate_state=torch.random.get_rng_state()
    torch.manual_seed(19);e.evaluate_mppi_candidates(torch.zeros(1,3),a,terminal_action='policy_prior',compute_cost=False);prior_state=torch.random.get_rng_state()
    assert torch.equal(candidate_state,prior_state)


@pytest.mark.parametrize('aggregation,expected',[('fixed_pair',6.),('ensemble_mean',6.),('random_pair',None)])
def test_q_decoding_and_prefix_horizon_are_correct(aggregation,expected):
    e=evaluator();a=torch.zeros(2,4,2)
    r=e.evaluate_mppi_candidates(torch.zeros(1,3),a,q_aggregation=aggregation,q_indices=(0,4) if aggregation=='fixed_pair' else None,
        compute_cost=False,capture_ensemble=True)
    assert e.agent.model.next_calls==e.agent.model.reward_calls==3
    torch.testing.assert_close(r.discounted_reward,torch.full((2,),2*(1+.9+.81)))
    torch.testing.assert_close(r.q_member_values[:,0],torch.arange(4.,9.))
    if expected is not None:torch.testing.assert_close(r.terminal_q,torch.full((2,),expected))
    torch.testing.assert_close(r.total_value,r.discounted_reward+r.terminal_discount*r.terminal_q)
    # This API is the frozen SAC feature path: preserve full ensemble mean.
    old=e.evaluate_trajectory_consequence(torch.zeros(1,3),a,compute_cost=False)
    torch.testing.assert_close(old.terminal_q,torch.full((2,),6.))


def test_fixed_pair_is_drawn_once_and_reused_for_all_five_evaluations():
    e=evaluator();seen=[];original=e.evaluate_mppi_candidates
    def score(*a,**kw):seen.append(kw.get('q_indices'));return original(*a,**kw)
    e.evaluate_mppi_candidates=score
    cfg=dict(num_samples=8,num_elites=4,iterations=4,q_aggregation='fixed_pair')
    p=LocalMPPI(e,cfg,horizon=3,action_dim=2,device='cpu')
    with patch.object(e,'new_mppi_q_pair',side_effect=[(0,1),(3,4)]) as draw:
        first=p.plan(torch.zeros(1,3),torch.zeros(4,2),eval_mode=True)
        assert seen==[(0,1)]*5 and draw.call_count==1
        seen.clear();p.plan(torch.zeros(1,3),torch.zeros(4,2),eval_mode=True)
        assert seen==[(3,4)]*5 and draw.call_count==2
    assert first.evaluation['q_indices']==(0,1)


def test_policy_terminal_has_no_unused_action_optimization_and_keeps_budget():
    e=evaluator();seen=[];original=e.evaluate_mppi_candidates
    def score(z,a,**kw):seen.append(a.clone());return original(z,a,**kw)
    e.evaluate_mppi_candidates=score
    cfg=dict(num_samples=64,num_elites=8,iterations=4,terminal_action='policy_prior')
    p=LocalMPPI(e,cfg,horizon=3,action_dim=2,device='cpu');base=torch.full((4,2),.2)
    r=p.plan(torch.zeros(1,3),base,eval_mode=True)
    assert len(seen)==5 and r.metrics['wm_evaluated_candidates']==262
    assert all(torch.equal(a[:,-1],base[-1:].expand(len(a),-1)) for a in seen)
    assert torch.equal(r.final_mean[-1],base[-1]) and (r.final_std[-1]==0).all()
    assert (r.final_std[:-1]>=p.min_std_vector).all()
    assert r.metrics['active_action_steps']==3
    assert len(r.iterations)==4 and all(1<=h['effective_sample_size']<=8.00001 and 0<=h['weight_entropy']<=math.log(8)+1e-6 for h in r.iterations)
    assert r.metrics['predicted_return_gain']==pytest.approx(r.metrics['prefix_reward_gain']+r.metrics['discounted_terminal_gain'],abs=1e-5)
    torch.testing.assert_close(seen[-1],torch.stack((base,r.final_mean)))


def test_one_factor_presets_preserve_all_sampling_parameters():
    base=dict(num_samples=64,num_elites=8,iterations=4,initial_std=[.04,.12],min_std=[.01,.03],max_std=[.08,.25],noise_rho=[.8,.8])
    configs={name:value_ablation_config(base,name) for name in VALUE_ABLATIONS}
    for name in ('D1','D2','D3','D4'):
        changed=[k for k in configs['D0'] if configs[name][k]!=configs['D0'][k]]
        assert len(changed)==1
    assert all(all(c[k]==v for k,v in base.items()) for c in configs.values())
