"""Six focused contracts for the sole SAC-centred TD-MPC2 optimiser."""
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.evaluator import TrajectoryWorldModelEvaluator
from tests.test_native_mppi_cost import RolloutModel


class Values:
    def __init__(self, value=None):
        self.calls=[]
        self.value=value or (lambda a: a[:, 0].sum(-1)*.01)
    def value_ranking_ready(self):
        raise AssertionError('No runtime Value-validation gate')
    def evaluate_mppi_candidates(self,z,actions,*,task=None,compute_cost=None):
        assert compute_cost is False
        self.calls.append(actions.clone());v=self.value(actions)
        return SimpleNamespace(total_value=v,discounted_reward=v,terminal_q=v*0,timings={})


def planner(e=None,**kwargs):
    cfg=dict(num_samples=8,num_elites=4,iterations=4,initial_std=[.04,.12],
             min_std=[.01,.03],max_std=[.08,.25],max_delta=[.1,.25],temperature=.5,eval_seed=9)
    cfg.update(kwargs)
    return LocalMPPI(e or Values(),cfg,horizon=3,action_dim=2,device='cpu')


def test_A_gaussian_stays_sac_anchored_and_single_warm_start():
    e=Values();p=planner(e);base=torch.tensor([[.97,-.96]]).expand(4,-1).clone()
    r=p.plan(torch.zeros(1,3),base,eval_mode=True)
    for batch in e.calls[:-1]:
        assert len(batch)==8 and torch.equal(batch[0],base)
        assert (batch.abs()<=1).all() and ((batch-base).abs()<=p.max_delta+1e-7).all()
    previous=p._prev_mean.clone();newbase=base*.9
    expected=newbase.clone();expected[:-1]=previous[1:]
    torch.testing.assert_close(p.initial_mean(newbase),p._project(expected,newbase))
    p.reset();assert torch.equal(p.initial_mean(base),base)
    first=p.plan(torch.zeros(1,3),base,eval_mode=True);p.reset()
    repeated=p.plan(torch.zeros(1,3),base,eval_mode=True)
    assert torch.equal(first.selected_sequence,repeated.selected_sequence)
    assert not hasattr(p,'candidate_safety') and not hasattr(p,'_stall_active')


def test_B_top_k_weights_mean_std_match_official_formula():
    p=planner(num_samples=4,num_elites=2)
    actions=torch.tensor([[.02,.10],[-.02,.1],[.06,.2],[-.08,-.1]])[:,None].expand(-1,4,-1).clone()
    mean,std,elite=p._update_distribution(actions,torch.tensor([3.,1.,2.,-1.]))
    w=np.array([1.,math.exp(-.5)]);w/=w.sum()+1e-9
    expected=(w[:,None]*np.array([[.02,.10],[.06,.2]])).sum(0)
    variance=(w[:,None]*(np.array([[.02,.10],[.06,.2]])-expected)**2).sum(0)
    np.testing.assert_allclose(mean.numpy(),np.tile(expected,(4,1)),rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(std.numpy(),np.tile(np.clip(np.sqrt(variance),[.01,.03],[.08,.25]),(4,1)),rtol=1e-6)
    assert elite.tolist()==[3.,2.]


def test_C_accept_improved_final_mean_without_extra_gates_or_elite_draw():
    e=Values();p=planner(e);base=torch.zeros(4,2)
    candidates=torch.zeros(8,4,2)
    candidates[1:5,:,0]=torch.tensor([.02,.04,.06,.08])[:,None]
    candidates[1:5,:,1]=.1
    p._candidate_batch=lambda *args:candidates.clone()
    r=p.plan(torch.zeros(1,3),base,eval_mode=True)
    assert not r.baseline_selected and 0<float(r.planner_gain)<.05
    assert torch.equal(r.selected_sequence,r.final_mean) and torch.equal(r.action,r.final_mean[0])
    assert not torch.equal(r.selected_sequence,candidates[4])
    assert r.metrics['iterations_completed']==4 and len(e.calls)==5
    assert ((r.selected_sequence-base).abs()<=p.max_delta+1e-7).all()


def test_D_equal_or_worse_final_value_keeps_exact_sac_action():
    for value in [lambda a:torch.ones(len(a)),lambda a:-a.square().sum((1,2))]:
        p=planner(Values(value));base=torch.zeros(4,2)
        r=p.plan(torch.zeros(1,3),base,eval_mode=True)
        assert r.baseline_selected and torch.equal(r.selected_sequence,base) and torch.equal(r.action,base[0])


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_E_invalid_wm_value_returns_baseline_without_recovery_or_failure(bad):
    p=planner(Values(lambda a:a.new_full((len(a),),bad)));base=torch.full((4,2),.1)
    r=p.plan(torch.zeros(1,3),base,eval_mode=True)
    assert torch.equal(r.action,base[0]) and torch.equal(r.selected_sequence,base)
    assert r.metrics['invalid_prediction']==1 and r.planning_mode=='baseline'
    assert not hasattr(r,'safety_failure') and not hasattr(r,'safety_certificate')
    assert not hasattr(r,'recovery_reason')


def test_F_h_plus_one_reward_dynamics_terminal_q_and_termination_mask():
    class Model(RolloutModel):
        def __init__(self):
            super().__init__();self.steps=[];self.rewards=[];self.terminal=[]
        def next(self,z,a,task):
            self.steps.append(a.clone());return z+1
        def reward(self,z,a,task):
            self.rewards.append(a.clone());return a.new_full((len(a),1),math.log1p(2))
        def Q(self,z,a,task,return_type='all'):
            assert return_type=='all';self.terminal.append(a.clone())
            return torch.log1p(torch.stack([a[:,:1]+4+i for i in (0,2,4)]))
        def cost(self,*args):raise AssertionError('Cost Head must not enter MPPI evaluation')
        def pi(self,*args):raise AssertionError('a_H must come from the candidate')
        def termination(self,z,task):return (z[:,:1]>=2).float()
    model=Model();cfg=SimpleNamespace(action_dim=2,latent_dim=3,multitask=False,episodic=True,num_bins=1,cost={'enabled':True})
    e=TrajectoryWorldModelEvaluator(SimpleNamespace(model=model,cfg=cfg,discount=.9,device='cpu'),3)
    actions=torch.zeros(2,4,2);actions[:,-1,0]=.7
    r=e.evaluate_mppi_candidates(torch.zeros(1,3),actions,compute_cost=False)
    assert len(model.steps)==len(model.rewards)==3 and len(model.terminal)==1
    assert torch.equal(model.terminal[0],actions[:,-1])
    torch.testing.assert_close(r.total_value,torch.full((2,),2*(1+.9+.81)+.9**3*6.7))
    assert all(torch.equal(a,actions[:,i]) for i,a in enumerate(model.steps))
    model._termination=True
    masked=e.evaluate_mppi_candidates(torch.zeros(1,3),actions,compute_cost=False)
    torch.testing.assert_close(masked.total_value,torch.full((2,),2*(1+.9)))
