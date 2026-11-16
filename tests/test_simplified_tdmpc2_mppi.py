"""Current SAC-initialised MPPI interface, formulas, indexing and deployment diagnostics."""
from types import SimpleNamespace
from unittest.mock import Mock
import gzip
import json
import math
import numpy as np
import pytest
import torch
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.evaluator import TrajectoryWorldModelEvaluator
from mpr_mpc.evaluation_diagnostics import EpisodeDiagnostics, StepDiagnosticsLogger
from tests.test_native_mppi_cost import RolloutModel


class Values:
    def __init__(self,value=None):
        self.calls=[]; self.value=value or (lambda a:a[:,0].sum(-1)*.01)
    def value_ranking_ready(self):
        raise AssertionError('Qualification must not control search.')
    def evaluate_mppi_candidates(self,z,actions,*,task=None,compute_cost=None):
        assert compute_cost is False
        self.calls.append(actions.clone());v=self.value(actions)
        return SimpleNamespace(total_value=v,discounted_reward=v,terminal_q=v*0,timings={})


def planner(e=None,**kwargs):
    cfg=dict(num_samples=8,num_elites=4,iterations=4,initial_std=[.04,.12],min_std=[.01,.03],
             max_std=[.08,.25],temperature=.5,noise_rho=[.8,.6],eval_seed=9)
    cfg.update(kwargs)
    return LocalMPPI(e or Values(),cfg,horizon=3,action_dim=2,device='cpu')


def test_sac_initializes_each_call_without_historical_mean_or_validation_gate():
    p=planner();a=torch.zeros(4,2);r=p.plan(torch.zeros(1,3),a,eval_mode=True)
    b=torch.full((4,2),.3);next=p.plan(torch.zeros(1,3),b,eval_mode=True)
    assert torch.equal(r.initial_mean,a) and torch.equal(next.initial_mean,b)
    assert not hasattr(p,'_prev_mean') and not hasattr(p,'candidate_safety')
    p.reset();one=p.plan(torch.zeros(1,3),a,eval_mode=True);p.reset();two=p.plan(torch.zeros(1,3),a,eval_mode=True)
    assert torch.equal(one.selected_sequence,two.selected_sequence)


def test_ar1_has_independent_channel_correlations():
    p=LocalMPPI(Values(),dict(num_samples=256,num_elites=8,noise_rho=[.8,0.]),horizon=30,action_dim=2,device='cpu')
    base=torch.zeros(31,2);batch=p._candidate_batch(base,torch.full_like(base,.02),base)[1:].numpy()
    correlations=[np.corrcoef(batch[:,:-1,c].ravel(),batch[:,1:,c].ravel())[0,1] for c in range(2)]
    assert .75<correlations[0]<.85 and abs(correlations[1])<.05
    assert abs(np.corrcoef(batch[:,:,0].ravel(),batch[:,:,1].ravel())[0,1])<.05


def test_sampling_uses_only_physical_bounds_not_legacy_trust_region():
    p=planner(initial_std=[.5,.5],max_std=[1.,1.]);base=torch.zeros(4,2)
    batch=p._candidate_batch(base,torch.full_like(base,.5),base)
    assert batch.shape==(9,4,2) and torch.equal(batch[0],base) and (batch.abs()<=1).all()
    assert (batch[1:].abs()>.25).any() and not hasattr(p,'max_delta')


def test_every_round_has_64_gaussians_plus_one_baseline_and_eight_elites():
    e=Values();p=planner(e,num_samples=64,num_elites=8);base=torch.zeros(4,2)
    r=p.plan(torch.zeros(1,3),base,eval_mode=True)
    assert len(e.calls)==5 and [len(a) for a in e.calls]==[65,65,65,65,2]
    assert all(torch.equal(a[0],base) for a in e.calls)
    assert all(h['candidate_count']==65 and h['elite_count']==8 for h in r.iterations)
    assert r.metrics['wm_evaluated_candidates']==262 and r.metrics['iterations_completed']==4


def test_top_k_softmax_and_weighted_distribution_follow_divisor_temperature():
    p=planner(num_samples=4,num_elites=2)
    actions=torch.tensor([[.02,.1],[-.02,.1],[.06,.2],[-.08,-.1]])[:,None].expand(-1,4,-1).clone()
    values=torch.tensor([3.,1.,2.,-1.]);indices,elite,w=p._elite_weights(values)
    assert indices.tolist()==[0,2] and elite.tolist()==[3.,2.] and w.sum()==pytest.approx(1.)
    weights=np.array([1.,math.exp(-2.)]);weights/=weights.sum()
    chosen=np.array([[.02,.1],[.06,.2]]);expected=(weights[:,None]*chosen).sum(0)
    variance=(weights[:,None]*(chosen-expected)**2).sum(0)
    mean,std,_=p._update_distribution(actions,values)
    np.testing.assert_allclose(mean.numpy(),np.tile(expected,(4,1)),rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(std.numpy(),np.tile(np.clip(np.sqrt(variance),[.01,.03],[.08,.25]),(4,1)),rtol=1e-6)
    assert torch.isfinite(p._elite_weights(values+1e6)[2]).all()


def test_final_mean_and_baseline_share_one_score_batch_and_no_extra_threshold():
    e=Values();p=planner(e);base=torch.zeros(4,2)
    candidates=torch.zeros(9,4,2);candidates[1:,:,0]=torch.linspace(.01,.08,8)[:,None]
    p._candidate_batch=lambda *args:candidates.clone()
    r=p.plan(torch.zeros(1,3),base,eval_mode=True)
    assert torch.equal(e.calls[-1],torch.stack((base,r.final_mean)))
    assert 0<float(r.planner_gain)<.05 and not r.baseline_selected
    assert torch.equal(r.action,r.final_mean[0]) and torch.equal(r.selected_sequence,r.final_mean)


def test_no_gain_keeps_exact_sac_and_records_unaccepted_optimized_value():
    for value in (lambda a:torch.ones(len(a)),lambda a:-a.square().sum((1,2))):
        p=planner(Values(value));base=torch.zeros(4,2);r=p.plan(torch.zeros(1,3),base,eval_mode=True)
        assert r.baseline_selected and torch.equal(r.action,base[0]) and torch.equal(r.selected_sequence,base)
        assert r.metrics['predicted_return_gain']<=0 and r.metrics['score_gain']==0


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_numerical_anomalies_are_reported_without_nonfinite_actions(bad):
    p=planner(Values(lambda a:a.new_full((len(a),),bad)));base=torch.full((4,2),.1)
    with pytest.warns(RuntimeWarning,match='numerical anomaly'):
        r=p.plan(torch.zeros(1,3),base,eval_mode=True)
    assert torch.isfinite(r.action).all() and torch.equal(r.action,base[0])
    assert r.metrics['invalid_prediction']==1 and r.numerical_error=='non_finite_candidate_return'
    assert not hasattr(r,'safety_certificate') and not hasattr(r,'safety_failure')


def test_h_plus_one_indexing_official_decoded_avg_q_and_frozen_sac_reducer():
    class Model(RolloutModel):
        def __init__(self):
            super().__init__();self.steps=[];self.rewards=[];self.terminal=[];self.q_modes=[]
        def next(self,z,a,task):self.steps.append(a.clone());return z+1
        def reward(self,z,a,task):self.rewards.append(a.clone());return a.new_full((len(a),1),math.log1p(2))
        def Q(self,z,a,task,return_type='all'):
            self.terminal.append(a.clone());self.q_modes.append(return_type)
            members=torch.stack([a[:,:1]+i for i in (4,6,10)])
            return members[:2].mean(0) if return_type=='avg' else torch.log1p(members)
        def cost(self,*a):raise AssertionError('Cost must not enter pure MPPI.')
        def pi(self,*a):raise AssertionError('Use candidate a_H.')
        def termination(self,z,task):return (z[:,:1]>=2).float()
    m=Model();cfg=SimpleNamespace(action_dim=2,latent_dim=3,multitask=False,episodic=True,num_bins=1,cost={'enabled':True})
    e=TrajectoryWorldModelEvaluator(SimpleNamespace(model=m,cfg=cfg,discount=.9,device='cpu'),3)
    a=torch.zeros(2,4,2);a[:,-1,0]=.7
    r=e.evaluate_mppi_candidates(torch.zeros(1,3),a,compute_cost=False)
    assert len(m.steps)==len(m.rewards)==3 and m.q_modes==['avg']
    assert torch.equal(m.terminal[0],a[:,-1]) and all(torch.equal(x,a[:,i]) for i,x in enumerate(m.steps))
    torch.testing.assert_close(r.total_value,torch.full((2,),2*(1+.9+.81)+.9**3*5.7))
    coarse=e.evaluate_trajectory_consequence(torch.zeros(1,3),a,compute_cost=False)
    assert m.q_modes[-1]=='all'
    torch.testing.assert_close(coarse.total_value,torch.full((2,),2*(1+.9+.81)+.9**3*(20/3+.7)))
    m._termination=True
    masked=e.evaluate_mppi_candidates(torch.zeros(1,3),a,compute_cost=False)
    torch.testing.assert_close(masked.total_value,torch.full((2,),2*(1+.9)))


def test_agent_executes_selected_first_action_without_certificate_interception():
    from mpr_mpc.agent import MPRMPCAgent
    from env.env import MetaDriveTDMPC2Env
    agent=object.__new__(MPRMPCAgent);sequence=torch.full((4,2),.2)
    result=SimpleNamespace(action=sequence[0],stage=SimpleNamespace(use_mppi=True),debug={'selected_mppi_actions':sequence})
    agent.planner=SimpleNamespace(enabled=True,plan=lambda *a,**k:result,last_context=None)
    agent.env=SimpleNamespace(unwrapped_metadrive=SimpleNamespace(agent=None),request_safety_failure=Mock(side_effect=AssertionError('interception')))
    agent.global_env_step=0;agent.set_global_step=lambda s:None;agent._maybe_freeze_tdmpc2=lambda s:None
    assert torch.equal(agent.act(torch.zeros(3),eval_mode=True),sequence[0])
    assert not hasattr(MetaDriveTDMPC2Env,'request_safety_failure')


def test_search_iteration_log_and_predicted_vs_actual_discounted_return(tmp_path):
    p=planner();r=p.plan(torch.zeros(1,3),torch.zeros(4,2),eval_mode=True)
    result=SimpleNamespace(stage=SimpleNamespace(use_mppi=True,use_residual=True),metrics=r.metrics.copy(),
        debug={'mppi_iterations':r.iterations,'wm_discount':.9},planning_mode=r.planning_mode)
    logger=StepDiagnosticsLogger(tmp_path,'search');logger.log(1,1,r.action,result,dict(tdmpc2_reward=2.,route_completion=.1));logger.close()
    with gzip.open(tmp_path/'search.search.jsonl.gz','rt') as f:record=json.loads(f.readline())
    assert len(record['iterations'])==4 and np.asarray(record['iterations'][0]['mean']).shape==(4,2)
    d=EpisodeDiagnostics()
    for actual,predicted in [(2.,3.),(3.,4.)]:
        result.metrics.update({'mppi/selected_predicted_return':10.,'mppi/predicted_first_reward':predicted})
        d.observe(torch.zeros(3),r.action,dict(tdmpc2_reward=actual),result)
    summary=d.summary({})
    assert summary['actual_discounted_return']==pytest.approx(4.7)
    assert summary['initial_predicted_return_bias']==pytest.approx(5.3)
    assert summary['one_step_reward_mae']==pytest.approx(1.)
