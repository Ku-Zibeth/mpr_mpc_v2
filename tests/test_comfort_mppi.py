"""Comfort-only task preservation, actual history and zero candidate-WM calls."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.comfort_evaluator import ComfortTrajectoryEvaluator

def configuration():
    return dict(enabled=True,steering_rate_scale=.6,steering_jerk_scale=8.5,steering_delta_bound=.043,
        path_extra_tolerance_m=.045,progress_loss_tolerance_m=.045,heading_tolerance_rad=.013)

def prepare():
    e=ComfortTrajectoryEvaluator(configuration(),control_dt=.1,device='cpu')
    v=SimpleNamespace(current_action=np.array([0.,-.5]),position=np.array([0.,0.]),velocity=np.array([8.,0.]),heading_theta=0.,LENGTH=4.515)
    c=e.context(v,np.array([[0.,0.],[20.,0.]]),width=1.852,wheelbase=2.46894,max_steering=.6981317)
    return e,v,c

def test_comfort_keeps_exact_sac_throttle_and_hplusone_padding():
    e,_,_=prepare();b=torch.zeros(11,2);b[:,1]=torch.linspace(-1.,1.,11)
    r=e.project(torch.rand(64,11,2)*2-1,b)
    assert torch.equal(r[:,:,1],b[None,:,1].expand(64,-1))
    assert torch.equal(r[:,-1],b[-1].expand(64,-1))
    assert (r[:,:,0]-b[None,:,0]).abs().max()<=.0430001

def test_history_reads_executed_controls_and_repeated_probe_does_not_advance():
    e,v,_=prepare();v.position[0]=.8;v.current_action=np.array([.2,-.8])
    c=e.context(v,np.array([[0.,0.],[20.,0.]]),width=1.852,wheelbase=2.46894,max_steering=.6981317)
    d=e.context(v,np.array([[0.,0.],[20.,0.]]),width=1.852,wheelbase=2.46894,max_steering=.6981317)
    assert np.array_equal(c.previous_two_actions,np.array([[0.,-.5],[.2,-.8]]))
    assert np.array_equal(c.previous_two_actions,d.previous_two_actions)
    e.reset();assert e._observed_action is None

def test_comfort_plan_never_calls_world_model_and_retains_budget():
    e,_,c=prepare()
    class Forbidden:
        def __getattr__(self,name):raise AssertionError('Comfort must not call candidate WM/Q')
    p=LocalMPPI(Forbidden(),dict(objective_mode='comfort',num_samples=64,num_elites=8,iterations=4,record_candidates=True),horizon=10,action_dim=2,device='cpu')
    p.reset(1);b=torch.zeros(11,2);b[::2,0]=.1;b[1::2,0]=-.1;b[:,1]=-.5
    r=p.plan(torch.zeros(1,512),b,eval_mode=True,comfort_evaluator=e,comfort_context=c)
    assert len(r.iterations)==4 and all(x['candidate_count']==65 and x['elite_count']==8 for x in r.iterations)
    assert torch.equal(r.selected_sequence[:,1],b[:,1])
    assert r.metrics['wm_evaluated_candidates']==0 and r.metrics['comfort_wm_rollout_calls']==0
    assert r.metrics['comfort_selected_cost']<=r.metrics['comfort_baseline_cost']
    assert torch.isnan(r.selected_wm_value) and r.evaluation['terminal_q_used'] is False

def test_smooth_baseline_is_kept_without_false_optimization_claim():
    e,_,c=prepare();p=LocalMPPI(None,dict(objective_mode='comfort'),horizon=10,action_dim=2,device='cpu')
    b=torch.zeros(11,2);b[:,1]=-.5;r=p.plan(torch.zeros(1,512),b,eval_mode=True,comfort_evaluator=e,comfort_context=c)
    assert r.baseline_selected and torch.equal(r.selected_sequence,b) and r.metrics['comfort_gain']==0

def test_physics_only_validated_one_step_and_common_baseline_context():
    e,_,c=prepare();old=e.motion.rollout;shapes=[]
    def check(a,c):shapes.append(a.shape);return old(a,c)
    e.motion.rollout=check;b=torch.zeros(11,2);b[:,1]=-.5;r=e.evaluate(b[None].repeat(3,1,1),b,c)
    assert all(s[1]==2 for s in shapes) and r['feasible'].all()
    assert torch.equal(r['score'],r['score'][0].expand(3))

def test_throttle_change_task_infeasible_even_if_comfort_would_improve():
    e,_,c=prepare();b=torch.zeros(11,2);b[:,1]=-.5;a=b[None].clone();a[0,0,1]=.2
    assert not e.evaluate(a,b,c)['feasible'][0]

@pytest.mark.parametrize('name',['steering_rate_scale','steering_jerk_scale','path_extra_tolerance_m'])
def test_missing_development_calibration_rejected(name):
    cfg=configuration();del cfg[name]
    with pytest.raises(ValueError):ComfortTrajectoryEvaluator(cfg,control_dt=.1,device='cpu')

def test_comfort_requires_explicit_context_and_evaluator():
    p=LocalMPPI(None,dict(objective_mode='comfort'),horizon=10,action_dim=2,device='cpu')
    with pytest.raises(ValueError):p.plan(torch.zeros(1,512),torch.zeros(11,2),eval_mode=True)
