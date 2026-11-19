"""Physical units, original-tracker parity, exact first action and score provenance."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from shapely.geometry import box
from mpr_mpc.planning.local_mppi import LocalMPPI,MPPI_DIAGNOSTIC_KEYS
from mpr_mpc.planning.tracking_dynamics import TrackingVehicle,rollout
from mpr_mpc.planning.tracking_feedback import BatchedLatticeTracker
from mpr_mpc.planning.tracking_evaluator import TrackingTrajectoryEvaluator,TrackingContext
from lattice.frenet_metadrive import FRENET_DEFAULT_CONFIG,MetaDriveFrenetController

def vehicle():return TrackingVehicle(2.46894,1.4166,np.deg2rad(40),1100,772.849731,102.849739,.9,.02,4.515,1.852)
def tracker_fixture(curve=False):
    x=np.linspace(0,30,81);y=.01*x*x if curve else np.zeros_like(x)
    path=SimpleNamespace(xy=np.column_stack((x,y)),target_speed=8.)
    controller=MetaDriveFrenetController(dict(FRENET_DEFAULT_CONFIG))
    return controller,path,BatchedLatticeTracker(controller,path)
def evaluator(feedback=True):
    cfg=dict(enabled=True,prediction_steps=5,lateral_scale=.36,heading_scale=.085,steering_delta_bound=.08,motion_error_p95=.18,action_scale=.08,
        motion_model='physical_uncapped',motion_coefficients=dict(drive_gain=.915,brake_gain=.762,slip_gain=.724),feedback_rollout=feedback)
    e=TrackingTrajectoryEvaluator(cfg,control_dt=.1);_,path,fb=tracker_fixture()
    c=TrackingContext(np.array([0.,0.,8.,0.]),vehicle(),path.xy,np.zeros(2),(),box(-10,-5,60,5),fb)
    return e,c

@pytest.mark.parametrize('curve',[False,True])
def test_batched_feedback_reproduces_unmodified_real_tracker(curve):
    original,path,batched=tracker_fixture(curve);rng=np.random.default_rng(37)
    states=np.column_stack((rng.uniform(0,25,40),rng.uniform(-2,5,40),rng.uniform(0,15,40),rng.uniform(-np.pi,np.pi,40)))
    actual=batched(states)
    expected=np.array([original._track_path(SimpleNamespace(position=s[:2],speed=s[2],heading_theta=s[3]),path)[0] for s in states])
    assert np.allclose(actual,expected,rtol=0,atol=2e-14)

def test_engine_force_and_brake_impulse_use_different_time_units():
    v=vehicle();u=np.zeros((2,1,2));u[0,0,1]=1;u[1,0,1]=-1
    result=rollout(u,np.array([0.,0.,20.,0.]),v,model='physical_uncapped')
    assert np.isclose(result[0,1,2]-20,4*v.engine_force/v.mass*.1)
    assert np.isclose(20-result[1,1,2],4*v.brake_impulse/(v.mass*.02)*.1)

def test_substep_rollout_and_control_endpoint_rollout_agree():
    rng=np.random.default_rng(2);u=rng.uniform(-.4,.4,(4,5,2));s=np.array([1.,2.,8.,.2])
    a=rollout(u,s,vehicle(),return_substeps=False);b=rollout(u,s,vehicle(),return_substeps=True)
    assert np.array_equal(a,b[:,::5]) and (b[:,:,2]>=0).all()

def test_braking_never_reverses_and_center_velocity_has_steering_slip():
    u=np.ones((1,5,2));u[:,:,1]=-1;s=np.array([0.,0.,.1,0.])
    r=rollout(u,s,vehicle(),model='physical_uncapped');assert (r[:,:,2]>=0).all()
    u=np.zeros((1,1,2));u[:,:,0]=.2;r=rollout(u,np.array([0.,0.,8.,0.]),vehicle())
    assert r[0,1,1]>0 and r[0,1,3]>0

def test_feedback_first_pedal_is_exact_sac_but_future_controls_use_predicted_state():
    e,c=evaluator();b=torch.zeros(11,2);b[:,1]=1
    result=e.evaluate(b[None],b,c)
    assert result['applied_actions'][0,0,1]==1 and result['applied_actions'][0,1,1]<1

def test_nominal_straight_reference_is_not_changed_to_claim_optimization():
    e,c=evaluator();b=torch.zeros(11,2)
    p=LocalMPPI(None,dict(objective_mode='tracking'),horizon=10,action_dim=2,device='cpu');p.reset(2)
    result=p.plan(torch.zeros(1,512),b,eval_mode=True,tracking_evaluator=e,tracking_context=c)
    assert torch.equal(result.action,b[0]) and result.baseline_selected
    assert result.metrics['wm_evaluated_candidates']==0 and result.metrics['iterations_completed']==4
    assert all(x['candidate_count']==65 and x['elite_count']==8 for x in result.iterations)
    assert torch.isnan(result.selected_wm_value) and result.evaluation['original_SAC_future_replanning_simulated'] is False

def test_body_road_and_obstacle_terms_are_real_geometry_not_terminal_q():
    e,c=evaluator(False);b=torch.zeros(11,2)
    good=e.evaluate(b[None],b,c)
    obstacle=SimpleNamespace(position=np.array([3.,0.]),velocity=np.zeros(2),heading=0.,length=4.,width=1.8)
    blocked=TrackingContext(c.state,c.vehicle,c.reference_path,c.previous_action,(obstacle,),c.road,c.feedback)
    bad=e.evaluate(b[None],b,blocked)
    assert bad['obstacle_cost'][0]>0 and bad['score'][0]<good['score'][0]

def test_scoring_requires_exact_baseline_comparison_and_explicit_context():
    e,c=evaluator();b=torch.zeros(11,2);u=b[None].clone();u[0,0,0]=.01
    with pytest.raises(ValueError):e.evaluate(u,b,c)
    p=LocalMPPI(None,dict(objective_mode='tracking'),horizon=10,action_dim=2,device='cpu')
    with pytest.raises(ValueError):p.plan(torch.zeros(1,512),b,eval_mode=True)

def test_diagnostic_keys_are_unique():assert len(MPPI_DIAGNOSTIC_KEYS)==len(set(MPPI_DIAGNOSTIC_KEYS))

def test_opt_in_pedal_search_is_bounded_and_first_execution_matches_candidate():
    e,c=evaluator();e.cfg.update(optimize_throttle=True,throttle_delta_bound=.25,speed_scale=1/.35)
    b=torch.zeros(11,2);u=b.repeat(2,1,1);u[1,:,1]=1
    u=e.project(u,b);assert u[1,0,1]==.25 and torch.equal(u[:,-1],b[-1].expand(2,2))
    result=e.evaluate(u,b,c)
    assert result['applied_actions'][1,0,1]==.25 and np.isfinite(result['speed_cost']).all()
    assert e.cfg['optimize_throttle'] and evaluator()[0].cfg.get('optimize_throttle',False) is False

def test_target_speed_is_original_curve_limited_target_not_a_new_goal():
    original,path,fb=tracker_fixture(True)
    states=np.array([[5.,0.,2.,0.],[15.,2.,10.,.2]])
    target=fb.target_speeds(states)
    assert np.all(target<=path.target_speed)
    assert np.allclose(fb(states)[:,1],np.clip(.35*(target-states[:,2]),-1,1),rtol=0,atol=1e-14)

def test_optional_prediction_resolution_preserves_exact_current_sac_action():
    e,c=evaluator();e.cfg['min_terminal_error_reduction_m']=10.
    b=torch.zeros(11,2);b[:,0]=.05
    p=LocalMPPI(None,dict(objective_mode='tracking'),horizon=10,action_dim=2,device='cpu')
    r=p.plan(torch.zeros(1,512),b,eval_mode=True,tracking_evaluator=e,tracking_context=c)
    assert r.baseline_selected and torch.equal(r.action,b[0])
    assert r.metrics['tracking_resolution_rejected_ratio']==1. and r.metrics['tracking_resolution_floor_m']==10.
