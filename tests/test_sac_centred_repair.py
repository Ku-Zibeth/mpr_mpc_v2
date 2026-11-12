"""Targeted certification regressions; synthetic evidence is never deployment evidence."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from shapely.geometry import box
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.mppi_safety import MPPISafetyContext, MPPISafetyFilter
from tests.test_v2_safety import context, obstacle
from tests.test_mppi_v3 import RankedEvaluator


def state(speed=5., obstacles=()):
    return MPPISafetyContext(**vars(context(speed=speed, road=box(-100,-100,200,100), obstacles=obstacles)),
        rear_axle_to_center=1.2, brake_action_mapping_verified=True)


def filter():
    return MPPISafetyFilter(dict(two_layer=True, staged=True, center_bicycle=True, center_model_guard=True,
        substeps=2, max_lateral_acceleration=6., model_margin=.05, road_margin=.05), control_dt=.1)


def evidence():
    return dict(deceleration_lower_mps2=2., acceleration_upper_mps2=4., validated_prediction_time_s=10.,
                domain=dict(max_curvature_inv_m=2., max_speed_mps=20.))


def test_adaptive_horizon_covers_stopping_and_preserves_committed_prefix():
    f=filter(); c=state(speed=8.); actions=torch.zeros(1,11,2)
    with patch.object(f,'braking_domain',return_value=((evidence(),'test'), 'validated_domain')):
        r=f.certify(actions,c)
    assert r.valid.all() and r.safety_horizon_steps>10
    witness=r.stopping_witnesses[0]
    assert torch.equal(witness[:1],actions[0,:1]) and (witness[1:,1]==-1).all()
    assert r.hold_braking_valid.all() and not r.safety_unknown.any()


def test_unknown_evidence_and_near_collision_cannot_authorize():
    f=filter(); a=torch.zeros(1,11,2); r=f.certify(a,state())
    assert r.near_valid.all() and not r.valid.any() and r.safety_unknown.all()
    hit=f.certify(a,state(obstacles=(obstacle(10.2,10.),)))
    assert hit.near_collision.all() and not hit.valid.any() and not hit.safety_unknown.any()
    limited=evidence();limited['domain']['max_speed_mps']=5.
    with patch.object(f,'braking_domain',return_value=((limited,'limited'), 'validated_domain')):
        blocked=f.certify(a,state())
    assert not blocked.valid.any() and blocked.certification_reason=='prefix_exits_braking_speed_domain'


def test_center_and_legacy_must_certify_the_same_stopping_witness():
    f=filter(); c=state(); c.lattice_actions=torch.zeros(11,2)
    # Each model approves a DIFFERENT branch. AND(exists) is insufficient.
    checker=SimpleNamespace(_strict_check=lambda a,c:SimpleNamespace(
        valid=torch.tensor([True,False]) if c.rear_axle_to_center else torch.tensor([False,True]),
        speeds=torch.zeros(2,2)))
    f._brake_filters['test']=checker
    with patch.object(f,'braking_domain',return_value=((evidence(),'test'), 'validated_domain')):
        r=f.certify(torch.zeros(1,11,2),c)
    assert not r.valid.any() and not r.hold_braking_valid.any() and not r.guided_braking_valid.any()


def fake_result(actions, valid):
    n=len(actions); zeros=torch.zeros(n,dtype=torch.bool)
    steer=actions[:,0,0].cpu()
    return SimpleNamespace(valid=valid,near_valid=torch.ones(n,dtype=torch.bool),full_valid=valid.clone(),
        near_collision=zeros,near_offroad=zeros,collision=~valid,offroad=zeros,kinematic_violation=zeros,
        safety_unknown=zeros,center_model_valid=valid,legacy_model_valid=valid,
        center_first_failure_substep=None,legacy_first_failure_substep=None,
        certification_reason='test',boundary_margin=1-(steer-.02).abs(),min_clearance=torch.full((n,),10.),
        route_progress_estimate=torch.ones(n),terminal_speed=torch.ones(n),
        stopping_witnesses=[a.clone() for a in actions.cpu()],evidence_sha256='test',safety_horizon_steps=30,
        hold_braking_valid=valid,guided_braking_valid=zeros,stopping_time_s=3.,road_rejection_kind='none')


def planner():
    p=LocalMPPI(RankedEvaluator(lambda a:100*a[:,0,0]),dict(num_samples=4,num_elites=2,iterations=4),
        horizon=3,action_dim=2,device='cpu')
    p.candidate_safety.preview=lambda a,c:fake_result(a,torch.ones(len(a),dtype=torch.bool))
    return p


def test_normal_unqualified_value_keeps_exact_sac_even_when_q_prefers_change():
    p=planner();p.candidate_safety.certify=lambda a,c:fake_result(a,torch.ones(len(a),dtype=torch.bool))
    base=torch.zeros(4,2);r=p.plan(torch.zeros(1,2),base,eval_mode=True,safety_context=state())
    assert torch.equal(r.selected_sequence,base) and r.baseline_selected and r.safety_certificate['valid']
    assert r.metrics['iterations_completed']==0 and r.metrics['value_ranking_validated']==0


def test_tail_repair_uses_certified_geometry_not_unvalidated_q():
    p=planner();base=torch.zeros(4,2);low=base+.02;high=base+.08
    p._structured=lambda *a:[('sac',base)]
    p._candidate_batch=lambda *a:torch.stack((base,low,high,low))
    p.candidate_safety.certify=lambda a,c:fake_result(a,a[:,0,0].cpu()>0)
    r=p.plan(torch.zeros(1,2),base,eval_mode=True,safety_context=state())
    assert r.decision_metadata['decision_state_name']=='TAIL_VIABILITY_REPAIR'
    assert torch.equal(r.selected_sequence,low) and r.safety_certificate['valid']
    assert r.metrics['tail_viability_repair_success']==1 and r.metrics['iterations_completed']==4


def test_uncertified_candidates_cannot_update_distribution_or_execute():
    p=planner();base=torch.zeros(4,2)
    p.candidate_safety.certify=lambda a,c:fake_result(a,torch.zeros(len(a),dtype=torch.bool))
    r=p.plan(torch.zeros(1,2),base,eval_mode=True,safety_context=state())
    assert r.safety_failure and not r.safety_certificate['valid']
    assert torch.equal(r.initial_mean,r.final_mean) and torch.equal(r.initial_std,r.final_std)
    assert r.metrics['elite_update_skipped']==4 and r.metrics['execution_authorized']==0


def test_ar_noise_preserves_action_bounds_and_temporal_correlation():
    p=LocalMPPI(RankedEvaluator(),dict(num_samples=64,num_elites=8,iterations=4,noise_rho=.8),
        horizon=20,action_dim=2,device='cpu')
    base=torch.zeros(21,2);a=p._candidate_batch(base,torch.full_like(base,.02),base)
    assert torch.equal(a[0],base) and (a.abs()<=p.max_delta).all()
    noise=a[1:].numpy()
    assert np.corrcoef(noise[:,:-1].ravel(),noise[:,1:].ravel())[0,1]>.6
