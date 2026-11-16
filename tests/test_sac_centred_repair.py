"""Targeted certification regressions; synthetic evidence is never deployment evidence."""
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from shapely.geometry import box
from mpr_mpc.planning.mppi_safety import MPPISafetyContext, MPPISafetyFilter
from tests.test_v2_safety import context, obstacle


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
