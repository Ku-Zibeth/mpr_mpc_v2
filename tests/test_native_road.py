"""Actual-box geometry and conditional optimization, no learned value claims."""
from dataclasses import replace
import numpy as np
import torch
from panda3d.core import NodePath,Vec3
from panda3d.bullet import BulletWorld,BulletGhostNode,BulletBoxShape
from types import SimpleNamespace
from metadrive.type import MetaDriveType
from mpr_mpc.planning.native_road import NativeRoadGeometry
from mpr_mpc.planning.local_mppi import LocalMPPI
from tests.test_tracking_mppi import evaluator

def line(y):
    return NativeRoadGeometry(np.array([[20.,y]]),np.array([[[1.,0.],[0.,1.]]]),np.array([[40.,.025]]),(MetaDriveType.LINE_SOLID_SINGLE_WHITE,),{})

def test_full_body_contacts_white_line_even_inside_large_road_polygon():
    g=line(1.)
    s=np.array([[0.,0.,8.,0.],[0.,.1,8.,0.]])
    d,_=g.clearance(s,4.515,1.852)
    assert d[0]>0 and d[1]<0

def test_rotated_body_front_corner_contact_is_not_just_center_distance():
    g=line(1.)
    d,_=g.clearance(np.array([[0.,0.,8.,.3]]),4.515,1.852)
    assert d[0]<0

def test_read_real_panda_shape_transform_preserves_axes_at_large_coordinates():
    world=BulletWorld();root=NodePath('root');node=BulletGhostNode(MetaDriveType.LINE_SOLID_SINGLE_WHITE)
    node.addShape(BulletBoxShape(Vec3(3.,.025,1.)));path=root.attachNewNode(node);path.setPos(10000.,-20000.,.5);path.setH(37.)
    world.attachGhost(node)
    engine=SimpleNamespace(physics_world=SimpleNamespace(static_world=world,dynamic_world=BulletWorld()))
    g=NativeRoadGeometry.from_engine(engine)
    assert len(g.centers)==1 and np.allclose(g.centers[0],[10000.,-20000.])
    assert abs(np.dot(*g.axes[0]))<1e-6
    d,_=g.clearance(np.array([[10000.,-20000.,0.,0.]]),4.515,1.852);assert d[0]<0

def test_empty_neighborhood_is_explicit_and_broad_phase_keeps_uncertainty_envelope():
    g=line(1.);near=g.nearby(np.array([0.,0.,8.,0.]),4.515,1.852,.5,3.,.25)
    assert len(near.centers)==1
    far=g.nearby(np.array([1000.,1000.,0.,0.]),4.515,1.852,.5,3.,.25)
    assert not len(far.centers)
    d,_=far.clearance(np.zeros((2,3,4)),4.515,1.852);assert d.shape==(2,3) and np.isinf(d).all()

def test_risk_only_mode_retains_exact_sac_without_useless_optimization():
    e,c=evaluator();e.cfg.update(native_road=True,native_risk_only=True,native_margin_m=.25)
    c=replace(c,native_road=line(20.));b=torch.zeros(11,2)
    p=LocalMPPI(None,dict(objective_mode='tracking'),horizon=10,action_dim=2,device='cpu')
    r=p.plan(torch.zeros(1,512),b,eval_mode=True,tracking_evaluator=e,tracking_context=c)
    assert r.baseline_selected and torch.equal(r.action,b[0]) and r.metrics['native_search_active']==0
    assert r.metrics['iterations_completed']==0 and r.metrics['wm_evaluated_candidates']==0

def test_active_risk_search_preserves_original_sampling_budget_and_current_pedal():
    e,c=evaluator();e.cfg.update(native_road=True,native_risk_only=True,native_margin_m=.25)
    c=replace(c,native_road=line(1.1));b=torch.zeros(11,2)
    p=LocalMPPI(None,dict(objective_mode='tracking'),horizon=10,action_dim=2,device='cpu');p.reset(1)
    r=p.plan(torch.zeros(1,512),b,eval_mode=True,tracking_evaluator=e,tracking_context=c)
    assert r.metrics['native_search_active']==1 and len(r.iterations)==4
    assert all(x['candidate_count']==65 and x['elite_count']==8 for x in r.iterations)
    assert r.action[1]==b[0,1] and r.metrics['native_selected_risk']<=r.metrics['native_baseline_risk']
