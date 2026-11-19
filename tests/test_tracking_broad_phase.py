"""Speed optimization must preserve all candidate scores and selected controls."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
from tests.test_tracking_mppi import evaluator
from mpr_mpc.planning.tracking_broad_phase import cull_zero_overlap_obstacles
from mpr_mpc.planning.local_mppi import LocalMPPI

def obstacle(position,velocity=(0.,0.)):
    return SimpleNamespace(position=np.array(position),velocity=np.array(velocity),heading=.2,length=4.5,width=1.8)

def test_approaching_far_object_and_horizon_boundary_are_retained():
    e,c=evaluator();c=replace(c,obstacles=(obstacle((50.,0.),(-100.,0.)),obstacle((1000.,0.))))
    result=cull_zero_overlap_obstacles(c,e)
    assert len(result.obstacles)==1 and result.obstacles[0] is c.obstacles[0]

def test_full_candidate_scores_and_distribution_are_exact_after_culling():
    rng=np.random.default_rng(6)
    for seed in range(5):
        e,c=evaluator();near=tuple(obstacle(rng.uniform([-3,-3],[15,3]),rng.uniform(-2,2,2)) for _ in range(5))
        far=tuple(obstacle(rng.uniform([100,100],[500,500]),rng.uniform(-2,2,2)) for _ in range(80))
        c=replace(c,obstacles=near+far);short=cull_zero_overlap_obstacles(c,e);assert len(short.obstacles)<len(c.obstacles)
        b=torch.zeros(11,2);u=b.repeat(65,1,1);u[1:]=torch.tensor(rng.uniform(-.08,.08,(64,11,2)),dtype=torch.float32);u=e.project(u,b)
        full=e.evaluate(u,b,c);fast=e.evaluate(u,b,short)
        for key in full:assert np.array_equal(full[key],fast[key]),key
        p=LocalMPPI(None,dict(objective_mode='tracking'),horizon=10,action_dim=2,device='cpu');p.reset(seed)
        a=p.plan(torch.zeros(1,512),b,eval_mode=True,tracking_evaluator=e,tracking_context=c);p.reset(seed)
        z=p.plan(torch.zeros(1,512),b,eval_mode=True,tracking_evaluator=e,tracking_context=short)
        assert torch.equal(a.selected_sequence,z.selected_sequence) and torch.equal(a.final_mean,z.final_mean) and torch.equal(a.final_std,z.final_std)

def test_factored_native_sat_preserves_full_arithmetic_exactly():
    from mpr_mpc.planning.native_road import NativeRoadGeometry
    from mpr_mpc.planning.tracking_broad_phase import exact_native_clearance
    rng=np.random.default_rng(28);heading=rng.uniform(-np.pi,np.pi,97);forward=np.column_stack((np.cos(heading),np.sin(heading)));normal=np.column_stack((-np.sin(heading),np.cos(heading)))
    g=NativeRoadGeometry(rng.uniform(-20,20,(97,2)),np.stack((forward,normal),axis=1),rng.uniform(.02,5,(97,2)),tuple('white' if i%2 else 'yellow' for i in range(97)),{})
    s=rng.uniform(-10,10,(65,25,4));a,aa=g.clearance(s,4.515,1.852);b,bb=exact_native_clearance(g,s,4.515,1.852)
    assert np.array_equal(a,b) and all(np.array_equal(aa[k],bb[k]) for k in aa)
