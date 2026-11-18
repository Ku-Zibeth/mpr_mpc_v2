"""Small regressions for the one Cartesian-virtual-speed correction."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mpr_mpc.planning.time_alignment import align_frenet_path
from mpr_mpc.planning.trajectory_adapter import LatticeActionAdapter
from mpr_mpc.planning.local_mppi import LocalMPPI


def path():
    t=np.arange(0,1.01,.1)
    return SimpleNamespace(t=t,xy=np.c_[10*t,3*t],s_d=np.full(len(t),10.),target_speed=12.)


def test_cartesian_ground_speed_differs_from_longitudinal_frenet_component():
    p=path();legacy=align_frenet_path(p,required_steps=11,control_dt=.1)
    actual=align_frenet_path(p,required_steps=11,control_dt=.1,cartesian_speed=True)
    assert np.allclose(legacy.speeds,10.)
    assert np.allclose(actual.speeds,np.sqrt(109),atol=1e-12)
    assert np.array_equal(actual.times,legacy.times) and np.array_equal(actual.positions,legacy.positions)
    assert np.array_equal(actual.headings,legacy.headings)


def test_curved_offset_path_speed_matches_cartesian_curve_not_reference_arclength():
    t=np.arange(0,1.01,.1)
    p=SimpleNamespace(t=t,xy=np.c_[8*np.cos(t),8*np.sin(t)],s_d=np.full(len(t),10.))
    a=align_frenet_path(p,required_steps=11,control_dt=.1,cartesian_speed=True)
    assert np.allclose(a.speeds,8.,atol=.03)


def test_default_and_explicit_false_are_bit_identical():
    p=path()
    a=align_frenet_path(p,required_steps=14,control_dt=.1)
    b=align_frenet_path(p,required_steps=14,control_dt=.1,cartesian_speed=False)
    for name in ['positions','headings','speeds','times']:assert np.array_equal(getattr(a,name),getattr(b,name))
    assert np.all(a.speeds==10.)


def test_held_endpoint_has_zero_virtual_velocity_only_after_its_time():
    p=path();a=align_frenet_path(p,required_steps=14,control_dt=.1,cartesian_speed=True)
    assert np.allclose(a.positions[11:],a.positions[10]) and np.all(a.speeds[11:]==0)
    assert a.speeds[10]>0


def test_real_first_action_unchanged_only_virtual_tail_speed_changes():
    class Tracker:
        def _track_path(self,vehicle,p):return [0.,.35*(p.target_speed-vehicle.speed)],0
    adapter=LatticeActionAdapter(Tracker());v=SimpleNamespace(speed=10.,position=np.zeros(2),heading_theta=0.)
    a=adapter.path_to_actions(path(),v,required_steps=11,control_dt=.1,device='cpu')
    b=adapter.path_to_actions(path(),v,required_steps=11,control_dt=.1,device='cpu',cartesian_virtual_speed=True)
    assert torch.equal(a[0],b[0]) and not torch.equal(a[1:],b[1:])


def test_actual_saved_trajectory_replay_changes_only_speed_field():
    root=Path(__file__).parent;source=next((root/'episodes').glob('B_*/snapshot_50.npz'))
    with np.load(source) as f:p=SimpleNamespace(t=f['refined_t'],xy=f['refined_xy'],s_d=f['refined_speed'])
    a=align_frenet_path(p,required_steps=11,control_dt=.1)
    b=align_frenet_path(p,required_steps=11,control_dt=.1,cartesian_speed=True)
    assert np.array_equal(a.positions,b.positions) and np.array_equal(a.headings,b.headings)
    assert np.all(b.speeds>=0) and np.isfinite(b.speeds).all()


def test_raw_sampler_audit_does_not_change_actions_or_rng():
    spec=importlib.util.spec_from_file_location('sampler_audit_for_test',Path(__file__).parent/'instrumentation.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    class Evaluator:
        def __init__(self):self.values=[]
        def evaluate_mppi_candidates(self,z,a,**kw):
            value=a[:,0,0];self.values.append(value.numpy().copy())
            return SimpleNamespace(total_value=value,discounted_reward=value,terminal_q=value*0,timings={})
    cfg=dict(num_samples=8,num_elites=4,iterations=4,initial_std=[.04,.12],min_std=[.01,.03],max_std=[.08,.25])
    one=LocalMPPI(Evaluator(),cfg,horizon=3,action_dim=2,device='cpu')
    evaluator=Evaluator();two=LocalMPPI(evaluator,cfg,horizon=3,action_dim=2,device='cpu');audit=mod.SamplerAudit(two)
    baseline=torch.zeros(4,2);baseline[:,1]=1.
    a=one.plan(torch.zeros(1,2),baseline,eval_mode=True);b=two.plan(torch.zeros(1,2),baseline,eval_mode=True)
    assert torch.equal(a.selected_sequence,b.selected_sequence)
    assert torch.equal(one.generator.get_state(),two.generator.get_state())
    before=two.generator.get_state().clone();rows=audit.rows({},evaluator.values,[(0,1)]*4)
    assert len(rows)==4 and all(r['exact_preclip_reconstruction'] for r in rows)
    assert torch.equal(before,two.generator.get_state())
