"""Regression tests for opt-in evidence-aware final selection; sampling remains native."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from mpr_mpc.planning.local_mppi import LocalMPPI

class Evaluator:
    def __init__(self):self.calls=[]
    def evaluate_mppi_candidates(self,z,actions,**kwargs):
        self.calls.append(len(actions));v=4*actions[:,0,0]+actions[:,0,1]
        return SimpleNamespace(total_value=v,discounted_reward=v,terminal_q=v*0,reward_sequence=v[:,None].expand(-1,10),timings={})

def solve(**options):
    evaluator=Evaluator();cfg=dict(num_samples=64,num_elites=8,iterations=4,initial_std=[.04,.12],min_std=[.01,.03],max_std=[.08,.25],**options)
    planner=LocalMPPI(evaluator,cfg,horizon=10,action_dim=2,device='cpu');planner.reset(17)
    result=planner.plan(torch.zeros(1,512),torch.zeros(11,2),eval_mode=True)
    return planner,evaluator,result

def test_unsupported_guard_preserves_baseline_but_does_not_skip_search():
    _,e,r=solve(final_selection_policy='baseline_guard',replacement_supported=False,final_selection_margin=13.235)
    assert torch.equal(r.selected_sequence,torch.zeros(11,2))
    assert e.calls==[65,65,65,65,2] and r.numerical_error is None
    assert r.metrics['replacement_supported']==0 and r.metrics['baseline_guard_blocked']==1

def test_zero_margin_supported_guard_matches_native_actions_and_rng():
    a,_,before=solve();b,_,after=solve(final_selection_policy='baseline_guard',replacement_supported=True,final_selection_margin=0.)
    assert torch.equal(before.selected_sequence,after.selected_sequence)
    assert torch.equal(a.generator.get_state(),b.generator.get_state())
    assert not after.baseline_selected

def test_positive_margin_can_preserve_baseline_without_claiming_numerical_failure():
    _,_,result=solve(final_selection_policy='baseline_guard',replacement_supported=True,final_selection_margin=100.)
    assert result.baseline_selected and result.numerical_error is None and result.metrics['baseline_guard_blocked']==1

def test_margin_boundary_is_strict_and_matches_frozen_h1_delta_definition():
    _,_,legacy=solve();delta=float(legacy.optimized_wm_value-legacy.baseline_wm_value)
    _,_,result=solve(final_selection_policy='baseline_guard',replacement_supported=True,final_selection_margin=delta)
    assert result.baseline_selected

@pytest.mark.parametrize('margin',[-1.,float('nan'),float('inf')])
def test_invalid_margin_rejected(margin):
    with pytest.raises(ValueError):solve(final_selection_policy='baseline_guard',final_selection_margin=margin)

def test_unknown_policy_rejected():
    with pytest.raises(ValueError):solve(final_selection_policy='misspelled')

def test_original_defaults_remain_native():
    p,_,r=solve();assert p.final_selection_policy=='legacy' and not r.baseline_selected

def test_map_bootstrap_does_not_treat_repeated_seeds_as_independent_maps():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parent))
    from h1_analysis import clustered
    import pandas as pd
    result=clustered(pd.DataFrame(dict(scene=[1,2,2,2],gain=[10.,0.,0.,0.])),['gain'])['gain']
    assert result['mean']==5 and np.array_equal(result['ci95'],[0.,10.]) and result['lomo']=={1:0.,2:10.}
