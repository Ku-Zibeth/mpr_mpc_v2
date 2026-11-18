"""Small invariants for H1 labels, aliases and unchanged candidate controls."""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from study import representative_roles,discounted_return,extended_environment

def test_dedup_keeps_baseline_and_preserves_selected_role_alias():
    data=[dict(role='A0',action=[.1,.2]),dict(role='A1',action=[.1,.2]),dict(role='A2',action=[.1,.2001])]
    kept,aliases=representative_roles(data,1e-5)
    assert len(kept)==2 and aliases==dict(A0='A0',A1='A0',A2='A2')
    assert kept[0]['roles']==['A0','A1']

def test_selected_mean_is_not_replaced_by_nearest_sample():
    data=[dict(role='A0',action=[0.,0.]),dict(role='A1',action=[.1,.2]),dict(role='A3',action=[.11,.21])]
    kept,_=representative_roles(data,1e-5)
    assert np.array_equal(kept[1]['action'],data[1]['action']) and len(kept)==3

@pytest.mark.parametrize('action',[[np.nan,0],[0,np.inf],[1.001,0],[0,0,0]])
def test_invalid_physical_actions_fail_before_environment_execution(action):
    with pytest.raises(AssertionError):representative_roles([dict(role='A0',action=action)],1e-5)

def test_rtg_starts_at_intervention_and_keeps_native_early_termination():
    assert discounted_return([2.])==2.
    assert np.isclose(discounted_return([2.,3.]),2+.995*3.)
    assert not np.isclose(discounted_return([100.,2.,3.]),discounted_return([2.,3.]))

def test_diagnostic_does_not_initialize_unused_gym_rng():
    from types import SimpleNamespace
    class Sim:
        _np_random=None
        episode_step=49
        engine=SimpleNamespace(managers={},get_objects=lambda:{})
        @property
        def np_random(self):raise AssertionError('Reading this property would initialize entropy-seeded RNG')
    sim=Sim();value=extended_environment(SimpleNamespace(unwrapped_metadrive=sim))
    assert value['values']['environment_rng'] is None and sim._np_random is None

def test_bootstrap_weights_complete_maps_not_number_of_rows():
    import pandas as pd
    from analyze import bootstrap_map
    frame=pd.DataFrame(dict(scene=[1,2,2,2],gain=[10.,0.,0.,0.]))
    actual=bootstrap_map(frame,['gain'])['gain']
    assert actual['mean']==5. and np.array_equal(actual['ci95'],[0.,10.])
    assert actual['leave_one_map_out']=={1:0.,2:10.}
