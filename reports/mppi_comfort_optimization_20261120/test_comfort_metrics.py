import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from metrics import comfort_metrics
from comfort_study import clustered

def test_constant_acceleration_and_linear_steering_have_correct_physical_derivatives():
    t=np.arange(21)*.1;states=np.column_stack((t,np.zeros(21),t,np.zeros(21)))
    actions=np.column_stack((np.arange(20)*.01,np.ones(20)))
    metrics,_=comfort_metrics(actions,states,.1,.7)
    assert np.isclose(metrics['steering_rate_rms_raw'],.07)
    assert metrics['steering_jerk_rms_raw']<1e-12
    assert np.isclose(metrics['longitudinal_acceleration_rms_raw'],1.)
    assert metrics['longitudinal_jerk_rms_raw']<1e-12
    assert metrics['lateral_acceleration_rms_raw']==0

def test_braking_count_is_event_count_not_number_of_bad_samples():
    states=np.column_stack((np.arange(11),np.zeros(11),10-np.arange(11),np.zeros(11)))
    actions=np.column_stack((np.zeros(10),-np.ones(10)))
    m,_=comfort_metrics(actions,states,.1,.7,braking_threshold=5)
    assert m['hard_braking_events']==1

def test_map_bootstrap_preserves_whole_map_and_does_not_count_seeds_as_maps():
    result=clustered(pd.DataFrame(dict(scene=[1,2,2,2],gain=[10.,0.,0.,0.])),['gain'])['gain']
    assert result['mean']==5 and np.array_equal(result['ci95'],[0.,10.])
