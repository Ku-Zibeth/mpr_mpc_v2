import numpy as np
from tools.mppi_diagnosis import relative_dynamics,grouped_folds,effective_df_ridge,angle_difference


def test_relative_dynamics_removes_common_error_and_identifies_direction_flip():
    actual=np.array([[1.,2.],[2.,2.],[1.,3.]])
    exact=relative_dynamics(actual+100.,actual)
    assert np.allclose(exact['error_l2'],0.)
    assert np.isnan(exact['cosine'][0]) and np.allclose(exact['cosine'][1:],1.)
    flipped=relative_dynamics(2*actual[0]-actual,actual)
    assert np.allclose(flipped['cosine'][1:],-1.)
    assert np.allclose(flipped['error_l2'][1:],2.)
    assert abs(angle_difference(-np.pi+.01,np.pi-.01)-.02)<1e-10


def test_scene_folds_and_label_free_equal_capacity_probe():
    scenes=np.repeat(np.arange(10),3);folds=grouped_folds(scenes,folds=5)
    assert sorted(s for f in folds for s in f)==list(range(10))
    assert all(not set(folds[i])&set(folds[j]) for i in range(5) for j in range(i))
    rng=np.random.default_rng(1);x=rng.normal(size=(80,20));y=x[:,0]-x[:,1]
    coef,info=effective_df_ridge(x,y,np.ones(80),degrees=4)
    flipped,other=effective_df_ridge(x,-y,np.ones(80),degrees=4)
    assert np.allclose(coef,-flipped) and info['alpha']==other['alpha']
    assert abs(info['effective_df']-4)<1e-6


def test_iteration_ablation_changes_no_other_sampling_or_value_options():
    from copy import deepcopy
    import pytest
    from planning.mppi_ablation import iterations_ablation_config
    original = dict(iterations=4,num_samples=64,num_elites=8,noise_rho=[.8,.8],
                    initial_std=[.04,.12],min_std=[.01,.03],max_std=[.08,.25],
                    q_aggregation='random_pair',temperature=.5,cost_weight=0.)
    for count in (1,2,4):
        result=iterations_ablation_config(original,count)
        expected=deepcopy(original);expected['iterations']=count
        assert result==expected
        result['noise_rho'][0]=0.
        assert original['noise_rho']==[.8,.8]
    assert original['iterations']==4
    with pytest.raises(ValueError):iterations_ablation_config(original,3)
