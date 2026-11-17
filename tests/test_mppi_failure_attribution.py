import numpy as np
import pytest
from tools.mppi_failure_attribution import ranking_metrics, scene_bootstrap, distribution_from_indices, replanning_error


def test_return_ties_are_excluded_and_score_ties_half_credit():
    r = ranking_metrics([0, 0, 1], [0, .5, 3])
    assert r['eligible_pairs'] == 2 and r['pair_accuracy'] == 1
    assert ranking_metrics([0, 0], [0, 3])['pair_accuracy'] == .5
    assert r['equal_fraction'] == .5 and r['top1_regret'] == 0
    with pytest.raises(ValueError): ranking_metrics([np.nan], [0])


def test_bootstrap_weights_maps_not_origins_or_pairs():
    a = [dict(scenario=1, x=0), dict(scenario=2, x=10)]
    b = a + [dict(scenario=1, x=0)]*100
    assert scene_bootstrap(a, ['x']) == scene_bootstrap(b, ['x'])
    assert scene_bootstrap(b, ['x'])['x']['mean'] == 5


def test_distribution_and_replanning_use_actual_selected_source():
    a = np.zeros((9, 11, 2)); a[1:, :, 0] = .1
    mean, std, _, row = distribution_from_indices(a, np.zeros(9), a[0], a[0], [.01,.03], [.08,.25], np.arange(8))
    assert row['effective_sample_size'] == 8 and row['baseline_in_elite'] == 1
    assert np.all(std >= [.01,.03]) and np.isclose(mean[0,0], .0875)
    plans = np.zeros((4,3,11,2)); plans[:,2,1] = [1,2]
    error = replanning_error(plans, np.full(4,2), np.zeros((4,2)), 1)
    assert error.shape == (3,2) and np.all(error == [1,2])
