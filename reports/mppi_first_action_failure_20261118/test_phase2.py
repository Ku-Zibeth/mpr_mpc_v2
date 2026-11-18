"""Minimal regressions for the new offline audit/statistics contracts."""
import csv
from pathlib import Path

import numpy as np
import pytest
from tools.first_action_failure import (assert_submitted_actions, cluster_bootstrap,
    discounted, isolated_state_inputs, oracle_values, paired_origins, ranking)


def test_real_branch_index_has_576_rows_and_separate_protocols():
    root = Path(__file__).parent
    with (root / 'branch_integrity.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 576
    assert sum(r['exact_replay'] == 'True' for r in rows) == 1
    assert sum(r['takeover'] != 'baseline' for r in rows) == 4
    assert len({r['scene_id'] for r in rows}) == 18
    assert all(r['checks_passed'] == 'True' for r in rows)


def test_reward_time_index_and_each_horizon():
    rewards = [2., 4., 8., 16., 32., 64., 128., 256., 512., 1024., 3.]
    done = [False] * 10 + [True]
    for h in [1, 2, 4, 10]:
        a = discounted(rewards, done, done, [False]*11, h, gamma=.5)
        assert a['prefix'] == 2*h
        assert a['total'] == pytest.approx(a['prefix'] + .5**h * a['after_prefix'])
        assert a['after_first'] == pytest.approx((a['total'] - 2.) / .5)


def test_native_termination_and_truncation_have_no_invented_tail():
    a = discounted([2., 5.], [False, True], [False, False], [False, True], 10)
    assert a['early_terminal'] and a['prefix_steps'] == 2
    assert a['after_prefix'] == 0
    assert a['total'] == pytest.approx(2 + .995*5)
    with pytest.raises(ValueError, match='done='):
        discounted([2., 5.], [False, True], [False, False], [False, False], 1)


def test_submitted_action_mapping_order_and_bounds():
    sequence = np.array([[.1, .7], [.2, .8], [.3, .9]], np.float32)
    assert_submitted_actions(sequence, sequence[:2], 10)  # native early stop
    with pytest.raises(ValueError, match='prefix'):
        assert_submitted_actions(sequence, sequence[:, ::-1], 1)
    with pytest.raises(ValueError, match='Invalid'):
        assert_submitted_actions(sequence, [[np.nan, .5]], 1)


def test_origin_pairing_rejects_other_maps_and_duplicate_keys():
    a = dict(scene_id=1, origin_id='1_50', horizon=1, pair_accuracy=.2, spearman=.1,
             kendall=.1, top1_regret=4.)
    b = a | dict(pair_accuracy=.7, top1_regret=2.)
    rows = paired_origins([a], [b])
    assert rows[0]['pair_accuracy_improvement'] == pytest.approx(.5)
    assert rows[0]['regret_improvement'] == 2
    with pytest.raises(ValueError, match='match'):
        paired_origins([a], [b | dict(scene_id=2)])
    with pytest.raises(ValueError, match='match'):
        paired_origins([a, a], [b, b])


def test_score_direction_ties_and_reverse_same_pairs():
    y, score = [0., .5, 4., 8.], [0., 0., 2., 1.]
    a, b = ranking(score, y), ranking(-np.array(score), y)
    assert a['raw_pairs'] == 6 and a['effective_pairs'] == 5 and a['tied_pairs'] == 1
    assert a['pair_accuracy'] + b['pair_accuracy'] == 1
    assert a['top1_regret'] == 4 and b['top1_regret'] == 8


def test_oracle_is_measured_subset_not_model_selected_or_random():
    result = oracle_values([10., 18., 5.], [0., -1., 2.], random_index=1)
    assert result['oracle_gain'] == 8 and result['random_gain'] == 8
    assert result['model_gain'] == -5 and result['top1_regret'] == 13
    assert result['positive_fraction'] == .5 and result['selected_negative'] == 1


def test_cluster_bootstrap_weights_maps_not_origins_or_pairs():
    rows = [dict(scene_id=1, metric=1.)]*20 + [dict(scene_id=2, metric=0.)]
    a = cluster_bootstrap(rows, ['metric'], samples=2000, seed=9)
    b = cluster_bootstrap(rows, ['metric'], samples=2000, seed=9)
    assert a == b and a['metric']['mean'] == .5
    assert a['metric']['maps'] == 2 and a['metric']['ci95'] == [0., 1.]


def test_current_qb_cannot_become_mixed_time_future535():
    current = np.arange(537, dtype=np.float32)
    pred, actual = isolated_state_inputs(current, np.zeros(512), np.ones(512), 'current_qb')
    assert np.array_equal(pred, current) and np.array_equal(actual, current)
    actual[0] = -999
    assert pred[0] == 0 and current[0] == 0  # separate copies, no future mutation


def test_endpoint_swap_is_isolated_and_horizon_matched():
    zp, zr = np.zeros((8, 512)), np.ones((8, 512))
    pred, actual = isolated_state_inputs(None, zp, zr, 'terminal_td_q')
    actual[0, 0] = 2
    assert pred[0, 0] == 0 and zr[0, 0] == 1
    with pytest.raises(ValueError, match='same horizon'):
        isolated_state_inputs(None, np.zeros((7, 512)), zr, 'terminal_td_q')


def test_original_and_experiment_default_planner_outputs_identical():
    root = Path(__file__).parent
    with np.load(root/'planner_parity_0.npz') as original, np.load(root/'planner_parity_1.npz') as current:
        assert original.files == current.files
        for name in original.files:
            assert np.array_equal(original[name], current[name]), name
        assert not bool(current['numerical_error'])
