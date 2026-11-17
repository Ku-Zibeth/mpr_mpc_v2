"""Opt-in selection must share Q members, preserve mean behavior and execute a0."""
import torch
from unittest.mock import patch
from tests.test_mppi_value_evaluation import evaluator
from mpr_mpc.planning.local_mppi import LocalMPPI


def test_captured_members_restore_rng_and_reproduce_final_values_for_both_terminals():
    for terminal in ('candidate','policy_prior'):
        e=evaluator();a=torch.rand(3,4,2);z=torch.zeros(1,3)
        torch.manual_seed(19);normal=e.evaluate_mppi_candidates(z,a,terminal_action=terminal,compute_cost=False);rng=torch.random.get_rng_state()
        torch.manual_seed(19);captured=e.evaluate_mppi_candidates(z,a,terminal_action=terminal,compute_cost=False,capture_q_indices=True)
        assert torch.equal(rng,torch.random.get_rng_state())
        assert torch.equal(normal.total_value,captured.total_value)
        assert len(captured.q_member_indices)==2
        fixed=e.evaluate_mppi_candidates(z,a,terminal_action=terminal,compute_cost=False,q_aggregation='fixed_pair',q_indices=captured.q_member_indices)
        torch.testing.assert_close(fixed.total_value,captured.total_value)


def test_mean_elite_and_baseline_use_identical_final_members_without_extra_WM_rollout():
    for terminal in ('candidate','policy_prior'):
        results={}
        for selection in ('weighted_mean','best_final_elite'):
            e=evaluator();base=torch.zeros(4,2)
            cfg=dict(num_samples=8,num_elites=4,iterations=2,terminal_action=terminal,selection_mode=selection,record_candidates=True)
            p=LocalMPPI(e,cfg,horizon=3,action_dim=2,device='cpu')
            def candidates(*args):
                a=base[None].expand(9,-1,-1).clone();a[1:,:,0]=torch.linspace(.1,.8,8)[:,None]
                if terminal=='policy_prior':a[:,-1]=base[-1]
                return a
            p._candidate_batch=candidates
            torch.manual_seed(4)
            with patch.object(e,'rescore_cached_candidates',wraps=e.rescore_cached_candidates) as rescore:
                r=p.plan(torch.zeros(1,3),base,eval_mode=True)
                assert rescore.call_count==1
                assert rescore.call_args.kwargs['q_indices']==r.selection_candidates['q_member_indices']
            expected=r.selection_candidates['actions'][1 if selection=='weighted_mean' else 2]
            assert torch.equal(r.selected_sequence,expected) and torch.equal(r.action,expected[0])
            assert r.selection_candidates['selected_source']==('mean' if selection=='weighted_mean' else 'elite')
            assert r.selected_wm_value>r.baseline_wm_value
            assert r.metrics['wm_evaluated_candidates']==20 and e.agent.model.next_calls==9
            assert len(r.candidate_trace['actions'])==2 and all(x.shape==(9,4,2) for x in r.candidate_trace['actions'])
            if terminal=='policy_prior':assert (r.final_std[-1]==0).all() and torch.equal(r.final_mean[-1],base[-1])
            results[selection]=r
        assert torch.equal(results['weighted_mean'].final_mean,results['best_final_elite'].final_mean)


def test_no_gain_keeps_baseline_and_recording_does_not_change_weighted_mean():
    base=torch.zeros(4,2);saved=[]
    for extra in ({},{'selection_mode':'weighted_mean'},{'selection_mode':'weighted_mean','record_candidates':True}):
        e=evaluator();cfg=dict(num_samples=8,num_elites=4,iterations=2,**extra);p=LocalMPPI(e,cfg,horizon=3,action_dim=2,device='cpu');p.reset(17);torch.manual_seed(5)
        r=p.plan(torch.zeros(1,3),base,eval_mode=True);saved.append((r.selected_sequence,torch.random.get_rng_state()))
    assert all(torch.equal(saved[0][0],x[0]) and torch.equal(saved[0][1],x[1]) for x in saved[1:])
    for mode in ('weighted_mean','best_final_elite'):
        e=evaluator();p=LocalMPPI(e,dict(num_samples=8,num_elites=4,iterations=2,selection_mode=mode),horizon=3,action_dim=2,device='cpu')
        p._candidate_batch=lambda *args:base[None].expand(9,-1,-1).clone()
        r=p.plan(torch.zeros(1,3),base,eval_mode=True)
        assert r.baseline_selected and torch.equal(r.action,base[0])
        assert r.selection_candidates['selected_source']=='baseline'
