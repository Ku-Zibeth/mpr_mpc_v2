"""Continuation-state semantics, cached common scoring and frozen feature regressions."""
from unittest.mock import patch
from types import SimpleNamespace
import torch,pytest
from tests.test_mppi_value_evaluation import evaluator
from mpr_mpc.planning.local_mppi import LocalMPPI
from mpr_mpc.planning.sac_value import SACContinuationValue,load_sac_value,file_sha256


def test_state_value_never_reads_a_H_or_Q_and_keeps_frozen_SAC_features():
    e=evaluator();a=torch.zeros(2,4,2);a[1,:3]=.1
    old=e.evaluate_trajectory_consequence(torch.zeros(1,3),a,compute_cost=False).features.clone()
    e.attach_offline_sac_value(torch.nn.Linear(3,1))
    with patch.object(e.agent.model,'Q',side_effect=AssertionError('Q must not be used')),patch.object(e.agent.model,'pi',side_effect=AssertionError('pi must not be used')):
        r=e.evaluate_mppi_candidates(torch.zeros(1,3),a,terminal_value='sac_value',capture_ensemble=True,compute_cost=False)
        b=a.clone();b[:,-1]=.9
        s=e.evaluate_mppi_candidates(torch.zeros(1,3),b,terminal_value='sac_value',compute_cost=False)
    assert torch.equal(r.total_value,s.total_value) and r.terminal_actions is None
    torch.testing.assert_close(r.total_value,r.discounted_reward+r.terminal_discount*r.terminal_q)
    assert torch.equal(old,e.evaluate_trajectory_consequence(torch.zeros(1,3),a,compute_cost=False).features)
    p=LocalMPPI(e,dict(terminal_value='sac_value',audit_candidates=True),horizon=3,action_dim=2,device='cpu')
    e.agent.model.next_calls=0
    out=p.plan(torch.zeros(1,3),a[0],eval_mode=True)
    assert (out.final_std[-1]==0).all() and torch.equal(out.final_mean[-1],a[0,-1])
    assert out.metrics['wm_evaluated_candidates']==262 and e.agent.model.next_calls==15
    assert len(out.candidate_audit['actions'])==258


def test_all_actual_samples_share_final_reference_and_best_is_not_a_stale_Q_score():
    e=evaluator();calls=[]
    def q(z,a,task,return_type='avg'):
        assert return_type=='avg';calls.append(len(a))
        return 100*len(calls)+a[:,:1]
    e.agent.model.Q=q
    p=LocalMPPI(e,dict(audit_candidates=True),horizon=3,action_dim=2,device='cpu');rounds=iter([.8,.7,.6,.5])
    def proposals(mean,std,base):
        x=base[None].expand(65,-1,-1).clone();x[1,:,0]=next(rounds);return x
    p._candidate_batch=proposals
    r=p.plan(torch.zeros(1,3),torch.zeros(4,2),eval_mode=True);audit=r.candidate_audit
    assert calls==[65]*4+[2,258] and r.metrics['wm_evaluated_candidates']==262
    assert audit['indices']['best_elite']==2 and audit['indices']['final_elite']==194
    assert float(audit['iteration_values'][0].max())<float(audit['iteration_values'][-1].max())
    assert torch.equal(r.selected_sequence,r.final_mean) or torch.equal(r.selected_sequence,r.initial_mean)
    assert not torch.equal(r.selected_sequence,audit['actions'][2])
    assert len(audit['iteration_actions'])==4 and all(len(x)==64 for x in audit['iteration_actions'])


def test_cost_is_continuous_only_with_ranking_evidence_and_validation_does_not_touch_SAC_model():
    e=evaluator();e.agent.cfg.cost=dict(enabled=True,gamma=.9)
    e.agent.model.cost=lambda z,a,task:torch.ones_like(z[:,:1])*.2
    a=torch.zeros(2,4,2);z=torch.zeros(1,3)
    with patch('mpr_mpc.planning.evaluator.candidate_ranking_ready',return_value=False):
        base=e.evaluate_mppi_candidates(z,a,q_aggregation='ensemble_mean',compute_cost=True,cost_weight=0)
        no=e.evaluate_mppi_candidates(z,a,q_aggregation='ensemble_mean',compute_cost=True,cost_weight=2)
    assert torch.equal(base.total_value,no.total_value) and no.effective_cost_weight==0
    with patch('mpr_mpc.planning.evaluator.candidate_ranking_ready',return_value=True):
        yes=e.evaluate_mppi_candidates(z,a,q_aggregation='ensemble_mean',compute_cost=True,cost_weight=2)
    torch.testing.assert_close(yes.total_value,base.total_value-2*base.cumulative_cost)
    assert yes.effective_cost_weight==2 and torch.isfinite(yes.total_value).all()
    def validate(validator,model,device,*,horizon):
        assert model is not e.agent.model;model.marker='changed_only_in_copy';return {'passed':0}
    with patch('mpr_mpc.planning.cost_validation.CostModelValidator.validate',new=validate):
        e.validate_mppi_cost([])
    assert not hasattr(e.agent.model,'marker') and e.cost_validation_model.marker=='changed_only_in_copy'


def test_unvalidated_or_incompatible_value_artifacts_cannot_be_loaded_for_online(tmp_path):
    wm=tmp_path/'original.pt';wm.write_bytes(b'frozen original')
    head=SACContinuationValue(3,(8,4));head.freeze();path=tmp_path/'value.pt'
    data=dict(architecture=dict(latent_dim=3,hidden=(8,4)),state_dict=head.state_dict(),
        context=dict(wm_checkpoint_sha256=file_sha256(wm),horizon=3,gamma=.9,input_kind='imagined_terminal_latent'),training_scenarios=[1],validation=dict(passed=False))
    torch.save(data,path)
    with pytest.raises(ValueError,match='independent'):load_sac_value(path,wm_checkpoint=wm,horizon=3,gamma=.9,device='cpu')
    loaded,_=load_sac_value(path,wm_checkpoint=wm,horizon=3,gamma=.9,device='cpu',require_validated=False)
    torch.testing.assert_close(loaded(torch.ones(2,3)),head(torch.ones(2,3)))
    with pytest.raises(ValueError,match='different'):load_sac_value(path,wm_checkpoint=wm,horizon=4,gamma=.9,device='cpu',require_validated=False)
