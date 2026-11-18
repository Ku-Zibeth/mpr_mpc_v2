from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.first_action_study import assert_pair,physical_first_action,feature_vectors,all_reducers,representative_candidates,terminal_policy_actions,check_value_context,assert_feedback_stage
from tools.policy_value_diagnosis import discounted_return

def test_origin_domains_are_all_strict():
    a={'hashes':dict(observation='o',vehicle='v',traffic='t',rng='r')};assert_pair(a,a)
    for k in a['hashes']:
        b={'hashes':a['hashes']|{k:'changed'}}
        with pytest.raises(ValueError):assert_pair(a,b)

def test_physical_first_action_is_not_residual_remapped():
    a=np.array([[.23,-.8],[.8,.9]],np.float32)
    assert np.array_equal(physical_first_action(a),a[0])
    assert np.isclose(discounted_return([1,2,3],.995),1+.995*discounted_return([2,3],.995))
    with pytest.raises(ValueError):physical_first_action([[1.1,0]])

def test_online_feature_schema_uses_current_state_and_refined_parameters():
    z=np.arange(512,dtype=np.float32);s=np.r_[z,np.ones(23)];p=[.5,8.,3.]
    r=feature_vectors(z,s,p,[0,.2],[.1,.4]);assert [r[k].shape for k in r]==[(514,),(537,),(542,)]
    assert np.array_equal(r['F3'][535:538],p)
    with pytest.raises(ValueError):feature_vectors(z+1,s,p,[0,.2],[.1,.4])

def test_q_reducers_do_not_confuse_pair_avg_and_min():
    q=np.array([[0,2,4,6,8]],float);r=all_reducers(q)
    assert len(r)==23 and r['avg01'][0]==1 and r['min01'][0]==0 and r['median'][0]==4

def test_representatives_preserve_selected_role_and_deduplicate_first_action():
    pool=np.zeros((20,11,2),np.float32);pool[:,0,0]=np.arange(20)/20;pool[1,0]=pool[0,0]
    r,aliases=representative_candidates(pool,np.arange(20),1,5)
    assert r[0]['roles']==['baseline','mppi_selected','worst_nonbaseline'] and len(r)<=8
    assert {a['role'] for a in aliases}=={'mppi_selected','worst_nonbaseline'}
    assert len({pool[x['pool_index'],0].tobytes() for x in r})==len(r)

def test_sampled_terminal_is_not_mean_and_rng_is_restored():
    class Model:
        def pi(self,z,task):return torch.randn(len(z),2),{'mean':torch.zeros(len(z),2)}
    state=torch.random.get_rng_state();r=terminal_policy_actions(Model(),torch.zeros(4,3),torch.ones(4,2),1)
    assert torch.equal(state,torch.random.get_rng_state()) and not torch.equal(r['policy_mean'],r['native_policy_sample'])

def test_gamma_checkpoint_context_and_residual_continuation_guard():
    d={k:'ok' for k in ['wm_checkpoint_sha','sac_checkpoint_sha','training_manifest_sha','feature_schema','gamma','reward_definition','baseline_policy_sha','intervention_horizon','continuation_policy']}
    check_value_context(d,d)
    for k in ['gamma','wm_checkpoint_sha','sac_checkpoint_sha','intervention_horizon','continuation_policy']:
        with pytest.raises(ValueError):check_value_context(d|{k:'wrong'},d)
    assert_feedback_stage(SimpleNamespace(use_residual=True,use_mppi=False))
    for residual,mppi in [(False,False),(True,True)]:
        with pytest.raises(ValueError):assert_feedback_stage(SimpleNamespace(use_residual=residual,use_mppi=mppi))

def test_actual_offline_artifact_loader_rejects_wrong_context(tmp_path):
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('offline_study_model_test',root/'reports/mppi_first_action_value_20261117/offline_model.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    ctx={k:'frozen' for k in ['wm_checkpoint_sha','sac_checkpoint_sha','training_manifest_sha','feature_schema','reward_definition','baseline_policy_sha','continuation_policy']};ctx['gamma']=.995;ctx['intervention_horizon']=1
    model=module.StudyModel(514,'ridge_huber_rank');path=tmp_path/'independent_test_model.pt'
    torch.save(dict(kind='ridge_huber_rank',dimension=514,state_dict=model.state_dict(),context=ctx),path)
    loaded,_=module.load_model(path,ctx)
    assert torch.equal(model(torch.zeros(2,514)),loaded(torch.zeros(2,514)))
    for key in ['gamma','wm_checkpoint_sha','sac_checkpoint_sha','training_manifest_sha','baseline_policy_sha','reward_definition','feature_schema','intervention_horizon','continuation_policy']:
        with pytest.raises(ValueError):module.load_model(path,ctx|{key:'incompatible'})

def test_rank_loss_balances_maps_even_when_an_origin_has_no_effective_pairs(monkeypatch):
    import importlib.util
    from pathlib import Path
    root=Path(__file__).resolve().parents[1];folder=root/'reports/mppi_first_action_value_20261117'
    monkeypatch.syspath_prepend(str(folder))
    spec=importlib.util.spec_from_file_location('offline_training_balance_test',folder/'train_offline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[dict(scenario=1,origin='a',actual=y) for y in [0,2,4]]+[dict(scenario=1,origin='ties',actual=y) for y in [0,.1]]+[dict(scenario=2,origin='b',actual=y) for y in [0,3]]
    roww=module.row_weights(rows);assert np.isclose(roww[:5].sum(),.5) and np.isclose(roww[5:].sum(),.5)
    pairs,w=module.paired_indices(rows);first_map=np.array([rows[int(i)]['scenario']==1 for i in pairs[:,0]])
    assert np.isclose(w[first_map].sum(),.5) and np.isclose(w[~first_map].sum(),.5)
    assert all(abs(rows[int(i)]['actual']-rows[int(j)]['actual'])>1 for i,j in pairs)
