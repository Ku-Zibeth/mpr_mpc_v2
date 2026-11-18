"""Required Phase3 regressions; all fixtures are offline/current-state only."""
import importlib.util,json
from pathlib import Path
import numpy as np
import pytest
import torch

R=Path(__file__).parent
spec=importlib.util.spec_from_file_location('phase3_advantage_models',R/'models.py')
models=importlib.util.module_from_spec(spec);spec.loader.exec_module(models)


def protocol():return json.loads((R/'protocol.json').read_text())


def test_baseline_delta_action_is_zero():
    for s in ['A1','A2']:
        x=models.features(np.ones(535),[.2,.7],[.2,.7],s)
        assert np.array_equal(x[-2:],[0.,0.])


def test_baseline_advantage_exactly_zero():
    for kind in ['linear','mlp']:
        model=models.Advantage(539,kind)
        x=torch.randn(7,539);x[:,-2:]=0
        assert torch.equal(model(x),torch.zeros(7))


def test_candidate_advantage_is_difference_of_raw_predictions():
    model=models.Advantage(539,'mlp');model.target_scale.fill_(2.5)
    x=torch.randn(3,539);reference=x.clone();reference[:,-2:]=0
    assert torch.equal(model(x),(model.raw(x)-model.raw(reference))*2.5)


def test_targets_use_unique_actual_same_origin_baseline():
    result=models.relative_targets([10,13,100,80],['a','a','b','b'],[True,False,True,False])
    assert np.array_equal(result,[0,3,0,-20])
    with pytest.raises(ValueError,match='ONE'):models.relative_targets([1,2],['a','a'],[True,True])


def test_original_gamma_reward_done_definition_unchanged():
    p=protocol();assert p['gamma']==.995 and p['horizon']==1 and p['reward']=='info.tdmpc2_reward'
    assert p['continuation']=='original frozen deterministic Lattice+SAC until native done'
    assert p['checkpoint_sha256']=='4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a'


def test_pairs_are_within_origin_and_keep_original_tolerance():
    scene=np.array([1,1,1,2,2]);origin=np.array(['a','a','a','b','b']);y=np.array([0.,1.,2.,100.,103.])
    pair,w=models.within_origin_pairs(scene,origin,y)
    assert pair.tolist()==[[0,2],[3,4]] and np.allclose(w,[.5,.5])
    assert all(origin[i]==origin[j] for i,j in pair)


def test_origin_balanced_candidate_weights():
    w=models.balanced_weights([1]*5,['a']*2+['b']*3)
    assert w[:2].sum()==pytest.approx(.5) and w[2:].sum()==pytest.approx(.5)


def test_scene_balanced_even_with_different_origin_counts():
    w=models.balanced_weights([1,1,1,2],['a','a','b','c'])
    assert w[:3].sum()==pytest.approx(.5) and w[3]==pytest.approx(.5)


def test_fit_only_normalization_excludes_heldout_values():
    x=np.array([[1.,0.],[3.,2.],[10000.,90000.]],np.float32)
    train,test=models.holdout({'scene':np.array([1,1,2])},2)
    mu,scale,target=models.fit_stats(x[train],[0.,4.],[.5,.5])
    assert np.allclose(mu.numpy(),[2.,1.]) and np.allclose(scale.numpy(),[1.,1.])
    assert float(target)==pytest.approx(np.sqrt(8),rel=1e-6)


def test_leave_one_map_out_never_fits_heldout_or_inner_maps():
    p=protocol();manifest=json.loads((R/'model_manifest.json').read_text())
    for m in manifest:
        if m['stage'].startswith('lomo_'):
            held=int(m['stage'].split('_')[1]);assert held not in m['fit_maps'] and len(m['fit_maps'])==7
        assert set(m['fit_maps'])<=set(p['fit_maps'])
        assert set(m['fit_maps']).isdisjoint(p['inner_stop_maps']+p['excluded_seen_validation'])


def test_knn_linear_use_identical_current_A2_features():
    x=np.stack([models.features(np.ones(535),[.2,.8],[a,.7],'A2') for a in [.1,.2,.3]])
    y=np.array([1.,0.,-1.],np.float32);w=np.ones(3,np.float32)/3
    knn=models.LocalKNN(x,y,w);mu,scale,target=models.fit_stats(x,y,w)
    linear=models.Advantage(539,'linear');linear.mu.copy_(mu);linear.scale.copy_(scale)
    assert torch.equal(knn.mu,linear.mu) and torch.equal(knn.scale,linear.scale)
    reference=torch.tensor(x);reference[:,-2:]=0
    assert torch.equal(knn(reference),torch.zeros(3))


def test_feature_schema_does_not_accept_future_information():
    state=np.arange(535,dtype=np.float32);x=models.features(state,[.1,.7],[.2,.9],'A2')
    assert len(x)==539 and np.array_equal(x[:535],state) and np.allclose(x[535:537],[.1,.7])
    assert np.allclose(x[-2:],[.1,.2])
    with pytest.raises(TypeError):models.features(state,[0,0],[0,0],'A2',future_reward=10)


def test_A0_frozen_result_is_unchanged():
    import hashlib
    p=protocol();assert hashlib.sha256(Path(p['A0_checkpoint']).read_bytes()).hexdigest()==p['A0_sha256']
    r=json.loads((R/'evaluation_results.json').read_text())
    assert r['summaries']['inner_stop/A0_frozen']['pair_accuracy']['mean']==pytest.approx(.590769019982575,abs=1e-12)
    assert r['A0_lomo_reference_is_in_sample']


def test_online_sources_and_default_config_are_unchanged():
    import hashlib
    root=R.parents[1];m=json.loads((R/'experiment_manifest.json').read_text())
    for path,digest in m['protected_source_sha256'].items():
        assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path


def test_seen_validation_not_used_for_selection_or_training():
    p=protocol();decision=json.loads((R/'decision.json').read_text())
    assert decision['no_old_seen_validation_used'] and 'inner-stop ONLY' in decision['selected_using']
    with np.load(R/'fit_dataset.npz') as fit,np.load(R/'inner_stop_dataset.npz') as inner:
        assert set(fit['scene'])==set(p['fit_maps']) and set(inner['scene'])==set(p['inner_stop_maps'])
        assert not (set(fit['scene'])|set(inner['scene']))&set(p['excluded_seen_validation'])


def test_centered_linear_cancels_context_and_baseline_terms():
    a=models.Advantage(537,'linear').double();b=models.Advantage(539,'linear').double()
    with torch.no_grad():
        a.net.weight.normal_();b.net.weight.normal_();a.net.weight[:,-2:]=torch.tensor([.3,-.2]);b.net.weight[:,-2:]=a.net.weight[:,-2:]
    delta=torch.randn(5,2,dtype=torch.float64)
    x=torch.cat([torch.randn(5,535,dtype=torch.float64),delta],1)
    xx=torch.cat([torch.randn(5,537,dtype=torch.float64),delta],1)
    assert torch.allclose(a(x),b(xx),atol=1e-12,rtol=1e-12)
