"""True fit-map LOMO, frozen A0 reference and once-only inner-stop selection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from study_io import R,ROOT,OLD,PRIOR,load,dump,table,sha
sys.path.insert(1,str(ROOT))
sys.path.insert(2,str(OLD))
import numpy as np
import torch
from models import Advantage,LocalKNN,balanced_weights
from tools.first_action_failure import ranking,cluster_bootstrap,paired_origins
from offline_model import load_model as load_A0

METRICS=['pair_accuracy','spearman','kendall','top1_regret','model_gain','baseline_gain',
         'selected_positive','selected_negative','oracle_gain','positive_fraction','baseline_gain_mae']
NAMES=['A0_frozen','A1_linear','A1_mlp','A2_linear','A2_mlp','A2_knn']


def load_relative(meta):
    assert sha(Path(meta['checkpoint']))==meta['sha256']
    artifact=torch.load(meta['checkpoint'],map_location='cpu',weights_only=True)
    assert artifact['meta']['protocol_sha256']==sha(R/'protocol.json')
    data=np.load(R/'fit_dataset.npz');ids=meta['fit_indices']
    model=(LocalKNN(data['A2'][ids],data['delta_y'][ids],balanced_weights(data['scene'][ids],data['origin'][ids]))
           if meta['kind']=='knn' else Advantage(meta['dimension'],meta['kind']))
    model.load_state_dict(artifact['state_dict']);model.eval().requires_grad_(False)
    return model


def per_origin_predict(model,x,data,ids):
    result=np.full(len(data['y']),np.nan,np.float32)
    for origin in sorted(set(data['origin'][ids])):
        ii=ids[data['origin'][ids]==origin]
        with torch.no_grad():result[ii]=model(torch.tensor(x[ii],dtype=torch.float32)).numpy()
    return result


def evaluate(data,pred,stage,name):
    rows=[]
    for origin in sorted(set(data['origin'])):
        ids=np.flatnonzero(data['origin']==origin)
        assert data['baseline'][ids][0] and data['delta_y'][ids][0]==0
        a=ranking(pred[ids],data['delta_y'][ids])
        rows.append(dict(scene_id=int(data['scene'][ids[0]]),origin_id=str(origin),horizon=1,
            stage=stage,model=name,A0_fit_overlap=(stage=='lomo' and name=='A0_frozen'),
            model_gain=a['selected_gain'],baseline_gain=0.,**a))
    return rows


def score_tuple(summary):
    a=summary['pair_accuracy']['mean'];r=summary['top1_regret']['mean'];g=summary['model_gain']['mean']
    return (a if a is not None else -.5,-r,g)


def main():
    torch.set_num_threads(1)
    p=load(R/'protocol.json');manifest=load(R/'model_manifest.json')
    fit={k:v.copy() for k,v in np.load(R/'fit_dataset.npz').items()}
    inner={k:v.copy() for k,v in np.load(R/'inner_stop_dataset.npz').items()}
    assert set(inner['scene'])==set(p['inner_stop_maps'])
    assert not (set(fit['scene'])|set(inner['scene']))&set(p['excluded_seen_validation'])
    A0,_=load_A0(p['A0_checkpoint'],p['A0_context']);A0.eval().requires_grad_(False)
    all_rows=[];predictions={};weights=[]
    for stage,data in [('lomo',fit),('inner_stop',inner)]:
        n=len(data['y']);preds={name:np.full(n,np.nan,np.float32) for name in NAMES}
        raw=per_origin_predict(A0,data['A0'],data,np.arange(n))
        for origin in sorted(set(data['origin'])):
            ids=np.flatnonzero(data['origin']==origin);preds['A0_frozen'][ids]=raw[ids]-raw[ids[0]]
        for scene in (p['fit_maps'] if stage=='lomo' else [None]):
            ids=np.flatnonzero(data['scene']==scene) if scene is not None else np.arange(n)
            prefix=f'lomo_{scene}' if stage=='lomo' else 'full'
            for name in NAMES[1:]:
                schema,kind=name.split('_')
                meta=next(m for m in manifest if m['name']==f'{prefix}_{schema}_{kind}')
                if scene is not None:assert scene not in meta['fit_maps']
                model=load_relative(meta)
                pred=per_origin_predict(model,data[schema],data,ids)
                preds[name][ids]=pred[ids]
        for name,pred in preds.items():
            assert np.isfinite(pred).all() and np.all(pred[data['baseline']]==0)
            all_rows.extend(evaluate(data,pred,stage,name))
        np.savez_compressed(R/f'{stage}_predictions.npz',**preds)
        predictions[stage]=preds
    # Record actual hierarchical candidate weights for every fold and full-fit.
    for meta in [m for m in manifest if m['schema']=='A2' and m['kind']=='linear']:
        ids=np.asarray(meta['fit_indices']);s,o=fit['scene'][ids],fit['origin'][ids]
        w=balanced_weights(s,o)
        for i,index in enumerate(ids):
            origin_count=len(set(o[s==s[i]]));candidate_count=int((o==o[i]).sum())
            weights.append(dict(stage=meta['stage'],scene=int(s[i]),origin=str(o[i]),candidate=int(fit['candidate'][index]),
                scene_weight=1/len(set(s)),origin_weight=1/len(set(s))/origin_count,candidate_weight=float(w[i]),candidate_count=candidate_count))
    table('training_weights.csv',weights)
    table('origin_results.csv',all_rows)
    summaries={}
    for stage in ['lomo','inner_stop']:
        for name in NAMES:
            rows=[r for r in all_rows if r['stage']==stage and r['model']==name]
            summaries[f'{stage}/{name}']=cluster_bootstrap(rows,METRICS)
    original_inner=load(OLD/'training_decision.json')['selected']['inner_stop']['summary']['pair_accuracy']['mean']
    assert abs(summaries['inner_stop/A0_frozen']['pair_accuracy']['mean']-original_inner)<1e-12
    model_rows=[]
    for key,stat in summaries.items():
        stage,name=key.split('/')
        model_rows.append(dict(stage=stage,model=name,A0_fit_overlap=stage=='lomo' and name=='A0_frozen',
            **{f'{k}_{suffix}':v for k,s in stat.items() for suffix,v in [('mean',s['mean']),('ci_low',s['ci95'][0]),('ci_high',s['ci95'][1])]}))
    table('model_comparison.csv',model_rows)
    table('inner_stop_results.csv',[r for r in model_rows if r['stage']=='inner_stop'])
    lomo=[]
    for scene in p['fit_maps']:
        for name in NAMES:
            rr=[r for r in all_rows if r['stage']=='lomo' and r['scene_id']==scene and r['model']==name]
            stat=cluster_bootstrap(rr,METRICS)
            lomo.append(dict(heldout_scene=scene,model=name,origins=len(rr),independent_lomo=name!='A0_frozen',
                A0_note='Seen-fit reference, NOT held-out' if name=='A0_frozen' else '',
                **{k:v['mean'] for k,v in stat.items()}))
    table('leave_one_map_out.csv',lomo)
    selected={}
    for schema in ['A1','A2']:
        names=[f'{schema}_linear',f'{schema}_mlp']
        selected[schema]=max(names,key=lambda name:score_tuple(summaries[f'inner_stop/{name}']))
    a2=selected['A2'];l=summaries[f'lomo/{a2}'];s=summaries[f'inner_stop/{a2}'];a0=summaries['inner_stop/A0_frozen']
    lomo_gate=dict(accuracy_ci_lower_above_half=l['pair_accuracy']['ci95'][0] is not None and l['pair_accuracy']['ci95'][0]>.5,
        spearman_ci_lower_positive=l['spearman']['ci95'][0] is not None and l['spearman']['ci95'][0]>0,
        gain_ci_lower_nonnegative=l['model_gain']['ci95'][0]>=0)
    inner_gate=dict(accuracy_above_half=s['pair_accuracy']['mean'] is not None and s['pair_accuracy']['mean']>.5,
        spearman_positive=s['spearman']['mean'] is not None and s['spearman']['mean']>0,
        regret_better_than_A0=s['top1_regret']['mean']<a0['top1_regret']['mean'],gain_nonnegative=s['model_gain']['mean']>=0,
        negative_rate_no_greater=s['selected_negative']['mean']<=a0['selected_negative']['mean'])
    proceed=all(lomo_gate.values()) and all(inner_gate.values())
    pairing={}
    for stage in ['lomo','inner_stop']:
        for name in NAMES[1:]:
            baseline=[r for r in all_rows if r['stage']==stage and r['model']=='A0_frozen']
            candidates=[r for r in all_rows if r['stage']==stage and r['model']==name]
            paired=paired_origins(baseline,candidates)
            pairing[f'{stage}/{name}_vs_A0']=dict(statistics=cluster_bootstrap(paired,['pair_accuracy_improvement','regret_improvement']),
                scope='A0fit-overlap, NOT fair heldout causalcomparison' if stage=='lomo' else '4 inner-stop maps, already used for historical A0 selection; development only')
    oracle=[]
    for stage,data in [('lomo',fit),('inner_stop',inner)]:
        for origin in sorted(set(data['origin'])):
            ids=np.flatnonzero(data['origin']==origin);y=data['delta_y'][ids]
            meta=load(OLD/'origins'/str(origin)/'catalog.json')
            random_idx=next(i for i,r in enumerate(meta['representatives']) if 'fixed_gaussian' in r['roles'])
            common=dict(stage=stage,scene_id=int(data['scene'][ids[0]]),origin_id=str(origin))
            for name in ['Baseline','Random','Oracle']+NAMES:
                index=0 if name=='Baseline' else random_idx if name=='Random' else int(y.argmax()) if name=='Oracle' else int(predictions[stage][name][ids].argmax())
                oracle.append(common|dict(model=name,gain=float(y[index]),oracle_gain=float(y.max()),
                    top1_regret=float(y.max()-y[index]),selected_positive=float(y[index]>1),selected_negative=float(y[index]<-1),
                    positive_fraction=float((y[1:]>1).mean()),any_positive=float((y[1:]>1).any()),candidate=int(data['candidate'][ids[index]])))
    table('oracle_comparison.csv',oracle)
    magnitude=[]
    for stage,data in [('lomo',fit),('inner_stop',inner)]:
        for origin in sorted(set(data['origin'])):
            ids=np.flatnonzero(data['origin']==origin);distance=np.linalg.norm(data['A2'][ids,-2:],axis=1)
            bands=np.searchsorted([.01,.05,.1],distance,side='right')
            for band in sorted(set(bands)):
                local=sorted(set([0]+np.flatnonzero(bands==band).tolist()))
                if len(local)<2:continue
                for name in NAMES:
                    stat=ranking(predictions[stage][name][ids][local],data['delta_y'][ids][local])
                    magnitude.append(dict(stage=stage,scene_id=int(data['scene'][ids[0]]),origin_id=str(origin),model=name,
                        action_band=['<.01','.01-.05','.05-.1','>=.1'][band],baseline_in_band=bool(bands[0]==band),
                        scope='Baseline retained as shared reference; restricted-pool Top1, not original model selection',
                        model_gain=stat['selected_gain'],**stat))
    table('action_magnitude_analysis.csv',magnitude)
    kind=a2.split('_')[1];frozen=next(m for m in manifest if m['name']==f'full_A2_{kind}')
    dump('evaluation_results.json',dict(summaries=summaries,paired_A0_comparison=pairing,selected=selected,
        oracle={f'{stage}/{name}':cluster_bootstrap([r for r in oracle if r['stage']==stage and r['model']==name],['gain','top1_regret','selected_positive','selected_negative','positive_fraction','any_positive']) for stage in ['lomo','inner_stop'] for name in ['Baseline','Random','Oracle']+NAMES},
        magnitude={f'{stage}/{name}/{band}':cluster_bootstrap([r for r in magnitude if r['stage']==stage and r['model']==name and r['action_band']==band],['pair_accuracy','spearman','top1_regret','model_gain']) for stage in ['lomo','inner_stop'] for name in NAMES for band in ['<.01','.01-.05','.05-.1','>=.1']},
        A0_inner_accuracy_unchanged=True,A0_lomo_reference_is_in_sample=True,seen_validation_opened=False))
    dump('decision.json',dict(selected_A1=selected['A1'],selected_A2=a2,frozen_A2=frozen,
        lomo_gate=lomo_gate,inner_stop_gate=inner_gate,proceed_to_new_validation=proceed,
        selected_using='inner-stop ONLY; pairaccuracy DESC,regret ASC,gain DESC; stable linear-first tie',
        no_A0_retraining=True,no_old_seen_validation_used=True,new_validation_performed=False,
        online_enabled=False,online_changes=False,reason='Both development gates passed: must preregister and finish one NEW independent validation.' if proceed else 'At least one preregistered LOMO/inner-stop gate failed. Stop; do not add validation or enlarge models.'))
    print('SELECTED',a2,'LOMO',lomo_gate,'INNER',inner_gate,'NEW_VALIDATION',proceed,flush=True)


if __name__=='__main__':main()
