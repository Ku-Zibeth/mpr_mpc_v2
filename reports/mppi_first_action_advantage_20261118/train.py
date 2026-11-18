"""Fixed-epoch, fit-only LOMO/full training. Never opens inner-stop/validation."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import numpy as np
import torch
from torch.nn import functional as F
from study_io import R,load,dump,sha,table
from models import Advantage,LocalKNN,fit_stats,balanced_weights,within_origin_pairs,holdout,SCHEMAS


def train_one(data,ids,schema,kind,stage,p):
    name=f'{stage}_{schema}_{kind}'
    output=R/'models'/f'{name}.pt';info=R/'models'/f'{name}.json'
    if output.exists():
        saved=load(info)
        assert saved['protocol_sha256']==sha(R/'protocol.json') and saved['sha256']==sha(output)
        return saved
    x=data[schema][ids].astype(np.float32);y=data['delta_y'][ids].astype(np.float32)
    scene,origin=data['scene'][ids],data['origin'][ids]
    weights=balanced_weights(scene,origin)
    pair,pair_weights=within_origin_pairs(scene,origin,y)
    torch.manual_seed(p['training']['seed'])
    if kind=='knn':
        model=LocalKNN(x,y,weights,p['knn']['k']);history=[];elapsed=0.
    else:
        model=Advantage(SCHEMAS[schema],kind)
        mu,scale,target=fit_stats(x,y,weights)
        model.mu.copy_(mu);model.scale.copy_(scale);model.target_scale.copy_(target)
        opt=torch.optim.Adam(model.parameters(),lr=p['training']['lr'],
            weight_decay=p['training'][f'{kind}_weight_decay'])
        xx,yy,ww=torch.tensor(x),torch.tensor(y),torch.tensor(weights)
        pp,pw=torch.tensor(pair),torch.tensor(pair_weights)
        sign=torch.sign(yy[pp[:,0]]/target-yy[pp[:,1]]/target) if len(pp) else torch.empty(0)
        history=[];begin=time.perf_counter()
        for epoch in range(p['training']['epochs']):
            normalized=model(xx)/target
            regression=(ww*F.smooth_l1_loss(normalized,yy/target,reduction='none')).sum()
            ranking=(pw*F.softplus(-sign*(normalized[pp[:,0]]-normalized[pp[:,1]]))).sum() if len(pp) else normalized.sum()*0
            loss=regression+p['training']['lambda_rank']*ranking
            opt.zero_grad();loss.backward();opt.step()
            if epoch in [0,49,99,199,299]:history.append(dict(epoch=epoch+1,loss=float(loss),relative_huber=float(regression),ranking=float(ranking)))
        elapsed=time.perf_counter()-begin
    model.eval().requires_grad_(False)
    with torch.no_grad():assert torch.equal(model(torch.tensor(x[data['baseline'][ids]])),torch.zeros(int(data['baseline'][ids].sum())))
    fit_maps=sorted(map(int,set(scene)))
    meta=dict(name=name,stage=stage,schema=schema,kind=kind,dimension=x.shape[1],fit_maps=fit_maps,
        fit_origins=sorted(set(origin.tolist())),fit_rows=len(ids),fit_indices=ids.tolist(),
        parameter_count=sum(q.numel() for q in model.parameters()),train_seconds=elapsed,history=history,
        protocol_sha256=sha(R/'protocol.json'),dataset_sha256=sha(R/'fit_dataset.npz'),
        gamma=p['gamma'],horizon=1,target='same-origin real Baseline-relative H1 Return',
        checkpoint_sha256=p['checkpoint_sha256'],parent_commit=p['parent_commit'],
        target_scale=float(model.target_scale),no_inner_stop_or_validation_normalization=True,
        no_future_inputs=True,baseline_centered=True,online_validated=False,
        k=model.k if kind=='knn' else None)
    torch.save(dict(meta=meta,state_dict=model.state_dict()),output)
    meta.update(checkpoint=str(output),sha256=sha(output))
    dump(f'models/{name}.json',meta)
    print('TRAIN',name,'maps',fit_maps,'rows',len(ids),'seconds',round(elapsed,2),flush=True)
    return meta


def main():
    torch.set_num_threads(1)
    p=load(R/'protocol.json')
    data={k:v.copy() for k,v in np.load(R/'fit_dataset.npz').items()}
    assert set(data['scene'])==set(p['fit_maps'])
    (R/'models').mkdir(exist_ok=True)
    reports=[];diagnostics=[];stages=[]
    for scene in p['fit_maps']:
        train,test=holdout(data,scene)
        assert set(data['scene'][train]).isdisjoint(set(data['scene'][test]))
        stages.append((f'lomo_{scene}',train))
    stages.append(('full',np.arange(len(data['y']))))
    for stage,ids in stages:
        for schema in ['A1','A2']:
            for kind in ['linear','mlp']:reports.append(train_one(data,ids,schema,kind,stage,p))
        reports.append(train_one(data,ids,'A2','knn',stage,p))
        y,origin,scene=data['delta_y'][ids],data['origin'][ids],data['scene'][ids]
        pairs,pw=within_origin_pairs(scene,origin,y);lookup={(int(i),int(j)):float(w) for (i,j),w in zip(pairs,pw)}
        scale=float(fit_stats(data['A2'][ids],y,balanced_weights(scene,origin))[2])
        for o in sorted(set(origin)):
            local=np.flatnonzero(origin==o)
            for n,i in enumerate(local):
                for j in local[n+1:]:diagnostics.append(dict(stage=stage,scene=int(scene[i]),origin=o,
                    candidate_i=int(data['candidate'][ids[i]]),candidate_j=int(data['candidate'][ids[j]]),
                    raw_delta_Y_i=float(y[i]),raw_delta_Y_j=float(y[j]),normalized_difference=float((y[i]-y[j])/scale),
                    eligible=abs(y[i]-y[j])>1,pair_weight=lookup.get((int(i),int(j)),0.),fit_target_scale=scale))
    table('advantage_pair_diagnostics.csv',diagnostics)
    dump('model_manifest.json',reports)
    dump('training_results.json',dict(models=reports,fixed_epochs=300,early_stop_on_heldout_map=False,
        inner_stop_used_during_training=False,seen_validation_opened=False,absolute_return_models_retrained=0))
    print('All8LOMO+full fits complete, ONLY original8fit maps used.',flush=True)


if __name__=='__main__':main()
