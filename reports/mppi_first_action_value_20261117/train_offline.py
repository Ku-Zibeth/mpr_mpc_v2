"""Conditional training on H1-only fit8/innerstop4 maps; NEVER opens newvalidation."""
import copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from tools.first_action_study import *
from tools.mppi_failure_attribution import ranking_metrics,scene_bootstrap
from tools.policy_value_diagnosis import calibration
from offline_model import StudyModel
from torch.nn import functional as F


def dataset(scenes):
    p=load(REPORT/'protocol.json');rows=[];arrays={k:[] for k in ['F1','F2','F3']}
    for scene in scenes:
        for before in p['before_steps']:
            folder=REPORT/'origins'/f'{scene}_{before}'
            if not (folder/'catalog.json').exists():continue
            c=np.load(folder/'catalog.npz');meta=load(folder/'catalog.json');temp=[]
            for i,candidate in enumerate(meta['representatives']):
                file=REPORT/'branches'/folder.name/f'H1_{candidate["candidate_id"]}_baseline.json'
                if not file.exists():raise ValueError('IncompleteH1trainingorigin')
                real=load(file);assert real['horizon']==1 and real['takeover']=='baseline' and real['native_done']
                temp.append(dict(scenario=scene,origin=folder.name,candidate=candidate['candidate_id'],actual=float(real['actual_return']),baseline=i==0,
                    original_h10_score=float(c['scores'][c['pool_indices'][i]]),branch_source=str(file.relative_to(ROOT)),branch_sha=file_sha(file),catalog_sha=file_sha(folder/'catalog.npz')))
            rows.extend(temp)
            for k in arrays:arrays[k].extend(c[k])
    return rows,{k:np.asarray(v,np.float32) for k,v in arrays.items()}


def row_weights(rows):
    scenes=sorted({r['scenario'] for r in rows});origins={s:{r['origin'] for r in rows if r['scenario']==s} for s in scenes};weights=[]
    for r in rows:weights.append(1/len(scenes)/len(origins[r['scenario']])/sum(q['origin']==r['origin'] for q in rows))
    return np.array(weights,np.float32)


def paired_indices(rows):
    pairs=[];weights=[];groups=[]
    for origin in sorted({r['origin'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['origin']==origin];ss=rows[ids[0]]['scenario']
        local=[(i,j) for n,i in enumerate(ids) for j in ids[n+1:] if abs(rows[i]['actual']-rows[j]['actual'])>1]
        if local:groups.append((ss,origin,local))
    scenes={g[0] for g in groups}
    for scene,origin,local in groups:
        valid_origins=sum(g[0]==scene for g in groups)
        for i,j in local:pairs.append((i,j));weights.append(1/len(scenes)/valid_origins/len(local))
    pairs=np.array(pairs,dtype=np.int64).reshape(-1,2);w=np.asarray(weights,np.float32)
    if len(w):w/=w.sum()
    return torch.tensor(pairs),torch.tensor(w)


def evaluate(predicted,rows):
    originrows=[]
    for origin in sorted({r['origin'] for r in rows}):
        ii=[i for i,r in enumerate(rows) if r['origin']==origin];y=np.array([rows[i]['actual'] for i in ii]);p=np.asarray(predicted)[ii];old=np.array([rows[i]['original_h10_score'] for i in ii]);a=ranking_metrics(p,y);b=ranking_metrics(old,y)
        originrows.append(dict(scenario=rows[ii[0]]['scenario'],origin=origin,**a,**calibration(p,y),
            old_pair_accuracy=b['pair_accuracy'],old_regret=b['top1_regret'],regret_improvement=b['top1_regret']-a['top1_regret']))
    return dict(summary=scene_bootstrap(originrows,['pair_accuracy','spearman','kendall','top1_regret','regret_improvement','mae','bias','calibration_error']),rows=originrows)


def main():
    p=load(REPORT/'protocol.json');gate=load(REPORT/'development_decision.json');assert gate['training_coverage_gate'] and gate['evaluation_gap_gate']
    assert not any((REPORT/'branches').glob('94*/*_baseline.json')),'Do notreadvalidationlabels beforefreeze'
    torch.set_num_threads(1);fit,xf=dataset(p['fit_scenes']);stop,xs=dataset(p['inner_stop_scenes']);assert set(r['scenario'] for r in fit).isdisjoint(r['scenario'] for r in stop)
    manifest=dict(fit_scenes=p['fit_scenes'],inner_stop_scenes=p['inner_stop_scenes'],validation_scenes_sealed=p['validation_scenes'],
        fit_rows=fit,inner_stop_rows=stop,features='F1=currentz512+a0; F2=current535+a0; F3=current535+actualpostSACparams3+baselinephysical2+a0',
        labels='H1realfirstaction+gamma995SACfeedbackONLY, native done; NOT historicalH10labels',
        target='Q_B currentstate,onephysicalaction,then deterministicoriginalLatticeSAC',
        design=dict(architectures=['linear','MLP64ReLU32ReLU'],objectives='scene-originbalancednormalizedHuber +.1pairwisesoftplus(rawReturnDiff/10), L2',
            max_epochs=500,check_every=5,patience_checks=12,seed=1,ridge_lambda=.01,mlp_weight_decay=.001,ridge_finetune_lr=.001,mlp_lr=.0003,
            model_choice='innerstop pairaccuracy DESC,top1regret ASC,MAE ASC. FreezeONE model before newvalidation.',
            inner_gate='pointpairaccuracy>=.5 andpairedRegretimprovement>0; otherwise noindependentvalidation.'),
        no_future_features=True,normalization='fitonly, per-scene+originbalanced',training_original_networks=False)
    dump(REPORT/'training_manifest.json',manifest);manifest_sha=file_sha(REPORT/'training_manifest.json')
    protected=['planning/coordinator.py','planning/evaluator.py','planning/trajectory_adapter.py','residual_rl/state_builder.py','residual_rl/features.py','residual_rl/action_adapter.py','lattice/frenet_metadrive.py','config.yaml']
    baseline_sha=digest(dict(code={n:file_sha(ROOT/n) for n in protected},
        effective_policy_context=load(SOURCE/'reports/sac_value_20261117/dataset_manifest.json')['context'],
        source_environment_config_sha=file_sha(p['source_environment_config'])))
    y=torch.tensor([r['actual'] for r in fit]);w=torch.tensor(row_weights(fit));mean=(w*y).sum();scale=(w*(y-mean).square()).sum().sqrt().clamp_min(1.)
    pairs,pairw=paired_indices(fit);sign=torch.sign(y[pairs[:,0]]-y[pairs[:,1]]) if len(pairs) else torch.zeros(0)
    reports=[];models=REPORT/'models';models.mkdir(exist_ok=False)
    for schema in ['F1','F2','F3']:
        xx=torch.tensor(xf[schema]);test=torch.tensor(xs[schema]);mu=(w[:,None]*xx).sum(0);sd=(w[:,None]*(xx-mu).square()).sum(0).sqrt().clamp_min(.001)
        for kind in ['ridge_huber_rank','mlp_huber_rank']:
            torch.manual_seed(1);model=StudyModel(xx.shape[1],kind);model.input_mean.copy_(mu);model.input_scale.copy_(sd);model.target_mean.copy_(mean);model.target_scale.copy_(scale)
            if kind=='ridge_huber_rank':
                x=((xx-mu)/sd).numpy().astype(np.float64);ww=w.numpy();xw=x*np.sqrt(ww[:,None]);yy=((y-mean)/scale).numpy()*np.sqrt(ww)
                coef=xw.T@np.linalg.solve(xw@xw.T+.01*np.eye(len(xw)),yy)
                with torch.no_grad():model.net.weight.copy_(torch.tensor(coef,dtype=torch.float32)[None]);model.net.bias.zero_()
            opt=torch.optim.Adam(model.parameters(),lr=.001 if kind.startswith('ridge') else .0003,weight_decay=.01 if kind.startswith('ridge') else .001)
            best=None;best_score=None;stale=0;history=[]
            for epoch in range(501):
                if epoch:
                    pred=model(xx);reg=(w*F.smooth_l1_loss((pred-mean)/scale,(y-mean)/scale,reduction='none')).sum()
                    rank=(pairw*F.softplus(-sign*(pred[pairs[:,0]]-pred[pairs[:,1]])/10)).sum() if len(pairs) else pred.sum()*0
                    loss=reg+.1*rank;opt.zero_grad();loss.backward();opt.step()
                if epoch%5==0:
                    with torch.no_grad():scores=model(test).numpy()
                    # Pointmetrics for earlystop; sceneCI only on final frozencheckpoint.
                    metrics=[]
                    for origin in sorted({r['origin'] for r in stop}):
                        ids=[i for i,r in enumerate(stop) if r['origin']==origin];actual=np.array([stop[i]['actual'] for i in ids]);metrics.append(dict(scenario=stop[ids[0]]['scenario'],**ranking_metrics(scores[ids],actual)))
                    pa_scene=[];regret_scene=[]
                    for scene in sorted({m['scenario'] for m in metrics}):
                        mm=[m for m in metrics if m['scenario']==scene];valid=[m['pair_accuracy'] for m in mm if m['pair_accuracy'] is not None]
                        if valid:pa_scene.append(float(np.mean(valid)))
                        regret_scene.append(float(np.mean([m['top1_regret'] for m in mm])))
                    pa=np.mean(pa_scene) if pa_scene else -.5
                    regret=np.mean(regret_scene);mae=float(row_weights(stop)@np.abs(scores-np.array([r['actual'] for r in stop])))
                    score=(float(pa),-float(regret),-float(mae));history.append(dict(epoch=epoch,pair_accuracy=float(pa),regret=float(regret),mae=float(mae)))
                    if best_score is None or score>best_score:best_score=score;best=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
                    else:stale+=1
                    if stale>=12:break
            model.load_state_dict(best);model.eval().requires_grad_(False)
            with torch.no_grad():pred=model(test).numpy()
            evaluation=evaluate(pred,stop)
            context=dict(wm_checkpoint_sha=p['checkpoint_sha256'],sac_checkpoint_sha=p['checkpoint_sha256'],training_manifest_sha=manifest_sha,feature_schema=schema,
                gamma=.995,reward_definition='info.tdmpc2_reward/native_done',baseline_policy_sha=baseline_sha,
                intervention_horizon=1,continuation_policy='frozen_deterministic_LatticeSAC')
            artifact=dict(kind=kind,dimension=xx.shape[1],state_dict=model.state_dict(),context=context,independent_validated=False,best_epoch=best_epoch)
            name=f'{schema}_{kind}';file=models/f'{name}.pt';torch.save(artifact,file)
            dump(models/f'{name}.json',dict(inner_stop=evaluation,history=history,context=context,sha256=file_sha(file),future_input=False))
            reports.append(dict(name=name,schema=schema,kind=kind,checkpoint=str(file.relative_to(ROOT)),checkpoint_sha=file_sha(file),context=context,score=best_score,inner_stop=evaluation,best_epoch=best_epoch))
            print(name,'bestepoch',best_epoch,'inner',evaluation['summary'],flush=True)
    chosen=max(reports,key=lambda r:r['score']);s=chosen['inner_stop']['summary'];proceed=bool(s['pair_accuracy']['mean'] is not None and s['pair_accuracy']['mean']>=.5 and s['regret_improvement']['mean']>0)
    dump(REPORT/'training_results.json',reports);dump(REPORT/'training_decision.json',dict(trained=True,selected=chosen,proceed_to_one_validation=proceed,online=False,
        reason='OnlyONE innerstopselectedcheckpoint may see new6mapvalidation, neverselectonthevalidation. NoonlineintegrationunlessALLindependentgatespass.' if proceed else 'No innerstopranking+regret pointimprovement; retain newvalidationsealed.'))

if __name__=='__main__':main()
