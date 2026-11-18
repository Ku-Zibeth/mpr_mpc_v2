"""Full old123 H1 score-component audit, frozen WM only, NO environment construction or learning."""
import itertools
import pandas as pd
from scipy.stats import spearmanr,kendalltau
from study import *

def annotate_native_scope():
    file=R/'score_component_ablation.csv';data=pd.read_csv(file)
    data['selector_scope']='diagnostic argmax over measured historical canonical pool, NOT native online final2'
    added=[]
    for directory in sorted((HIST/'origins').iterdir()):
        meta=load(directory/'catalog.json');records={x['candidate_id']:load(HIST/'branches'/directory.name/x['candidate_id']/'episode.json') for x in meta['representatives']}
        base=records[meta['aliases']['A0']]['discounted_return'];selected=records[meta['aliases']['A1']]['discounted_return']
        z=np.load(directory/'catalog.npz')
        added.append(dict(scene=meta['scene'],step=meta['step'],method='native_actual_final2',selector_scope='EXACT actual native Baseline/WeightedMean output',
            selected_gain=selected-base,top1_regret=max(x['discounted_return'] for x in records.values())-selected,
            negative_selection_rate=float(selected-base<-1),score_margin=meta['C']['score']-float(z['C_final_values'][0]),
            interpretation='Actual executed H1 consequence, not a matched H1 model ranking metric'))
    file.rename(R/'score_component_pool_only.csv')
    table(file,data.to_dict('records')+added)

def ranking(y,s,base):
    y=np.asarray(y);s=np.asarray(s);i,j=np.triu_indices(len(y),1);dy=y[i]-y[j];ds=s[i]-s[j];eligible=np.abs(dy)>1.
    chosen=int(s.argmax());positive=y-y[base]>1.;predpositive=s-s[base]>0
    return dict(spearman=float(spearmanr(s,y).statistic) if np.ptp(y)>0 and np.ptp(s)>0 else None,
        kendall=float(kendalltau(s,y).statistic) if np.ptp(y)>0 and np.ptp(s)>0 else None,
        descriptive_pair_accuracy=float(np.mean((np.sign(ds[eligible])*np.sign(dy[eligible])+1)/2)) if eligible.any() else None,
        effective_pairs=int(eligible.sum()),top1_regret=float(y.max()-y[chosen]),selected_gain=float(y[chosen]-y[base]),
        positive_candidate_recall=float(predpositive[positive].mean()) if positive.any() else None,
        negative_selection_rate=float(y[chosen]-y[base]<-1),score_margin=float(s[chosen]-s[base]),chosen=chosen,
        interpretation='DESCRIPTIVE: model target differs from H1 + original SAC continuation')

@torch.no_grad()
def main():
    required=['FINAL_REPORT.md','PROTOCOL.md','paired_statistics.json','candidate_ranking.csv','candidate_actions.csv','counterfactual_episodes.csv','counterfactual_steps.csv','STATE_RECONSTRUCTION_AUDIT.md','code_changes.patch']
    read=[]
    for name in required:
        file=HIST/name;data=file.read_bytes()
        if file.suffix=='.csv':count=len(pd.read_csv(file))
        elif file.suffix=='.json':json.loads(data);count=1
        else:data.decode();count=data.count(b'\n')
        read.append(dict(path=str(file),sha256=file_sha(file),bytes=len(data),rows_or_lines=count,fully_read=True))
    if (R/'historical_inputs_read.json').exists():assert load(R/'historical_inputs_read.json')==read
    else:dump(R/'historical_inputs_read.json',read)
    evaluator=load_offline_evaluator();model=evaluator.agent.model
    from common import math as td_math
    rows=[];metrics=[];native=[];stability=[]
    for directory in sorted((HIST/'origins').iterdir()):
        meta=load(directory/'catalog.json');z=np.load(directory/'catalog.npz');scene=meta['scene'];step=meta['step']
        sequences=[];ys=[]
        # Preserve the original all8-role GPU batch BEFORE deduplication.
        for raw in meta['raw_roles']:
            cid=raw['role']
            seq=z['baseline_sequence'] if cid=='A0' else z['C_selected'] if cid=='A1' else z['D_selected'] if cid=='A2' else z['C_iteration_actions'][-1,raw['original_row']]
            sequences.append(seq)
        indices=[next(i for i,x in enumerate(meta['raw_roles']) if x['role']==rep['candidate_id']) for rep in meta['representatives']]
        for rep in meta['representatives']:ys.append(load(HIST/'branches'/directory.name/rep['candidate_id']/'episode.json')['discounted_return'])
        actions=torch.tensor(np.stack(sequences),device=evaluator.agent.device);latent=torch.tensor(z['latent'],device=actions.device)
        pair=tuple(meta['C']['final_pair']);out=evaluator.evaluate_mppi_candidates(latent,actions,compute_cost=False,q_aggregation='fixed_pair',q_indices=pair,capture_ensemble=True)
        zz=latent.expand(len(actions),-1);alive=torch.ones(len(actions),device=actions.device);prefix=torch.zeros_like(alive);shorts={}
        for t in range(10):
            reward=td_math.two_hot_inv(model.reward(zz,actions[:,t],None),evaluator.agent.cfg).reshape(-1)
            prefix+=.995**t*alive*reward
            if t<4:shorts[t+1]=prefix.clone()
            zz=model.next(zz,actions[:,t],None)
            if evaluator.agent.cfg.episodic:alive*=~(model.termination(zz,None).reshape(-1)>.5)
        assert torch.allclose(prefix,out.discounted_reward,atol=1e-3,rtol=1e-5)
        scores=dict(native_h10=out.total_value.cpu().numpy()[indices],immediate=shorts[1].cpu().numpy()[indices],
            short_h2=shorts[2].cpu().numpy()[indices],short_h3=shorts[3].cpu().numpy()[indices],short_h4=shorts[4].cpu().numpy()[indices],
            reward_h10=out.discounted_reward.cpu().numpy()[indices],terminal_contribution=(out.total_value-out.discounted_reward).cpu().numpy()[indices])
        original=np.array([r['common_h10_score'] for r in meta['representatives']]);assert np.allclose(scores['native_h10'],original,atol=.01,rtol=0)
        ids=[r['candidate_id'] for r in meta['representatives']];ys=np.array(ys);base=ids.index(meta['aliases']['A0']);sel=ids.index(meta['aliases']['A1'])
        raw_gain=meta['C']['score']-float(z['C_final_values'][0]);native.append(dict(scene=scene,step=step,predicted_gain=raw_gain,actual_gain=ys[sel]-ys[base],
            crash_delta=load(HIST/'branches'/directory.name/ids[sel]/'episode.json')['crash']-load(HIST/'branches'/directory.name/'A0'/'episode.json')['crash'],
            oor_delta=load(HIST/'branches'/directory.name/ids[sel]/'episode.json')['out_of_road']-load(HIST/'branches'/directory.name/'A0'/'episode.json')['out_of_road']))
        for k,s in scores.items():metrics.append(dict(scene=scene,step=step,method=k,**ranking(ys,s,base)))
        for j,rep in enumerate(meta['representatives']):
            actual_row=indices[j]
            rows.append(dict(scene=scene,step=step,candidate_id=ids[j],actual_h1_return=ys[j],actual_h1_gain=ys[j]-ys[base],
                steering=float(actions[actual_row,0,0]),throttle_brake=float(actions[actual_row,0,1]),
                first_action_l2=float((actions[actual_row,0]-actions[indices[base],0]).norm()),
                native_selected_first_distance=float((actions[actual_row,0]-actions[indices[sel],0]).norm()),
                original_stored_h10_score=float(original[j]),**{k:float(s[j]) for k,s in scores.items()},
                **{f'delta_{k}':float(s[j]-s[base]) for k,s in scores.items()},
                q_ensemble_std=float(out.terminal_q_std[actual_row]),formal_h1_rank='NOT ESTIMABLE'))
        # Candidate ranking changes under Q pairs, without any environment execution.
        qs=out.q_member_values.cpu().numpy();qs=qs if qs.shape[0]==5 else qs.T;qs=qs[:,indices]
        for members in itertools.combinations(range(5),2):
            value=scores['reward_h10']+out.terminal_discount.cpu().numpy()[indices]*qs[list(members)].mean(0)
            stability.append(dict(scene=scene,step=step,q_pair=list(members),top1=ids[int(value.argmax())],
                selected_gain=float(ys[int(value.argmax())]-ys[base]),score_span=float(np.ptp(value))))
    table(R/'historical_h1_analysis.csv',rows);table(R/'score_component_ablation.csv',metrics);table(R/'q_rank_stability.csv',stability)
    df=pd.DataFrame(native);grid=np.unique(np.r_[0,np.quantile(df.predicted_gain,[.25,.5,.75,.9])]);margin_rows=[]
    for threshold in grid:
        replace=df.predicted_gain>threshold;g=df.actual_gain*replace
        margin_rows.append(dict(margin=float(threshold),replacement_rate=float(replace.mean()),selected_gain=float(g.groupby(df.scene).mean().mean()),
            negative_selection_rate=float(((df.actual_gain<-1)&replace).mean()),
            crash_delta=float((df.crash_delta*replace).mean()),oor_delta=float((df.oor_delta*replace).mean()),
            source='old8maps development only',eligible_nontrivial=bool(.125<=replace.mean()<=.875)))
    feasible=[x for x in margin_rows if x['eligible_nontrivial'] and x['crash_delta']<=0 and x['oor_delta']<=0]
    eligible=feasible or [x for x in margin_rows if x['eligible_nontrivial']]
    choice=max(eligible,key=lambda x:(x['selected_gain'],-x['margin']))
    cv=[]
    for scene in sorted(df.scene.unique()):
        train=df[df.scene!=scene];test=df[df.scene==scene];options=[]
        for threshold in grid:
            mask=train.predicted_gain>threshold
            if .125<=mask.mean()<=.875:options.append((float((train.actual_gain*mask).groupby(train.scene).mean().mean()),-float(threshold)))
        threshold=-max(options)[1]
        cv.append(dict(scene=int(scene),margin=threshold,gain=float((test.actual_gain*(test.predicted_gain>threshold)).mean())))
    table(R/'baseline_margin_analysis.csv',margin_rows);table(R/'development_margin_lomo.csv',cv)
    dump(R/'frozen_margin.json',dict(margin=choice['margin'],development_choice=choice,development_lomo_gain=float(np.mean([x['gain'] for x in cv])),
        development_positive=bool(choice['selected_gain']>0 and np.mean([x['gain'] for x in cv])>0),
        rule='Grid=0 plus predicted-score gain quartiles/90th; replacement12.5%-87.5%, prefer nonworse old safety, maximize map-mean actual gain; no new-map labels.',
        unsupported_if_negative='F remains an explicitly unvalidated experimental finite margin; if new methods all fail, opt-in deployment guard denies unsupported replacements, never claims performance improvement.',
        no_new_map_outcomes_seen=True))
    annotate_native_scope()
    print('OLD123_READ',len(rows),'MARGIN',choice,'LOMO',cv,flush=True)
if __name__=='__main__':main()
