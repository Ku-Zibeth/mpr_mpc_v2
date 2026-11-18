"""Same-origin Y1 coverage/ranking; horizons remain distinct native interventions."""
import sys,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *
from tools.mppi_failure_attribution import ranking_metrics,scene_bootstrap
from tools.policy_value_diagnosis import calibration
from scipy.stats import spearmanr,kendalltau


def eligible_folders(scenes):
    p=load(REPORT/'protocol.json')
    for scene in scenes:
        for before in p['before_steps']:
            folder=REPORT/'origins'/f'{scene}_{before}'
            if (folder/'catalog.json').exists():yield folder


def analyze(scenes,phase):
    p=load(REPORT/'protocol.json');coverage=[];rankrows=[];effects=[];returns=[];candidates=[];hrows=[];noise=[]
    for folder in eligible_folders(scenes):
        meta=load(folder/'catalog.json');c=np.load(folder/'catalog.npz');q=np.load(folder/'offline_scores.npz');ids=np.array(c['pool_indices']);scene=meta['scene'];catalog=meta['representatives'];pair=np.asarray(meta['actual_final_pair']);branches=REPORT/'branches'/folder.name
        values={};branch_rows={}
        selected_index=next(i for i,r in enumerate(catalog) if 'mppi_selected' in r['roles'])
        for h in p['horizons']:
            files=[branches/f'H{h}_{r["candidate_id"]}_baseline.json' for r in catalog]
            if not all(f.exists() for f in files):continue
            rr=[load(f) for f in files];y=np.array([r['actual_return'] for r in rr]);values[h]=y;branch_rows[h]=rr;returns.extend(rr)
            selected_gain=float(y[selected_index]-y[0]);best=float(y.max()-y[0])
            coverage.append(dict(scenario=scene,origin=folder.name,horizon=h,candidates=len(y),oracle_best_gain=best,actual_selected_gain=selected_gain,
                actual_selected_regret=float(y.max()-y[selected_index]),positive_fraction=float(np.mean(y[1:]>y[0]+1)),negative_fraction=float(np.mean(y[1:]<y[0]-1)),
                baseline_win_rate=float(np.mean(y[0]>y[1:]+1)),has_positive_candidate=float(best>1),selected_positive=float(selected_gain>1),
                maximum_first_action_delta=float(np.linalg.norm(c['actions'][:,0]-c['baseline'][0],axis=1).max())))
            for i,r in enumerate(rr):effects.append(dict(scenario=scene,origin=folder.name,horizon=h,candidate=r['candidate_id'],roles=' '.join(catalog[i]['roles']),
                actual_gain=float(y[i]-y[0]),actual_return=float(y[i]),predicted_h10_gain=float(c['scores'][ids[i]]-c['scores'][0]),
                first_action_delta=float(np.linalg.norm(c['actions'][i,0]-c['baseline'][0])),selected_by_original_mppi=i==selected_index,
                crash=r['crash'],out_of_road=r['out_of_road'],route_completion=r['route_completion'],success=r['success'],
                success_delta=r['success']-rr[0]['success'],crash_delta=r['crash']-rr[0]['crash'],out_of_road_delta=r['out_of_road']-rr[0]['out_of_road'],
                safety_cost_delta=r['safety_cost']-rr[0]['safety_cost']))
            scores={'current_mppi_h10':c['scores'][ids],'constant_baseline':np.zeros(len(ids))}
            scores.update({'current_q_'+k:v[ids] for k,v in all_reducers(q['q_current']).items()})
            for mode in ['candidate','policy_mean','native_policy_sample']:
                qs=q['q_'+mode]
                scores['terminal_'+mode]=c['prefix'][ids]+c['discount'][ids]*qs[ids[:,None],pair[None,:]].mean(1)
            for name,score in scores.items():rankrows.append(dict(scenario=scene,origin=folder.name,horizon=h,score=name,
                **ranking_metrics(score,y),**calibration(score,y)))
            if h==1:
                qstd=q['q_current'][ids].std(1)[1:];gain=np.abs(y[1:]-y[0]);noise.append(dict(scenario=scene,origin=folder.name,
                    q_std_vs_actual_gain_spearman=float(spearmanr(qstd,gain).statistic) if min(np.ptp(qstd),np.ptp(gain))>0 else None))
        for i,r in enumerate(catalog):
            idx=ids[i];candidates.append(dict(scenario=scene,origin=folder.name,candidate=r['candidate_id'],roles=' '.join(r['roles']),pool_index=int(idx),pool_size=meta['pool_size'],
                original_roles=8,unique_representatives=meta['unique'],first_action=c['actions'][i,0].tolist(),baseline_first=c['baseline'][0].tolist(),
                first_action_delta=float(np.linalg.norm(c['actions'][i,0]-c['baseline'][0])),predicted_h10_value=float(c['scores'][idx]),
                discounted_prefix=float(c['prefix'][idx]),terminal_q=float(c['q_members'][idx,pair].mean()),q_pair=pair.tolist(),
                predicted_common_rank=int(np.sum(c['scores']>c['scores'][idx]))+1,
                original_iteration_q_members=meta['iteration_q_pairs'],full_sequence_source=str((folder/'catalog.npz').relative_to(ROOT)),
                sac_critic_candidate_score=None if i else min(load(folder/'offline_scores.json')['sac_q1'],load(folder/'offline_scores.json')['sac_q2']),
                sac_critic_reason='physicalcandidate cannot be inputasresidual; only originalresidual baseline evaluation known'))
        if 1 in values:
            y1=values[1]
            for h,y in values.items():
                valid=[(i,j) for i,j in itertools.combinations(range(len(y)),2) if min(abs(y[i]-y[j]),abs(y1[i]-y1[j]))>1]
                hrows.append(dict(scenario=scene,origin=folder.name,horizon=h,spearman_H1=float(spearmanr(y1,y).statistic),kendall_H1=float(kendalltau(y1,y).statistic),
                    effective_pairs=len(valid),pair_direction_agreement=float(np.mean([(y[i]-y[j])*(y1[i]-y1[j])>0 for i,j in valid])) if valid else None,
                    top1_same=float(y.argmax()==y1.argmax()),baseline_return=float(y[0]),baseline_minus_H1=float(y[0]-y1[0]),selected_gain=float(y[selected_index]-y[0])))
    agreements=[]
    for origin in sorted({r['origin'] for r in rankrows}):
        rr=[r for r in rankrows if r['origin']==origin and r['horizon']==1]
        if not rr:continue
        row=dict(scenario=rr[0]['scenario'],origin=origin)
        for family in ['avg','min']:
            group=[r for r in rr if r['score'].startswith('current_q_'+family) and len(r['score'])==15]
            row['top1_agreement_'+family]=float(np.mean([a['selected_index']==b['selected_index'] for a,b in itertools.combinations(group,2)]))
        agreements.append(row)
    terminal_paired=[]
    for origin in sorted({r['origin'] for r in rankrows}):
        rr={r['score']:r for r in rankrows if r['origin']==origin and r['horizon']==1}
        if not rr:continue
        base=rr['terminal_candidate']
        for mode in ['terminal_policy_mean','terminal_native_policy_sample']:
            r=rr[mode];terminal_paired.append(dict(scenario=r['scenario'],origin=origin,mode=mode,
                pair_accuracy_gain=None if r['pair_accuracy'] is None or base['pair_accuracy'] is None else r['pair_accuracy']-base['pair_accuracy'],
                regret_improvement=base['top1_regret']-r['top1_regret'],argmax_gain_difference=r['selected_gain']-base['selected_gain']))
    keys=['oracle_best_gain','actual_selected_gain','actual_selected_regret','positive_fraction','negative_fraction','baseline_win_rate','has_positive_candidate','selected_positive','maximum_first_action_delta']
    rankkeys=['pair_accuracy','spearman','kendall','top1_regret','selected_gain','oracle_best_gain','baseline_accuracy','false_improvement_rate','mae','bias','calibration_error']
    result=dict(phase=phase,scenes=scenes,coverage={str(h):scene_bootstrap([r for r in coverage if r['horizon']==h],keys) for h in sorted({r['horizon'] for r in coverage})},
        rankings={str(h):{name:scene_bootstrap([r for r in rankrows if r['horizon']==h and r['score']==name],rankkeys) for name in sorted({r['score'] for r in rankrows})} for h in sorted({r['horizon'] for r in rankrows})},
        horizon_agreement={str(h):scene_bootstrap([r for r in hrows if r['horizon']==h],['spearman_H1','kendall_H1','pair_direction_agreement','top1_same','baseline_minus_H1','selected_gain']) for h in sorted({r['horizon'] for r in hrows})},
        q_variance_relationship=scene_bootstrap(noise,['q_std_vs_actual_gain_spearman']),q_pair_top1_agreement=scene_bootstrap(agreements,['top1_agreement_avg','top1_agreement_min']),
        terminal_action_paired={mode:scene_bootstrap([r for r in terminal_paired if r['mode']==mode],['pair_accuracy_gain','regret_improvement','argmax_gain_difference']) for mode in ['terminal_policy_mean','terminal_native_policy_sample']},
        observed_h1_positive_candidates=sum(r['horizon']==1 and r['has_positive_candidate'] for r in coverage),
        coverage_rows=coverage,ranking_rows=rankrows,horizon_rows=hrows,scope='Measuredrepresentatives only; nofullGaussianoracle. H1isonephysicalactionthenfeedbackSAC, not closedloopMPPI. Argmax scorer metrics separatefromactualMPPIweightedMeanoutput.')
    dump(REPORT/f'{phase}_statistics.json',result)
    table(REPORT/f'{phase}_candidate_effects.csv',effects);table(REPORT/f'{phase}_ranking_rows.csv',rankrows)
    table(REPORT/'first_action_candidates.csv',candidates);table(REPORT/'first_action_returns.csv',returns);table(REPORT/'paired_candidate_effects.csv',effects);table(REPORT/'horizon_protocol_comparison.csv',hrows)
    h1=result['coverage']['1'];actualmaps=len(set(r['scenario'] for r in coverage if r['horizon']==1));positive_maps=len(set(r['scenario'] for r in coverage if r['horizon']==1 and r['has_positive_candidate']))
    original_rank=result['rankings']['1']['current_mppi_h10']['pair_accuracy']
    evaluation_gap=bool(original_rank['ci95'][0] is None or original_rank['ci95'][0]<=.5 or h1['actual_selected_gain']['ci95'][0]<=0)
    positive_coverage=bool(h1['oracle_best_gain']['ci95'][0]>0 and positive_maps>=actualmaps/2)
    decision=dict(phase=phase,expand=bool(result['observed_h1_positive_candidates']),positive_coverage_gate=positive_coverage,
        training_coverage_gate=bool(positive_coverage and evaluation_gap),
        evaluation_gap_gate=evaluation_gap,original_rank=original_rank,
        maps=actualmaps,positive_maps=positive_maps,default_output_gain=h1['actual_selected_gain'],oracle=h1['oracle_best_gain'],online=False)
    dump(REPORT/f'{phase}_decision.json',decision);print(phase,clean(decision),flush=True)
    return result

if __name__=='__main__':
    p=load(REPORT/'protocol.json');phase=sys.argv[1]
    if phase=='pilot':analyze(p['pilot_scenes'],phase)
    elif phase=='development':analyze(p['pilot_scenes']+p['extra_development_scenes'],phase)
    elif phase=='validation':analyze(p['validation_scenes'],phase)
    else:raise ValueError(phase)
