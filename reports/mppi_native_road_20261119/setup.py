from study import *
import pandas as pd
if __name__=='__main__':
    p=load(PREV/'protocol.json');rng=np.random.default_rng(20261121)
    excluded=set(load(PREV/'exclusions.json')['scene_ids'])|set(p['development_scenes'])|set(p['holdout_scenes'])
    maps=[int(s) for s in rng.permutation(np.arange(600000,601000)) if int(s) not in excluded][:12]
    cfg=load(PREV/'corrected_variant_configs.json')['T5_feedback_corrected'];cfg['tracking'].pop('min_terminal_error_reduction_m',None)
    errors=pd.read_csv(PREV/'motion_prediction_errors_corrected.csv');check=errors[(errors.split=='check_old12')&(errors.model=='calibrated_cg')&(errors.horizon==5)]
    margin=float(check[check.metric=='position_m'].p95.mean()+.5*4.515*check[check.metric=='heading_rad'].p95.mean())
    variants={'Ttrack':cfg}
    import copy
    for tag,conditional in [('Tnative',False),('Tconditional',True)]:
        c=copy.deepcopy(cfg);c['tracking'].update(native_road=True,native_risk_only=conditional,native_margin_m=margin)
        variants[tag]=c
    dump(R/'variant_configs.json',variants)
    p.update(development_scenes=maps,development_methods=['A',*variants],development_episodes=48,holdout_methods=['A','C'],holdout_episodes_if_promising=96,
        selection_rule='development: no success/crash/OOR deterioration; route>=A-0.005; mean discounted gain>0; positive gains on >=6/12 maps. Choose success then return. Otherwise do not use holdout.',
        native_hypothesis='Actual physics solid-line/sidewalk boxes represent native line contacts better than the road union. Add the same continuous coefficient20 term, compare always-search and nominal-risk-only search, steering-only, no reward/done modifications.',
        native_prediction_envelope_m=margin,native_envelope_source='old24 trajectories only: mean map H5 position P95 + half vehicle length*heading P95; NOT a probabilistic safety guarantee',
        no_development_tuning=True,technical_gate='Validate actual-state planar contact vs original native contact labels on existing traces before collecting any new driving outcomes.',
        native_zero_risk_policy='Risk-only variant executes exact SAC if baseline line/road/obstacle modeled risk is zero; when active retain64 proposals+baseline,8elites,4iterations.',
        whole_road_union_not_native_OOR=True,previous_holdout_preserved=True)
    dump(R/'protocol.json',p)
    dump(R/'preregistration.json',dict(development_maps=maps,variants=variants,episodes=48,controls='same checkpoint, scenario+traffic RNG, physical calibration, sampling parameters, SAC goal/path, current pedal',
        stop='If no variant passes the frozen development gate, stop this hypothesis without independent validation. No outcome-driven threshold tuning.',holdout=p['holdout_scenes'],independent_holdout_not_viewed=True))
    print('NEW_MAPS',maps,'MARGIN',margin,flush=True)
