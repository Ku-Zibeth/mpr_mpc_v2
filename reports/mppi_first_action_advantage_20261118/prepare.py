"""Freeze Phase3 protocol BEFORE reading new fit/inner-stop advantage labels."""
import csv
import importlib.metadata as md
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from study_io import R,ROOT,PRIOR,OLD,PHASE2,load,sha,dump,table
import numpy as np
from models import features,relative_targets,balanced_weights


def main():
    parent='7d15cdda92ff434024bb72c5dd324d05fa3fe3d0'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==parent
    assert subprocess.check_output(['git','diff','--name-only'],cwd=ROOT,text=True).strip()==''
    old=load(OLD/'protocol.json');chosen=load(OLD/'training_decision.json')['selected']
    assert sha(Path(old['checkpoint']))=='4d0f1d43749794d1d1c28533309cbe75bc3712d25a92936ca7b477c30014810a'
    assert sha(PRIOR/chosen['checkpoint'])==chosen['checkpoint_sha']
    protocol=dict(parent_commit=parent,phase2_report_commit=parent,date='2026-11-18',
        old_report_generation_base='27f8131d95a39bfe2f4a95d101fef1774903b907',
        checkpoint=old['checkpoint'],checkpoint_sha256=old['checkpoint_sha256'],
        A0_checkpoint=str(PRIOR/chosen['checkpoint']),A0_sha256=chosen['checkpoint_sha'],A0_context=chosen['context'],
        data_root=str(OLD),fit_maps=old['fit_scenes'],inner_stop_maps=old['inner_stop_scenes'],
        excluded_seen_validation=old['validation_scenes'],horizon=1,gamma=.995,
        reward='info.tdmpc2_reward',continuation='original frozen deterministic Lattice+SAC until native done',
        target='actual Y(candidate)-actual Y(unique same-origin Baseline)',tolerance=1.,
        feature_schemas={'A0':'CURRENT535 + absolute physical action2, frozen OLD control only',
                         'A1':'CURRENT535 + candidate-minus-Baseline action2 (537)',
                         'A2':'CURRENT535 + Baseline physical2 + action delta2 (539)'},
        architectures=['linear','MLP32SiLU16SiLU1'],new_absolute_return_training=False,
        output='(raw_f(x)-raw_f(same context, zero delta))*fit_target_RMS; exact Baseline0',
        training=dict(epochs=300,seed=1,optimizer='Adam',lr=.001,linear_weight_decay=.001,mlp_weight_decay=.0001,
            lambda_rank=.25,regression='SmoothL1 on advantage / fit-only weighted RMS',
            ranking='softplus(-sign(normalized actual difference)*normalized predicted difference)',
            pair_scope='same origin, exclude raw abs(delta_Y_i-delta_Y_j)<=1',
            normalization='fit-only weighted input mean/std(min.001), label weighted RMS(min1); no label mean shift',
            weights='each scene equal, each origin equal inside scene, each candidate equal inside origin; pair weights balance valid origins/maps',
            LOMO='8 fixed folds, seven fit maps train, one fit map test; fixed300 epochs; no inner-stop early stopping'),
        knn=dict(schema='A2',k=8,distance='block-normalized Euclidean of current535,baseline2,delta2; fit-only normalization',
            weighting='fit scene/origin/candidate weight divided by distance(min1e-6)',baseline_centered=True,diagnostic_only=True),
        A0_lomo_scope='Frozen A0 already trained on all8fit maps. Show as in-sample reference, NOT an independent LOMO model. No new absolute-return refit.',
        selection='Once, inner-stop4: pair accuracy DESC,Top1Regret ASC,ModelGain DESC; stable tie linear before MLP; KNN not eligible. No old seen-validation data.',
        gate=dict(lomo='For inner-stop-selected A2 kind: accuracy CI lower>.5,Spearman CI lower>0,ModelGain CI lower>=0; do NOT use leaked A0 LOMO as a held-out comparator.',
            inner_stop='A2 accuracy point>.5,Spearman point>0, paired Top1Regret improvement vs frozen A0 point>0, ModelGain>=0, selected_negative no greater than A0.',
            validation='Only if BOTH prior gates pass: freeze selected model SHA; preregister6 unused maps/12origins from110000..110999, seed20261118, before50/150. No scene replacement.',
            final='Accuracy CI lower>.5,Spearman CI lower>0,Regret improvement vs A0 CI lower>0, ModelGain>=0, selected-negative increase CI upper<=0, gains not single-map-dependent; never deploy in this offline task.'),
        bootstrap=dict(samples=2000,seed=20261118,unit='whole map; origins averaged inside map'),
        magnitude_edges=[.01,.05,.1],benchmark=dict(warmup=20,repeats=200,batches=[1,258],not_full_planner=True),
        new_environment_runs_before_gates=0,online_changes=False,commit=False,push=False)
    dump('protocol.json',protocol)
    shutil.copyfile('/home/kzb/.codex/attachments/57f96211-5de8-4730-b15e-b16c88627326/已粘贴的文本.txt',R/'task_prompt.txt')
    required=['MPPI_FIRST_ACTION_FAILURE_REPORT_20261118.md','decision.json','verification.json',
        'model_implementation_audit.json','oracle_candidate_report.json','state_oracle_ranking.json',
        'dynamics_error_analysis.json','ranking_analysis.json','stratified_ranking_statistics.json','feature_shift_and_sensitivity.csv']
    inspected=[]
    for f in [*[PHASE2/name for name in required],*[OLD/name for name in ['train_offline.py','offline_model.py','validate_offline.py','training_results.json','training_decision.json']]]:
        if f.suffix=='.json':content=load(f);summary=list(content)[:20] if isinstance(content,dict) else f'{len(content)} rows'
        elif f.suffix=='.csv':content=list(csv.DictReader(f.open()));summary=f'{len(content)} rows'
        else:content=f.read_text();summary=f'{len(content)} characters read'
        inspected.append(dict(path=str(f),sha256=sha(f),review=summary))
    protected=[ROOT/'config.yaml']
    for directory in ['planning','residual_rl','lattice','mpr_mpc','vendor/tdmpc2_runtime']:
        protected.extend((ROOT/directory).rglob('*.py'))
    packages={n:md.version(n) for n in ['torch','numpy','metadrive-simulator']}
    dump('experiment_manifest.json',dict(parent_commit=parent,phase2_report_commit=parent,
        remote_main=dict(sha=parent,verified_via='GitHub connector fetch_commit(main)',date='2026-11-18'),
        report_version_note='Phase2 report described its generation-time27f8131 base; published by later7d15cdd. Historical report not edited.',
        checkpoint_sha256=old['checkpoint_sha256'],config_sha256=sha(Path(old['source_environment_config'])),
        effective_default_config_sha256=sha(ROOT/'config.yaml'),A0_sha256=chosen['checkpoint_sha'],
        binary_provenance='Git-ignored .pt/.npz remain in prior worktree. Read in place; no re-created substitute binaries.',
        python=sys.version,packages=packages,prior_reports_read=inspected,
        protected_source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted(set(protected))},
        prompt_sha256=sha(R/'task_prompt.txt'),protocol_sha256=sha(R/'protocol.json')))
    rows=[];input_files=[]
    for split,scenes in [('fit',protocol['fit_maps']),('inner_stop',protocol['inner_stop_maps'])]:
        arrays={k:[] for k in ['A0','A1','A2','y','delta_y','scene','origin','candidate','baseline']}
        for scene in scenes:
            for before in old['before_steps']:
                name=f'{scene}_{before}';folder=OLD/'origins'/name
                c=np.load(folder/'catalog.npz');meta=load(folder/'catalog.json')
                baselines=[i for i,r in enumerate(meta['representatives']) if 'baseline' in r['roles']]
                assert len(baselines)==1 and baselines[0]==0
                baseline=c['baseline'][0];local=[]
                input_files.extend([folder/'catalog.npz',folder/'catalog.json'])
                for i,rep in enumerate(meta['representatives']):
                    f=OLD/'branches'/name/f"H1_{rep['candidate_id']}_baseline.json";rec=load(f)
                    archive=PRIOR/rec['source'];input_files.extend([f,archive])
                    with np.load(archive) as a:
                        assert a['done'][-1] and not a['done'][:-1].any()
                        assert np.array_equal(a['done'],a['terminated']|a['truncated'])
                        total=0.
                        for reward in a['rewards'][::-1]:total=float(reward)+.995*total
                        assert abs(total-rec['actual_return'])<1e-8
                        assert np.array_equal(a['actions'][0],c['actions'][i,0])
                    assert rec['horizon']==1 and rec['takeover']=='baseline' and not rec['replay']
                    assert np.array_equal(rec['baseline_first_action'],baseline)
                    local.append(rec)
                y=np.array([r['actual_return'] for r in local]);dy=y-y[0]
                for i,(rep,rec) in enumerate(zip(meta['representatives'],local)):
                    candidate=c['actions'][i,0]
                    for schema in ['A0','A1','A2']:arrays[schema].append(features(c['state535'],baseline,candidate,schema))
                    arrays['y'].append(y[i]);arrays['delta_y'].append(dy[i]);arrays['scene'].append(scene);arrays['origin'].append(name)
                    arrays['candidate'].append(i);arrays['baseline'].append(i==0)
                    rows.append(dict(split=split,scene=scene,origin=name,candidate=i,baseline=i==0,
                        action=candidate.tolist(),baseline_action=baseline.tolist(),delta_action=(candidate-baseline).tolist(),
                        Y=float(y[i]),Y_B=float(y[0]),delta_Y=float(dy[i]),first_reward=rec['actual_first_reward'],
                        gamma=.995,native_done=rec['native_done'],source_json=str(OLD/'branches'/name/f'H1_c{i}_baseline.json'),
                        features_source=str(folder/'catalog.npz')+'::state535,baseline,actions',features_only_current=True))
        arrays={k:np.asarray(v) for k,v in arrays.items()}
        assert np.array_equal(relative_targets(arrays['y'],arrays['origin'],arrays['baseline']),arrays['delta_y'])
        w=balanced_weights(arrays['scene'],arrays['origin']);arrays['weights']=w
        np.savez_compressed(R/f'{split}_dataset.npz',**arrays)
        chosen_rows=[r for r in rows if r['split']==split]
        for i,r in enumerate(chosen_rows):
            count=int((arrays['origin']==r['origin']).sum());os=len(set(arrays['origin'][arrays['scene']==r['scene']]))
            r.update(scene_weight=1/len(scenes),origin_weight=1/len(scenes)/os,candidate_weight=float(w[i]),candidate_count_origin=count)
    table('advantage_dataset.csv',rows)
    table('input_sha256.csv',[dict(path=str(f),sha256=sha(f)) for f in sorted(set(input_files))])
    dump('feature_schema.json',dict(A0=dict(dimension=537,feature='current535+absolute action2',frozen=True),
        A1=dict(dimension=537,feature='current535+action delta2'),A2=dict(dimension=539,feature='current535+Baseline2+action delta2'),
        no_future_features=True,baseline_reference='unique actual same-origin candidate',
        note='Centered linear A1/A2 cancel context and Baseline terms; do not interpret equality as lack of useful context.'))
    print('Protocol frozen; dataset rows:',len(rows),'fit/inner only. No seen-validation branches opened.',flush=True)


if __name__=='__main__':main()
