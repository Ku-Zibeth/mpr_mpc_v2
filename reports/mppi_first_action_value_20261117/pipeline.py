"""Evidence-gated experiment continuation; no online changes without validation."""
import os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *
from batch import run_jobs
ENV=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')

def run(script,args,log):
    print('START',script,args,flush=True)
    with (REPORT/log).open('x') as f:subprocess.run([sys.executable,'-u',str(REPORT/script),*args],cwd=ROOT,env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
    print('FINISH',script,args,flush=True)

p=load(REPORT/'protocol.json')
while not (REPORT/'pilot_decision.json').exists():time.sleep(1)
pilot=load(REPORT/'pilot_decision.json')
jobs=[]
for scene in p['pilot_scenes'][:2]:
    before=p['before_steps'][0];folder=REPORT/'origins'/f'{scene}_{before}'
    if not (folder/'catalog.json').exists():continue
    for policy in ['td_actor','td_mpc']:
        key=f'{scene}_{before}_{policy}';jobs.append((key,'collect.py',['--scene',scene,'--before',before,'--candidate','c0','--horizon',1,'--takeover',policy],REPORT/'branches'/folder.name/f'H1_c0_{policy}.json'))
scene=p['pilot_scenes'][0];before=p['before_steps'][0]
jobs.append(('deterministic_replay','collect.py',['--scene',scene,'--before',before,'--candidate','c0','--horizon',1,'--replay'],REPORT/'branches'/f'{scene}_{before}'/'H1_c0_baseline_replay.json'))
run_jobs(jobs,'native_takeover_and_replay',workers=2)
# H2/4/10 are always completed as controls; never called H1 labels.
run('batch.py',['pilot_other_h'],'pilot_other_h.log')
run('analyze.py',['pilot'],'pilot_full_horizon_analysis.log')
if pilot['expand']:
    run('batch.py',['prepare_extra'],'prepare_extra.log')
    run('enrich.py',[],'enrich_development.log')
    run('batch.py',['extra_h1'],'extra_h1.log')
    run('analyze.py',['development'],'development_analysis.log')
    decision=load(REPORT/'development_decision.json')
    if decision['training_coverage_gate'] and decision['evaluation_gap_gate']:
        run('train_offline.py',[],'train_offline.log')
        trained=load(REPORT/'training_decision.json')
        if trained['proceed_to_one_validation']:
            run('batch.py',['prepare_validation'],'prepare_validation.log')
            run('enrich.py',[],'enrich_validation.log')
            run('batch.py',['validation_h1'],'validation_h1.log')
            run('analyze.py',['validation'],'validation_analysis.log')
            run('validate_offline.py',[],'validation_value.log')
    else:dump(REPORT/'training_decision.json',dict(proceed_to_one_validation=False,trained=False,reason='Either no stablemulti-map measuredH1coverage, or no demonstrated evaluationgap; do nottrain.'))
else:dump(REPORT/'training_decision.json',dict(proceed_to_one_validation=False,trained=False,reason='Pilot has no >1 measuredH1candidategain; stopnoexpansion/training.'))
dump(REPORT/'collection_pipeline_complete.json',dict(pilot_expanded=pilot['expand'],completed=True,online_defaults_changed=False))
print('COLLECTION/TRAINING/VALIDATION GATES COMPLETE',flush=True)
