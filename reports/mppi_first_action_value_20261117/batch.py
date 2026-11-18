"""Bounded resumable subprocess queue; never overwrites completed native branches."""
import os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.first_action_study import *

ENV=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MUJOCO_GL='egl')

def run_jobs(jobs,stage,workers=4):
    logs=REPORT/'logs'/stage;logs.mkdir(parents=True,exist_ok=True);queue=[];done={};running=[]
    for key,script,args,target in jobs:
        if Path(target).exists():done[key]=0;continue
        queue.append((key,script,args,target))
    expected=len(jobs)
    while queue or running:
        while queue and len(running)<workers and not any(done.values()):
            key,script,args,target=queue.pop(0);attempt=0
            while (logs/f'{key}_attempt{attempt}.log').exists():attempt+=1
            stream=(logs/f'{key}_attempt{attempt}.log').open('x')
            job=subprocess.Popen([sys.executable,'-u',str(REPORT/script),*map(str,args)],cwd=ROOT,env=ENV,stdout=stream,stderr=subprocess.STDOUT)
            running.append((key,job,stream,target))
        for key,job,stream,target in running[:]:
            status=job.poll()
            if status is not None:
                running.remove((key,job,stream,target));stream.close();done[key]=status;dump(REPORT/f'{stage}_status.json',done)
                print(stage,len(done),'/',expected,key,status,flush=True)
        if any(done.values()) and not running:break
        if running:time.sleep(1)
    assert len(done)==expected and not any(done.values()),done
    dump(REPORT/f'{stage}_status.json',done)

def preparation(scenes,stage):
    p=load(REPORT/'protocol.json');jobs=[]
    for scene in scenes:
        for before in p['before_steps']:
            folder=REPORT/'origins'/f'{scene}_{before}'
            if (folder/'unreachable.json').exists():continue
            if folder.exists() and not (folder/'catalog.json').exists():
                archive=REPORT/'failed_prepare'/folder.name;archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists();folder.rename(archive)
            jobs.append((folder.name,'prepare.py',['--scene',scene,'--before',before],folder/'catalog.json'))
    run_jobs(jobs,stage)

def branches(scenes,horizons,stage):
    p=load(REPORT/'protocol.json');jobs=[]
    for scene in scenes:
        for before in p['before_steps']:
            folder=REPORT/'origins'/f'{scene}_{before}'
            if (folder/'unreachable.json').exists():continue
            meta=load(folder/'catalog.json')
            for c in meta['representatives']:
                for h in horizons:
                    key=f'{scene}_{before}_H{h}_{c["candidate_id"]}'
                    jobs.append((key,'collect.py',['--scene',scene,'--before',before,'--candidate',c['candidate_id'],'--horizon',h],REPORT/'branches'/folder.name/f'H{h}_{c["candidate_id"]}_baseline.json'))
    run_jobs(jobs,stage)

if __name__=='__main__':
    p=load(REPORT/'protocol.json');stage=sys.argv[1]
    if stage=='prepare_pilot':preparation(p['pilot_scenes'],stage)
    elif stage=='pilot_h1':branches(p['pilot_scenes'],[1],stage)
    elif stage=='pilot_other_h':branches(p['pilot_scenes'],[2,4,10],stage)
    elif stage=='prepare_extra':preparation(p['extra_development_scenes'],stage)
    elif stage=='extra_h1':branches(p['extra_development_scenes'],[1],stage)
    elif stage=='prepare_validation':preparation(p['validation_scenes'],stage)
    elif stage=='validation_h1':branches(p['validation_scenes'],[1],stage)
    else:raise ValueError(stage)
