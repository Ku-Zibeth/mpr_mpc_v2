"""Lock development-only selection before analyzing held-out validation."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent;meta=json.loads((R/'protocol.json').read_text())
def complete(split):
 for s in meta[split+'_scenarios']:
  for before in meta['origin_steps']:
   p=R/'probes'/f'{split}_{s}_{before}'
   for b in ['B',*meta['groups']]+(['elite_Q1'] if s in meta['elite_scenarios'] else []):
    if not any(x.exists() for x in (p/(b+'.json'),p/('unavailable_'+b+'.json'),p/'unavailable.json')):return False
 return True
def loop_done(split,count):
 p=R/(split+'_worker_statuses.json')
 if not p.exists():return False
 x=json.loads(p.read_text());return len(x)==count and not any(x.values())
def run(args,name):
 with (R/name).open('x') as f:
  p=subprocess.run([sys.executable,'-u',*args],stdout=f,stderr=subprocess.STDOUT,env=os.environ);assert p.returncode==0,(name,p.returncode)
while not (complete('dev') and loop_done('dev',20)):time.sleep(5)
run([str(R/'analyze.py'),'--phase','dev'],'dev_analysis.log')
while not (complete('validation') and loop_done('validation',30)):time.sleep(5)
run([str(R/'analyze.py'),'--phase','validation'],'validation_analysis.log')
selection=json.loads((R/'selection.json').read_text())
if selection['validation_gate_passed']:
 run([str(R/'run.py'),'--kind','test','--jobs','3'],'test_run.log')
 run([str(R/'analyze.py'),'--phase','test'],'test_analysis.log')
(R/'analysis_complete.json').write_text(json.dumps(dict(selection=selection,formal100=False,training=False),indent=2))
