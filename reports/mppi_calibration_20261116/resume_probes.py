"""Drain two active workers, then resume only the queue at higher concurrency."""
import json,os,shlex,signal,subprocess,sys,time
from pathlib import Path
R=Path(__file__).parent;ledger=json.loads((R/'orchestrator_restart.json').read_text());parent=ledger['parent'];children=ledger['children']
while True:
 states=[]
 for line in children:
  pid=int(line.split()[0]);parts=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();states.append((parts[0],int(parts[49])))
 if all(state=='Z' for state,_ in states):break
 time.sleep(2)
assert all(code==0 for _,code in states),states
statusfile=R/'probes_worker_statuses.json';statuses=json.loads(statusfile.read_text());meta=json.loads((R/'protocol.json').read_text())
for line in children:
 words=shlex.split(line);s=int(words[words.index('--scenario')+1]);before=int(words[words.index('--before')+1]);b=words[words.index('--branch')+1];split='dev' if s in meta['dev_scenarios'] else 'validation'
 assert (R/'probes'/f'{split}_{s}_{before}'/(b+'.json')).exists();statuses[f'{split}_{s}_{before}_{b}']=0
statusfile.write_text(json.dumps(statuses,indent=2));ledger['drained_exit_codes']=[code for _,code in states]
os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT);ledger['resumed_jobs']=4
(R/'orchestrator_restart.json').write_text(json.dumps(ledger,indent=2))
with (R/'probes_resume.log').open('x') as f:
 result=subprocess.run([sys.executable,'-u',str(R/'run.py'),'--kind','probes','--jobs','4','--resume'],stdout=f,stderr=subprocess.STDOUT,env=os.environ)
assert result.returncode==0
