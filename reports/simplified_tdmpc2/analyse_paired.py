"""Reuse existing scenario-paired bootstrap statistics; never select best seeds."""
import json,csv,math,sys,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation_statistics import summarize_runs,paired_comparisons,METRICS
METRICS["reward"]=("return",None)
from tools.benchmark_mppi import source_fingerprint
REPORT=Path(__file__).parent;m=json.load(open(REPORT/'paired_run.json'));out=Path(m['output'])
if (out/'summary.json').exists():raise FileExistsError('Refusing to overwrite final evaluation')
assert hashlib.sha256(Path(m['baseline_csv']).read_bytes()).hexdigest()==m['baseline_csv_sha256']
assert source_fingerprint(ROOT)==m['source_snapshot_sha256']
def convert(v):
 if v=='':return None
 try:
  n=float(v);return n if math.isfinite(n) else None
 except (ValueError,TypeError):return v
rows=[];groups={}
for label,files in [('B_lattice_sac',[Path(m['baseline_csv'])]),('D_lattice_sac_mppi',sorted(out.glob('part[1-4]/episodes_D_lattice_sac_mppi_seed1.csv')))]:
 data=[{k:convert(v) for k,v in row.items()} for p in files for row in csv.DictReader(p.open())]
 assert len(data)==100 and {int(r['scenario']) for r in data}==set(m['scenarios'])
 assert len({(int(r['scenario']),int(r['test_seed'])) for r in data})==100
 for r in data:r.update(configuration=label,scenario=int(r['scenario']),test_seed=int(r['test_seed']))
 data.sort(key=lambda r:m['scenarios'].index(r['scenario']));groups[label]=data;rows+=data
keys=list(dict.fromkeys(k for r in rows for k in r))
for name,data in [('episodes.csv',rows)]+[(f'episodes_{k}_seed1.csv',v) for k,v in groups.items()]:
 with (out/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
summaries=[summarize_runs(k,v,samples=2000,seed=1) for k,v in groups.items()]
comparisons=paired_comparisons(rows,samples=2000,seed=1,pairs=[('B_lattice_sac','D_lattice_sac_mppi')])
contributions=[]
for a,b in zip(*groups.values()):
 r=dict(scenario=a['scenario'],test_seed=a['test_seed'])
 for name,key in [('success','final_success'),('crash','episode_crash'),('out_of_road','episode_out_of_road'),('route','final_route_completion'),('return','return'),('timeout','timeout')]:
  r['B_'+name]=a.get(key);r['D_'+name]=b.get(key)
  if a.get(key) is not None and b.get(key) is not None:r[name+'_delta']=b[key]-a[key]
 r.update(D_score_gain=b.get('mppi_score_gain'),D_first_action_delta=b.get('mppi_first_action_delta_norm'),
          D_sequence_rms_delta=b.get('mppi_correction_magnitude'),D_planning_ms=b.get('mean_mpr_total_planning_ms'))
 contributions.append(r)
for name,data in [('summary.csv',summaries),('paired_comparisons.csv',comparisons),('scenario_contributions.csv',contributions)]:
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in data for k in r)));w.writeheader();w.writerows(data)
result=dict(manifest=m,configurations=summaries,paired_comparisons=comparisons,
 terminal_reasons={k:dict(Counter(str(r.get('end_reason')) for r in v)) for k,v in groups.items()},
 cases=contributions,scope='B reused from immutable complete100, D newly evaluated. Same frozen checkpoint/env/scenario IDs/test seed/horizon. No candidate safety certificates; event rates come from the environment. Parallel latency is not an isolated real-time benchmark.')
(out/'summary.json').write_text(json.dumps(result,indent=2));(out/'scenarios.json').write_text(json.dumps({'scenarios':m['scenarios']},indent=2))
(REPORT/'summary.json').write_text(json.dumps(result,indent=2));print(out,flush=True)
