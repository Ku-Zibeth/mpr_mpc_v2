"""All100 scenario-paired statistics; preserve failures and original seed order."""
import csv, hashlib, json, math, sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation_statistics import summarize_runs, paired_comparisons
from tools.benchmark_mppi import source_fingerprint
REPORT=Path(__file__).parent
m=json.loads((REPORT/'paired_run.json').read_text());out=Path(m['output'])
if (out/'summary.json').exists():raise FileExistsError('Refusing to overwrite a completed summary.')
statuses=json.loads((out/'worker_statuses.json').read_text());assert len(statuses)==8 and not any(statuses.values()),statuses
assert source_fingerprint(ROOT)==m['source_snapshot_sha256']
assert hashlib.sha256(Path(m['checkpoint']).read_bytes()).hexdigest()==m['checkpoint_sha256']
def convert(value):
 if value=='':return None
 try:
  n=float(value);return n if math.isfinite(n) else None
 except (ValueError,TypeError):return value
all_rows=[];groups={}
for group,label in [('B','B_lattice_sac'),('D','D_lattice_sac_mppi')]:
 files=sorted(out.glob(f'{group}_part*/episodes_{label}_seed1.csv'));assert len(files)==4
 data=[{k:convert(v) for k,v in row.items()} for p in files for row in csv.DictReader(p.open())]
 assert len(data)==len({(r['scenario'],r['test_seed']) for r in data})==100
 assert {int(r['scenario']) for r in data}==set(m['scenarios'])
 for r in data:r.update(configuration=label,scenario=int(r['scenario']),test_seed=int(r['test_seed']))
 data.sort(key=lambda r:m['scenarios'].index(r['scenario']));groups[label]=data;all_rows+=data
fields=list(dict.fromkeys(k for r in all_rows for k in r))
def csv_write(name,rows,keys=None):
 with (out/name).open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=keys or list(dict.fromkeys(k for r in rows for k in r)))
  writer.writeheader();writer.writerows(rows)
csv_write('episodes.csv',all_rows,fields)
for name,rows in groups.items():csv_write(f'episodes_{name}_seed1.csv',rows,fields)
summaries=[summarize_runs(k,rows,samples=2000,seed=1) for k,rows in groups.items()]
paired=paired_comparisons(all_rows,samples=2000,seed=1,pairs=[('B_lattice_sac','D_lattice_sac_mppi')])
contributions=[]
for b,d in zip(*groups.values()):
 row=dict(scenario=b['scenario'],test_seed=1)
 for name,key in [('success','final_success'),('crash','episode_crash'),('out_of_road','episode_out_of_road'),
                  ('route_completion','final_route_completion'),('reward','return'),('actual_discounted_return','actual_discounted_return'),('timeout','timeout')]:
  row['B_'+name],row['D_'+name]=b.get(key),d.get(key)
  row[name+'_delta']=d[key]-b[key] if b.get(key) is not None and d.get(key) is not None else None
 for name,key in [('predicted_gain','mppi_score_gain'),('optimized_gain','mppi_predicted_return_gain'),
                  ('diversity','mppi_sampling_diversity'),('first_action_correction','mppi_action_correction_norm'),
                  ('prediction_bias','initial_predicted_return_bias'),('reward_mae','one_step_reward_mae'),('planning_ms','mean_mpr_total_planning_ms')]:
  row['D_'+name]=d.get(key)
 contributions.append(row)
csv_write('summary.csv',summaries);csv_write('paired_comparisons.csv',paired);csv_write('scenario_contributions.csv',contributions)
observations=[(r['D_predicted_gain'],r['reward_delta']) for r in contributions if r['D_predicted_gain'] is not None and r['reward_delta'] is not None]
correlation=float(np.corrcoef(np.asarray(observations).T)[0,1]) if len(observations)>2 else None
result=dict(manifest=m,configurations=summaries,paired_comparisons=paired,cases=contributions,
 end_reasons={label:dict(Counter(r['end_reason'] for r in rows)) for label,rows in groups.items()},
 episode_prediction_gain_vs_actual_reward_delta_correlation=correlation,
 interpretation='Correlation of episode-average predicted gain and paired episode reward change is descriptive, not matched-state candidate ranking validation. Forecast-vs-realized return biases compare frozen-Q continuation with deployed receding MPC/SAC continuation and cannot isolate dynamics, reward, or Q errors.',
 scope='Fresh B100 and D100, original fixed IDs/order, seed1, same evaluator/environment/checkpoint/horizon1000. No training or candidate certificates. Timings are recorded under shared RTX3070 load.')
(out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False));(REPORT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(out,flush=True)
