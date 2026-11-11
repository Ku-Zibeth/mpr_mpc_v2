"""Merge completed disjoint B/D shards; reuse existing paired bootstrap statistics."""
import csv,json,sys,math,hashlib,subprocess
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluation_statistics import summarize_runs,paired_comparisons
from tools.benchmark_mppi import source_fingerprint
folder=Path(__file__).parent;meta=json.load(open(folder/'paired_run.json'))
scenes=json.load(open(ROOT/'scenarios/fixed100.json'));scenes=scenes['scenarios'] if isinstance(scenes,dict) else scenes
rows=[];groups={}
def number(x):
 if x=='':return None
 try:
  f=float(x);return f if math.isfinite(f) else None
 except (ValueError,TypeError):return x
for label,base in [('B_lattice_sac',meta['baseline_output']),('D_lattice_sac_mppi',meta['output'])]:
 found=[]
 for file in sorted(Path(base).glob(f'part[1-4]/episodes_{label}_seed1.csv')):
  for row in csv.DictReader(file.open()):
   row={k:number(v) for k,v in row.items()};row['scenario']=int(row['scenario']);row['test_seed']=int(row['test_seed']);row['configuration']=label
   if label=='D_lattice_sac_mppi':
    fix=json.load(open(folder/'diagnostic_attribution_fix.json'))
    assert float(row.get('mppi_mode_recovery_lattice_rate') or 0)==0
    for key in fix['normalized_fields']:
     if key in row and row[key] is not None:
      row['recorded_'+key]=row[key];row[key]=0.
   found.append(row)
 if len(found)!=100 or {r['scenario'] for r in found}!=set(scenes) or len({(r['scenario'],r['test_seed']) for r in found})!=100:
  raise SystemExit(f'{label}: incomplete or duplicate observations ({len(found)}/100); no final summary written.')
 found.sort(key=lambda r:scenes.index(r['scenario']));groups[label]=found;rows+=found
out=ROOT/'logs/benchmark'/('sac_refinement_BD_final100_'+Path(meta['output']).name.rsplit('_',2)[-2]+'_'+Path(meta['output']).name.rsplit('_',1)[-1])
out.mkdir(exist_ok=True)
if (out/'summary.json').exists():raise FileExistsError('Final results already exist; refusing overwrite')
keys=list(dict.fromkeys(k for row in rows for k in row))
for name,data in [('episodes.csv',rows)]+[(f'episodes_{label}_seed1.csv',data) for label,data in groups.items()]:
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
summaries=[summarize_runs(k,v,samples=2000,seed=1) for k,v in groups.items()]
comparisons=paired_comparisons(rows,samples=2000,seed=1,pairs=[('B_lattice_sac','D_lattice_sac_mppi')])
for name,data in [('summary.csv',summaries),('paired_comparisons.csv',comparisons)]:
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in data for k in row)));w.writeheader();w.writerows(data)
contributions=[]
for a,b in zip(*groups.values()):
 contributions.append(dict(scenario=a['scenario'],test_seed=a['test_seed'],
  B_success=a['final_success'],D_success=b['final_success'],success_delta=b['final_success']-a['final_success'],
  B_route=a['final_route_completion'],D_route=b['final_route_completion'],route_delta=b['final_route_completion']-a['final_route_completion'],
  B_crash=a['episode_crash'],D_crash=b['episode_crash'],B_offroad=a['episode_out_of_road'],D_offroad=b['episode_out_of_road'],
  B_timeout=a.get('timeout'),D_timeout=b.get('timeout'),D_safety_failure=b.get('safety_failure'),
  D_baseline_selected=b.get('mppi_baseline_selected'),D_performance_modification=b.get('mppi_performance_modification'),
  D_safety_modification=b.get('mppi_safety_modification'),D_model_uncertainty=b.get('mppi_model_uncertainty')))
with (out/'scenario_contributions.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(contributions[0]));w.writeheader();w.writerows(contributions)
diagnostics={}
for label,data in groups.items():
 diagnostics[label]=dict(terminal_reasons=dict(Counter(str(r.get('end_reason')) for r in data)),
  safety_failure_episodes=sum(float(r.get('safety_failure') or 0) for r in data),
  unverified_action_executions=sum(float(r.get('unverified_action_executions') or 0) for r in data),
  emergency_planning_attempts=sum(float(r.get('emergency_steps') or 0) for r in data),
  physical_controls=sum(float(r.get('physical_steps') or 0) for r in data))
summary=dict(diagnostics=diagnostics,configurations=summaries,paired_comparisons=comparisons,scenarios=scenes,
 checkpoint_sha256=meta['checkpoint_sha256'],source_snapshot_sha256=meta['source_snapshot_sha256'],
 current_source_matches_frozen=source_fingerprint(ROOT)==meta['source_snapshot_sha256'],
 source_directories={'B':meta['baseline_output'],'D':meta['output']},
 protocol='100 independent fixed scenario IDs, seed1, original 1000-control episode limit. Four disjoint shards; latency under load is not an isolated speed benchmark.',
 diagnostic_attribution_fix=json.load(open(folder/'diagnostic_attribution_fix.json')),
 action_selection_source_equivalent=json.load(open(folder/'diagnostic_attribution_fix.json'))['action_selection_unchanged'],
 scope='Unqualified Value ranking keeps certified SAC unchanged; D changes are safety interventions, not demonstrated Value optimisation.')
(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'manifest.json').write_text(json.dumps(meta,indent=2));(out/'scenarios.json').write_text(json.dumps({'scenarios':scenes},indent=2))
(folder/'paired_results.json').write_text(json.dumps(dict(output=str(out),**summary),indent=2));print(out,flush=True)
