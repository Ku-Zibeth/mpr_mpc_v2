"""Predeclared one-factor validation; optional sealed B/one-candidate test."""
import argparse, hashlib, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.benchmark_mppi import source_fingerprint
from planning.mppi_ablation import VALUE_ABLATIONS, value_ablation_config
REPORT=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validation','test'],default='validation')
p.add_argument('--worker',choices=['B',*VALUE_ABLATIONS]);p.add_argument('--output',type=Path);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
protocol=json.loads((REPORT/'protocol.json').read_text());checkpoint=Path(protocol['checkpoint'])
if a.phase=='validation':groups=protocol['validation_groups'];scenarios=protocol['validation_scenarios']
else:
 selection=json.loads((REPORT/'selection.json').read_text())
 if not selection.get('provisional_candidate'):raise SystemExit('No eligible validation candidate. No sealed test or combination is justified.')
 groups=['B',selection['provisional_candidate']];scenarios=protocol['sealed_test_scenarios']
if a.worker:
 import torch
 from omegaconf import OmegaConf
 from mpr_mpc._bootstrap import bootstrap
 bootstrap()
 import evaluate_model_suite as suite
 from evaluate_ablation import _EpisodeCSVWriter
 out=a.output/a.worker;out.mkdir()
 raw=OmegaConf.load(protocol['source_environment_config']);base=OmegaConf.to_container(OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi,resolve=True)
 raw.mpr_mpc.mppi=OmegaConf.create(value_ablation_config(base,a.worker if a.worker!='B' else 'D0'))
 raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.seed=1
 raw.evaluation_stack='world_model_residual_rl' if a.worker=='B' else 'full_mpr_mpc'
 raw.enable_wandb=raw.save_video=raw.save_agent=False;raw.residual_rl.buffer_size=8
 raw.metadrive.simulator.start_seed=min(scenarios);raw.metadrive.simulator.num_scenarios=max(scenarios)-min(scenarios)+1
 raw.metadrive.simulator.horizon=1000;raw.metadrive.simulator.use_render=False
 raw.benchmark.output_dir=str(out);raw.benchmark.save_step_diagnostics=True;raw.benchmark.paired_episode_seeds=True
 raw.benchmark.compact_step_logs=True;raw.benchmark.step_log_stride=10;raw.benchmark.cost_ablation=False
 parsed,cfg=suite._make_cfg(raw,checkpoint.parent.parent.name,checkpoint,scenarios)
 assert cfg.horizon==10 and cfg.metadrive['simulator']['horizon']==1000
 OmegaConf.save(parsed,out/'config.yaml');torch.set_float32_matmul_precision('high')
 with (out/'episodes.csv').open('w',newline='') as f:
  writer=_EpisodeCSVWriter(f)
  def save(row):
   assert 0<row['length']<=1000
   row.update(configuration=a.worker,test_seed=1);writer.writerow(row)
  rows,summary=suite._evaluate_checkpoint(raw,checkpoint.parent.parent.name,checkpoint,scenarios,on_episode=save)
 (out/'summary.json').write_text(json.dumps(summary,indent=2));raise SystemExit()
assert a.output is None and 1<=a.jobs<=4
assert '40 passed' in (REPORT/'tests.txt').read_text()
assert all(all(v==0 for v in row['max_errors'].values()) for row in json.loads((REPORT/'sac_interface_verification.json').read_text()))
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==protocol['checkpoint_sha256']
fingerprint=source_fingerprint(ROOT)
name='mppi_value_'+a.phase+'_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S');out=ROOT/'logs/benchmark'/name;out.mkdir()
manifest=dict(protocol=protocol,phase=a.phase,groups=groups,scenarios=scenarios,source_snapshot_sha256=fingerprint,
 output=str(out),variant_definitions=VALUE_ABLATIONS,checkpoint_sha256=protocol['checkpoint_sha256'],
 note='One group per fresh process, common scenario/episode seeds, four workers maximum, no model updates. No combinations or re-use of the old100 for selection. Shared-load latency, not an isolated benchmark.')
(REPORT/f'{a.phase}_run.json').write_text(json.dumps(manifest,indent=2));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(out,flush=True)
waiting=list(groups);running=[];statuses={}
while waiting or running:
 while waiting and len(running)<a.jobs:
  group=waiting.pop(0);log=(out/f'{group}.log').open('w')
  proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--phase',a.phase,'--worker',group,'--output',str(out)],cwd=ROOT,
   env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
  running.append((group,proc,log))
 for group,proc,log in running[:]:
  status=proc.poll()
  if status is not None:
   log.close();running.remove((group,proc,log));statuses[group]=status
   (out/'worker_statuses.json').write_text(json.dumps(statuses,indent=2));print(group,'exit',status,flush=True)
 if running:time.sleep(2)
assert not any(statuses.values()),statuses
assert source_fingerprint(ROOT)==fingerprint
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==protocol['checkpoint_sha256']
