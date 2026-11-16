"""Fresh B100/D100 through one evaluator, original fixed IDs, frozen weights."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.benchmark_mppi import source_fingerprint
REPORT=Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--worker',choices=['B','D']);p.add_argument('--part',type=int)
p.add_argument('--output',type=Path);p.add_argument('--jobs',type=int,default=4);a=p.parse_args()
reference=json.loads((ROOT/'reports/simplified_tdmpc2/paired_run.json').read_text())
checkpoint=Path(reference['checkpoint']);source=Path(reference['source_environment_config'])
scenarios=json.loads((ROOT/'scenarios/fixed100.json').read_text());scenarios=scenarios.get('scenarios',[]) if isinstance(scenarios,dict) else scenarios
assert len(scenarios)==len(set(scenarios))==100
if a.worker:
 import torch
 from omegaconf import OmegaConf
 from mpr_mpc._bootstrap import bootstrap
 bootstrap()
 import evaluate_model_suite as suite
 from evaluate_ablation import _EpisodeCSVWriter
 label='B_lattice_sac' if a.worker=='B' else 'D_lattice_sac_mppi'
 out=a.output/f'{a.worker}_part{a.part+1}';out.mkdir()
 raw=OmegaConf.load(source);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
 raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.seed=1
 raw.evaluation_stack='world_model_residual_rl' if a.worker=='B' else 'full_mpr_mpc'
 raw.enable_wandb=raw.save_video=raw.save_agent=False;raw.residual_rl.buffer_size=8
 raw.metadrive.simulator.start_seed=300;raw.metadrive.simulator.num_scenarios=2000
 raw.metadrive.simulator.horizon=1000;raw.metadrive.simulator.use_render=False
 raw.benchmark.output_dir=str(out);raw.benchmark.save_step_diagnostics=True
 raw.benchmark.paired_episode_seeds=True;raw.benchmark.compact_step_logs=True;raw.benchmark.step_log_stride=10
 raw.benchmark.cost_ablation=False
 subset=scenarios[a.part::4]
 parsed,cfg=suite._make_cfg(raw,checkpoint.parent.parent.name,checkpoint,subset)
 assert cfg.metadrive['simulator']['horizon']==1000 and cfg.horizon==10
 OmegaConf.save(parsed,out/'config.yaml')
 torch.set_float32_matmul_precision('high')
 with (out/f'episodes_{label}_seed1.csv').open('w',newline='') as stream:
  writer=_EpisodeCSVWriter(stream)
  def save(row):
   assert 0 < row['length'] <= 1000
   row.update(configuration=label,test_seed=1);writer.writerow(row)
  rows,summary=suite._evaluate_checkpoint(raw,checkpoint.parent.parent.name,checkpoint,subset,on_episode=save)
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
 raise SystemExit()
assert a.output is None and a.part is None and 1<=a.jobs<=4
assert '31 passed' in (REPORT/'tests.txt').read_text()
smoke=json.loads((REPORT/'smoke_final.json').read_text())
assert len(smoke['results'])==2 and all(c['summary']['steps']==60 and not any(c['summary'][k] for k in ('crash','offroad','safety_failure')) for c in smoke['results'])
assert all(all(error==0 for error in r['max_errors'].values()) for r in json.loads((REPORT/'sac_interface_verification.json').read_text()))
checksum=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert checksum==reference['checkpoint_sha256']
fingerprint=source_fingerprint(ROOT)
name='sac_diffusion_BD_fixed100_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
out=ROOT/'logs/benchmark'/name;out.mkdir()
metadata=dict(output=str(out),base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
 checkpoint=str(checkpoint),checkpoint_sha256=checksum,source_snapshot_sha256=fingerprint,
 source_environment_config=str(source),scenarios=scenarios,test_seed=1,episode_seed_rule='(1+scenario) modulo 2**32',
 episode_horizon=1000,WM_horizon=10,B_reused=False,B_new_episodes=100,D_new_episodes=100,
 max_concurrent_workers=a.jobs,shards_per_group=4,search_budget=dict(gaussian_samples=64,baseline_candidates=1,elites=8,iterations=4),
 temperature_definition='softmax((J-Jmax)/tau), tau=0.5; official beta=1/tau=2',
 Q_reducer='Official random-two decoded average for MPPI only; unchanged full-ensemble mean for SAC features',
 latency_scope='Shared RTX3070; loaded timings, not an isolated real-time benchmark')
(REPORT/'paired_run.json').write_text(json.dumps(metadata,indent=2));(out/'manifest.json').write_text(json.dumps(metadata,indent=2))
pending=[(group,part) for part in range(4) for group in ('B','D')];running=[];statuses={}
print(out,flush=True)
while pending or running:
 while pending and len(running)<a.jobs:
  group,part=pending.pop(0);log=(out/f'{group}_part{part+1}.log').open('w')
  job=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',group,'--part',str(part),'--output',str(out)],cwd=ROOT,
   env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='egl',PYTHONUNBUFFERED='1'),stdout=log,stderr=subprocess.STDOUT)
  running.append((group,part,job,log))
 for item in running[:]:
  group,part,job,log=item;status=job.poll()
  if status is not None:
   log.close();running.remove(item);statuses[f'{group}_part{part+1}']=status
   print(group,part+1,'exit',status,flush=True)
   (out/'worker_statuses.json').write_text(json.dumps(statuses,indent=2))
 if running:time.sleep(2)
assert all(status==0 for status in statuses.values()),statuses
assert source_fingerprint(ROOT)==fingerprint
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checksum
print('B100/D100 complete:',out,flush=True)
