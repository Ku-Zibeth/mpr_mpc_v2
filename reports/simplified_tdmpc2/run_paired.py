"""Evaluate D on fixed100; reuse the immutable, interface-equivalent B control."""
import argparse,csv,json,hashlib,subprocess,sys,os,time,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.benchmark_mppi import source_fingerprint
p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--output',type=Path);a=p.parse_args()
REPORT=Path(__file__).parent
CK=ROOT/'logs/metadrive-risk/1/mpr_mpc_v2_1.0_wm200k_rl300k_20261103_195237/models/milestone_500k_full_mpr_mpc.pt'
BASE=ROOT/'logs/benchmark/sac_refinement_BD_final100_20261111_154509/episodes_B_lattice_sac_seed1.csv'
OLD=json.load(open(ROOT/'reports/sac_refinement/paired_run.json'))
SOURCE_CFG=Path(OLD['baseline_output'])/'part1/config_B_lattice_sac_seed1.yaml'
scenes=json.load(open(ROOT/'scenarios/fixed100.json'));scenes=scenes['scenarios'] if isinstance(scenes,dict) else scenes
if a.worker is not None:
 import torch
 from omegaconf import OmegaConf
 from mpr_mpc._bootstrap import bootstrap
 bootstrap()
 import evaluate_model_suite as suite
 from evaluate_ablation import _EpisodeCSVWriter
 out=a.output/f'part{a.worker+1}';out.mkdir()
 raw=OmegaConf.load(SOURCE_CFG);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi
 raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.seed=1
 raw.evaluation_stack='full_mpr_mpc';raw.enable_wandb=raw.save_video=raw.save_agent=False
 raw.residual_rl.buffer_size=8
 raw.metadrive.simulator.start_seed=300;raw.metadrive.simulator.num_scenarios=2000;raw.metadrive.simulator.horizon=1000;raw.metadrive.simulator.use_render=False
 raw.benchmark.output_dir=str(out);raw.benchmark.save_step_diagnostics=True;raw.benchmark.cost_ablation=False
 OmegaConf.save(raw,out/'config.yaml');torch.set_float32_matmul_precision('high')
 part=scenes[a.worker::4]
 with (out/'episodes_D_lattice_sac_mppi_seed1.csv').open('w',newline='') as f:
  writer=_EpisodeCSVWriter(f)
  def save(row):row.update(configuration='D_lattice_sac_mppi',test_seed=1);writer.writerow(row)
  rows,summary=suite._evaluate_checkpoint(raw,CK.parent.parent.name,CK,part,on_episode=save)
 (out/'summary.json').write_text(json.dumps(summary,indent=2));raise SystemExit()
if a.output is not None:raise SystemExit('--output is reserved for workers')
assert len(scenes)==len(set(scenes))==100
rows=list(csv.DictReader(BASE.open()));assert len(rows)==100 and {int(r['scenario']) for r in rows}==set(scenes)
verify=json.load(open(REPORT/'sac_interface_verification.json'));assert all(all(v==0 for v in r['max_errors'].values()) for r in verify)
smoke=json.load(open(REPORT/'smoke.json'));assert smoke['checkpoint_sha256']==hashlib.sha256(CK.read_bytes()).hexdigest()
assert '27 passed' in (REPORT/'tests.txt').read_text()
fingerprint=source_fingerprint(ROOT)
accept=dict(formal_evaluation_ready=True,blocking_reasons=[],source_snapshot_sha256=fingerprint,
 checkpoint_sha256=smoke['checkpoint_sha256'],tests='27 passed',smoke=str(REPORT/'smoke.json'),
 protocol='B immutable prior 100, D current full100. Original fixed scenarios/test seed=1/horizon1000; no training.')
(REPORT/'acceptance.json').write_text(json.dumps(accept,indent=2))
out=ROOT/'logs/benchmark'/('simplified_tdmpc2_BD_fixed100_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir()
meta=dict(output=str(out),checkpoint=str(CK),checkpoint_sha256=accept['checkpoint_sha256'],
 source_snapshot_sha256=fingerprint,base_commit='618eebfe6c15123134ee18a04e0a83762b06acc0',
 baseline_csv=str(BASE),baseline_csv_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
 source_environment_config=str(SOURCE_CFG),B_reused=True,D_new_episodes=100,test_seed=1,
 scenarios=scenes,workers=4,episode_horizon=1000,
 note='B is explicitly reused, not described as newly rerun. SAC input/output before/after bitwise equality is recorded. D restores exactly the archived B environment and frozen checkpoint; only MPPI configuration/pipeline changes. Parallel planning timings are under shared load.')
(REPORT/'paired_run.json').write_text(json.dumps(meta,indent=2));(out/'manifest.json').write_text(json.dumps(meta,indent=2))
handles=[];jobs=[]
for i in range(4):
 log=(out/f'part{i+1}.log').open('w');handles.append(log)
 jobs.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(i),'--output',str(out)],cwd=ROOT,
  env=dict(os.environ,MUJOCO_GL='egl',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT))
print(out,flush=True)
status=[j.wait() for j in jobs]
for h in handles:h.close()
(out/'worker_statuses.json').write_text(json.dumps(status));print(status,flush=True)
if any(status):raise SystemExit(1)
assert source_fingerprint(ROOT)==fingerprint
