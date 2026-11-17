"""Read-only audit and frozen-WM reconstruction of ALL previous training windows."""
import contextlib,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common.seed import set_seed
from common import math as wm_math
from env import make_env
import evaluate_model_suite as suite
from tools.value_calibration import assert_origin_match
from planning.candidate_value import aligned_returns
from planning.sac_value import file_sha256
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump

@torch.no_grad()
def main():
 p=json.loads((R/'protocol.json').read_text());old=Path(p['old_data']);op=json.loads((old/'protocol.json').read_text());manifest=json.loads((old/'dataset_manifest.json').read_text());assert file_sha256(p['checkpoint'])==p['checkpoint_sha256']
 raw=OmegaConf.load(p['source_environment_config']);raw.mpr_mpc.mppi=OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi;raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False;raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8
 ck=Path(p['checkpoint']);_,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[op['training_scenarios'][0]]);set_seed(1);torch.set_num_threads(1);torch.set_float32_matmul_precision('high');env=make_env(cfg)
 try:
  with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);agent.load(ck)
  agent.eval();model=agent.tdmpc_agent.model;model.requires_grad_(False);H=10;rows=[];features=[];pairs=[];windows=[];sources=[];unavailable=[];seenorigins={};worst=dict(predicted_latent=0.,actual_latent=0.,label=0.)
  expected={x['path']:x['sha256'] for split in ['fit','inner_stop'] for x in manifest[split]}
  for scene in op['training_scenarios']:
   for before in op['origin_steps']:
    folder=old/'training_data'/f'{scene}_{before}'
    if list(folder.glob('unavailable_*.json')):
     unavailable.extend([dict(path=str(f),record=json.loads(f.read_text())) for f in sorted(folder.glob('unavailable_*.json'))]);continue
    reference=json.loads((folder/'B_origin.json').read_text());group=[]
    for branch in op['train_branches']:
     origin=json.loads((folder/(branch+'_origin.json')).read_text());assert_origin_match(reference,origin);path=folder/(branch+'.npz');assert file_sha256(path)==expected[str(path)];d=np.load(path);m=json.loads((folder/(branch+'_meta.json')).read_text());assert m['native_done'];n=len(d['actions']);done=np.zeros(n,bool);done[-1]=True
     label=aligned_returns(d['rewards'],d['starts'],horizon=H,gamma=float(d['gamma']),done=done);assert label['valid_terminal'].all();assert np.allclose(label['rtg'],d['rtg'],atol=1e-8,rtol=0);assert np.allclose(label['continuation'],d['targets'],atol=1e-8,rtol=0)
     obs=torch.tensor(d['observations'],device=agent.device);z=torch.cat([model.encode(obs[i:i+256],None) for i in range(0,len(obs),256)])
     predictions=[];prefixes=[]
     for b in range(0,len(d['starts']),256):
      starts=d['starts'][b:b+256];zz=z[starts].clone();prefix=torch.zeros(len(starts),device=agent.device)
      for t in range(H):
       actions=torch.tensor(d['actions'][starts+t],device=agent.device);prefix+=float(d['gamma'])**t*wm_math.two_hot_inv(model.reward(zz,actions,None),agent.tdmpc_agent.cfg).reshape(-1);zz=model.next(zz,actions,None)
      predictions.append(zz.cpu().numpy());prefixes.append(prefix.cpu().numpy())
     pred=np.concatenate(predictions);true=z[d['starts']+H].cpu().numpy();pe=float(np.max(abs(pred-d['predicted_latents'])));ae=float(np.max(abs(true-d['actual_latents'])));worst['predicted_latent']=max(worst['predicted_latent'],pe);worst['actual_latent']=max(worst['actual_latent'],ae);print('max reconstruction',pe,ae,flush=True);assert pe<1e-4 and ae<1e-4
     i=len(features);commands=np.asarray(origin['candidate_actions'][op['train_branches'].index(branch)],np.float32);assert np.array_equal(d['actions'][:min(H,n)],commands[:min(H,n)]);assert n>=H
     features.append(dict(z0=z[0].cpu().numpy(),terminal=pred[0],actual_terminal=true[0],actions=commands,baseline=np.asarray(origin['candidate_actions'][0],np.float32),target=float(label['total'][0]),continuation=float(label['continuation'][0]),predicted_prefix=float(np.concatenate(prefixes)[0]),scene=scene,before=before,branch=branch,split='fit' if scene in op['fit_scenarios'] else 'inner_stop'))
     group.append(i);sources.append(dict(path=str(path),sha256=expected[str(path)],samples=len(d['starts'])))
     row=dict(scenario=scene,before=before,branch=branch,windows=len(d['starts']),length=n,actual_return=float(label['total'][0]),success=m['success'],crash=m['crash'],out_of_road=m['out_of_road'],action_rmse=float(np.sqrt(np.mean((commands[:H]-features[-1]['baseline'][:H])**2))),coverage=origin['coverage'],prefix_hash=commands[:H].tobytes().hex(),position_error=pe,actual_error=ae,
      labels_native_flags_last=dict(terminated=bool(d['terminated'][-1]),truncated=bool(d['truncated'][-1])),consecutive_window_overlap_ratio=max(0.,(H-op['window_stride'])/H))
     rows.append(row);windows.append(d['targets']);print(scene,before,branch,'audit',pe,ae,flush=True)
    for a in range(len(group)):
     for b in range(a+1,len(group)):
      left,right=group[a],group[b];delta=features[left]['target']-features[right]['target'];pairs.append(dict(scenario=scene,before=before,left=features[left]['branch'],right=features[right]['branch'],delta_real=delta,effective=abs(delta)>p['pair_tolerance'],identical_prefix=bool(np.array_equal(features[left]['actions'][:H],features[right]['actions'][:H]))))
  arr=dict(z0=np.stack([f['z0'] for f in features]),terminal=np.stack([f['terminal'] for f in features]),actual_terminal=np.stack([f['actual_terminal'] for f in features]),actions=np.stack([f['actions'] for f in features]),baseline=np.stack([f['baseline'] for f in features]))
  for k in ['target','continuation','predicted_prefix','scene','before','branch','split']:arr[k]=np.array([f[k] for f in features])
  np.savez_compressed(R/'old_origin_features.npz',**arr)
  scene_pairs={str(s):dict(pairs=sum(x['scenario']==s for x in pairs),effective_pairs=sum(x['scenario']==s and x['effective'] for x in pairs)) for s in op['training_scenarios']}
  history=json.loads((old/'predicted_training.json').read_text())['history'];dup=sum(x['identical_prefix'] for x in pairs)
  report=dict(independent_scenes=len(op['training_scenarios']),fit_scenes=8,inner_stop_scenes=4,origins=len(rows)//3,native_branches=len(rows),planned_tasks=72,unavailable_tasks=unavailable,fit_windows=sum(x['windows'] for x in rows if x['scenario'] in op['fit_scenarios']),stop_windows=sum(x['windows'] for x in rows if x['scenario'] in op['early_stop_scenarios']),matched_fit_pairs=sum(x['scenario'] in op['fit_scenarios'] for x in pairs),all_pairs=len(pairs),effective_pairs=sum(x['effective'] for x in pairs),tolerance=p['pair_tolerance'],identical_prefix_pairs=dup,by_scene=scene_pairs,pairs=pairs,branches=rows,reconstruction_max_error=worst,
   diagnosis=['Only24 origin states, not3693 independent windows. Sliding windows at stride5 overlap H10 prefixes by50%; nearby continuation RTGs are autocorrelated.','Original optimized loss puts absolute regression on2676 fit windows but difference supervision on48 matched-origin pairs. Independent early-stop maps differ from fit maps.','Training loss is sampled minibatch mean, early-stop is exact scene-weighted objective: raw scales are not perfectly identical; train decreases while held-out-map loss rises. This supports overfit/distribution shift, not a proven label error.','All saved starts+H are native available, all actual/predicted latents reconstructed; early failures shorter thanH cannot invent terminal latent. Actual total labels would be retained even when excluded from H-endpoint head fitting.'],old_training_history=history)
  dump(R/'data_audit.json',report);dump(R/'old_sources.json',sources)
 finally:env.close()
if __name__=='__main__':main()
