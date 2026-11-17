"""Train ONLY new small sidecars. No original WM/SAC object or optimizer is created."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from torch.nn import functional as F
from planning.sac_value import SACContinuationValue,state_sha256,file_sha256
from tools.value_calibration import assert_origin_match

def dataset(scenes):
 zs=[];actual=[];ys=[];scene_ids=[];pair_groups={};contexts=set();files=[];length=0
 for scene in scenes:
  for before in protocol['origin_steps']:
   folder=R/'training_data'/f'{scene}_{before}'
   if list(folder.glob('unavailable_*.json')):continue
   origin=json.loads((folder/'B_origin.json').read_text());positions=[]
   for branch in protocol['train_branches']:
    assert_origin_match(origin,json.loads((folder/(branch+'_origin.json')).read_text()))
    path=folder/(branch+'.npz');data=np.load(path);meta=json.loads((folder/(branch+'_meta.json')).read_text());assert meta['native_done'];contexts.add(meta['policy_context_sha256'])
    gamma=float(data['gamma']);n=len(data['rewards']);rtg=np.zeros(n+1,np.float64)
    for t in range(n-1,-1,-1):rtg[t]=data['rewards'][t]+gamma*rtg[t+1]
    assert np.allclose(rtg,data['rtg'],rtol=0,atol=1e-8)
    starts=data['starts'];assert (starts+10<=n).all();assert np.allclose(data['targets'],rtg[starts+10],rtol=0,atol=1e-8)
    assert len(data['observations'])==n+1 and len(data['actions'])==n
    if len(starts) and starts[0]==0:positions.append(length)
    zs.append(data['predicted_latents']);actual.append(data['actual_latents']);ys.append(data['targets']);scene_ids.extend([scene]*len(starts));length+=len(starts)
    files.append(dict(path=str(path),sha256=file_sha256(path),samples=len(starts)))
   pair_groups[f'{scene}_{before}']=positions
 pairs=[]
 for group in pair_groups.values():
  for i in range(len(group)):
   for j in range(i+1,len(group)):pairs.append((group[i],group[j]))
 assert len(contexts)==1
 return dict(predicted=torch.tensor(np.concatenate(zs),dtype=torch.float32),actual=torch.tensor(np.concatenate(actual),dtype=torch.float32),
  target=torch.tensor(np.concatenate(ys),dtype=torch.float32),scene=torch.tensor(scene_ids),pairs=torch.tensor(pairs,dtype=torch.long).reshape(-1,2),files=files,context=next(iter(contexts)))

def scene_weights(scene):
 return torch.tensor([1./int((scene==s).sum()) for s in scene],dtype=torch.float32)

def fit(kind,train,stop,device):
 torch.manual_seed(1);rng=torch.Generator().manual_seed(1)
 head=SACContinuationValue(512,(128,64)).to(device)
 z=train[kind].to(device);y=train['target'].to(device);weight=scene_weights(train['scene']).to(device);weight/=weight.sum()
 vz=stop[kind].to(device);vy=stop['target'].to(device);vw=scene_weights(stop['scene']).to(device);vw/=vw.sum()
 with torch.no_grad():
  mu=(weight[:,None]*z).sum(0);std=((weight[:,None]*(z-mu).square()).sum(0)).sqrt().clamp_min(.001)
  ym=(weight*y).sum();sd=(weight*(y-ym).square()).sum().sqrt().clamp_min(1.)
  head.input_mean.copy_(mu);head.input_std.copy_(std);head.target_mean.copy_(ym);head.target_std.copy_(sd)
 optim=torch.optim.Adam(head.parameters(),lr=.0003);pairs=train['pairs'].to(device);vpairs=stop['pairs'].to(device)
 best=float('inf');snapshot=None;history=[];stale=0
 def objective(z,y,w,p):
  pred=head(z);loss=(w*F.smooth_l1_loss((pred-head.target_mean)/head.target_std,(y-head.target_mean)/head.target_std,reduction='none')).sum()
  if len(p):loss+=F.smooth_l1_loss((pred[p[:,0]]-pred[p[:,1]])/head.target_std,(y[p[:,0]]-y[p[:,1]])/head.target_std)
  return loss
 for epoch in range(100):
  head.train();draw=torch.multinomial(weight.cpu(),len(z),replacement=True,generator=rng)
  losses=[]
  for i in range(0,len(draw),256):
   index=draw[i:i+256].to(device);pred=head(z[index]);loss=F.smooth_l1_loss((pred-head.target_mean)/head.target_std,(y[index]-head.target_mean)/head.target_std)
   if len(pairs):
    pp=pairs[torch.randint(len(pairs),(min(32,len(pairs)),),generator=rng).to(device)]
    loss+=F.smooth_l1_loss((head(z[pp[:,0]])-head(z[pp[:,1]]))/head.target_std,(y[pp[:,0]]-y[pp[:,1]])/head.target_std)
   optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optim.step();losses.append(float(loss))
  head.eval()
  with torch.no_grad():valid=float(objective(vz,vy,vw,vpairs));mae=float((vw*(head(vz)-vy).abs()).sum())
  history.append(dict(epoch=epoch+1,train_loss=float(np.mean(losses)),inner_stop_loss=valid,inner_stop_mae=mae))
  if valid<best-1e-6:
   best=valid;snapshot={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};stale=0;best_epoch=epoch+1
  else:stale+=1
  if stale>=12:break
 assert snapshot is not None
 context=dict(wm_checkpoint_sha256=protocol['checkpoint_sha256'],policy_context_sha256=train['context'],horizon=10,
              gamma=gamma_value,input_kind='imagined_terminal_latent' if kind=='predicted' else 'actual_terminal_latent_oracle_only')
 artifact=dict(architecture=dict(latent_dim=512,hidden=(128,64)),state_dict=snapshot,context=context,
  training_scenarios=protocol['training_scenarios'],fit_scenarios=protocol['fit_scenarios'],early_stop_scenarios=protocol['early_stop_scenarios'],
  validation=dict(passed=False),training=dict(best_epoch=best_epoch,loss='scene-balanced normalized SmoothL1 + matched-origin difference SmoothL1',history=history,
  training_examples=len(z),inner_stop_examples=len(vz),training_pairs=len(pairs),head_state_sha256=state_sha256(snapshot),torch_version=str(torch.__version__)))
 out=R/'models';out.mkdir(exist_ok=True);path=out/('sac_value_candidate.pt' if kind=='predicted' else 'actual_latent_oracle.pt');assert not path.exists();torch.save(artifact,path)
 (R/(kind+'_training.json')).write_text(json.dumps(artifact['training'],indent=2));print(kind,'samples',len(z),'best_epoch',best_epoch,'stop_loss',best,flush=True)

if __name__=='__main__':
 protocol=json.loads((R/'protocol.json').read_text());assert file_sha256(protocol['checkpoint'])==protocol['checkpoint_sha256']
 assert not set(protocol['training_scenarios'])&set(protocol['development_scenarios']+protocol['validation_scenarios'])
 train=dataset(protocol['fit_scenarios']);stop=dataset(protocol['early_stop_scenarios']);assert train['context']==stop['context']
 sample=next((R/'training_data').glob('*/*.npz'));gamma_value=float(np.load(sample)['gamma'])
 device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
 fit('predicted',train,stop,device);fit('actual',train,stop,device)
 (R/'dataset_manifest.json').write_text(json.dumps(dict(fit=train['files'],inner_stop=stop['files'],fit_scenarios=protocol['fit_scenarios'],inner_stop_scenarios=protocol['early_stop_scenarios'],
  context=train['context'],gamma=gamma_value,train_examples=len(train['target']),inner_stop_examples=len(stop['target']),training_pairs=len(train['pairs']),notes='Only independent sidecars trained. Native early failures retained; no samples beyond done. No development/validation scene used in weights or epoch selection.'),indent=2))
