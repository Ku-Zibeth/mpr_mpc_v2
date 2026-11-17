"""Predeclared ablations; fit inputs/normalizers use only fit maps."""
import copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from torch.nn import functional as F
from dataset import build
from planning.candidate_value import CandidateValue,pairwise_objective
from planning.sac_value import SACContinuationValue,state_sha256


def to_device(d,device):
 return {k:v.to(device) if torch.is_tensor(v) else v for k,v in d.items()}


def scores(head,kind,d):
 if kind.startswith('absolute'):
  value=head(d['terminal']);return d['prefix']+d['discount']*value,value
 value=head(d['z0'],d['terminal'],d['actions']);return value,None


def loss(head,kind,d,scale,tolerance):
 score,value=scores(head,kind,d);w=d['sample_weight'];w=w/w.sum()
 _,terms=pairwise_objective(score,d['target'],d['pairs'],weights=d['pair_weight'],tolerance=tolerance,difference_scale=scale,difference_weight=.25)
 _,rank_terms=pairwise_objective(score,d['target'],d['pairs'],weights=d['rank_weight'],tolerance=tolerance,difference_scale=scale,difference_weight=0.)
 terms['rank']=rank_terms['rank'];total=terms['rank']+.25*terms['difference']
 if value is not None:
  mask=d['complete'];vweights=w[mask]/w[mask].sum();absolute=(vweights*F.smooth_l1_loss((value[mask]-head.target_mean)/head.target_std,(d['continuation'][mask]-head.target_mean)/head.target_std,reduction='none')).sum()
  total=absolute+(terms['difference'] if kind=='absolute_plus_difference' else 0.)
 return total,terms,score


def main():
 p=json.loads((R/'protocol.json').read_text());torch.set_num_threads(1);torch.set_float32_matmul_precision('high');train=build('fit');stop=build('inner_stop');device='cuda' if torch.cuda.is_available() else 'cpu';train=to_device(train,device);stop=to_device(stop,device)
 assert not set(train['scene'].tolist())&set(stop['scene'].tolist());assert not (set(train['scene'].tolist())|set(stop['scene'].tolist()))&set(p['validation_scenarios']);modeldir=R/'models';modeldir.mkdir(exist_ok=True)
 tr=train['target'];pp=train['pairs'];pw=train['pair_weight'];pw=pw/pw.sum();deltas=tr[pp[:,0]]-tr[pp[:,1]];scale=max(1.,float((pw*deltas.square()).sum().sqrt()))
 records=[]
 for kind,options in p['models'].items():
  torch.manual_seed(1)
  if kind.startswith('absolute'):head=SACContinuationValue(512,options['hidden']).to(device);x=train['terminal']
  else:head=CandidateValue(input_kind=options['input_kind'],hidden=options['hidden']).to(device);x=head.features(train['z0'],train['terminal'],train['actions'])
  with torch.no_grad():
   w=train['sample_weight'];w=w/w.sum();mu=(w[:,None]*x).sum(0);sd=(w[:,None]*(x-mu).square()).sum(0).sqrt().clamp_min(.001);head.input_mean.copy_(mu);head.input_std.copy_(sd)
   if kind.startswith('absolute'):
    mask=train['complete'];vw=w[mask]/w[mask].sum();y=train['continuation'][mask];ym=(vw*y).sum();ys=(vw*(y-ym).square()).sum().sqrt().clamp_min(1.);head.target_mean.copy_(ym);head.target_std.copy_(ys)
   else:head.difference_scale.fill_(scale)
  optim=torch.optim.Adam(head.parameters(),lr=p['training']['lr']);best=float('inf');best_epoch=0;stale=0;history=[];best_state=None
  # Full grouped batch is <200 candidates; every update includes every origin.
  # Exact scene-origin balancing avoids repeatedly sampling correlated windows.
  for epoch in range(p['training']['epochs']):
   head.train();objective,_,_=loss(head,kind,train,scale,p['pair_tolerance']);optim.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optim.step();head.eval()
   with torch.no_grad():
    tr_loss,tr_terms,tr_score=loss(head,kind,train,scale,p['pair_tolerance']);va_loss,va_terms,va_score=loss(head,kind,stop,scale,p['pair_tolerance'])
    def accuracy(s,d):
     pairs=d['pairs'];delta=d['target'][pairs[:,0]]-d['target'][pairs[:,1]];active=delta.abs()>p['pair_tolerance'];weights=d['rank_weight'][active];ok=((s[pairs[:,0]]-s[pairs[:,1]])[active]*delta[active]>0).float();return float((weights*ok).sum()/weights.sum()) if active.any() else None
    record=dict(epoch=epoch+1,train_loss=float(tr_loss),inner_stop_loss=float(va_loss),train_rank_loss=float(tr_terms['rank']),stop_rank_loss=float(va_terms['rank']),train_pair_accuracy=accuracy(tr_score,train),stop_pair_accuracy=accuracy(va_score,stop));history.append(record)
   value=float(va_loss)
   if value<best-1e-6:best=value;best_epoch=epoch+1;stale=0;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
   else:stale+=1
   if stale>=p['training']['patience']:break
  context=dict(wm_checkpoint_sha256=p['checkpoint_sha256'],policy_context_sha256=json.loads((Path(p['old_data'])/'dataset_manifest.json').read_text())['context'],horizon=10,gamma=p['gamma'],input_kind=options['input_kind'],label='native actual total discounted H-prefix + frozen SAC suffix; no unexecuted a_H',training_only_normalization=True)
  artifact=dict(kind=kind,architecture=dict(latent_dim=512,horizon=10,action_dim=2,input_kind=options['input_kind'],hidden=options['hidden']) if not kind.startswith('absolute') else dict(latent_dim=512,hidden=options['hidden']),state_dict=best_state,context=context,validation=dict(passed=False),
   training_scenarios=sorted(set(train['scene'].tolist()+stop['scene'].tolist())),fit_scenarios=sorted(set(train['scene'].tolist())),inner_stop_scenarios=sorted(set(stop['scene'].tolist())),training=dict(best_epoch=best_epoch,epochs=len(history),history=history,difference_scale=scale,pair_tolerance=p['pair_tolerance'],parameters=sum(x.numel() for x in head.parameters()),head_state_sha256=state_sha256(best_state),objective=options['loss']))
  target=modeldir/(kind+'.pt');assert not target.exists();torch.save(artifact,target);record=dict(model=kind,best_epoch=best_epoch,epochs=len(history),parameters=artifact['training']['parameters'],train=history[best_epoch-1],artifact=str(target),head_state_sha256=state_sha256(best_state));records.append(record);(R/(kind+'_training.json')).write_text(json.dumps(artifact['training'],indent=2));print(kind,best_epoch,record['train'],flush=True)
 manifest=dict(fit_sources=train['sources'],stop_sources=stop['sources'],fit_scenes=sorted(set(train['scene'].tolist())),stop_scenes=sorted(set(stop['scene'].tolist())),fit_branches=len(train['items']),stop_branches=len(stop['items']),fit_origins=len(train['groups']),stop_origins=len(stop['groups']),fit_pairs=len(train['pairs']),stop_pairs=len(stop['pairs']),effective_fit_pairs=int((deltas.abs()>p['pair_tolerance']).sum()),fit_deduplicated=train['deduplicated'],stop_deduplicated=stop['deduplicated'],unavailable=train['unavailable']+stop['unavailable'],difference_scale=scale,protocol='One sample per origin-candidate; no sliding-window multiplication. Full grouped batches with exact scene/origin balance; no stochastic minibatch sampling needed at this size.',models=records)
 (R/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2));print('Models frozen; fresh validation not used.',flush=True)
if __name__=='__main__':main()
