"""Cached-pool GPU batch and original elite-update checks, not online deployment."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from omegaconf import OmegaConf
from planning.candidate_value import CandidateValue
from planning.sac_value import SACContinuationValue
from planning.local_mppi import LocalMPPI
sys.path.insert(0,str(ROOT/'reports/mppi_calibration_20261116'));from probe import dump


@torch.no_grad()
def main():
 p=json.loads((R/'protocol.json').read_text());validation=json.loads((R/'ranking_evaluation.json').read_text());assert validation['head_hashes'];device='cuda';torch.set_num_threads(1);torch.set_float32_matmul_precision('high');a=torch.load(R/'models/rank_context_actions.pt',map_location=device,weights_only=True);head=CandidateValue(**a['architecture']).to(device);head.load_state_dict(a['state_dict']);head.eval().requires_grad_(False)
 a=torch.load(Path(p['old_data'])/'models/sac_value_candidate.pt',map_location=device,weights_only=True);old=SACContinuationValue(**a['architecture']).to(device);old.load_state_dict(a['state_dict']);old.freeze();cfg=OmegaConf.to_container(OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi);planner=LocalMPPI(object(),cfg,horizon=10,action_dim=2,device=device);rows=[]
 for s in p['validation_scenarios']:
  for before in p['origin_steps']:
   folder=R/'data'/f'{s}_{before}'
   if not (folder/'audit.npz').exists():continue
   data=np.load(folder/'audit.npz');origin=np.load(folder/'baseline.npz');z0=torch.tensor(origin['z0'],device=device).reshape(1,-1);terminal=torch.tensor(data['latents'],device=device);actions=torch.tensor(data['actions'],device=device);scores=head(z0,terminal,actions)
   chunk=torch.cat([head(z0,terminal[i:i+65],actions[i:i+65]) for i in range(0,len(actions),65)])
   indices=torch.topk(scores,8).indices;shifted=torch.topk(scores-scores[0],8).indices;batchindices=torch.topk(chunk,8).indices
   rounds=[]
   for i in range(4):
    ids=torch.tensor([0,*range(2+64*i,2+64*(i+1))],device=device);aa=actions[ids];vv=scores[ids];mean,std,elite=planner._update_distribution(aa,vv);mean[-1]=actions[0,-1];std[-1]=0.;weights=planner._weights(elite)
    rounds.append(dict(iteration=i,finite=bool(torch.isfinite(mean).all() and torch.isfinite(std).all()),valid_actions=bool((mean.abs()<=1).all()),std_min=float(std[:10].min()),std_max=float(std[:10].max()),effective_samples=float(weights.square().sum().reciprocal()),entropy=float(-(weights*weights.clamp_min(1e-30).log()).sum()),first_action=mean[0].cpu().tolist(),padding_fixed=bool(torch.equal(mean[-1],actions[0,-1]))))
   times={}
   for name,fn in [('relative_score',lambda:head(z0,terminal,actions)),('previous_terminal_V',lambda:old(terminal))]:
    for _ in range(10):fn()
    torch.cuda.synchronize();wall=[];cuda=[]
    for _ in range(30):
     start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);now=time.perf_counter();start.record();fn();end.record();end.synchronize();wall.append(1000*(time.perf_counter()-now));cuda.append(start.elapsed_time(end))
    times[name]=dict(batch=len(terminal),wall_p50_ms=float(np.median(wall)),wall_p95_ms=float(np.quantile(wall,.95)),cuda_p50_ms=float(np.median(cuda)),cuda_p95_ms=float(np.quantile(cuda,.95)))
   rows.append(dict(scenario=s,before=before,top8_shift_set_equal=set(indices.cpu().tolist())==set(shifted.cpu().tolist()),top8_batch_jaccard=len(set(indices.cpu().tolist())&set(batchindices.cpu().tolist()))/len(set(indices.cpu().tolist())|set(batchindices.cpu().tolist())),batch_score_max_error=float((chunk-scores).abs().max()),score_range=[float(scores.min()),float(scores.max())],score_delta_quantiles=np.quantile((scores-scores[0]).cpu().numpy(),[0,.25,.5,.75,1]).tolist(),rounds=rounds,times=times))
 dump(R/'search_diagnostics.json',dict(rows=rows,all_updates_valid=all(x['finite'] and x['valid_actions'] and x['padding_fixed'] for r in rows for x in r['rounds']),note='Original Q-generated cached candidate pools only: these are scoring/Top-K/update diagnostics, NOT new-score sequential search or executed driving. Same original temperature/std/rho/budget; H-only score ignores a_H. No per-trajectory safety gate. Head microtimings after collection completes are not full planning time.'))
if __name__=='__main__':main()
