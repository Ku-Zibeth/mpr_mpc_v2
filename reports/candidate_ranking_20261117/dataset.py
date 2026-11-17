"""One supervised sample per matched candidate, grouped by true origin, not window."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from planning.candidate_value import aligned_returns
from planning.sac_value import file_sha256
from tools.value_calibration import assert_origin_match


def build(split, *, require_complete=True):
 p=json.loads((R/'protocol.json').read_text());items=[];sources=[];unavailable=[]
 old=np.load(R/'old_origin_features.npz')
 if split in ('fit','inner_stop'):
  for i in np.flatnonzero(old['split']==split):
   items.append(dict(z0=old['z0'][i],terminal=old['terminal'][i],actual_terminal=old['actual_terminal'][i],actions=old['actions'][i],baseline=old['baseline'][i],target=float(old['target'][i]),continuation=float(old['continuation'][i]),prefix=float(old['predicted_prefix'][i]),discount=p['gamma']**10,scene=int(old['scene'][i]),before=int(old['before'][i]),branch=str(old['branch'][i]),source='old',complete=True,old_q=float('nan')))
  sources.extend([x for x in json.loads((R/'old_sources.json').read_text()) if int(Path(x['path']).parent.name.split('_')[0]) in (p['old_fit_scenarios'] if split=='fit' else p['old_stop_scenarios'])])
 scenes=p['new_fit_scenarios'] if split=='fit' else p['new_stop_scenarios'] if split=='inner_stop' else p['validation_scenarios']
 for s in scenes:
  for before in p['origin_steps']:
   folder=R/'data'/f'{s}_{before}';missing=[b for b in p['branches'] if not (folder/(b+'.npz')).exists()]
   if missing:
    if all((folder/('unavailable_'+b+'.json')).exists() for b in p['branches']):unavailable.extend([json.loads((folder/('unavailable_'+b+'.json')).read_text()) for b in p['branches']]);continue
    if require_complete:raise RuntimeError(f'Incomplete prespecified origin {s}:{before}: {missing}')
    continue
   reference=json.loads((folder/'baseline_origin.json').read_text());audit=json.loads((folder/'audit.json').read_text())
   for b in p['branches']:
    origin=json.loads((folder/(b+'_origin.json')).read_text());assert_origin_match(reference,origin);path=folder/(b+'.npz');d=np.load(path);m=json.loads((folder/(b+'_meta.json')).read_text());assert m['native_done'];assert m['wm_checkpoint_sha256']==p['checkpoint_sha256'];assert m['policy_context_sha256']==json.loads((Path(p['old_data'])/'dataset_manifest.json').read_text())['context']
    labels=aligned_returns(d['rewards'],[0],horizon=10,gamma=float(d['gamma']),done=d['done']);assert np.allclose(labels['rtg'],d['rtg'],atol=1e-8,rtol=0);assert np.array_equal(d['planned_actions'][:min(10,len(d['actions']))],d['actions'][:10]);assert np.array_equal(d['planned_actions'],np.array(reference['candidate_actions'][p['branches'].index(b)],np.float32))
    items.append(dict(z0=d['z0'],terminal=d['predicted_terminal'],actual_terminal=d['actual_terminal'] if len(d['actual_terminal']) else np.full(512,np.nan),actions=d['planned_actions'],baseline=d['baseline_actions'],target=float(labels['total'][0]),continuation=float(labels['continuation'][0]),prefix=float(d['predicted_prefix']),discount=float(d['terminal_discount']),scene=s,before=before,branch=b,source='new',complete=bool(labels['valid_terminal'][0]),old_q=float(d['predicted_return'])))
    sources.append(dict(path=str(path),sha256=file_sha256(path),samples=1))
 # Exact duplicate first-H controls are one training unit, irrespective of label.
 # Same observations/RNG make their labels deterministic; a mismatch is an error.
 groups=[];deduplicated=[]
 for key in sorted({(i['scene'],i['before']) for i in items}):
  indices=[j for j,i in enumerate(items) if (i['scene'],i['before'])==key];seen={};members=[]
  for j in indices:
   token=items[j]['actions'][:10].tobytes()
   if token in seen:
    other=seen[token];assert abs(items[j]['target']-items[other]['target'])<1e-5
    deduplicated.append(dict(scene=key[0],before=key[1],branch=items[j]['branch'],same_as=items[other]['branch']));continue
   seen[token]=j;members.append(j)
  groups.append(members)
 assert items and all(len(g)>=1 for g in groups)
 # Normalize scenario contribution, then origin, then candidate, only on fit.
 counts={s:sum(items[g[0]]['scene']==s for g in groups) for s in {i['scene'] for i in items}}
 sample_weights=np.zeros(len(items));pairs=[];pair_weights=[];rank_weights=[]
 for g in groups:
  factor=1./counts[items[g[0]]['scene']]
  for i in g:sample_weights[i]=factor/len(g)
  local=[(g[a],g[b]) for a in range(len(g)) for b in range(a+1,len(g))]
  pairs.extend(local);pair_weights.extend([factor/len(local)]*len(local) if local else [])
  active=[abs(items[a]['target']-items[b]['target'])>p['pair_tolerance'] for a,b in local]
  rank_weights.extend([factor/max(1,sum(active)) if ok else 0. for ok in active])
 arrays={k:torch.tensor(np.stack([i[k] for i in items]),dtype=torch.float32) for k in ('z0','terminal','actual_terminal','actions','baseline','target','continuation','prefix','discount')}
 for key in ('z0','terminal','actions','baseline','target','prefix','discount'):assert torch.isfinite(arrays[key]).all(),key
 arrays.update(scene=torch.tensor([i['scene'] for i in items]),complete=torch.tensor([i['complete'] for i in items]),sample_weight=torch.tensor(sample_weights,dtype=torch.float32),pairs=torch.tensor(pairs,dtype=torch.long).reshape(-1,2),pair_weight=torch.tensor(pair_weights,dtype=torch.float32),rank_weight=torch.tensor(rank_weights,dtype=torch.float32),groups=groups,items=items,sources=sources,deduplicated=deduplicated,unavailable=unavailable)
 return arrays
