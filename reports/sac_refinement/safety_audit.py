"""Near/full paired-model confusion on actual B controls; no threshold fitting."""
import collections,gzip,json,math,sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from shapely import wkb
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from mpr_mpc.planning.mppi_safety import MPPISafetyFilter,MPPISafetyContext
folder=Path(__file__).parent
rows=list(map(json.loads,gzip.open(folder/'b_motion/measured_windows.jsonl.gz','rt')))
f=MPPISafetyFilter(OmegaConf.to_container(OmegaConf.load(ROOT/'config.yaml').mpr_mpc.mppi.safety),control_dt=.1)
actions={(r['scenario'],r['index']+i):t['action'] for r in rows for i,t in enumerate(r['states'])}
geometry={};counts={};rejects=collections.Counter();details=[]
def record(group,pred,actual):
 c=counts.setdefault(group,collections.Counter());c[('TP' if actual else 'FP') if pred else ('FN' if actual else 'TN')]+=1
for row in rows:
 sig=row['vehicle_signature'];states=row['states'];x,y,h,v=states[0]['before'];H=len(states)
 road=geometry.setdefault(row['road_wkb'],wkb.loads(row['road_wkb'],hex=True))
 ctx=MPPISafetyContext(position=np.array([x,y]),heading=h,speed=v,length=sig['length'],width=sig['width'],
  wheelbase=sig['wheelbase'],max_steering=sig['steering'],rear_axle_to_center=sig['rear_axle'],
  previous_action=np.array(actions.get((row['scenario'],row['index']-1),[0.,0.])),road=road,
  obstacles=tuple(SimpleNamespace(**{k:np.asarray(v) if k in ('position','velocity') else v for k,v in o.items()}) for o in row['obstacles']),
  brake_action_mapping_verified=True,motion_signature=sig)
 a=torch.tensor([r['action'] for r in states]+[states[-1]['action']])[None]
 models={'center':f._check_nominal(a,ctx),'legacy':f._check_nominal(a,replace(ctx,rear_axle_to_center=None,cache={}))}
 near=f.committed_controls(H);both_reject=True
 distance=sum(np.linalg.norm(np.asarray(r['ticks'][-1][:2])-r['before'][:2]) for r in states)
 yaw=states[-1]['ticks'][-1][2]-h;yaw=abs(math.atan2(math.sin(yaw),math.cos(yaw)))
 speed_bin='stopped' if v<.3 else 'low' if v<5 else 'medium' if v<10 else 'high'
 curve_bin='curve' if yaw>.05 else 'straight'
 margins=[float(r.boundary_margin[0]) for r in models.values() if r.boundary_margin is not None]
 boundary_bin='boundary_adjacent' if margins and min(margins)<ctx.width/2 else 'interior'
 per={}
 for model,r in models.items():
  both_reject &= not bool(r.valid[0])
  per[model]={k:bool(getattr(r,k)[0]) for k in ('valid','near_valid','full_valid','collision','offroad','near_collision','near_offroad','braking_continuation_valid')}
  for horizon,actual_states in [('near',states[:near]),('full',states)]:
   for event,field in [('crash','collision'),('offroad','offroad')]:
    pred=bool(getattr(r,('near_' if horizon=='near' else '')+field)[0]);actual=any(s[event] for s in actual_states)
    record(f"{row['split']}/{model}/{horizon}/{event}",pred,actual)
    for key,label in [('speed',speed_bin),('curvature',curve_bin),('boundary',boundary_bin)]:
     record(f"{row['split']}/{model}/{horizon}/{event}/{key}={label}",pred,actual)
 no_event=not any(t['crash'] or t['offroad'] for t in states)
 rejects['windows']+=1;rejects['both_reject_no_event']+=int(both_reject and no_event)
 rejects['disagree']+=int(per['center']['valid']!=per['legacy']['valid'])
 details.append(dict(scenario=row['scenario'],index=row['index'],split=row['split'],speed=v,curvature_rad_per_m=yaw/max(distance,1e-8),
  speed_stratum=speed_bin,curvature_stratum=curve_bin,boundary_stratum=boundary_bin,models=per,no_actual_event=no_event))
result=dict(source='B_lattice_sac actual actions; disjoint scene holdout',fit_scenes=[1525,539],heldout_scenes=[1714,559],
 near_controls=near,full_controls=10,counts={k:dict(v) for k,v in counts.items()},rejections=dict(rejects),
 both_reject_no_event_rate=rejects['both_reject_no_event']/rejects['windows'],
 limits='Finite-window event agreement is not recall certification. No guard/threshold/vehicle-model changes were made.',windows=details)
(folder/'b_safety_confusion.json').write_text(json.dumps(result,indent=2));print(result['rejections'],flush=True)
