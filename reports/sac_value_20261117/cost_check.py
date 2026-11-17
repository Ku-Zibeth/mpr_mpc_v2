"""Read-only original weights; candidate-rank validation on a separate WM copy."""
import contextlib,io,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));R=Path(__file__).parent
import numpy as np,torch
from omegaconf import OmegaConf
from mpr_mpc._bootstrap import bootstrap
bootstrap()
import evaluate_model_suite as suite
from env import make_env
from planning.cost_validation import cost_definition
from tools.value_calibration import assert_origin_match
old=ROOT/'reports/mppi_calibration_20261116';meta=json.loads((old/'protocol.json').read_text());ck=Path(meta['checkpoint'])
raw=OmegaConf.load(meta['source_environment_config']);raw.enable_wandb=raw.save_video=raw.save_agent=False;raw.residual_rl.buffer_size=8;raw.restore_mppi_runtime=False
_,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[meta['dev_scenarios'][0]]);env=make_env(cfg)
try:
 with contextlib.redirect_stdout(io.StringIO()):agent=suite._make_agent(cfg,env);agent.load(ck)
 agent.eval();model=agent.tdmpc_agent.model;e=agent.planner.evaluator
 before={k:v.clone() for k,v in model.state_dict().items() if '_cost_' in k}
 stats=dict(checkpoint=str(ck),cost_updates=int(model._cost_updates),initial_cost_ready=bool(model.cost_ready),initial_validation_schema=int(model._cost_validation_schema),initial_validated_horizons=int(model._cost_validated_horizons),splits={})
 for split in ('dev','validation'):
  rows=[]
  for path in sorted((old/'probes').glob(split+'_*')):
   origin=json.loads((path/'B_origin.json').read_text());state_id=json.dumps(dict(state=origin['state'],rng=origin['rng']),sort_keys=True)
   for label in ['B','Q1','Q05','Q025','Q0']+(['elite_Q1'] if (path/'elite_Q1.json').exists() else []):
    data=json.loads((path/(label+'.json')).read_text());assert_origin_match(origin,json.loads((path/(label+'_origin.json')).read_text()))
    count=min(10,len(data['trace']));rows.append(dict(executed=True,origin_id=path.name,initial_state_id=state_id,observation=origin['observation'],
      actions=data['actions'][:count],costs=[x['cost'] for x in data['trace'][:count]],cost_definition=cost_definition(cfg),label=label))
  metrics=e.validate_mppi_cost(rows)
  stats['splits'][split]=dict(candidates=len(rows),origins=len({r['origin_id'] for r in rows}),metrics=metrics,effective_cost_weight=0. if not e.cost_ranking_ready() else None)
  (R/(split+'_cost_candidates.json')).write_text(json.dumps(rows,indent=2))
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
 stats['original_cost_buffers_unchanged']=True
 def clean(x):
  if isinstance(x,float) and not np.isfinite(x):return None
  if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
  if isinstance(x,list):return [clean(v) for v in x]
  return x
 (R/'cost_validation.json').write_text(json.dumps(clean(stats),indent=2,allow_nan=False));print(json.dumps(clean(stats),indent=2))
finally:env.close()
