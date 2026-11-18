"""Isolated, frozen-checkpoint root-cause study; no training entry points."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent
ROOT=R.parents[1]
MAIN=ROOT.parent/'mpr_mpc_v2'
PRIOR=ROOT.parent/'mpr_mpc_v2_first_action_20261117'

def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,Path):return str(x)
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def dump(name,value):
    f=R/name;f.parent.mkdir(parents=True,exist_ok=True)
    with f.open('x') as stream:json.dump(clean(value),stream,indent=2,allow_nan=False)
def table(name,rows):
    with (R/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(clean(rows))


def make_runtime(scene,seed,group,record=True):
    import contextlib,io,sys,torch
    code_root = ROOT if group == 'D' else MAIN
    sys.path.insert(0,str(code_root))
    from omegaconf import OmegaConf
    from mpr_mpc._bootstrap import bootstrap
    bootstrap()
    from common.seed import set_seed
    from env import make_env
    import evaluate_model_suite as suite
    from planning.sac_value import policy_context_sha256
    p=load(R/'protocol.json');ck=Path(p['checkpoint'])
    raw=OmegaConf.load(p['source_environment_config'])
    raw.mpr_mpc.mppi=OmegaConf.load(code_root/'config.yaml').mpr_mpc.mppi
    if group == 'D':
        raw.mpr_mpc.mppi.cartesian_virtual_speed = True
    # Do NOT insert selection_mode: preserve current default's legacy two-way comparison.
    raw.mpr_mpc.mppi.record_candidates=bool(record)
    raw.restore_checkpoint_runtime=True;raw.restore_mppi_runtime=False
    raw.enable_wandb=raw.save_agent=raw.save_video=False;raw.residual_rl.buffer_size=8;raw.seed=seed
    parsed,cfg=suite._make_cfg(raw,ck.parent.parent.name,ck,[scene])
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high');set_seed(seed)
    env=make_env(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        agent=suite._make_agent(cfg,env);state=agent.load(ck)
    agent.eval();suite._apply_evaluation_stack(agent.planner,'world_model_residual_rl' if group=='B' else 'full_mpr_mpc',state)
    agent.set_global_step(max(agent.global_env_step,agent.planner.mppi_start_step))
    assert agent.planner.residual_state_dim==535 and agent.planner.allow_residual
    assert agent.planner.allow_mppi==(group!='B')
    if group == 'D':assert agent.planner.mppi_cfg['cartesian_virtual_speed']
    assert policy_context_sha256(cfg)==load(MAIN/'reports/sac_value_20261117/dataset_manifest.json')['context']
    assert agent.planner.control_dt==.1 and int(cfg.metadrive['simulator']['horizon'])==1000
    return parsed,cfg,env,agent
