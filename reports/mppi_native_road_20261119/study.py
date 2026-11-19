"""Tracking experiment utilities, isolated from all production control modules."""
import csv,hashlib,json,os,sys,subprocess,io,contextlib
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent;ROOT=R.parents[1];MAIN=ROOT.parent/'mpr_mpc_v2'
PREV=ROOT.parent/'mpr_mpc_v2_tracking_20261119'/'reports/mppi_tracking_20261119'
OLD=ROOT.parent/'mpr_mpc_v2_comfort_20261120'/'reports/mppi_comfort_optimization_20261120'
REDESIGN=ROOT.parent/'mpr_mpc_v2_evaluation_redesign_20261119'/'reports/mppi_evaluation_redesign_20261119'
sys.path.insert(0,str(ROOT))
from tools.first_action_study import clean,digest,snapshot,tensor_hash,file_sha
def load(p):return json.loads(Path(p).read_text())
def dump(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(clean(value),f,indent=2,allow_nan=False)
def table(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(clean(rows))
def discounted(rewards,gamma=.995):
    value=0.
    for x in reversed(rewards):value=float(x)+gamma*value
    return value
def clustered(frame,columns):
    means=frame.groupby('scene')[columns].mean();rng=np.random.default_rng(20261119);idx=rng.integers(len(means),size=(2000,len(means)));out={}
    for name in columns:
        x=means[name].to_numpy(float);out[name]=dict(mean=float(x.mean()),ci95=np.quantile(x[idx].mean(1),[.025,.975]),maps={int(k):float(v) for k,v in means[name].items()},lomo={int(k):float(np.delete(x,i).mean()) for i,k in enumerate(means.index)})
    return out
