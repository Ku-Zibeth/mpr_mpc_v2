"""Isolated comfort study utilities; do not shadow the vendor common package."""
import csv, hashlib, json, os, sys, subprocess
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parent
ROOT=R.parents[1]
MAIN=ROOT.parent/'mpr_mpc_v2'
OLD=ROOT.parent/'mpr_mpc_v2_evaluation_redesign_20261119'/'reports/mppi_evaluation_redesign_20261119'
sys.path.insert(0,str(ROOT))
from tools.first_action_study import clean, digest, snapshot, tensor_hash, file_sha
def load(p): return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(clean(x),f,indent=2,allow_nan=False)
def table(p,rows):
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader();w.writerows(clean(rows))
def discounted(rewards,gamma=.995):
    total=0.
    for r in reversed(rewards):total=float(r)+gamma*total
    return total
def clustered(frame,columns):
    maps=frame.groupby('scene')[columns].mean();indices=np.random.default_rng(20261120).integers(len(maps),size=(2000,len(maps)))
    result={}
    for key in columns:
        v=maps[key].to_numpy(float)
        result[key]=dict(mean=float(v.mean()),ci95=np.quantile(v[indices].mean(1),[.025,.975]),
            maps={int(s):float(x) for s,x in maps[key].items()},
            lomo={int(s):float(np.delete(v,i).mean()) for i,s in enumerate(maps.index)})
    return result
