"""Phase 3 paths/helpers; distinct from the runtime's common package."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

R=Path(__file__).resolve().parent
ROOT=R.parents[1]
PRIOR=ROOT.parent/'mpr_mpc_v2_first_action_20261117'
OLD=PRIOR/'reports/mppi_first_action_value_20261117'
PHASE2=PRIOR/'reports/mppi_first_action_failure_20261118'


def load(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
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
    with (R/name).open('x') as f:json.dump(clean(value),f,indent=2,allow_nan=False)
def table(name,rows):
    with (R/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(clean(rows))
