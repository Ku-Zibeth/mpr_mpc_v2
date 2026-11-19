"""Cross-worktree exact outputs; no environment or changed defaults."""
import sys,json
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.path.insert(0,str(root))
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from mpr_mpc.planning.local_mppi import LocalMPPI
from tests.test_simplified_tdmpc2_mppi import Values
from tests.test_comfort_mppi import prepare
rows={}
for mode in ['legacy','comfort','tracking']:
    for seed in [0,1,7,42]:
        cfg={} if mode=='legacy' else dict(objective_mode=mode)
        model=Values();p=LocalMPPI(model,cfg,horizon=10,action_dim=2,device='cpu');p.reset(seed)
        b=torch.zeros(11,2);b[:,0]=torch.linspace(-.1,.1,11);b[:,1]=torch.linspace(-.3,.7,11)
        opts={}
        if mode=='comfort':
            e,_,c=prepare();opts=dict(comfort_evaluator=e,comfort_context=c)
        if mode=='tracking':
            from tests.test_tracking_mppi import evaluator
            e,c=evaluator();opts=dict(tracking_evaluator=e,tracking_context=c)
        result=p.plan(torch.zeros(1,512),b,eval_mode=True,**opts)
        for name in ['action','selected_sequence','initial_mean','final_mean','initial_std','final_std']:
            rows[f'{mode}_{seed}_{name}']=getattr(result,name).numpy()
        for i,call in enumerate(model.calls):rows[f'{mode}_{seed}_wm_call_{i}']=call.numpy()
np.savez_compressed(out,**rows)
print('PARITY_EXPORT',root,len(rows),flush=True)
