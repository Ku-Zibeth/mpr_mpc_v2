"""Prove nominal-only prefix parity, not just first reset fingerprint."""
from study import *
import pandas as pd
if __name__=='__main__':
    rows=[]
    for split in (['development'] if '--development-only' in sys.argv else ['development','holdout']):
        for path in (R/split).glob('*/episode.json'):
            row=load(path);method=row['configuration']
            if method=='A':continue
            directory=path.parent;base=R/split/f'A_{row["scene"]}_r{row["seed"]}'
            if not (base/'episode.json').exists():continue
            a=np.load(base/'trace.npz');b=np.load(directory/'trace.npz');n=min(len(a['actions']),len(b['actions']))
            changes=np.where(np.any(a['actions'][:n]!=b['actions'][:n],axis=1))[0];first=int(changes[0]) if len(changes) else n
            assert np.array_equal(a['observations'][:first+1],b['observations'][:first+1]),(split,method,row['scene'],'Observation divergence before action intervention')
            steps=pd.read_csv(directory/'steps.csv')
            if first<n:
                expected=np.array([steps.sac_steering.iloc[first],steps.sac_throttle.iloc[first]],dtype=a['actions'].dtype)
                assert np.array_equal(expected,a['actions'][first]),(split,method,row['scene'],'SAC changed before first intervention')
            rows.append(dict(split=split,configuration=method,scene=row['scene'],seed=row['seed'],first_changed_control=first+1 if first<n else None,exact_nominal_prefix_controls=first,observations_equal_until_first_intervention=True,SAC_first_intervention_baseline_equal=True))
    prefix='development_' if '--development-only' in sys.argv else ''
    table(R/f'{prefix}nominal_prefix_verification.csv',rows);dump(R/f'{prefix}nominal_prefix_verification.json',dict(comparisons=len(rows),all_passed=True,description='Exact action/observation prefix and same SAC command at first actual intervention; not merely same initial state.'))
