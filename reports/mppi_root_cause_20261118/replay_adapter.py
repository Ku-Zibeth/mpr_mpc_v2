"""CPU-only replay of preselected real snapshots through the unchanged Lattice tracker."""
import sys
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from lattice.frenet_metadrive import FRENET_DEFAULT_CONFIG,MetaDriveFrenetController
from planning.trajectory_adapter import LatticeActionAdapter
from study_io import R,table,dump

def main():
    evidence=pd.read_csv(R/'virtual_speed_evidence.csv');rows=[]
    for rel in evidence.source:
        file=R.parents[1]/rel;z=np.load(file);step=int(file.stem.split('_')[1])
        cfg=OmegaConf.to_container(OmegaConf.load(file.parent/'config.yaml'))
        controller=MetaDriveFrenetController(dict(FRENET_DEFAULT_CONFIG,**cfg['lattice']))
        record=pd.read_csv(file.parent/'steps.csv').set_index('step').loc[step]
        path=SimpleNamespace(t=z['refined_t'],xy=z['refined_xy'],s_d=z['refined_speed'],target_speed=float(record.refined_target_speed))
        vehicle=SimpleNamespace(position=z['position'],speed=float(z['speed']),heading_theta=float(z['heading']))
        adapter=LatticeActionAdapter(controller)
        a=adapter.path_to_actions(path,vehicle,required_steps=11,control_dt=.1,device='cpu').numpy()
        b=adapter.path_to_actions(path,vehicle,required_steps=11,control_dt=.1,device='cpu',cartesian_virtual_speed=True).numpy()
        assert np.array_equal(a,z['refined_actions']),(rel,'tracker replay mismatch')
        assert np.array_equal(a[0],b[0])
        rows.append(dict(source=rel,legacy_exactly_reproduced=True,first_action_unchanged=True,
            future_control_l2_mean=float(np.linalg.norm(a[1:]-b[1:],axis=1).mean()),
            future_control_max_difference=float(np.abs(a[1:]-b[1:]).max()),
            changed_tail_steps=int(np.any(a[1:]!=b[1:],axis=1).sum())))
    table('same_path_adapter_replay.csv',rows)
    dump('same_path_adapter_replay.json',dict(snapshots=len(rows),all_legacy_exact=True,all_first_actions_unchanged=True,
        snapshots_with_changed_tail=sum(r['changed_tail_steps']>0 for r in rows),
        max_future_control_difference=max(r['future_control_max_difference'] for r in rows),
        protocol='Same saved path and current state, actual original Lattice tracker. No environment execution or alternative return labels.'))
if __name__=='__main__':main()
