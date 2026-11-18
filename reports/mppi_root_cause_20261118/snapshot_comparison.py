"""Compare recorded Lattice, SAC, Gaussian/Mean and executed first controls at fixed sample times."""
import numpy as np
from study_io import R,table

def main():
    rows=[]
    for file in sorted((R/'episodes').glob('*/snapshot_*.npz')):
        z=np.load(file);group,scene,repeat=file.parent.name.split('_');step=int(file.stem.split('_')[1])
        parameters=np.array([z[k] for k in z.files if k.startswith('lattice_') and k.endswith('_parameters')])
        endpoints=np.array([z[k][-1] for k in z.files if k.startswith('lattice_') and k.endswith('_xy')])
        pairwise=np.linalg.norm(endpoints[:,None]-endpoints[None,:],axis=-1)
        coarse,refined,selected=[z[k] for k in ['coarse_actions','refined_actions','selected_actions']]
        r=dict(configuration=group,scene=int(scene),repeat_seed=int(repeat[1:]),step=step,
            raw_lattice_candidates=len(parameters),distinct_raw_lateral_targets=len(np.unique(parameters[:,0])),
            distinct_raw_speed_targets=len(np.unique(parameters[:,1])),distinct_raw_horizons=len(np.unique(parameters[:,2])),
            raw_lateral_parameter_range=float(np.ptp(parameters[:,0])),raw_endpoint_pairwise_max=float(pairwise.max()),
            sac_vs_lattice_first_delta=float(np.linalg.norm(refined[0]-coarse[0])),
            sac_vs_lattice_sequence_rmse=float(np.sqrt(np.mean((refined-coarse)**2))),
            final_vs_sac_first_steer=float(selected[0,0]-refined[0,0]),final_vs_sac_first_throttle=float(selected[0,1]-refined[0,1]),
            final_vs_sac_sequence_rmse=float(np.sqrt(np.mean((selected-refined)**2))),
            note='Raw path diversity, not true-return ranking of discarded Lattice paths')
        if 'weighted_mean' in z:
            mean=z['weighted_mean'];r.update(weighted_mean_vs_sac_rmse=float(np.sqrt(np.mean((mean-refined)**2))),
                mean_vs_final_first_delta=float(np.linalg.norm(mean[0]-selected[0])))
        rows.append(r)
    table('path_and_action_comparison.csv',rows)
if __name__=='__main__':main()
