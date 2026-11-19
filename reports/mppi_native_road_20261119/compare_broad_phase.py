"""Same real contexts/candidates/seeds, compare full versus exact broad phase."""
from study import *
import pickle,time,shapely
from mpr_mpc.planning.tracking_evaluator import TrackingTrajectoryEvaluator
from mpr_mpc.planning.tracking_broad_phase import cull_zero_overlap_obstacles
from mpr_mpc.planning.local_mppi import LocalMPPI
if __name__=='__main__':
    files=list((R/'performance_states').glob('*.pkl'));assert len(files)==8
    rows=[]
    for f in files:
        data=pickle.loads(f.read_bytes());c=data['context'];b=data['baseline'];e=TrackingTrajectoryEvaluator(data['configuration'],control_dt=.1);shapely.prepare(c.road)
        short=cull_zero_overlap_obstacles(c,e);p=LocalMPPI(None,dict(objective_mode='tracking'),horizon=10,action_dim=2,device='cpu');p.reset(17)
        candidates=e.project(p._candidate_batch(b,p.initial_std_vector.expand_as(b),b),b)
        original=e.evaluate(candidates,b,c);fast=e.evaluate(candidates,b,short)
        assert all(np.array_equal(original[k],fast[k]) for k in original),(data['scene'],data['control'],'candidate scores')
        p.reset(17);a=p.plan(torch.zeros(1,512),b,eval_mode=True,tracking_evaluator=e,tracking_context=c)
        p.reset(17);z=p.plan(torch.zeros(1,512),b,eval_mode=True,tracking_evaluator=e,tracking_context=short)
        assert torch.equal(a.selected_sequence,z.selected_sequence) and torch.equal(a.final_mean,z.final_mean) and torch.equal(a.final_std,z.final_std)
        rows.append(dict(scene=data['scene'],control=data['control'],original_obstacles=len(c.obstacles),retained_obstacles=len(short.obstacles),candidate_count=len(candidates),all_candidate_arrays_exact=True,full_optimization_action_mean_std_exact=True,source_sha256=file_sha(f)))
    table(R/'broad_phase_real_state_parity.csv',rows);dump(R/'broad_phase_real_state_parity.json',dict(contexts=8,candidates_per_context=65,all_exact=True,not_used_in_frozen_driving_validation=True,mathematical_scope='exact zero overlap contribution under unchanged bicycle+CV model, not a new safety guarantee'))
    print('REAL_CONTEXT_BROAD_PHASE_EXACT',rows,flush=True)
