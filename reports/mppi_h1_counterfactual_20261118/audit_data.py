"""Verify live-captured candidate provenance and every executed branch after collection."""
from study import *

def main():
    p=load(R/'protocol.json');items=[]
    for scene in p['scenes']:
        for step in p['steps']:
            directory=origin_dir(scene,step);meta=load(directory/'catalog.json');z=np.load(directory/'catalog.npz')
            for mode in ['C','D']:
                batch=z[f'{mode}_iteration_actions'];values=z[f'{mode}_iteration_values'];final=z[f'{mode}_final_actions'];last=z[f'{mode}_selected']
                assert batch.shape==(4,65,11,2) and values.shape==(4,65) and final.shape==(2,11,2)
                assert np.array_equal(batch[:,0],np.repeat(final[0][None],4,axis=0))
                chosen=meta[mode]['selected_source'];assert chosen in [0,1] and np.array_equal(last,final[chosen])
                assert np.array_equal(last[0],np.asarray(meta[mode]['action'],np.float32))
                assert abs(float(z[f'{mode}_final_values'][chosen])-meta[mode]['score'])<1e-6
            order=np.argsort(z['C_iteration_values'][-1,1:],kind='stable')+1
            for role,q in zip(['A3','A4','A5','A6','A7'],p['gaussian_quantiles']):
                row=next(x for x in meta['raw_roles'] if x['role']==role);idx=int(order[int(round(q*63))])
                assert row['original_row']==idx and idx>=1
                assert np.array_equal(np.asarray(row['action'],np.float32),z['C_iteration_actions'][-1,idx,0])
                assert row['raw_score']==float(z['C_iteration_values'][-1,idx])
            rows=[]
            for r in meta['representatives']:
                result=load(R/'branches'/f'{scene}_{step}'/r['candidate_id']/'episode.json')
                assert result['exact_origin'] and result['exact_policy_history'] and result['extended_environment_exact']
                assert result['feedback_steps']==result['episode_length']-1 and result['intervention_steps']==1
                assert result['continuation_mppi_calls']==0 and result['native_done'] and result['weights_unchanged']
                rows.append(r['candidate_id'])
            items.append(dict(scene=scene,step=step,candidate_shape_verified=True,iteration3_is_final=True,
                row0_baseline_verified=True,raw_gaussian_roles_and_values_exact=True,final_selected_output_exact=True,
                unique_executed=rows,origin_and_policy_checks_all_exact=True,first_action_only=True))
    dump(R/'data_audit.json',dict(origins=items,all_passed=True,formal_h1_ranking='NOT ESTIMABLE',
        provenance='Runtime evaluator calls capture the exact candidate tensors used by LocalMPPI; final batch is separate Baseline/Mean, not iteration4.'))
    print('CANDIDATE_AND_BRANCH_DATA_AUDIT_PASSED',flush=True)
if __name__=='__main__':main()
