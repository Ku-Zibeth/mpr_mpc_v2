"""Repair a proven diagnostic-column alignment bug; preserve all raw data and decisions."""
import pandas as pd
from study import *

def main():
    path=R/'new_map_candidate_results.csv';rows=pd.read_csv(path);before=rows.copy();changed=0
    for i,row in rows.iterrows():
        scene,seed,cid=int(row.scene),int(row.seed),row.candidate_id
        catalog=load(origin_dir(scene,seed)/'catalog.json')
        candidate=next(x for x in catalog['representatives'] if x['candidate_id']==cid)
        action=np.asarray(candidate['action'],dtype=np.float32)
        trace=np.load(R/'branches'/f'{scene}_r{seed}'/cid/'trace.npz')
        assert np.array_equal(action,trace['actions'][0]),(scene,seed,cid)
        base=np.asarray(next(x['action'] for x in catalog['raw_roles'] if x['role']=='A'),dtype=np.float32)
        expected=[float(action[0]),float(action[1]),float(np.linalg.norm(action-base))]
        cols=['steering','throttle_brake','first_action_l2']
        changed+=int(not np.allclose(row[cols].to_numpy(float),expected,rtol=0,atol=1e-8))
        rows.loc[i,cols]=expected
    others=[x for x in rows if x not in cols]
    assert before[others].equals(rows[others]),'Only action diagnostic columns may change'
    archived=R/'source_versions/new_map_candidate_results_before_action_column_fix.csv'
    path.rename(archived)
    rows.to_csv(path,index=False,mode='x')
    reloaded=pd.read_csv(path)
    for _,row in reloaded.iterrows():
        actual=np.load(R/'branches'/f'{int(row.scene)}_r{int(row.seed)}'/row.candidate_id/'trace.npz')['actions'][0]
        assert np.allclose(actual,[row.steering,row.throttle_brake],rtol=0,atol=1e-15)
    dump(R/'candidate_action_columns_audit.json',dict(passed=True,branches_checked=len(rows),corrected_rows=changed,
        root_cause='h1_analysis reused the last candidate loop variable for diagnostic columns only',
        original_table_preserved=str(archived.relative_to(R)),raw_execution_labels_and_decision_unchanged=True))
    print('ACTION_COLUMNS_VERIFIED',len(rows),'corrected',changed,flush=True)
if __name__=='__main__':main()
