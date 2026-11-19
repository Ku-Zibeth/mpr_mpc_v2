"""Actual-state geometry labels, no new driving outcomes or candidate tuning."""
from runtime import *
from mpr_mpc.planning.native_road import NativeRoadGeometry
import pandas as pd
if __name__=='__main__':
    rows=[];meta=[]
    for scene in load(PREV/'protocol.json')['development_scenes']:
        _,_,env,agent,_=make_runtime(scene,1,'A')
        try:
            env.reset(seed=scene);sim=env.unwrapped_metadrive;g=NativeRoadGeometry.from_engine(sim.engine)
            np.savez_compressed(R/f'native_boxes_{scene}.npz',centers=g.centers,axes=g.axes,extents=g.extents,kinds=g.kinds)
            for method in ['A','T5_resolution_corrected']:
                directory=PREV/'development'/f'{method}_{scene}_r1';trace=np.load(directory/'trace.npz');steps=pd.read_csv(directory/'steps.csv')
                clearance,by=g.clearance(trace['states'][1:],sim.vehicle.LENGTH,sim.vehicle.WIDTH)
                white=by.get('LINE_SOLID_SINGLE_WHITE',None)
                from metadrive.type import MetaDriveType
                prediction=by[MetaDriveType.LINE_SOLID_SINGLE_WHITE]<=0;actual=steps.on_white_continuous_line.to_numpy(bool)
                for i in range(len(steps)):
                    rows.append(dict(scene=scene,method=method,step=i+1,minimum_contact_clearance=float(clearance[i]),predicted_white=bool(prediction[i]),actual_white=bool(actual[i]),
                        on_yellow=bool(steps.on_yellow_continuous_line.iloc[i]),native_OOR=bool(steps.out_of_road.iloc[i])))
            meta.append(dict(scene=scene,**g.metadata))
            print('CONTACT_AUDIT_MAP',scene,g.metadata,flush=True)
        finally:env.close()
    table(R/'actual_state_contact_audit.csv',rows);f=pd.DataFrame(rows)
    tp=int((f.predicted_white&f.actual_white).sum());fp=int((f.predicted_white&~f.actual_white).sum());fn=int((~f.predicted_white&f.actual_white).sum())
    dump(R/'contact_audit.json',dict(rows=len(rows),maps=meta,true_positive=tp,false_positive=fp,false_negative=fn,true_negative=int((~f.predicted_white&~f.actual_white).sum()),
        passed=fn==0 and tp>0 and fp/len(rows)<.01,criterion_fixed_before_data='No false-negative recorded white contact, at least one true contact, false-positive<1% over old actual-state audit; flat-world approximation only.'))
    print('CONTACT_AUDIT',tp,fp,fn,flush=True)
