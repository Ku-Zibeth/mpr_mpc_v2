"""Actual-state comfort metrics, with common raw and five-point SG derivatives."""
import numpy as np
from scipy.signal import savgol_filter

def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def p95(x):return float(np.quantile(np.abs(x),.95))
def reversals(x,deadband=0.):
    s=np.sign(np.asarray(x));s=s[np.abs(x)>deadband]
    return int(np.count_nonzero(np.diff(s))) if len(s)>1 else 0
def series(actions,states,dt,max_steer,filtered):
    """states include s0; control t executes between states[t] and states[t+1]."""
    assert len(states)==len(actions)+1 and len(actions)>=5
    speed=np.asarray(states[:,2]);yaw=np.unwrap(states[:,3]);steer=np.asarray(actions[:,0])*max_steer
    if filtered:
        speed=savgol_filter(speed,5,2,mode='interp');yaw=savgol_filter(yaw,5,2,mode='interp')
        steer=savgol_filter(steer,5,2,mode='interp')
    ax=np.diff(speed)/dt;wx=np.diff(yaw)/dt;ay=.5*(speed[:-1]+speed[1:])*wx
    # No fictitious pre-episode velocity or acceleration; adjacent real intervals only.
    return dict(speed=speed[1:],steering=steer,steering_rate=np.diff(steer)/dt,
        steering_jerk=np.diff(steer,n=2)/dt**2,longitudinal_acceleration=ax,
        longitudinal_jerk=np.diff(ax)/dt,lateral_acceleration=ay,lateral_jerk=np.diff(ay)/dt,
        yaw_rate=wx,yaw_acceleration=np.diff(wx)/dt)
def comfort_metrics(actions,states,dt,max_steer,*,scales=None,braking_threshold=None):
    out={};all_series={}
    for filtered,label in [(False,'raw'),(True,'filtered')]:
        values=series(actions,states,dt,max_steer,filtered);all_series[label]=values
        for key,value in values.items():
            out[f'{key}_rms_{label}']=rms(value);out[f'{key}_p95_{label}']=p95(value)
        # Separate normalized command units from mapped wheel-angle units.
        out[f'normalized_steering_rate_rms_{label}']=out[f'steering_rate_rms_{label}']/max_steer
    out['steering_direction_reversals']=reversals(actions[:,0])
    out['steering_rate_reversals']=reversals(np.diff(actions[:,0]))
    out['throttle_brake_reversals']=reversals(actions[:,1])
    out['average_speed']=float(np.mean(states[1:,2]));out['maximum_speed']=float(np.max(states[1:,2]))
    ax=all_series['filtered']['longitudinal_acceleration']
    if braking_threshold is not None:
        hard=(ax < -braking_threshold)&(actions[:,1]<0)
        out['hard_braking_events']=int(np.sum(hard & ~np.r_[False,hard[:-1]]))
    if scales:
        keys=['steering_rate_rms_raw','longitudinal_jerk_rms_filtered','lateral_jerk_rms_filtered','lateral_acceleration_rms_filtered']
        out['comfort_index']=float(np.mean([out[k]/scales[k] for k in keys]))
    return out,all_series

def path_error(point,path):
    a=np.asarray(path)[:-1];d=np.diff(path,axis=0);l2=np.sum(d*d,axis=1)
    f=np.clip(np.sum((point-a)*d,axis=1)/np.maximum(l2,1e-12),0,1)
    e=point-a-f[:,None]*d;idx=int(np.argmin(np.sum(e*e,axis=1)))
    lateral=float((d[idx,0]*e[idx,1]-d[idx,1]*e[idx,0])/max(np.sqrt(l2[idx]),1e-12))
    return lateral,float(np.sqrt(np.sum(e[idx]**2)))
