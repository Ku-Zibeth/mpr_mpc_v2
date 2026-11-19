"""Independent CG bicycle approximation for opt-in tracking MPPI.

Uses live wheel forces, mass and physics timestep. This does not change the
legacy SAC risk features or neural world model, and is not a safety guarantee.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TrackingVehicle:
    wheelbase: float
    rear_axle: float
    max_steering: float
    mass: float
    engine_force: float
    brake_impulse: float
    friction: float
    physics_dt: float
    length: float
    width: float

    @classmethod
    def from_vehicle(cls,vehicle,physics_dt):
        cfg=vehicle.config
        return cls(float(vehicle.FRONT_WHEELBASE+vehicle.REAR_WHEELBASE),float(vehicle.REAR_WHEELBASE),
            float(np.deg2rad(vehicle.max_steering)),float(cfg.get('mass') or vehicle.MASS),
            float(cfg['max_engine_force']),float(cfg['max_brake_force']),float(cfg['wheel_friction']),
            float(physics_dt),float(vehicle.LENGTH),float(vehicle.WIDTH))

def rollout(actions,state,vehicle,*,dt=.1,model='physical_cg',coefficients=None,return_substeps=False):
    """states=[x,y,speed,heading]; batch actions [N,H,2], output includes s0.

    Bullet's engine force is multiplied by physics dt; its rolling brake limit
    is an impulse. Tire contact/slip can limit the actual response, so accuracy
    is validated separately against real observations.
    """
    actions=np.asarray(actions,dtype=np.float64)
    if actions.ndim!=3 or actions.shape[-1]!=2 or not np.isfinite(actions).all() or np.max(np.abs(actions))>1+1e-7:
        raise ValueError('Tracking rollout needs finite normalized [N,H,2] controls.')
    if dt<=0 or min(vehicle.mass,vehicle.wheelbase,vehicle.rear_axle,vehicle.physics_dt)<=0:
        raise ValueError('Invalid tracking vehicle geometry or timestep.')
    n,h,_=actions.shape;s=np.asarray(state,dtype=np.float64)
    if s.shape==(4,):s=np.broadcast_to(s,(n,4)).copy()
    elif s.shape==(n,4):s=s.copy()
    else:raise ValueError('Tracking state must be [4] or [N,4].')
    if not np.isfinite(s).all() or (s[:,2]<0).any():raise ValueError('Invalid live tracking state.')
    out=np.empty((n,h+1,4));out[:,0]=s;dense=[s.copy()] if return_substeps else None
    steps=max(1,int(round(dt/vehicle.physics_dt)));subdt=dt/steps
    coefficients=coefficients or {};drive_gain=float(coefficients.get('drive_gain',1.));brake_gain=float(coefficients.get('brake_gain',1.));slip_gain=float(coefficients.get('slip_gain',1.))
    for t in range(h):
        u=actions[:,t];steer=u[:,0]*vehicle.max_steering
        beta=np.arctan(vehicle.rear_axle/vehicle.wheelbase*np.tan(steer))*slip_gain if model!='legacy' else np.zeros(n)
        if model=='legacy':acc=u[:,1]*np.where(u[:,1]>=0,4.,6.)
        else:
            drive=4*vehicle.engine_force/vehicle.mass*drive_gain
            braking=4*vehicle.brake_impulse/(vehicle.mass*vehicle.physics_dt)*brake_gain
            decel=np.maximum(-u[:,1],0)*braking
            if model=='physical_cg':decel=np.minimum(decel,vehicle.friction*9.81)
            coast=4*2./(vehicle.mass*vehicle.physics_dt)
            acc=np.where(u[:,1]>0,u[:,1]*drive,np.where(u[:,1]<0,-decel,-coast))
        for _ in range(steps):
            vnext=np.maximum(s[:,2]+subdt*acc,0);vm=.5*(s[:,2]+vnext)
            turn=vm/vehicle.wheelbase*np.cos(beta)*np.tan(steer)*subdt
            direction=s[:,3]+.5*turn+beta
            s[:,0]+=subdt*vm*np.cos(direction);s[:,1]+=subdt*vm*np.sin(direction)
            s[:,2]=vnext;s[:,3]+=turn
            if return_substeps:dense.append(s.copy())
        out[:,t+1]=s
    return np.stack(dense,axis=1) if return_substeps else out
