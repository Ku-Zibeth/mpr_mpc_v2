"""Opt-in control smoothness objective; short bicycle checks preserve reference intent.

Neither the approximation nor baseline preservation is a safety certificate.
Throttle is held identical to SAC: the legacy longitudinal dynamics are not
accurate enough to optimize physical longitudinal/lateral jerk online.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import math
import numpy as np
import torch
from .safety import TrajectorySafetyEvaluator

@dataclass(frozen=True)
class ComfortContext:
    motion: object
    previous_two_actions: np.ndarray
    reference_path: np.ndarray

class ComfortTrajectoryEvaluator:
    def __init__(self,cfg,*,control_dt,device):
        self.cfg=dict(cfg);self.dt=float(control_dt)
        names=('steering_rate_scale','steering_jerk_scale','steering_delta_bound',
               'path_extra_tolerance_m','progress_loss_tolerance_m','heading_tolerance_rad')
        for name in names:
            value=float(self.cfg.get(name,float('nan')))
            if not math.isfinite(value) or value<=0:raise ValueError(f'Comfort {name} must be explicitly calibrated and positive.')
        if not self.cfg.get('enabled',False):raise ValueError('Comfort objective requires comfort.enabled=true.')
        if not math.isfinite(self.dt) or self.dt<=0:raise ValueError('Comfort control_dt must be positive.')
        self.motion=TrajectorySafetyEvaluator(self.cfg.get('kinematic_cfg',{}),control_dt=self.dt,device=device)
        self.reset()

    def reset(self):
        self._observed_key=None;self._observed_action=None;self._penultimate=None

    def context(self,vehicle,reference_path,*,width,wheelbase,max_steering):
        previous=np.asarray(vehicle.current_action,dtype=float).reshape(2).copy()
        position=np.asarray(vehicle.position,dtype=float)[:2].copy()
        speed=float(np.linalg.norm(np.asarray(vehicle.velocity,dtype=float)[:2]))
        # Observe EXECUTED controls from the vehicle, not prior planned controls.
        # Repeated planner-only latency probes do not invent extra history ticks.
        key=(*position,float(vehicle.heading_theta),speed,*previous)
        if self._observed_action is None:self._penultimate=previous.copy()
        elif key!=self._observed_key:self._penultimate=self._observed_action.copy()
        self._observed_key=key;self._observed_action=previous.copy()
        history=np.stack((self._penultimate,previous))
        path=np.asarray(reference_path,dtype=float)
        if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all():
            raise ValueError('Comfort needs a finite refined_path, not a new route proposal.')
        values=np.asarray([*position,float(vehicle.heading_theta),speed,width,wheelbase,max_steering,*history.flatten()])
        if not np.isfinite(values).all() or min(width,wheelbase)<=0 or not 0<max_steering<math.pi/2:
            raise ValueError('Invalid live comfort vehicle geometry/history.')
        motion=SimpleNamespace(position=position,heading=float(vehicle.heading_theta),speed=speed,
            length=float(vehicle.LENGTH),width=width,wheelbase=wheelbase,max_steering=max_steering,previous_action=previous)
        return ComfortContext(motion,history,path.copy())

    def project(self,actions,baseline):
        result=actions.clone();bound=float(self.cfg['steering_delta_bound'])
        result[:,:,0]=torch.maximum(torch.minimum(result[:,:,0],baseline[None,:,0]+bound),baseline[None,:,0]-bound).clamp(-1,1)
        result[:,:,1]=baseline[None,:,1] # preserve braking/target-speed decisions exactly
        result[:,-1]=baseline[-1]       # H+1 padding is not a comfort optimization variable
        return result

    def evaluate(self,actions,baseline,context):
        if context is None:raise ValueError('Comfort scoring needs actual vehicle state and executed-action history.')
        n=len(actions);finite=torch.isfinite(actions).all((1,2))&(actions.abs()<=1).all((1,2))
        controls=actions[:,:-1];history=actions.new_tensor(context.previous_two_actions[:,0])*context.motion.max_steering
        steer=controls[:,:,0]*context.motion.max_steering
        sequence=torch.cat((history[None].expand(n,-1),steer),1)
        rate=torch.diff(sequence[:,1:],dim=1)/self.dt
        jerk=torch.diff(sequence,n=2,dim=1)/self.dt**2
        rate_cost=(rate/float(self.cfg['steering_rate_scale'])).square()
        jerk_cost=(jerk/float(self.cfg['steering_jerk_scale'])).square()
        cost=.5*(rate_cost+jerk_cost).mean(1)
        immediate=.5*(rate_cost[:,0]+jerk_cost[:,0])
        # Only H1 position/heading has acceptable measured accuracy. No H10
        # acceleration/jerk rollout is used to rank controls.
        short=torch.cat((actions[:,:1],actions[:,-1:]),1)
        base_short=torch.cat((baseline[None,:1],baseline[None,-1:]),1)
        xy,yaw,speed=self.motion.rollout(torch.cat((base_short,short),0),context.motion)
        endpoint=xy[:,-1];base_xy=endpoint[0];candidate_xy=endpoint[1:]
        path=actions.new_tensor(context.reference_path);start=path[:-1];direction=path[1:]-start
        length=direction.square().sum(-1).clamp_min(1e-12)
        delta=endpoint[:,None]-start
        frac=(delta*direction).sum(-1)/length
        error=delta-frac.clamp(0,1)[:,:,None]*direction
        distances=error.square().sum(-1);indices=distances.argmin(1)
        tracking=distances.gather(1,indices[:,None]).squeeze(1).sqrt()
        tangent=direction[indices[0]]/length[indices[0]].sqrt()
        progress=(candidate_xy-base_xy)*tangent
        yaw_delta=torch.atan2((yaw[1:,-1]-yaw[0,-1]).sin(),(yaw[1:,-1]-yaw[0,-1]).cos()).abs()
        feasible=finite & ((actions[:,:,1]-baseline[None,:,1]).abs().amax(1)==0)
        feasible &= (actions[:,:,0]-baseline[None,:,0]).abs().amax(1)<=float(self.cfg['steering_delta_bound'])+1e-7
        feasible &= tracking[1:]<=tracking[0]+float(self.cfg['path_extra_tolerance_m'])
        feasible &= progress.sum(-1)>=-float(self.cfg['progress_loss_tolerance_m'])
        feasible &= yaw_delta<=float(self.cfg['heading_tolerance_rad'])
        # Score the baseline with the same executed history; prevent H10-only
        # improvements that worsen the one control actually submitted now.
        base_controls=baseline[None]
        bs=torch.cat((history[None],base_controls[:,:-1,0]*context.motion.max_steering),1)
        br=torch.diff(bs[:,1:],dim=1)/self.dt;bj=torch.diff(bs,n=2,dim=1)/self.dt**2
        base_immediate=.5*((br[:,0]/float(self.cfg['steering_rate_scale']))**2+(bj[:,0]/float(self.cfg['steering_jerk_scale']))**2)
        feasible &= immediate<=base_immediate[0]+torch.finfo(actions.dtype).eps
        return dict(score=-cost,feasible=feasible,steering_rate_cost=rate_cost.mean(1),steering_jerk_cost=jerk_cost.mean(1),
            immediate_cost=immediate,reference_error=tracking[1:],relative_progress=progress.sum(-1),positions=xy[1:],headings=yaw[1:],speeds=speed[1:])
