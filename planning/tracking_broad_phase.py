"""Exact zero-contribution culling for the existing CV overlap cost.

Not a new safety margin: every possibly overlapping body is retained under
exactly the same candidate dynamics and constant-velocity obstacle model.
This does not guarantee that the model describes real traffic correctly.
"""
from dataclasses import replace
import numpy as np


def cull_zero_overlap_obstacles(context, evaluator):
    obstacles=context.obstacles
    if not obstacles:return context
    duration=evaluator.horizon*evaluator.dt
    vehicle=context.vehicle
    acceleration=4. if evaluator.model=='legacy' else 4*vehicle.engine_force/vehicle.mass*float(evaluator.coefficients.get('drive_gain',1.))
    margin=float(evaluator.cfg.get('model_margin',.3))
    ego_radius=np.hypot(vehicle.length+2*margin,vehicle.width+2*margin)/2
    reach=float(context.state[2])*duration+.5*max(acceleration,0.)*duration**2+ego_radius
    kept=[]
    for obstacle in obstacles:
        p=np.asarray(obstacle.position,dtype=np.float64);v=np.asarray(obstacle.velocity,dtype=np.float64)
        dimensions=np.array([obstacle.length,obstacle.width],dtype=np.float64)
        if not np.isfinite(np.r_[p,v,dimensions]).all() or (dimensions<=0).any():raise ValueError('Invalid obstacle in exact broad phase.')
        distance=float(np.linalg.norm(p-context.state[:2]));bound=reach+float(np.linalg.norm(v))*duration+float(np.linalg.norm(dimensions))/2
        # Keep equality and several ulps of numerical slack. Any omitted body
        # has strictly positive SAT separation throughout this model's horizon.
        slack=16*np.spacing(max(distance,bound,1.))
        if distance<=bound+slack:kept.append(obstacle)
    return replace(context,obstacles=tuple(kept))


def exact_native_clearance(geometry,states,length,width):
    """Same SAT arithmetic, without allocating [...,4,2] repeated axis tensors.

    Self/cross dot products are retained, including floating-point non-unit
    norms. Do not substitute ideal orthogonality or change contact tolerances.
    """
    shape=np.asarray(states).shape[:-1];s=np.asarray(states,dtype=float).reshape(-1,4)
    forward=np.column_stack((np.cos(s[:,3]),np.sin(s[:,3])));normal=np.column_stack((-np.sin(s[:,3]),np.cos(s[:,3])))
    out=np.full(len(s),np.inf);by={k:np.full(len(s),np.inf) for k in set(geometry.kinds)}
    dot=lambda a,b:np.abs(np.sum(a*b,axis=-1))
    ff=dot(forward,forward)[:,None];fn=dot(forward,normal)[:,None];nn=dot(normal,normal)[:,None]
    for start in range(0,len(geometry.centers),32):
        center=geometry.centers[start:start+32];axes=geometry.axes[start:start+32];extents=geometry.extents[start:start+32]
        of,on=axes[:,0],axes[:,1];delta=center[None]-s[:,None,:2];hx,hy=extents[None,:,0],extents[None,:,1]
        f0=dot(forward[:,None],of[None]);f1=dot(forward[:,None],on[None]);n0=dot(normal[:,None],of[None]);n1=dot(normal[:,None],on[None])
        oo=dot(of,of)[None];ox=dot(of,on)[None];xx=dot(on,on)[None]
        r0=(length/2*ff+width/2*fn)+(hx*f0+hy*f1)
        r1=(length/2*fn+width/2*nn)+(hx*n0+hy*n1)
        r2=(length/2*f0+width/2*n0)+(hx*oo+hy*ox)
        r3=(length/2*f1+width/2*n1)+(hx*ox+hy*xx)
        separation=np.max(np.stack((dot(delta,forward[:,None])-r0,dot(delta,normal[:,None])-r1,dot(delta,of[None])-r2,dot(delta,on[None])-r3),-1),axis=-1)
        out=np.minimum(out,separation.min(axis=1));kinds=np.array(geometry.kinds[start:start+len(center)])
        for kind in by:
            mask=kinds==kind
            if mask.any():by[kind]=np.minimum(by[kind],np.min(separation[:,mask],axis=1))
    return out.reshape(shape),{k:v.reshape(shape) for k,v in by.items()}


class ExactNativeRoad:
    def __init__(self,geometry):self.geometry=geometry
    def clearance(self,states,length,width):return exact_native_clearance(self.geometry,states,length,width)


def exact_geometry_context(context,evaluator):
    result=cull_zero_overlap_obstacles(context,evaluator)
    if result.native_road is not None:result=replace(result,native_road=ExactNativeRoad(result.native_road))
    return result
