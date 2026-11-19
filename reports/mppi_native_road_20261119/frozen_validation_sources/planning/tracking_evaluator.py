"""Geometry-based local execution of the frozen Lattice+SAC reference.

No learned return, terminal Q, Recovery or emergency state machine. Road and
obstacle terms are continuous model costs, not a certificate of real safety.
"""
from dataclasses import dataclass
import math
import numpy as np
import shapely
from .tracking_dynamics import TrackingVehicle,rollout
from .tracking_feedback import BatchedLatticeTracker

@dataclass(frozen=True)
class TrackingContext:
    state: np.ndarray
    vehicle: TrackingVehicle
    reference_path: np.ndarray
    previous_action: np.ndarray
    obstacles: tuple
    road: object
    feedback: object
    native_road: object = None

class TrackingTrajectoryEvaluator:
    def __init__(self,cfg,*,control_dt):
        self.cfg=dict(cfg);self.dt=float(control_dt)
        if not self.cfg.get('enabled',False):raise ValueError('Tracking mode must be explicitly enabled.')
        self.horizon=int(self.cfg.get('prediction_steps',5))
        if self.horizon not in (5,10):raise ValueError('Tracking prediction_steps must be 5 or 10.')
        for name in ['lateral_scale','heading_scale','steering_delta_bound','motion_error_p95','action_scale']:
            value=float(self.cfg.get(name,float('nan')))
            if not math.isfinite(value) or value<=0:raise ValueError(f'Explicit positive tracking calibration required: {name}')
        self.model=str(self.cfg.get('motion_model','physical_uncapped'));self.coefficients=dict(self.cfg.get('motion_coefficients',{}))
        self._road_cache={};self._native_cache={}

    def reset(self):
        self._road_cache.clear();self._native_cache.clear()

    def context(self,vehicle,refined_path,safety_context,physics_dt,controller):
        if safety_context is None:raise ValueError('Tracking mode needs existing road/obstacle context; never invent missing geometry.')
        shape=TrackingVehicle.from_vehicle(vehicle,physics_dt)
        path=np.asarray(refined_path.xy,dtype=float)
        if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or not np.isfinite(path).all():raise ValueError('Invalid SAC reference path.')
        state=np.r_[np.asarray(vehicle.position)[:2],np.linalg.norm(np.asarray(vehicle.velocity)[:2]),float(vehicle.heading_theta)]
        key=id(safety_context.road)
        if key not in self._road_cache:
            inset=safety_context.road.buffer(-float(self.cfg.get('road_margin',.15)))
            if inset.is_empty:raise ValueError('Existing road inset is empty; cannot invent drivable geometry.')
            shapely.prepare(inset);self._road_cache[key]=(safety_context.road,inset)
        native=None
        if self.cfg.get('native_road',False):
            from .native_road import NativeRoadGeometry
            engine=vehicle.engine;map_key=id(engine.current_map)
            if map_key not in self._native_cache:self._native_cache[map_key]=NativeRoadGeometry.from_engine(engine)
            acceleration=4*shape.engine_force/shape.mass*float(self.coefficients.get('drive_gain',1.))
            native=self._native_cache[map_key].nearby(state,shape.length,shape.width,self.horizon*self.dt,acceleration,float(self.cfg['native_margin_m']))
        return TrackingContext(state.astype(float),shape,path.copy(),np.asarray(vehicle.current_action,dtype=float).copy(),safety_context.obstacles,self._road_cache[key][1],BatchedLatticeTracker(controller,refined_path),native)

    def project(self,actions,baseline):
        result=actions.clone();bound=float(self.cfg['steering_delta_bound'])
        result[:,:,0]=(result[:,:,0]-baseline[None,:,0]).clamp(-bound,bound)+baseline[None,:,0]
        result[:,:,0].clamp_(-1,1)
        if self.cfg.get('optimize_throttle',False):
            pedal_bound=float(self.cfg['throttle_delta_bound'])
            if not 0<pedal_bound<=.25:raise ValueError('Explicit bounded pedal experiment required.')
            result[:,:,1]=(result[:,:,1]-baseline[None,:,1]).clamp(-pedal_bound,pedal_bound)+baseline[None,:,1]
            result[:,:,1].clamp_(-1,1)
        else:result[:,:,1]=baseline[None,:,1]
        result[:,-1]=baseline[-1]
        return result

    @staticmethod
    def reference_projection(points,path):
        start=path[:-1];direction=path[1:]-start;length=np.linalg.norm(direction,axis=1);valid=length>1e-7
        start=start[valid];direction=direction[valid];length=length[valid]
        if not len(length):raise ValueError('Reference path has no nonzero segment.')
        arc=np.r_[0,np.cumsum(length)[:-1]];delta=points[...,None,:]-start
        f=np.clip(np.sum(delta*direction,axis=-1)/(length**2),0,1)
        residual=delta-f[...,None]*direction;dist2=np.sum(residual**2,axis=-1);index=np.argmin(dist2,axis=-1)
        distance=np.sqrt(np.take_along_axis(dist2,index[...,None],axis=-1)[...,0])
        progress=arc[index]+np.take_along_axis(f,index[...,None],axis=-1)[...,0]*length[index]
        yaw=np.arctan2(direction[index,1],direction[index,0])
        return distance,progress,yaw

    def evaluate(self,actions,baseline,context):
        u=actions.detach().cpu().numpy().astype(float);b=baseline.detach().cpu().numpy().astype(float)
        if not np.array_equal(u[0],b):raise ValueError('Tracking comparison must include the exact SAC baseline as row0.')
        H=self.horizon
        substeps=max(1,int(round(self.dt/context.vehicle.physics_dt)))
        if self.cfg.get('feedback_rollout',True):
            current=np.broadcast_to(context.state,(len(u),4)).copy();states=[current.copy()];dense=[current.copy()];actual=[]
            for t in range(H):
                nominal=context.feedback(current)
                nominal[:,0]=np.clip(nominal[:,0]+u[:,t,0]-b[t,0],-1,1)
                if self.cfg.get('optimize_throttle',False):
                    nominal[:,1]=np.clip(nominal[:,1]+u[:,t,1]-b[t,1],-1,1)
                if t==0:nominal=u[:,0].copy()
                segment=rollout(nominal[:,None],current,context.vehicle,dt=self.dt,model=self.model,coefficients=self.coefficients,return_substeps=True)
                current=segment[:,-1];states.append(current.copy());dense.extend(np.moveaxis(segment[:,1:],1,0));actual.append(nominal)
            states=np.stack(states,1);dense=np.stack(dense,1);actual=np.stack(actual,1)
        else:
            dense=rollout(u[:,:H],context.state,context.vehicle,dt=self.dt,model=self.model,coefficients=self.coefficients,return_substeps=True)
            states=dense[:,::substeps];actual=u[:,:H]
        distance,progress,yaw=self.reference_projection(states[:,1:,:2],context.reference_path)
        beta=np.arctan(context.vehicle.rear_axle/context.vehicle.wheelbase*np.tan(actual[:,:,0]*context.vehicle.max_steering))*float(self.coefficients.get('slip_gain',1.))
        heading=states[:,1:,3]+beta-yaw;heading=np.arctan2(np.sin(heading),np.cos(heading))
        track=(distance/float(self.cfg['lateral_scale']))**2
        align=(heading/float(self.cfg['heading_scale']))**2
        delta=(u[:,:H,0]-b[None,:H,0])/float(self.cfg['action_scale'])
        change=np.diff(np.concatenate((np.full((len(u),1),context.previous_action[0]),u[:,:H,0]),axis=1),axis=1)/float(self.cfg['action_scale'])
        # The immediate pedal is exact SAC. Future feedback may brake, so
        # explicitly penalize loss of progress rather than rewarding stopping.
        cost=track.mean(1)+track[:,-1]+.25*align.mean(1)+.05*np.mean(delta**2,1)+.01*np.mean(change**2,1)
        advancement=progress[:,-1]-progress[:,0]
        cost+=.25*(np.maximum(advancement[0]-advancement,0)/float(self.cfg['lateral_scale']))**2
        speed_cost=np.zeros(len(u))
        if self.cfg.get('optimize_throttle',False):
            desired=context.feedback.target_speeds(states[:,1:].reshape(-1,4)).reshape(len(u),H)
            speed_cost=np.mean(((states[:,1:,2]-desired)/float(self.cfg['speed_scale']))**2,axis=1)
            cost+=speed_cost
        road_cost=np.zeros(len(u));obstacle_cost=np.zeros(len(u))
        ego_length=context.vehicle.length+2*float(self.cfg.get('model_margin',.3))
        ego_width=context.vehicle.width+2*float(self.cfg.get('model_margin',.3))
        if context.road is not None:
            # Whole body corners; reuse the road assembled by unchanged SAC geometry.
            theta=dense[:,1:,3];forward=np.stack((np.cos(theta),np.sin(theta)),axis=-1);normal=np.stack((-np.sin(theta),np.cos(theta)),axis=-1)
            corners=np.stack([dense[:,1:,:2]+x*ego_length/2*forward+y*ego_width/2*normal for x,y in [(1,1),(1,-1),(-1,-1),(-1,1)]],axis=-2)
            flat=corners.reshape(-1,2);inside=shapely.intersects_xy(context.road,flat[:,0],flat[:,1]);outside=np.zeros(len(flat))
            outside[~inside]=shapely.distance(shapely.points(flat[~inside]),context.road)
            outside=outside.reshape(len(u),H*substeps,4)
            road_cost=np.mean((outside.max(-1)/float(self.cfg['lateral_scale']))**2,axis=1)
        if context.obstacles:
            obs=context.obstacles;times=np.arange(1,H*substeps+1)*self.dt/substeps
            op=np.asarray([x.position for x in obs]);ov=np.asarray([x.velocity for x in obs]);oy=np.asarray([x.heading for x in obs])
            op=op[None]+times[:,None,None]*ov[None];relative=op[None]-dense[:,1:,None,:2]
            ego_yaw=dense[:,1:,3];ef=np.stack((np.cos(ego_yaw),np.sin(ego_yaw)),-1);en=np.stack((-np.sin(ego_yaw),np.cos(ego_yaw)),-1)
            of=np.stack((np.cos(oy),np.sin(oy)),-1);on=np.stack((-np.sin(oy),np.cos(oy)),-1)
            axes=np.stack((np.broadcast_to(ef[:,:,None],relative.shape),np.broadcast_to(en[:,:,None],relative.shape),np.broadcast_to(of,relative.shape),np.broadcast_to(on,relative.shape)),-2)
            center=np.abs(np.sum(relative[...,None,:]*axes,-1))
            radius=ego_length/2*np.abs(np.sum(ef[:,:,None,None]*axes,-1))+ego_width/2*np.abs(np.sum(en[:,:,None,None]*axes,-1))
            length=np.asarray([x.length for x in obs]);width=np.asarray([x.width for x in obs])
            radius+=length[None,None,:,None]/2*np.abs(np.sum(of[None,None,:,None]*axes,-1))+width[None,None,:,None]/2*np.abs(np.sum(on[None,None,:,None]*axes,-1))
            separation=(center-radius).max(-1)
            # Only potential body overlap contributes; distant CV uncertainty
            # never turns ordinary driving into blanket candidate rejection.
            obstacle_cost=np.mean(np.max(np.maximum(-separation,0)**2,axis=-1),axis=1)/context.vehicle.width**2
        cost+=20*road_cost+20*obstacle_cost
        native_cost=np.zeros(len(u));native_clearance=np.full(len(u),np.inf)
        if self.cfg.get('native_road',False):
            if context.native_road is None:raise ValueError('Native road scoring requires actual contact geometry.')
            margin=float(self.cfg['native_margin_m'])
            if not math.isfinite(margin) or margin<=0:raise ValueError('Native envelope must be explicitly calibrated.')
            clearance,_=context.native_road.clearance(dense[:,1:],context.vehicle.length,context.vehicle.width)
            native_clearance=clearance.min(axis=1)
            native_cost=np.max(np.maximum(1-clearance/margin,0)**2,axis=1)
            cost+=20*native_cost
        finite=np.isfinite(cost)&np.isfinite(states).all((1,2))
        return dict(score=-cost,finite=finite,states=states,applied_actions=actual,tracking_cost=track.mean(1),heading_cost=align.mean(1),speed_cost=speed_cost,road_cost=road_cost,obstacle_cost=obstacle_cost,native_line_cost=native_cost,native_clearance=native_clearance,
            terminal_tracking_error=distance[:,-1],predicted_progress=progress[:,-1]-progress[:,0])
