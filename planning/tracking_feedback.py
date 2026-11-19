"""Vectorized existing Frenet tracker for candidate feedback rollouts."""
import numpy as np

class BatchedLatticeTracker:
    def __init__(self,controller,path):
        self.points=np.asarray(path.xy,dtype=float);self.cfg=controller.config
        self.arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(self.points,axis=0),axis=1))]
        self.curvature=np.array([controller._local_path_curvature(self.points,self.arc,i) for i in range(len(self.points))])
        resolver=self.cfg.get('_tracking_target_speed')
        self.target_speed=max(float(path.target_speed),0) if resolver is None else float(resolver(path.target_speed,controller.planner.target_speed))

    def __call__(self,states):
        p=self.points;n=len(p)
        if n<2:return np.zeros((len(states),2))
        position=states[:,:2];speed=states[:,2];yaw=states[:,3]
        closest=np.argmin(np.sum((position[:,None]-p[None])**2,axis=2),axis=1)
        look=float(self.cfg['frenet_follow_lookahead'])+np.maximum(speed,0)*float(self.cfg['frenet_follow_speed_gain'])
        look=np.clip(look,float(self.cfg['frenet_min_follow_lookahead']),float(self.cfg['frenet_max_follow_lookahead']))
        look=np.where(self.curvature[closest]>1e-6,look/(1+float(self.cfg['frenet_curve_lookahead_gain'])*self.curvature[closest]),look)
        look=np.maximum(float(self.cfg['frenet_min_follow_lookahead']),look)
        target=np.minimum(np.searchsorted(self.arc,self.arc[closest]+look,side='left'),n-1)
        delta=p[target]-position;fallback=np.linalg.norm(delta,axis=1)<1e-6
        delta[fallback]=p[np.minimum(target[fallback]+1,n-1)]-position[fallback]
        bearing=np.arctan2(delta[:,1],delta[:,0]);tangent=p[np.minimum(target+1,n-1)]-p[np.maximum(target-1,0)]
        path_yaw=np.where(np.linalg.norm(tangent,axis=1)>1e-6,np.arctan2(tangent[:,1],tangent[:,0]),bearing)
        tangent=p[np.minimum(closest+1,n-1)]-p[np.maximum(closest-1,0)]
        near_yaw=np.where(np.linalg.norm(tangent,axis=1)>1e-6,np.arctan2(tangent[:,1],tangent[:,0]),path_yaw)
        normal=np.stack((-np.sin(near_yaw),np.cos(near_yaw)),-1);lateral=np.sum((position-p[closest])*normal,axis=1)
        heading=(path_yaw-yaw+np.pi)%(2*np.pi)-np.pi;bearing=(bearing-yaw+np.pi)%(2*np.pi)-np.pi
        steer=np.clip(.60*bearing+.35*heading-.08*lateral,-1,1)
        target_speed=self.target_speeds(states,closest)
        return np.stack((steer,np.clip(.35*(target_speed-speed),-1,1)),-1)

    def target_speeds(self,states,closest=None):
        if closest is None:
            closest=np.argmin(np.sum((states[:,None,:2]-self.points[None])**2,axis=2),axis=1)
        target_speed=np.full(len(states),self.target_speed)
        if self.cfg['frenet_curve_speed_limit']:
            curve=self.curvature[closest];limit=np.sqrt(max(.1,float(self.cfg['frenet_max_lateral_accel']))/np.maximum(curve,1e-12))
            limit=np.maximum(float(self.cfg['frenet_min_curve_speed']),limit)
            target_speed=np.where(curve>1e-6,np.minimum(target_speed,limit),target_speed)
        return target_speed
