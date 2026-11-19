"""Read actual MetaDrive contact boxes, independent of route-polygon unions.

Planar box overlap reproduces line contacts only under the validated flat-map
approximation. It is not Bullet itself and is not a safety certificate.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class NativeRoadGeometry:
    centers: np.ndarray
    axes: np.ndarray
    extents: np.ndarray
    kinds: tuple
    metadata: dict

    @classmethod
    def from_engine(cls,engine):
        from panda3d.core import NodePath,Point3,Vec3
        from panda3d.bullet import BulletBoxShape
        from metadrive.type import MetaDriveType
        names={MetaDriveType.LINE_SOLID_SINGLE_WHITE,MetaDriveType.LINE_SOLID_SINGLE_YELLOW,MetaDriveType.BOUNDARY_LINE}
        centers=[];axes=[];extents=[];kinds=[];unknown=[];seen=set()
        for world in [engine.physics_world.static_world,engine.physics_world.dynamic_world]:
            for node in [*world.getGhosts(),*world.getRigidBodies()]:
                if node.getName() not in names or int(node.this) in seen:continue
                seen.add(int(node.this));matrix=NodePath(node).getNetTransform().getMat()
                for i in range(node.getNumShapes()):
                    shape=node.getShape(i)
                    if not isinstance(shape,BulletBoxShape):
                        unknown.append((node.getName(),type(shape).__name__));continue
                    transform=node.getShapeTransform(i).getMat();half=np.asarray(shape.getHalfExtentsWithMargin(),dtype=float)
                    def point(p):return np.asarray(matrix.xformPoint(transform.xformPoint(Point3(*p))),dtype=float)
                    center=point([0,0,0])
                    # Transform vectors directly: subtracting float32 world
                    # points loses axis precision at large map coordinates.
                    basis=np.array([np.asarray(matrix.xformVec(transform.xformVec(Vec3(*v))),dtype=float) for v in [(1,0,0),(0,1,0)]])
                    norm=np.linalg.norm(basis[:,:2],axis=1)
                    if min(norm)<1e-8 or np.abs(basis[:,2]).max()>1e-5:raise ValueError('Nonplanar native contact geometry not validated.')
                    unit=basis[:,:2]/norm[:,None]
                    if abs(np.dot(*unit))>1e-5:raise ValueError('Native contact boxes are not orthogonal.')
                    centers.append(center[:2]);axes.append(unit);extents.append(half[:2]*norm);kinds.append(node.getName())
        if unknown or not centers:raise ValueError(f'Incomplete native geometry: {unknown}, boxes={len(centers)}')
        return cls(np.asarray(centers),np.asarray(axes),np.asarray(extents),tuple(kinds),dict(boxes=len(centers),unknown=unknown,source='actual Bullet world ghosts/rigid bodies and shape/net transforms'))

    def nearby(self,state,length,width,horizon_s,acceleration,extra_margin=0.):
        radius=float(state[2])*horizon_s+.5*max(float(acceleration),0)*horizon_s**2+np.hypot(length,width)/2+float(extra_margin)
        # AABB broad phase includes complete box radius; no near obstacle is lost.
        reach=radius+np.linalg.norm(self.extents,axis=1)
        keep=np.linalg.norm(self.centers-state[:2],axis=1)<=reach
        return NativeRoadGeometry(self.centers[keep],self.axes[keep],self.extents[keep],tuple(k for k,v in zip(self.kinds,keep) if v),self.metadata)

    def clearance(self,states,length,width):
        """Signed SAT separation. <=0 means planar overlap, not Euclidean distance.

        Returns minimum separation to all contact boxes and separately by type.
        Vectorized over arbitrary [...,4] states, chunk boxes to bound memory.
        """
        shape=np.asarray(states).shape[:-1];s=np.asarray(states,dtype=float).reshape(-1,4)
        f=np.column_stack((np.cos(s[:,3]),np.sin(s[:,3])));n=np.column_stack((-np.sin(s[:,3]),np.cos(s[:,3])))
        out=np.full(len(s),np.inf);by={k:np.full(len(s),np.inf) for k in set(self.kinds)}
        for start in range(0,len(self.centers),32):
            center=self.centers[start:start+32];ob=self.axes[start:start+32];ex=self.extents[start:start+32]
            delta=center[None]-s[:,None,:2];count=len(center)
            axes=np.stack((np.broadcast_to(f[:,None],delta.shape),np.broadcast_to(n[:,None],delta.shape),np.broadcast_to(ob[None,:,0],delta.shape),np.broadcast_to(ob[None,:,1],delta.shape)),-2)
            projected=np.abs(np.sum(delta[:,:,None]*axes,axis=-1))
            radius=length/2*np.abs(np.sum(f[:,None,None]*axes,-1))+width/2*np.abs(np.sum(n[:,None,None]*axes,-1))
            radius+=ex[None,:,None,0]*np.abs(np.sum(ob[None,:,None,0]*axes,-1))+ex[None,:,None,1]*np.abs(np.sum(ob[None,:,None,1]*axes,-1))
            separation=np.max(projected-radius,axis=-1);out=np.minimum(out,np.min(separation,axis=1))
            kinds=np.array(self.kinds[start:start+count])
            for kind in by:
                mask=kinds==kind
                if mask.any():by[kind]=np.minimum(by[kind],np.min(separation[:,mask],axis=1))
        return out.reshape(shape),{k:v.reshape(shape) for k,v in by.items()}
