"""Read-only value/return diagnostics; never an online value function."""
import numpy as np
import torch

CALIBRATION_EDGES = np.array([-np.inf,-200,-100,0,50,100,150,200,300,500,np.inf])


def discounted_return(rewards, gamma, done=None):
    rewards=np.asarray(rewards,dtype=np.float64)
    if rewards.ndim!=1 or not np.isfinite(rewards).all() or not 0<=gamma<=1:
        raise ValueError('Require finite reward vector and gamma in[0,1].')
    if done is not None:
        done=np.asarray(done,bool)
        if done.shape!=rewards.shape:raise ValueError('done/reward shape mismatch')
        ends=np.flatnonzero(done)
        if len(ends):rewards=rewards[:ends[0]+1]
    return float(rewards @ (float(gamma)**np.arange(len(rewards))))


def calibration(predicted, actual):
    p,y=np.asarray(predicted,float),np.asarray(actual,float)
    if p.shape!=y.shape or p.ndim!=1 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Require finite paired predictions and labels.')
    bins=np.digitize(p,CALIBRATION_EDGES[1:-1]);error=0.
    for b in np.unique(bins):
        mask=bins==b;error+=mask.mean()*abs(p[mask].mean()-y[mask].mean())
    return dict(mae=float(np.mean(np.abs(p-y))),bias=float(np.mean(p-y)),
                calibration_error=float(error),rmse=float(np.sqrt(np.mean((p-y)**2))))


@torch.no_grad()
def residual_critic_readout(actor,q1,q2,target_q1,target_q2,state):
    """Q(s,deterministic residual mean); not a physical-action critic or soft V."""
    action=actor(state,deterministic=True)[0]
    first,second=q1(state,action),q2(state,action)
    return dict(actor_action=action,q1=first,q2=second,q_min=torch.minimum(first,second),
        q_mean=.5*(first+second),target_q1=target_q1(state,action),target_q2=target_q2(state,action))
