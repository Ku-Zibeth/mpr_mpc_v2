"""Offline paired diagnostics only. Never imported by the execution planner."""
from __future__ import annotations
import numpy as np


def relative_dynamics(predicted, actual, baseline=0):
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    if predicted.shape != actual.shape or predicted.ndim != 2 or not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError('Require paired finite [candidate, latent] observations.')
    dp, dr = predicted - predicted[baseline], actual - actual[baseline]
    denominator = np.linalg.norm(dp, axis=1) * np.linalg.norm(dr, axis=1)
    cosine = np.divide((dp*dr).sum(1), denominator, out=np.full(len(dp), np.nan), where=denominator>1e-12)
    return dict(predicted_delta=dp, actual_delta=dr, error=dp-dr,
                error_l2=np.linalg.norm(dp-dr,axis=1), cosine=cosine,
                actual_delta_l2=np.linalg.norm(dr,axis=1), predicted_delta_l2=np.linalg.norm(dp,axis=1))


def grouped_folds(scenes, *, folds=5, seed=2026111703):
    unique = np.unique(scenes).copy()
    if len(unique)<folds:raise ValueError('Not enough independent maps.')
    np.random.default_rng(seed).shuffle(unique)
    return [part.tolist() for part in np.array_split(unique, folds)]


def effective_df_ridge(x, y, weight, *, degrees=16):
    """Small OFFLINE linear probe; label-free regularization, no saved head.

    Same effective degrees of freedom across512/535 features. Normalize from
    training observations upstream. Choose L2 strength from training X alone,
    never from held-out returns or the old validation maps.
    """
    x,y,w=np.asarray(x,float),np.asarray(y,float),np.asarray(weight,float)
    if x.ndim!=2 or y.shape!=(len(x),) or w.shape!=y.shape or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid weighted pairwise probe samples.')
    w=w/w.sum();xx=x*np.sqrt(w[:,None]);yy=y*np.sqrt(w)
    u,s,vt=np.linalg.svd(xx,full_matrices=False);sq=s*s
    target=min(float(degrees),max(1.,np.count_nonzero(sq>1e-12)-1.))
    lo,hi=1e-12,max(1.,float(sq.sum())*100)
    for _ in range(80):
        alpha=(lo*hi)**.5
        if np.sum(sq/(sq+alpha))>target:lo=alpha
        else:hi=alpha
    alpha=(lo*hi)**.5
    coef=vt.T@((s/(sq+alpha))*(u.T@yy))
    return coef,dict(alpha=alpha,effective_df=float(np.sum(sq/(sq+alpha))),rank=int(np.count_nonzero(sq>1e-12)))


def angle_difference(left, right):
    difference=np.asarray(left)-np.asarray(right)
    return np.arctan2(np.sin(difference),np.cos(difference))
