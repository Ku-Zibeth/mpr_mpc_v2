"""Offline value-error probes. Never imported by online action selection."""
from __future__ import annotations
import hashlib,json
import numpy as np
import torch
from mpr_mpc._bootstrap import bootstrap
bootstrap()
from common import math as wm_math


class PrefixValueProbe:
    """reward(z_t,a_t), then next(), then encode(o_{t+1}); strict step pairing."""
    def __init__(self,evaluator,z0):
        self.evaluator=evaluator;self.imagined=z0.detach().clone();self.actual=z0.detach().clone()
        self.records=[];self.pending=None

    @torch.no_grad()
    def before(self,observation,action):
        if self.pending is not None:raise RuntimeError('Complete the preceding transition first.')
        model=self.evaluator.agent.model;cfg=self.evaluator.agent.cfg
        action=action.to(self.imagined.device).reshape(1,-1)
        self.actual=self.evaluator.encode(observation)
        self.pending=dict(step=len(self.records),action=action.cpu().tolist()[0],
            reward_open_loop=float(wm_math.two_hot_inv(model.reward(self.imagined,action,None),cfg).item()),
            reward_teacher=float(wm_math.two_hot_inv(model.reward(self.actual,action,None),cfg).item()),
            predicted_z_before=self.imagined.cpu().tolist(),actual_z_before=self.actual.cpu().tolist())
        self.imagined=model.next(self.imagined,action,None)

    @torch.no_grad()
    def after(self,observation,info):
        if self.pending is None:raise RuntimeError('Missing pre-action prediction.')
        # This is the exact WM training/evaluation label, not base MetaDrive reward.
        reward=float(info['tdmpc2_reward'])
        self.actual=self.evaluator.encode(observation);error=self.imagined-self.actual
        row=dict(self.pending,reward=reward,metadrive_reward=info.get('metadrive_reward'),
            latent_l2=float(error.norm()),latent_rmse=float(error.square().mean().sqrt()),
            predicted_z_after=self.imagined.cpu().tolist(),actual_z_after=self.actual.cpu().tolist())
        row.update(reward_teacher_error=row['reward_teacher']-reward,
                   reward_open_loop_error=row['reward_open_loop']-reward,
                   reward_rollout_shift=row['reward_open_loop']-row['reward_teacher'])
        self.records.append(row);self.pending=None
        return row


@torch.no_grad()
def terminal_q_audit(agent,imagined,actual,terminal_action,next_sac_action):
    """Keep candidate a_H fixed between predicted/real z_H; SAC action is diagnostic."""
    def q(z,a):
        return wm_math.two_hot_inv(agent.model.Q(z,a.to(z.device),None,return_type='all'),agent.cfg).reshape(-1).cpu().tolist()
    return dict(predicted_z_candidate_q=q(imagined,terminal_action),actual_z_candidate_q=q(actual,terminal_action),
        actual_z_sac_q=q(actual,next_sac_action),candidate_action=terminal_action.cpu().tolist(),
        next_sac_action=next_sac_action.cpu().tolist())


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def assert_origin_match(a,b):
    """Observed state/RNG/control match is not a full hidden simulator snapshot."""
    for key in ('episode_config','observation','state','rng','candidate_actions'):
        if a[key]!=b[key]:raise ValueError(f'Counterfactual origin mismatch: {key}')
    return True
