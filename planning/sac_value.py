"""Independent frozen-encoder SAC-continuation value, never a SAC feature head."""
from __future__ import annotations
import hashlib,math,json
from pathlib import Path
import torch
from torch import nn


class SACContinuationValue(nn.Module):
    def __init__(self,latent_dim=512,hidden=(128,64)):
        super().__init__()
        self.latent_dim=int(latent_dim);self.hidden=tuple(hidden)
        self.register_buffer('input_mean',torch.zeros(self.latent_dim))
        self.register_buffer('input_std',torch.ones(self.latent_dim))
        self.register_buffer('target_mean',torch.zeros(()))
        self.register_buffer('target_std',torch.ones(()))
        sizes=(self.latent_dim,*self.hidden,1);layers=[]
        for i,(left,right) in enumerate(zip(sizes,sizes[1:])):
            layers.append(nn.Linear(left,right))
            if i<len(sizes)-2:layers.append(nn.SiLU())
        self.net=nn.Sequential(*layers)

    def forward(self,z):
        if z.shape[-1]!=self.latent_dim:raise ValueError('SAC value latent dimension mismatch.')
        value=self.net((z-self.input_mean)/self.input_std.clamp_min(1e-4)).squeeze(-1)
        return value*self.target_std+self.target_mean

    def freeze(self):
        self.eval();self.requires_grad_(False);return self


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_sha256(state):
    digest=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def policy_context_sha256(cfg):
    from .config import section
    meta=section(cfg,'metadrive');sim=dict(meta.get('simulator',{}))
    for key in ('start_seed','num_scenarios','use_render','image_observation','manual_control'):
        sim.pop(key,None)
    residual=dict(section(cfg,'residual_rl'));residual.pop('buffer_size',None)
    payload=dict(metadrive={**meta,'simulator':sim},residual=residual,lattice=section(cfg,'lattice'),
                 refinement=section(cfg,'mpr_mpc').get('refinement',{}))
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()


def load_sac_value(path,*,wm_checkpoint,horizon,gamma,device,require_validated=True,policy_context=None):
    """Statistical model readiness/compatibility, not per-trajectory approval."""
    artifact=torch.load(path,map_location=device,weights_only=True)
    context=artifact['context'];validation=artifact.get('validation',{})
    if (context['wm_checkpoint_sha256']!=file_sha256(wm_checkpoint)
            or int(context['horizon'])!=int(horizon) or not math.isclose(float(context['gamma']),float(gamma),abs_tol=1e-9)
            or context['input_kind']!='imagined_terminal_latent'):
        raise ValueError('SAC value was fitted to a different WM/policy, horizon, gamma or latent definition.')
    if policy_context is not None and context.get('policy_context_sha256')!=policy_context:
        raise ValueError('SAC value reward, simulator, Lattice or residual-policy context mismatch.')
    if require_validated:
        if not validation.get('passed',False):raise ValueError('SAC value has not passed independent predicted-latent ranking validation.')
        if set(artifact['training_scenarios']) & set(validation['scenarios']):raise ValueError('Training/validation scene leakage.')
        if len(set(validation['scenarios']))<6 or validation.get('head_state_sha256')!=state_sha256(artifact['state_dict']):
            raise ValueError('SAC value validation does not identify sufficient independent scenes and these exact weights.')
        if not (validation['pair_accuracy_ci95_low']>.5 and validation['spearman_ci95_low']>0
                and validation['j_mae']<validation['old_q_j_mae']):
            raise ValueError('SAC value independent ranking/error evidence is insufficient.')
    head=SACContinuationValue(**artifact['architecture']).to(device)
    head.load_state_dict(artifact['state_dict']);head.freeze()
    return head,artifact
