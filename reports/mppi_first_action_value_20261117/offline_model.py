"""Small OFFLINE study models. No imports from this file in online planner."""
import torch
from torch import nn
import numpy as np
from tools.first_action_study import check_value_context


class StudyModel(nn.Module):
    def __init__(self,dimension,kind):
        super().__init__();self.dimension=int(dimension);self.kind=kind
        self.register_buffer('input_mean',torch.zeros(dimension));self.register_buffer('input_scale',torch.ones(dimension))
        self.register_buffer('target_mean',torch.zeros(()));self.register_buffer('target_scale',torch.ones(()))
        self.net=nn.Linear(dimension,1) if kind=='ridge_huber_rank' else nn.Sequential(nn.Linear(dimension,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):
        if x.shape[-1]!=self.dimension:raise ValueError('Featuredimension mismatch')
        return self.net((x-self.input_mean)/self.input_scale).squeeze(-1)*self.target_scale+self.target_mean


def load_model(path,expected):
    artifact=torch.load(path,map_location='cpu',weights_only=True);check_value_context(artifact['context'],expected)
    model=StudyModel(artifact['dimension'],artifact['kind']);model.load_state_dict(artifact['state_dict']);model.eval().requires_grad_(False)
    return model,artifact
