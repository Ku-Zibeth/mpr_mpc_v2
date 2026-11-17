import numpy as np
import pytest
import torch
from tools.policy_value_diagnosis import discounted_return,calibration,residual_critic_readout


def test_return_uses_native_done_and_explicit_discount():
    assert discounted_return([1,2,999],.5,[False,True,False])==2.
    assert discounted_return([1,2],.99)!=discounted_return([1,2],.995)
    with pytest.raises(ValueError):discounted_return([float('nan')],.99)


def test_calibration_is_not_ranking_oracle():
    a=calibration([0,0],[10,20]);assert a['mae']==15 and a['calibration_error']==15
    assert calibration([10,20],[10,20])['mae']==0


def test_critic_gets_residual_action_and_readout_preserves_rng():
    class Actor:
        def __call__(self,state,deterministic=False):
            assert deterministic;return (state[:,:2],)
    class Q:
        def __call__(self,state,action):return action.sum(1,keepdim=True)
    rng=torch.random.get_rng_state();state=torch.tensor([[.1,.2,8.]])
    r=residual_critic_readout(Actor(),Q(),Q(),Q(),Q(),state)
    assert torch.allclose(r['q_min'],torch.tensor([[.3]]))
    assert torch.equal(r['actor_action'],state[:,:2]) and torch.equal(rng,torch.random.get_rng_state())
