from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from mpr_mpc._bootstrap import bootstrap


bootstrap()

from common import layers  # noqa: E402
from mpr_mpc.residual_rl.action_adapter import ResidualActionAdapter  # noqa: E402
from mpr_mpc.residual_rl.networks import GaussianActor  # noqa: E402


def _cfg(*, grouped: bool = True):
    return SimpleNamespace(
        obs_shape={"state": (259,)},
        task_dim=0,
        num_enc_layers=2,
        enc_dim=256,
        latent_dim=512,
        simnorm_dim=8,
        num_channels=32,
        state_encoder={
            "grouped": grouped,
            "ego_dim": 9,
            "nav_dim": 10,
            "lidar_dim": 240,
            "group_embed_dim": 128,
        },
    )


def test_grouped_state_encoder_shapes_and_slices():
    encoder = layers.enc(_cfg())["state"]
    assert isinstance(encoder, layers.GroupedStateEncoder)
    assert encoder.ego_slice.start == 0 and encoder.ego_slice.stop == 9
    assert encoder.nav_slice.start == 9 and encoder.nav_slice.stop == 19
    assert encoder.lidar_slice.start == 19 and encoder.lidar_slice.stop == 259
    assert encoder.ego_dim + encoder.nav_dim + encoder.lidar_dim == 259

    batch = torch.randn(4, 259)
    sequence = torch.randn(3, 4, 259)
    assert encoder(batch).shape == (4, 512)
    assert encoder(sequence).shape == (3, 4, 512)

    try:
        encoder(torch.randn(4, 258))
    except ValueError as exc:
        assert "expected last dim 259" in str(exc)
    else:
        raise AssertionError("GroupedStateEncoder must reject non-259 observations.")


def test_grouped_false_keeps_original_state_encoder():
    encoder = layers.enc(_cfg(grouped=False))["state"]
    assert not isinstance(encoder, layers.GroupedStateEncoder)
    assert encoder(torch.randn(2, 259)).shape == (2, 512)


def test_latent_tanh_zero_residual_preserves_reference():
    adapter = ResidualActionAdapter()
    coarse = torch.tensor([[0.25, 10.0], [4.0, 20.0], [-4.0, 0.5]], dtype=torch.float32)
    lower = torch.tensor([[-4.0, 0.5]], dtype=torch.float32)
    upper = torch.tensor([[4.0, 20.0]], dtype=torch.float32)
    refined, residual = adapter.refine_parameters(
        torch.zeros(3, 2),
        coarse,
        lower,
        upper,
        torch.tensor([1.0, 1.0]),
        1e-6,
    )
    torch.testing.assert_close(refined, coarse, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-5, rtol=1e-5)


def test_latent_tanh_bounds_direction_and_finite_near_edges():
    adapter = ResidualActionAdapter()
    lower = torch.tensor([[-4.0, 0.5]], dtype=torch.float32)
    upper = torch.tensor([[4.0, 20.0]], dtype=torch.float32)
    scale = torch.tensor([1.0, 1.0])
    coarse = torch.tensor(
        [
            [0.0, 8.0],
            [0.0, 8.0],
            [3.999999, 19.999998],
            [-3.999999, 0.500001],
        ],
        dtype=torch.float32,
    )
    action = torch.tensor(
        [
            [0.75, 0.50],
            [-0.75, -0.50],
            [1.0, 1.0],
            [-1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    refined, residual = adapter.refine_parameters(action, coarse, lower, upper, scale, 1e-6)
    assert torch.isfinite(refined).all()
    assert torch.isfinite(residual).all()
    assert torch.all(refined >= lower.expand_as(refined))
    assert torch.all(refined <= upper.expand_as(refined))
    assert refined[0, 0] >= coarse[0, 0]
    assert refined[0, 1] >= coarse[0, 1]
    assert refined[1, 0] <= coarse[1, 0]
    assert refined[1, 1] <= coarse[1, 1]


def test_sac_actor_keeps_single_raw_action_tanh_squash():
    source = inspect.getsource(GaussianActor.forward)
    assert "action = torch.tanh(raw_action)" in source
    assert "torch.tanh(action)" not in source
