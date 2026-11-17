"""Small regressions for the offline ranking objective and native label alignment."""
import numpy as np
import pytest
import torch
from planning.candidate_value import CandidateValue, aligned_returns, pairwise_objective


def test_native_h_prefix_alignment_and_early_failure():
    label = aligned_returns([1., 2., 3., 4.], [0, 2], horizon=3, gamma=.5, done=[0, 0, 0, 1])
    assert np.allclose(label['total'], [3.25, 5.])
    assert label['continuation'][0] == 4.
    assert label['total'][0] == label['prefix'][0] + .5**3 * label['continuation'][0]
    assert label['valid_terminal'].tolist() == [True, False]
    assert np.isnan(label['continuation'][1])
    assert label['prefix'][1] == label['total'][1]  # retain real early failure total
    with pytest.raises(ValueError, match='post-terminal'):
        aligned_returns([1., 2.], [0], horizon=1, gamma=.5, done=[1, 1])


def test_pairwise_ranking_is_consistent_balanced_and_excludes_near_ties():
    pairs = torch.tensor([[0, 1], [1, 2], [0, 2]])
    target = torch.tensor([4., 2., 0.])
    correct = torch.tensor([3., 1.5, 0.], requires_grad=True)
    good, detail = pairwise_objective(correct, target, pairs, tolerance=1.)
    wrong, _ = pairwise_objective(-correct, target, pairs, tolerance=1.)
    shifted, _ = pairwise_objective(correct + 100., target, pairs, tolerance=1.)
    assert good < wrong and torch.allclose(good, shifted)
    assert (correct[0]-correct[2]) == (correct[0]-correct[1])+(correct[1]-correct[2])
    good.backward(); assert torch.isfinite(correct.grad).all()
    ties, info = pairwise_objective(correct, torch.zeros(3), pairs, tolerance=1.)
    assert info['effective_pairs'] == 0 and info['rank'] == 0 and torch.isfinite(ties)
    duplicate = torch.tensor([[0, 1], [0, 1], [1, 2], [0, 2]])
    weighted, _ = pairwise_objective(correct, target, duplicate, weights=torch.tensor([.5,.5,1.,1.]))
    assert torch.allclose(good, weighted)


def test_candidate_inputs_and_tail_action_do_not_leak_future():
    torch.manual_seed(1)
    head = CandidateValue(latent_dim=5, horizon=3, hidden=(8,4))
    z0, terminal, actions = torch.rand(1,5), torch.rand(4,5), torch.rand(4,4,2)
    score = head(z0, terminal, actions)
    actions[:,3] = 99.  # Coordinator-only padding is not an optimized input
    assert torch.equal(score, head(z0,terminal,actions))
    assert score.shape == (4,)
    assert torch.equal(torch.topk(score,3).indices, torch.topk(score-score[0],3).indices)
    head.zero_grad(); score.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    for kind in ('terminal','context_terminal'):
        model = CandidateValue(latent_dim=5,horizon=3,input_kind=kind,hidden=(8,4))
        assert model(z0,terminal,actions).shape == (4,)
