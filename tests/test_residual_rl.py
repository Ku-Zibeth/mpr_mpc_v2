from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mpr_mpc.agent import MPRMPCAgent
from mpr_mpc.planning.coordinator import MPRMPCPlanner
from mpr_mpc.planning.types import PlanningStage
from mpr_mpc.residual_rl import ResidualReplayBuffer, ResidualSACAgent, ResidualStateBuilder
from mpr_mpc.training_schedule import tdmpc_update_gate


def _batch(batch_size=8, state_dim=6, *, done=0.0):
    return {
        "state": torch.randn((batch_size, state_dim)),
        "action": torch.randn((batch_size, 2)).clamp(-1.0, 1.0),
        "reward": torch.randn((batch_size, 1)),
        "cost": torch.rand((batch_size, 1)),
        "next_state": torch.randn((batch_size, state_dim)),
        "done": torch.full((batch_size, 1), float(done)),
        "discount": torch.full((batch_size, 1), 0.99),
    }


def _bare_agent(*, state_dim=3, batch_size=2, stage_name="C"):
    stage_flags = {
        "A": PlanningStage("A", False, False),
        "B": PlanningStage("B", True, False),
        "C": PlanningStage("C", True, True),
    }
    agent = MPRMPCAgent.__new__(MPRMPCAgent)
    agent.device = torch.device("cpu")
    agent.planner = SimpleNamespace(
        enabled=True,
        stage_for_step=lambda _step: stage_flags[stage_name],
    )
    agent.residual_rl_enabled = True
    agent.residual_learning_starts = 0
    agent.residual_batch_size = batch_size
    agent.residual_update_per_step = 1.0
    agent.residual_update_budget = 0.0
    agent.residual_gamma = 0.99
    agent.residual_rl_step = batch_size
    agent.residual_sac = ResidualSACAgent(
        state_dim,
        {"hidden_dims": [16], "lr": 1e-3, "use_lagrangian": True},
        device="cpu",
    )
    agent.residual_replay = ResidualReplayBuffer(16, state_dim=state_dim)
    for _ in range(batch_size):
        agent.residual_replay.add(
            np.ones(state_dim, dtype=np.float32),
            np.asarray([0.25, -0.5], dtype=np.float32),
            1.0,
            0.25,
            np.full(state_dim, 2.0, dtype=np.float32),
            False,
        )
    agent.last_residual_transition_metrics = {}
    agent._residual_stats = []
    agent.last_plan = None
    agent.global_env_step = 60000
    agent.env = SimpleNamespace(last_info={})
    agent.tdmpc_agent = SimpleNamespace(
        update=lambda _buffer: (_ for _ in ()).throw(AssertionError("TD-MPC2 update called"))
    )
    return agent


def _checkpoint_agent():
    agent = _bare_agent(state_dim=3, batch_size=2, stage_name="C")
    model = torch.nn.Linear(3, 2)
    agent.tdmpc_agent = SimpleNamespace(
        model=model,
        optim=torch.optim.Adam(model.parameters(), lr=1e-3),
        pi_optim=torch.optim.Adam(model.parameters(), lr=1e-3),
        scale=SimpleNamespace(
            state_dict=lambda: {"scale": torch.tensor(1.0)},
            load_state_dict=lambda state: None,
        ),
        load=lambda state: model.load_state_dict(state["model"], strict=False),
    )
    agent.cfg = SimpleNamespace(
        algorithm_version="test_algo",
        latent_dim=512,
        model_size=5,
        num_q=5,
        horizon=10,
    )
    agent.planner = SimpleNamespace(
        enabled=True,
        residual_state_dim=3,
        mpr_cfg={"stages": {"residual_start_step": 20000, "mppi_start_step": 50000}},
        mppi_cfg={"num_samples": 64, "num_elites": 8, "iterations": 4},
        stage_for_step=lambda step: (
            PlanningStage("A", False, False)
            if step < 20000
            else PlanningStage("B", True, False)
            if step < 50000
            else PlanningStage("C", True, True)
        ),
        residual_invalid_count=1,
        residual_attempt_count=2,
        mppi_call_count=3,
        baseline_selected_count=4,
    )
    agent.residual_cfg = {"enabled": True}
    agent._planner_calls = 5
    agent._tdmpc_frozen = False
    agent.global_env_step = 0
    agent.best_eval_key = None
    return agent


def test_use_lagrangian_false_disables_cost_training():
    agent = ResidualSACAgent(
        6,
        {"hidden_dims": [16], "lr": 1e-3, "use_lagrangian": False},
        device="cpu",
    )
    before = [parameter.detach().clone() for parameter in agent.cost_q1.parameters()]
    info = agent.update(_batch(state_dim=6))
    after = list(agent.cost_q1.parameters())
    assert float(info["sac/loss_actor_safety"]) == 0.0
    assert float(info["lagrangian/value"]) == 0.0
    assert float(info["sac/loss_cost_critic"]) == 0.0
    for lhs, rhs in zip(before, after):
        torch.testing.assert_close(lhs, rhs)
    metrics = agent.update_lagrangian(episode_cost=10.0, episode_steps=5)
    assert metrics["lagrangian/value"] == 0.0
    assert metrics["lagrangian/update_delta"] == 0.0


def test_lagrangian_increases_above_cost_limit():
    agent = ResidualSACAgent(
        4,
        {"hidden_dims": [16], "lagrangian_lr": 0.01, "cost_limit": 5.0},
        device="cpu",
    )
    metrics = agent.update_lagrangian(episode_cost=10.0, episode_steps=5)
    assert abs(metrics["lagrangian/value"] - 0.05) < 1e-8
    assert abs(metrics["lagrangian/update_delta"] - 0.05) < 1e-8
    assert metrics["lagrangian/episode_cost_minus_limit"] == 5.0
    assert metrics["lagrangian/episode_cost_per_step"] == 2.0


def test_lagrangian_never_negative():
    agent = ResidualSACAgent(
        4,
        {"hidden_dims": [16], "lagrangian_lr": 0.01, "cost_limit": 5.0},
        device="cpu",
    )
    metrics = agent.update_lagrangian(episode_cost=-100.0, episode_steps=4)
    assert metrics["lagrangian/value"] == 0.0
    assert metrics["lagrangian/new_value"] == 0.0


def test_pending_transition_completed_on_next_state():
    agent = MPRMPCAgent.__new__(MPRMPCAgent)
    agent.residual_replay = ResidualReplayBuffer(8, state_dim=3)
    agent.invalid_residual_penalty = 0.2
    agent._pending_residual = {
        "state": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "action": np.asarray([0.4, -0.3], dtype=np.float32),
        "valid": True,
    }
    agent._residual_episode_cost = 0.0
    agent._residual_episode_steps = 0
    agent.last_residual_transition_metrics = {}
    agent.residual_sac = SimpleNamespace(constraint_alignment=False)
    agent.observe_transition(reward=2.0, cost=0.5, done=False)
    assert len(agent.residual_replay) == 0
    agent._complete_pending_with_next_state(torch.tensor([4.0, 5.0, 6.0]))
    assert len(agent.residual_replay) == 1
    np.testing.assert_allclose(agent.residual_replay.states[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(agent.residual_replay.actions[0], [0.4, -0.3])
    np.testing.assert_allclose(agent.residual_replay.rewards[0], [2.0])
    np.testing.assert_allclose(agent.residual_replay.costs[0], [0.5])
    np.testing.assert_allclose(agent.residual_replay.next_states[0], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(agent.residual_replay.dones[0], [0.0])
    assert agent._pending_residual is None
    agent._complete_pending_with_next_state(torch.tensor([7.0, 8.0, 9.0]))
    assert len(agent.residual_replay) == 1


def test_terminal_transition_targets_do_not_bootstrap():
    agent = ResidualSACAgent(6, {"hidden_dims": [16], "lr": 1e-3}, device="cpu")
    batch = _batch(batch_size=5, state_dim=6, done=1.0)
    batch["reward"].fill_(3.0)
    batch["cost"].fill_(0.75)
    info = agent.update(batch)
    torch.testing.assert_close(info["sac/target_q_mean"], torch.tensor(3.0))
    torch.testing.assert_close(info["sac/cost_target_mean"], torch.tensor(0.75))


def test_invalid_residual_replay_keeps_requested_action():
    agent = MPRMPCAgent.__new__(MPRMPCAgent)
    agent.residual_replay = ResidualReplayBuffer(8, state_dim=2)
    agent.invalid_residual_penalty = 0.2
    agent._pending_residual = {
        "state": np.asarray([1.0, 2.0], dtype=np.float32),
        "action": np.asarray([0.8, -0.6], dtype=np.float32),
        "valid": False,
    }
    agent._residual_episode_cost = 0.0
    agent._residual_episode_steps = 0
    agent.last_residual_transition_metrics = {}
    agent.residual_sac = SimpleNamespace(update_lagrangian=lambda *_args: {}, constraint_alignment=False)
    agent.observe_transition(reward=1.5, cost=0.7, done=True)
    assert len(agent.residual_replay) == 1
    np.testing.assert_allclose(agent.residual_replay.actions[0], [0.8, -0.6])
    np.testing.assert_allclose(agent.residual_replay.rewards[0], [1.3])
    np.testing.assert_allclose(agent.residual_replay.costs[0], [0.7])
    np.testing.assert_allclose(agent.residual_replay.dones[0], [1.0])


def test_include_wm_cost_adds_value_and_availability_mask():
    builder = ResidualStateBuilder({"include_wm_cost_in_state": True}, latent_dim=4, device="cpu")
    assert builder.state_dim == 4 + 12 + 1 + 2 + 2


def test_tdmpc_freezes_once_on_stage_c():
    agent = MPRMPCAgent.__new__(MPRMPCAgent)
    agent.freeze_tdmpc2_on_start = True
    agent._tdmpc_frozen = False
    agent.planner = SimpleNamespace(stage_for_step=lambda _step: PlanningStage("C", True, True))
    model = torch.nn.Linear(3, 2)
    agent.tdmpc_agent = SimpleNamespace(model=model)
    agent.cfg = SimpleNamespace(cost={"enabled": False})
    agent.residual_sac = ResidualSACAgent(4, {"hidden_dims": [16]}, device="cpu")
    agent._maybe_freeze_tdmpc2(60000)
    assert agent._tdmpc_frozen
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(parameter.requires_grad for parameter in agent.residual_sac.actor.parameters())


def test_residual_update_not_blocked_by_tdmpc_replay():
    agent = _bare_agent(state_dim=3, batch_size=2)
    info = agent.update(None, tdmpc_update_enabled=False, residual_update_enabled=True)
    assert float(info["wm/replay_ready"]) == 0.0
    assert agent.residual_sac.update_steps == 1
    assert float(info["residual_rl/updates"]) == 1.0


def test_new_stage_boundaries_and_mppi_activation():
    fake = SimpleNamespace(
        residual_start_step=20000,
        mppi_start_step=50000,
        allow_mppi=True,
        allow_residual=True,
    )
    expected = {
        0: ("A", False, False),
        19999: ("A", False, False),
        20000: ("B", True, False),
        49999: ("B", True, False),
        50000: ("C", True, True),
    }
    for step, value in expected.items():
        stage = MPRMPCPlanner.stage_for_step(fake, step)
        assert (stage.name, stage.use_residual, stage.use_mppi) == value


def test_tdmpc_stage_update_ratio_budget():
    ratios = {"A": 1.0, "B": 1.0, "C": 0.25}
    enabled, budget, ratio, updates = tdmpc_update_gate(
        stage_name="A", replay_ready=True, budget=0.0, ratios=ratios
    )
    assert (enabled, ratio, updates) == (True, 1.0, 1)
    enabled, budget, ratio, updates = tdmpc_update_gate(
        stage_name="B", replay_ready=True, budget=0.0, ratios=ratios
    )
    assert (enabled, ratio, updates) == (True, 1.0, 1)
    budget = 0.0
    updates = []
    for _ in range(8):
        _, budget, _, update_count = tdmpc_update_gate(
            stage_name="C", replay_ready=True, budget=budget, ratios=ratios
        )
        updates.append(update_count)
    assert sum(updates) == 2


def test_residual_training_stage_a_off_stage_b_and_c_on():
    stage_a = _bare_agent(stage_name="A")
    assert stage_a.update_residual(enabled=True) == {}
    stage_b = _bare_agent(stage_name="B")
    info_b = stage_b.update_residual(enabled=True)
    assert stage_b.residual_sac.update_steps == 1
    assert float(info_b["residual_rl/updates"]) == 1.0
    stage_c = _bare_agent(stage_name="C")
    info_c = stage_c.update_residual(enabled=True)
    assert stage_c.residual_sac.update_steps == 1
    assert float(info_c["residual_rl/updates"]) == 1.0


def test_milestone_component_metadata():
    agent = _checkpoint_agent()
    world = agent.milestone_state("world_model", global_env_step=20000)
    assert world["format"] == "mpr_mpc_milestone_v1"
    assert world["milestone_type"] == "world_model"
    assert world["stage"] == "A"
    assert world["metadata"]["latent_dim"] == 512
    assert world["metadata"]["model_size"] == 5
    assert world["metadata"]["num_q"] == 5
    assert world["metadata"]["horizon"] == 10
    assert world["metadata"]["residual_state_dim"] == 3
    assert world["metadata"]["residual_action_dim"] == 2
    assert world["metadata"]["use_lagrangian"]
    assert world["components"] == {
        "world_model": True,
        "residual_rl": False,
        "planner": False,
        "mppi": False,
    }
    residual = agent.milestone_state("world_model_residual_rl", global_env_step=50000)
    assert residual["milestone_type"] == "world_model_residual_rl"
    assert residual["components"]["world_model"]
    assert residual["components"]["residual_rl"]
    assert not residual["components"]["mppi"]
    full = agent.milestone_state("full_mpr_mpc", global_env_step=1000000)
    assert full["milestone_type"] == "full_mpr_mpc"
    assert full["components"]["world_model"]
    assert full["components"]["residual_rl"]
    assert full["components"]["planner"]
    assert full["components"]["mppi"]
    assert "mppi_config" in full["planner"]


def test_milestone_component_loading():
    source = _checkpoint_agent()
    target = _checkpoint_agent()
    with tempfile.TemporaryDirectory() as tmpdir:
        world_path = Path(tmpdir) / "milestone_20k_world_model.pt"
        residual_path = Path(tmpdir) / "milestone_50k_world_model_residual_rl.pt"
        full_path = Path(tmpdir) / "milestone_1m_full_mpr_mpc.pt"
        source.save_milestone(world_path, milestone_type="world_model", global_env_step=20000)
        source.save_milestone(
            residual_path,
            milestone_type="world_model_residual_rl",
            global_env_step=50000,
        )
        source.save_milestone(full_path, milestone_type="full_mpr_mpc", global_env_step=1000000)
        target.load_milestone(world_path, components=["world_model"])
        try:
            target.load_milestone(world_path, components=["residual_rl"])
        except ValueError as exc:
            assert "does not contain" in str(exc)
        else:
            raise AssertionError("20K world-model milestone must not load residual_rl")
        target.load_milestone(residual_path, components=["world_model", "residual_rl"])
        assert target.residual_rl_step == source.residual_rl_step
        target.load_milestone(full_path, components=["world_model", "residual_rl", "planner"])
    assert target.planner.mppi_call_count == source.planner.mppi_call_count


def test_milestone_evaluation_stack():
    from mpr_mpc.evaluate import _apply_evaluation_stack

    planner = SimpleNamespace(allow_residual=True, allow_mppi=True)
    info = _apply_evaluation_stack(
        planner,
        "checkpoint",
        {"milestone_type": "world_model"},
    )
    assert info["stack"] == "world_model"
    assert not planner.allow_residual
    assert not planner.allow_mppi

    info = _apply_evaluation_stack(
        planner,
        "checkpoint",
        {"milestone_type": "world_model_residual_rl"},
    )
    assert info["stack"] == "world_model_residual_rl"
    assert planner.allow_residual
    assert not planner.allow_mppi

    info = _apply_evaluation_stack(
        planner,
        "checkpoint",
        {"milestone_type": "full_mpr_mpc"},
    )
    assert info["stack"] == "full_mpr_mpc"
    assert planner.allow_residual
    assert planner.allow_mppi


def test_checkpoint_config_mismatch():
    agent = _checkpoint_agent()
    state = agent.milestone_state("full_mpr_mpc", global_env_step=1000000)
    agent.planner.mppi_cfg = {"num_samples": 128, "num_elites": 8, "iterations": 4}
    with pytest.raises(ValueError, match="Checkpoint/config mismatch: .*mppi_config"):
        agent._check_checkpoint_compatibility(state, strict_planner_config=True)
