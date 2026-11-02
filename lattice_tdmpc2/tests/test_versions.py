from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from lattice_tdmpc2.versions import V2, V3, V4, normalize_version, sac_agent_class
from lattice_tdmpc2.versions.v2 import PIDLagrangianV2, SACAgentV2, V2TrainingSemantics
from lattice_tdmpc2.versions.v2.cost_v2 import SafetyCostV2
from lattice_tdmpc2.versions.v3 import (
    ProjectedLagrangianV3,
    SACAgentV3,
    V3TrainingSemantics,
)
from lattice_tdmpc2.versions.v3.cost_v3 import SafetyCostV3
from lattice_tdmpc2.versions.v4 import (
    ProjectedLagrangianV4,
    SACAgentV4,
    V4TrainingSemantics,
)


def flag(info, key):
    return float(bool(info.get(key, False)))


def cost_config():
    return {
        "enabled": True,
        "risk_field_weight": 1.0,
        "event_weights": {
            "cost": 1.0,
            "crash": 1.0,
            "crash_vehicle": 1.0,
            "crash_object": 1.0,
            "out_of_road": 1.0,
            "out_of_road_warning": 1.0,
        },
    }


class VersionDispatchTests(unittest.TestCase):
    def test_registry_dispatches_explicit_versions(self):
        self.assertEqual(normalize_version("v2"), V2)
        self.assertEqual(normalize_version("v3"), V3)
        self.assertEqual(normalize_version("v4"), V4)
        self.assertIs(sac_agent_class(SimpleNamespace(algorithm_version=V2)), SACAgentV2)
        self.assertIs(sac_agent_class(SimpleNamespace(algorithm_version=V3)), SACAgentV3)
        self.assertIs(sac_agent_class(SimpleNamespace(algorithm_version=V4)), SACAgentV4)

    def test_multiplier_updates_are_version_specific(self):
        v2 = PIDLagrangianV2((0.05, 0.0005, 0.1))
        v3 = ProjectedLagrangianV3(learning_rate=0.01)
        v4 = ProjectedLagrangianV4(learning_rate=0.01)
        self.assertAlmostEqual(v2.step(12.0, 10.0), 0.301)
        self.assertAlmostEqual(v3.step(12.0, 10.0), 0.02)
        self.assertAlmostEqual(v4.step(12.0, 10.0), 0.02)
        self.assertEqual(v2.step(8.0, 10.0), 0.0)
        self.assertEqual(v3.step(8.0, 10.0), 0.0)
        self.assertEqual(v4.step(8.0, 10.0), 0.0)


class VersionCostTests(unittest.TestCase):
    def test_v2_repeats_alias_cost_while_v3_deduplicates_one_contact(self):
        collision = {"cost": 1.0, "crash": 1.0, "crash_vehicle": 1.0}
        v2 = SafetyCostV2(cost_config(), flag)
        v3 = SafetyCostV3(cost_config(), flag)

        self.assertEqual(v2.compute(collision, 0.2), (3.2, 0.2, 3.0))
        self.assertEqual(v2.compute(collision, 0.3), (3.3, 0.3, 3.0))
        self.assertEqual(v3.compute(collision, 0.2), (1.2, 0.2, 1.0))
        self.assertEqual(v3.compute(collision, 0.3), (0.3, 0.3, 0.0))

    def test_replay_and_zero_speed_semantics_differ(self):
        bounds = SimpleNamespace(
            low=np.asarray([-0.4, -10.0], dtype=np.float32),
            high=np.asarray([0.4, 10.0], dtype=np.float32),
        )
        result = SimpleNamespace(unsafe_residual=True, bounds=bounds)
        v2 = V2TrainingSemantics({"unsafe_residual_cost": 1.0})
        v3 = V3TrainingSemantics({})

        replay_cost_v2, extras_v2 = v2.transition(2.0, result)
        replay_cost_v3, extras_v3 = v3.transition(2.0, result)
        self.assertEqual(replay_cost_v2, 3.0)
        self.assertEqual(set(extras_v2), {"residual_low", "residual_high", "unsafe_residual"})
        self.assertEqual(replay_cost_v3, 2.0)
        self.assertEqual(extras_v3, {})
        self.assertEqual(v2.tracking_target_speed(0.0, 13.89), 13.89)
        self.assertEqual(v3.tracking_target_speed(0.0, 13.89), 0.0)

    def test_v4_uses_refined_world_model_cost_and_episode_mean(self):
        semantics = V4TrainingSemantics({
            "constraint_cost_source": "world_model_cost_to_go",
        })
        replay_cost, extras = semantics.transition(
            9.0, SimpleNamespace(refined_cost=2.5),
        )
        self.assertEqual(replay_cost, 2.5)
        self.assertEqual(extras, {})
        self.assertEqual(semantics.lagrangian_episode_cost(9.0, 12.0, 4), 3.0)


class VersionSACUpdateTests(unittest.TestCase):
    def _batch(self, with_v2_extras: bool):
        batch_size = 8
        batch = {
            "obs": torch.randn(batch_size, 4),
            "action": torch.empty(batch_size, 2).uniform_(-1.0, 1.0),
            "reward": torch.randn(batch_size, 1),
            "cost": torch.rand(batch_size, 1),
            "next_obs": torch.randn(batch_size, 4),
            "done": torch.zeros(batch_size, 1),
            "discount": torch.full((batch_size, 1), 0.99),
        }
        if with_v2_extras:
            batch.update(
                residual_low=torch.tensor([[-0.4, -10.0]]).repeat(batch_size, 1),
                residual_high=torch.tensor([[0.4, 10.0]]).repeat(batch_size, 1),
                unsafe_residual=torch.zeros(batch_size, 1),
            )
        return batch

    def test_both_versioned_updates_execute_and_report_distinct_metrics(self):
        common = dict(
            obs_dim=4,
            action_dim=2,
            device="cpu",
            hidden_sizes=(16, 16),
            auto_alpha=False,
            use_lagrangian=True,
        )
        v2 = SACAgentV2(
            **common,
            trust_loss_weight=0.01,
            residual_sigma=(0.4, 2.0),
        )
        v3 = SACAgentV3(**common, lagrangian_lr=0.01)
        v4 = SACAgentV4(**common, lagrangian_lr=0.01)

        metrics_v2 = v2.update(self._batch(with_v2_extras=True))
        metrics_v3 = v3.update(self._batch(with_v2_extras=False))
        v4_batch = self._batch(with_v2_extras=False)
        expected_wm_cost = float(v4_batch["cost"].mean())
        metrics_v4 = v4.update(v4_batch)
        self.assertIn("loss/trust", metrics_v2)
        self.assertNotIn("loss/trust", metrics_v3)
        self.assertNotIn("loss/trust", metrics_v4)
        self.assertAlmostEqual(metrics_v4["train/cost_target_q"], expected_wm_cost, places=6)
        self.assertEqual(v2.update_steps, 1)
        self.assertEqual(v3.update_steps, 1)
        self.assertEqual(v4.update_steps, 1)
        self.assertTrue(np.isfinite(list(metrics_v2.values())).all())
        self.assertTrue(np.isfinite(list(metrics_v3.values())).all())
        self.assertTrue(np.isfinite(list(metrics_v4.values())).all())


if __name__ == "__main__":
    unittest.main()
