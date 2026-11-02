from __future__ import annotations

import unittest

from env.env import DEFAULT_COST_CONFIG, MetaDriveTDMPC2Env


class SafetyCostUnitTests(unittest.TestCase):
    def setUp(self):
        self.env = object.__new__(MetaDriveTDMPC2Env)
        self.env.cost_config = {
            "enabled": DEFAULT_COST_CONFIG["enabled"],
            "risk_field_weight": DEFAULT_COST_CONFIG["risk_field_weight"],
            "event_weights": dict(DEFAULT_COST_CONFIG["event_weights"]),
        }
        self.env._active_cost_events = set()

    def test_collision_flags_are_deduplicated_and_charged_once(self):
        collision = {
            "cost": 1.0,
            "crash": 1.0,
            "crash_vehicle": 1.0,
            "crash_object": 0.0,
            "out_of_road": 0.0,
        }
        total, risk, event = self.env._safety_cost(collision, normalized_risk=0.2)
        self.assertAlmostEqual(total, 1.2)
        self.assertAlmostEqual(risk, 0.2)
        self.assertAlmostEqual(event, 1.0)

        total, risk, event = self.env._safety_cost(collision, normalized_risk=0.3)
        self.assertAlmostEqual(total, 0.3)
        self.assertAlmostEqual(risk, 0.3)
        self.assertAlmostEqual(event, 0.0)

    def test_collision_is_charged_again_after_contact_clears(self):
        collision = {"cost": 1.0, "crash": 1.0, "crash_object": 1.0}
        clear = {"cost": 0.0, "crash": 0.0, "crash_object": 0.0}
        self.assertAlmostEqual(self.env._safety_cost(collision, 0.0)[2], 1.0)
        self.assertAlmostEqual(self.env._safety_cost(clear, 0.0)[2], 0.0)
        self.assertAlmostEqual(self.env._safety_cost(collision, 0.0)[2], 1.0)

    def test_distinct_events_do_not_recharge_an_active_collision(self):
        collision = {"cost": 1.0, "crash": 1.0, "crash_vehicle": 1.0}
        collision_and_warning = {
            **collision,
            "out_of_road_warning": 1.0,
        }
        self.assertAlmostEqual(self.env._safety_cost(collision, 0.0)[2], 1.0)
        self.assertAlmostEqual(self.env._safety_cost(collision_and_warning, 0.0)[2], 1.0)


if __name__ == "__main__":
    unittest.main()
