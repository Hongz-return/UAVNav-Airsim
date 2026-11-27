"""Seed and geometry contracts for the NavRL training routes."""

from configparser import ConfigParser
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from tools.navrl_forest_runtime import forest_config, parse_route_portals, sample_training_route


class NavRLForestRouteTests(unittest.TestCase):
    portals = (22.0, (-12.0, 0.0, 12.0), 0.75)

    def test_v1_replays_the_original_seed(self):
        start, goal, yaw = sample_training_route(np.random.default_rng(0))
        np.testing.assert_allclose(start, (12.949762260665775, -24.0, 0.9180529521276106), atol=1e-12, rtol=0)
        np.testing.assert_allclose(goal, (0.7933265053693965, 24.0, -0.6265404784005448), atol=1e-12, rtol=0)
        self.assertAlmostEqual(yaw, 1.8188399976909377)

    def test_v2_routes_cover_both_axes_directions_and_all_portals(self):
        rng = np.random.default_rng(17)
        edge, offsets, jitter = self.portals
        starts, goals = set(), set()
        for _ in range(200):
            start, goal, yaw = sample_training_route(rng, self.portals)
            world_start = (start[0] - 24.0, start[1])
            world_goal = (goal[0] - 24.0, goal[1])
            axis = 0 if abs(world_start[0]) == edge else 1
            self.assertEqual(abs(world_start[axis]), edge)
            self.assertEqual(world_goal[axis], -world_start[axis])
            for ned, world, visited in ((start, world_start, starts), (goal, world_goal, goals)):
                tangent = world[1 - axis]
                center = min(offsets, key=lambda value: abs(tangent - value))
                self.assertLessEqual(abs(tangent - center), jitter)
                self.assertGreaterEqual(ned[2], -1.0)
                self.assertLessEqual(ned[2], 1.0)
                visited.add((axis, world[axis], center))
            self.assertAlmostEqual(math.atan2(goal[1] - start[1], goal[0] - start[0]), yaw)
        self.assertEqual(len(starts), 12)
        self.assertEqual(len(goals), 12)

    def test_saved_config_carries_v2_portals_and_legacy_config_drops_them(self):
        cfg = ConfigParser()
        cfg.read(Path(__file__).resolve().parents[1] / "configs" / "ch4_train_trainenv.ini", encoding="utf-8")
        stage = SimpleNamespace(number=1, static_count=120, dynamic_count=20, map_path="test_map")
        version2 = forest_config(cfg, stage, route_portals=self.portals)
        self.assertEqual(parse_route_portals(version2["navrl_forest"]["route_portals"]), self.portals)
        self.assertEqual(version2["environment"]["start_position"], "[2.0, 0.0, 0.0]")
        legacy = forest_config(version2, stage)
        self.assertNotIn("route_portals", legacy["navrl_forest"])
        self.assertEqual(legacy["environment"]["start_position"], "[0.0, 0.0, 0.0]")


if __name__ == "__main__":
    unittest.main()
