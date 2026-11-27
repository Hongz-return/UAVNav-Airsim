"""Protect the geometry seams and the per-family training envelopes."""

from collections import Counter
from configparser import ConfigParser
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from gymnasium import spaces

from gym_env.algorithms.mamba_sac import MambaSAC
from gym_env.replay.sequence_buffer import SequenceReplayBuffer
from tools.ue_benchmark.ch4_stage3_v2_course import FLOOR_Z, TUNNEL_Z, sample_corridor, tunnel_roof
from tools.ue_benchmark.ch4_stage3_v2_training import ROOFED_FAMILIES, Stage3V2Course


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    cfg = ConfigParser()
    cfg.read([ROOT / "configs/ch4_train_trainenv.ini", ROOT / "configs/ch4_stage3_v2_overrides.ini"], encoding="utf-8")
    cfg["airsim"]["rpc_port"] = "42072"
    return cfg


class Stage3V2Tests(unittest.TestCase):
    def test_meshes_overlap_all_potential_side_and_roof_seams(self):
        for seed in range(15):
            corridor = sample_corridor("over_under", np.random.default_rng(seed))
            left, right, roof, over, under = corridor["boxes"]
            ymin, ymax = corridor["y_bounds"]
            self.assertGreaterEqual(ymax - ymin, 8)
            self.assertLessEqual(ymax - ymin, 12)
            for gate in (over, under):
                self.assertLess(gate.y[0], left.y[1])
                self.assertGreater(gate.y[1], right.y[0])
            self.assertLess(over.z[0], FLOOR_Z)
            self.assertGreater(under.z[1], roof.z[0])
            self.assertLess(roof.y[0], ymin)
            self.assertGreater(roof.y[1], ymax)

    def test_alternating_envelopes_do_not_carry_climb_reward_into_roofed_flights(self):
        course = Stage3V2Course(configuration())
        course.geometry = Mock()
        env = SimpleNamespace(np_random=np.random.default_rng(7), task=SimpleNamespace(),
                              work_space_x=(-7, 169), work_space_y=(-58, 58),
                              training_routes=[([0, -15, 1], [162, 15, 1]), ([0, 0, 1], [162, 0, 1])],
                              random_route_reverse=True, start_random_angle=.45, dynamic_model=Mock())
        families = Counter()
        for _ in range(8):
            course.select_route(env)
            family = course.scenario["family"]
            families[family] += 1
            if family in ROOFED_FAMILIES:
                self.assertEqual(env.task.work_space_z, TUNNEL_Z)
                self.assertEqual(env.task.altitude_deviation_weight, .35)
                self.assertEqual(course.tracking_reward((0, 0, -.2)), 0)
                self.assertEqual(env.max_episode_steps, 1000)
            else:
                self.assertEqual(env.task.accept_altitude_tolerance, .5)
                self.assertEqual(env.task.altitude_deviation_weight, 0)
                self.assertEqual(env.max_episode_steps, 500)
        self.assertEqual(dict(families), {"original_roofed": 2, "side_roofed": 2, "over": 1, "under": 1,
                                          "over_under": 1, "under_over": 1})
        self.assertAlmostEqual(tunnel_roof(1.4).z[0], .4)

    def test_gate_is_credited_only_for_crossing_clear_space_inside_the_corridor(self):
        course = Stage3V2Course(configuration())
        scenario = sample_corridor("under", np.random.default_rng(38))
        course.scenario = scenario
        gate = scenario["gates"][0]
        direction = scenario["direction"]
        enter, leave = sorted(direction * x for x in gate.x)
        center = scenario["center_y"]
        safe = gate.z[0] - .8
        a, b = direction * (enter - .1), direction * (leave + .1)
        for y, z, expected in [(center, safe, True), (center, gate.z[0] - .1, False),
                               (scenario["y_bounds"][1] + 1, safe, False)]:
            course.passed_gates = 0
            course.valid_crossing = False
            course._cross_gate((a, y, z), (b, y, z))
            self.assertEqual(course.route_complete, expected)

    def test_sample_episode_ids_survive_eviction(self):
        observation = spaces.Dict({"state": spaces.Box(-10, 10, (1,), dtype=np.float32)})
        actions = spaces.Box(-1, 1, (1,), dtype=np.float32)
        replay = SequenceReplayBuffer(2, observation, actions, seq_len=1, device="cpu")
        for index in range(3):
            obs = {"state": np.array([index], dtype=np.float32)}
            replay.add(obs, obs, np.array([0], dtype=np.float32), 0.0, True)
        batch = replay.sample(2, history_len=1, compact=True, include_episode_indices=True)
        self.assertEqual(set(batch["episode_indices"].tolist()), {1, 2})
        torch.testing.assert_close(batch["episode_indices"].float(), batch["context_observations"]["state"][:, 0, 0])

    def test_replay_reserves_the_requested_demo_sequences(self):
        observation = spaces.Dict({"state": spaces.Box(-10, 10, (1,), dtype=np.float32)})
        actions = spaces.Box(-1, 1, (1,), dtype=np.float32)
        replay = SequenceReplayBuffer(8, observation, actions, seq_len=1, device="cpu")
        for index in range(6):
            obs = {"state": np.array([index], dtype=np.float32)}
            replay.add(obs, obs, np.array([0], dtype=np.float32), 0.0, True)
        batch = replay.sample(8, history_len=1, compact=True, include_episode_indices=True, demo_prefix=(2, .25))
        self.assertEqual(int((batch["episode_indices"] < 2).sum()), 2)

    def test_action_anchor_ignores_online_samples_and_padded_transitions(self):
        actor = SimpleNamespace(distribution_parameters=lambda context: (context, None),
                                unscale_action=lambda actions: actions)
        algorithm = SimpleNamespace(demo_action_anchor=(1, 2, .75, 1.5), demo_terminal_altitude_weight=0.0,
                                    policy=SimpleNamespace(actor=actor))
        context = torch.zeros((3, 2, 1), requires_grad=True)
        loss = MambaSAC._demo_action_loss(algorithm, context, torch.ones_like(context),
                                         torch.tensor([[1., 0.], [1., 1.], [1., 1.]]), torch.tensor([0, 1, 2]), None)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 1.25)
        self.assertEqual(context.grad[0, 1, 0].item(), 0.0)
        self.assertTrue(torch.all(context.grad[2] == 0))
        self.assertLess(context.grad[1, 0, 0].item(), context.grad[0, 0, 0].item())

    def test_terminal_anchor_only_adjusts_vertical_action_near_demonstration_finish(self):
        actor = SimpleNamespace(distribution_parameters=lambda context: (context, None),
                                unscale_action=lambda actions: actions)
        algorithm = SimpleNamespace(demo_action_anchor=(1, 2, 0.0, 0.0), demo_terminal_altitude_weight=.7,
                                    policy=SimpleNamespace(actor=actor))
        context = torch.zeros((3, 2, 3), requires_grad=True)
        loss = MambaSAC._demo_action_loss(algorithm, context, torch.ones_like(context), torch.ones((3, 2)),
                                         torch.tensor([0, 1, 2]), torch.tensor([[20, 19], [41, 40], [30, 29]]))
        loss.backward()
        self.assertAlmostEqual(loss.item(), .7)
        self.assertEqual(context.grad.nonzero().tolist(), [[1, 1, 1]])

    def test_tight_goal_mode_enforces_2m_roofed_and_1p6m_corridor_with_side_and_endwall(self):
        cfg = ConfigParser()
        cfg.read([ROOT / "configs/ch4_train_trainenv.ini", ROOT / "configs/ch4_stage3_v3_tight_overrides.ini"], encoding="utf-8")
        cfg["airsim"]["rpc_port"] = "42072"
        course = Stage3V2Course(cfg)
        course.geometry = Mock()
        env = SimpleNamespace(np_random=np.random.default_rng(19), task=SimpleNamespace(),
                              work_space_x=(-7, 169), work_space_y=(-58, 58),
                              training_routes=[([0, -15, 1], [162, 15, 1]), ([0, 0, 1], [162, 0, 1])],
                              random_route_reverse=True, start_random_angle=.45, dynamic_model=Mock())
        families = Counter()
        for _ in range(10):
            course.select_route(env)
            fam = course.scenario["family"]
            families[fam] += 1
            if fam in ROOFED_FAMILIES:
                self.assertEqual(env.task.accept_radius, 2.0)
                self.assertEqual(env.task.accept_altitude_tolerance, .5)
                self.assertEqual(env.task.work_space_z, TUNNEL_Z)
            else:
                self.assertEqual(env.task.accept_radius, 1.6)
                self.assertEqual(env.task.accept_altitude_tolerance, .4)
                self.assertIn("goal_end_wall", [b.key for b in course.scenario["boxes"]])
        self.assertEqual(dict(families), {
            "original_roofed": 3, "side_roofed": 2,
            "over": 1, "under": 1, "side": 1, "over_under": 1, "under_over": 1,
        })

    def test_side_gate_requires_passing_outside_central_pillar_and_inside_walls(self):
        cfg = ConfigParser()
        cfg.read([ROOT / "configs/ch4_train_trainenv.ini", ROOT / "configs/ch4_stage3_v3_tight_overrides.ini"], encoding="utf-8")
        cfg["airsim"]["rpc_port"] = "42072"
        course = Stage3V2Course(cfg)
        scenario = sample_corridor("side", np.random.default_rng(42), tight_goal=True)
        course.scenario = scenario
        gate = scenario["gates"][0]
        direction = scenario["direction"]
        enter, leave = sorted(direction * x for x in gate.x)
        z_mid = -scenario["start_position"][2]
        a, b = direction * (enter - .1), direction * (leave + .1)
        for y, expected in [(scenario["center_y"], False), (gate.y[1] + 1.0, True), (gate.y[0] - 1.0, True)]:
            course.passed_gates = 0
            course.valid_crossing = False
            course._cross_gate((a, y, z_mid), (b, y, z_mid))
            self.assertEqual(course.route_complete, expected)

    def test_terminal_precision_anchor_adjusts_all_three_actions_near_finish(self):
        actor = SimpleNamespace(distribution_parameters=lambda context: (context, None),
                                unscale_action=lambda actions: actions)
        algorithm = SimpleNamespace(demo_action_anchor=(1, 2, 0.0, 0.0), demo_terminal_altitude_weight=0.0,
                                    demo_terminal_precision_weight=.6, policy=SimpleNamespace(actor=actor))
        context = torch.zeros((3, 2, 3), requires_grad=True)
        loss = MambaSAC._demo_action_loss(algorithm, context, torch.ones_like(context), torch.ones((3, 2)),
                                         torch.tensor([0, 1, 2]), torch.tensor([[55, 48], [41, 40], [30, 29]]))
        loss.backward()
        self.assertAlmostEqual(loss.item(), .6)
        self.assertEqual(set(tuple(p) for p in context.grad.nonzero().tolist()),
                         {(0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2)})


if __name__ == "__main__":
    unittest.main()
