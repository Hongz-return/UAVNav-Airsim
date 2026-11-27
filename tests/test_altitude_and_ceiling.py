"""Altitude shaping and virtual ceiling regressions without an AirSim server."""

from configparser import ConfigParser
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cosysairsim as airsim
from gymnasium import spaces
import numpy as np

from gym_env.envs.airsim_env import AirsimGymEnv
from gym_env.envs.dynamics.multirotor_airsim import MultirotorDynamicsAirsim
from gym_env.envs.perception.depth_state_perception import DepthStatePerception
from gym_env.envs.tasks.base_task import BaseTask
from scripts.evaluation.evaluate_single import apply_evaluation_overrides


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ACTION = np.array([1.0, 0.0, 0.0], dtype=np.float32)
ALTITUDE_KEYS = ("altitude_deviation_weight", "altitude_boundary_margin", "altitude_boundary_weight")


def training_config():
    cfg = ConfigParser()
    cfg.read(CONFIGS / "ch4_train_trainenv.ini", encoding="utf-8")
    return cfg


def mock_environment(cfg):
    client = Mock()
    client.simGetVehiclePose.return_value = airsim.Pose(airsim.Vector3r(), airsim.Quaternionr())
    client.getMultirotorState.return_value = SimpleNamespace(
        timestamp=0,
        kinematics_estimated=SimpleNamespace(
            linear_velocity=airsim.Vector3r(), angular_velocity=airsim.Vector3r()
        ),
    )
    client.armDisarm.return_value = True
    client.moveByVelocityAsync.return_value.get.return_value = True
    client.simGetCollisionInfo.return_value.has_collided = False
    client.simGetImages.return_value = [
        SimpleNamespace(width=32, height=18, image_data_float=[25.0] * (32 * 18))
    ]
    env = AirsimGymEnv()
    with patch("cosysairsim.MultirotorClient", return_value=client):
        env.set_config(cfg)
    env.perception._camera_warmed = True
    client.reset_mock()
    return env, client


class AltitudeRewardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = training_config()
        self.cfg["environment"]["work_space_z"] = "[-2.0, 2.0]"
        self.cfg["reward"].update({
            "altitude_deviation_weight": "0.4",
            "altitude_boundary_margin": "0.5",
            "altitude_boundary_weight": "1.2",
        })
        self.dynamics = SimpleNamespace(
            goal_position=[10.0, 0.0, 0.0], max_vertical_difference=2.0, get_position=Mock()
        )
        self.env = SimpleNamespace(
            dynamic_model=self.dynamics, action_space=spaces.Box(-1, 1, shape=(3,)),
            min_distance_to_obstacles=25.0,
        )

    def set_altitude(self, z):
        self.dynamics.get_position.return_value = [0.0, 0.0, z]
        self.env.previous_distance_from_des_point = math.hypot(10.0, z)

    def test_vertical_scale_default_custom_and_state_normalization(self):
        cfg = training_config()
        client = Mock()
        client.simGetVehiclePose.return_value = airsim.Pose(airsim.Vector3r(), airsim.Quaternionr())
        cfg.remove_option("environment", "max_vertical_difference")
        for scale, expected_z in ((5.0, 0.6), (2.0, 0.75)):
            with self.subTest(scale=scale):
                dynamics = MultirotorDynamicsAirsim(cfg, client=client)
                dynamics._set_goal_pose_single([10.0, 0.0, 1.0])
                dynamics.get_velocity = Mock(return_value=[1.0, 0.0, 0.0])
                self.assertEqual(dynamics.max_vertical_difference, scale)
                self.assertAlmostEqual(dynamics._get_state_feature()[1], expected_z)
            cfg.set("environment", "max_vertical_difference", "2.0")

    def test_invalid_vertical_scale_and_reward_values(self):
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with self.subTest(key="max_vertical_difference", value=value):
                self.cfg.set("environment", "max_vertical_difference", value)
                with self.assertRaisesRegex(ValueError, "max_vertical_difference"):
                    MultirotorDynamicsAirsim(self.cfg, client=Mock())
        for key in ALTITUDE_KEYS:
            original = self.cfg.get("reward", key)
            for value in ("-1", "nan", "inf", "-inf"):
                with self.subTest(key=key, value=value):
                    self.cfg.set("reward", key, value)
                    with self.assertRaisesRegex(ValueError, key):
                        BaseTask(self.cfg, self.env)
            self.cfg.set("reward", key, original)

    def test_exact_altitude_penalties_at_center_floor_and_ceiling(self):
        task = BaseTask(self.cfg, self.env)
        cases = ((0.0, 0.0), (1.0, -0.1), (1.75, -0.60625), (-1.75, -0.60625))
        for z, expected in cases:
            with self.subTest(z=z):
                self.set_altitude(z)
                reward = task.compute_reward(False, ACTION)
                self.assertAlmostEqual(task.last_reward_components["altitude"], expected)
                self.assertAlmostEqual(reward, expected - task.time_penalty)

    def test_altitude_saturates_before_dense_clip_and_leaves_terminal_reward(self):
        self.cfg.set("reward", "dense_reward_clip", "0.2")
        self.dynamics.max_vertical_difference = 1.0
        self.set_altitude(1.75)
        task = BaseTask(self.cfg, self.env)
        self.assertEqual(task.compute_reward(False, ACTION), -0.2)
        self.assertAlmostEqual(task.last_reward_components["altitude"], -0.7)
        self.assertAlmostEqual(task.last_reward_components["clipping"], 0.515)
        self.assertAlmostEqual(sum(task.last_reward_components.values()), -0.2)

        self.dynamics.get_position.reset_mock()
        task._cached_outside = True
        self.assertEqual(task.compute_reward(True, ACTION), task.r_fail_outside)
        self.assertNotIn("altitude", task.last_reward_components)
        self.dynamics.get_position.assert_not_called()

    def test_disabled_altitude_preserves_reward_components_and_pose_reads(self):
        self.set_altitude(1.75)
        for settings in ({}, {"altitude_boundary_margin": "0.5"}, {"altitude_boundary_weight": "1.2"}):
            with self.subTest(settings=settings):
                for key in ALTITUDE_KEYS:
                    self.cfg.remove_option("reward", key)
                self.cfg["reward"].update(settings)
                task = BaseTask(self.cfg, self.env)
                self.dynamics.get_position.reset_mock()
                self.assertAlmostEqual(task.compute_reward(False, ACTION), -0.015)
                self.assertEqual(task.last_reward_components, {
                    "progress": 0.0, "obstacle": 0.0, "time": -0.015, "smoothness": 0.0,
                    "clipping": 0.0, "terminal": 0.0, "timeout": 0.0,
                })
                self.dynamics.get_position.assert_called_once_with()

    def test_altitude_component_is_accumulated_and_cleared_on_episode_reset(self):
        self.cfg.set("environment", "max_episode_steps", "2")
        self.cfg.set("perception", "virtual_ceiling_prob", "0.0")
        env, client = mock_environment(self.cfg)
        env.reset(seed=19)
        pose = client.simGetVehiclePose.return_value
        for z in (0.0, 0.25):
            pose.position.z_val = z
            _, _, terminated, truncated, info = env.step(ACTION)
            self.assertFalse(terminated)
            self.assertAlmostEqual(info["reward_components"]["altitude"], -0.4 * ((1.0 - z) / 2.0) ** 2)
        self.assertTrue(truncated)
        self.assertAlmostEqual(info["episode_reward_components"]["altitude"], -0.15625)
        env.reset(seed=19)
        self.assertEqual(env._episode_reward_components, {})


class VirtualCeilingTests(unittest.TestCase):
    def test_invalid_ceiling_config_values(self):
        invalid = {
            "virtual_ceiling_prob": (-0.1, 1.1, math.nan, math.inf),
            "virtual_ceiling_height_min": (0, -1, math.nan, math.inf),
            "virtual_ceiling_height_max": (1, math.nan, math.inf),
            "virtual_ceiling_arch_drop_max": (-1, math.nan, math.inf),
            "virtual_ceiling_hfov_deg": (0, 179, math.nan, math.inf),
            "virtual_ceiling_aspect_ratio": (0, -1, math.nan, math.inf),
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    cfg = training_config()
                    cfg.set("perception", key, str(value))
                    with self.assertRaisesRegex(ValueError, key):
                        DepthStatePerception(cfg, client=None)

    def test_pinhole_geometry_arch_and_nearer_real_depth(self):
        cfg = training_config()
        cfg["environment"].update(screen_height="4", screen_width="5")
        cfg.set("perception", "virtual_ceiling_aspect_ratio", "2.0")
        perception = DepthStatePerception(cfg, client=None)
        rng = Mock()
        rng.random.return_value = 0.1
        rng.uniform.side_effect = [1.5, 0.6]
        perception.reset(rng=rng)
        rng.random.assert_called_once_with()
        self.assertEqual(rng.uniform.call_args_list[0].args, (1.5, 3.0))
        self.assertEqual(rng.uniform.call_args_list[1].args, (0.0, 0.6))

        raw = np.full((4, 5), 20.0, dtype=np.float32)
        raw[0, :3] = [np.inf, np.nan, -np.inf]
        raw[1, 2] = 1.0
        raw[3, 2] = np.inf
        slopes = np.array([[3.0 / 8.0], [1.0 / 8.0]])
        arch = 0.6 * np.array([0.8, 0.4, 0.0, 0.4, 0.8]) ** 2
        with (
            patch.object(perception, "_get_depth", return_value=raw),
            patch.object(perception, "preprocess_depth", wraps=perception.preprocess_depth) as preprocess,
        ):
            for dz in (0.0, 2.0):
                with self.subTest(dz=dz):
                    perception.get_observation(dz_above_start=dz)
                    depth = preprocess.call_args.args[0]
                    expected = np.maximum(0.35, max(0.35, 1.5 - dz) - arch) / slopes
                    expected[1, 2] = 1.0
                    np.testing.assert_allclose(depth[:2], expected, rtol=1e-6)
                    np.testing.assert_array_equal(depth[2:], raw[2:])
                    np.testing.assert_array_equal(depth[:2], np.fliplr(depth[:2]))
        rng.random.assert_called_once_with()

    def test_live_climb_descent_and_unchanged_airsim_rpc_sequence(self):
        def fly(probability):
            cfg = training_config()
            cfg["perception"].update({
                "virtual_ceiling_prob": str(probability), "virtual_ceiling_height_min": "2.0",
                "virtual_ceiling_height_max": "2.0", "virtual_ceiling_arch_drop_max": "0.0",
            })
            env, client = mock_environment(cfg)
            initial, _ = env.reset(seed=42)
            clearance_at_start = env.min_distance_to_obstacles
            start_z_ned = env.dynamic_model.start_position[2]
            pose = client.simGetVehiclePose.return_value
            depths = [initial["image"][0, 45, 0]]
            distances = [clearance_at_start]
            for climb in (1.0, -1.0):
                pose.position.z_val = start_z_ned - climb
                obs, _, terminated, truncated, _ = env.step(ACTION)
                self.assertFalse(terminated or truncated)
                depths.append(obs["image"][0, 45, 0])
                distances.append(env.min_distance_to_obstacles)
            return depths, distances, [rpc[0] for rpc in client.method_calls]

        active_depth, active_distance, active_calls = fly(1.0)
        off_depth, off_distance, off_calls = fly(0.0)
        self.assertLess(active_depth[1], active_depth[0])
        self.assertLess(active_depth[0], active_depth[2])
        self.assertGreater(active_distance[0], 2.5)
        self.assertLess(active_distance[1], 2.5)
        self.assertEqual(off_depth, [255, 255, 255])
        self.assertEqual(off_distance, [25.0, 25.0, 25.0])
        self.assertIn("simGetVehiclePose", active_calls)
        self.assertEqual(active_calls, off_calls)

    def test_seeded_reset_and_evaluation_scenario_override(self):
        cfg = training_config()
        cfg.set("perception", "virtual_ceiling_prob", "1.0")
        env, _ = mock_environment(cfg)
        first, _ = env.reset(seed=8)
        different, _ = env.reset(seed=97)
        repeated, _ = env.reset(seed=8)
        self.assertFalse(np.array_equal(first["image"], different["image"]))
        np.testing.assert_array_equal(first["image"], repeated["image"])
        self.assertLess(first["image"][0, 45, 0], 255)

        env.set_evaluation_scenarios([{
            "scenario_id": "held-out", "start_position": [0.0, 0.0, 1.0],
            "goal_position": [20.0, 0.0, 1.0], "yaw_rad": 0.0,
        }])
        np.testing.assert_array_equal(env._get_obs()["image"], 255)
        evaluation_obs, info = env.reset(seed=8)
        self.assertEqual(info["scenario_id"], "held-out")
        np.testing.assert_array_equal(evaluation_obs["image"], 255)
        self.assertIsNone(env.perception._virtual_ceiling)
        self.assertEqual(env.np_random.bit_generator.state, np.random.default_rng(8).bit_generator.state)

    def test_virtual_ceiling_probability_zero_or_unsampled(self):
        cfg = training_config()
        cfg.remove_section("perception")
        perception = DepthStatePerception(cfg, client=None)
        rng = Mock()
        perception.reset(rng=rng)
        rng.random.assert_not_called()
        cfg.read_dict({"perception": {"virtual_ceiling_prob": "0.75"}})
        active_perception = DepthStatePerception(cfg, client=None)
        rng.random.return_value = 0.75
        active_perception.reset(rng=rng)
        rng.uniform.assert_not_called()
        raw = np.full((5, 9), 25.0, dtype=np.float32)
        for item in (perception, active_perception):
            with patch.object(item, "_get_depth", return_value=raw):
                np.testing.assert_array_equal(item.get_observation(dz_above_start=0.0)["image"], 255)


class CeilingEvaluationConfigTests(unittest.TestCase):
    def test_all_chapter_four_scenes_disable_training_ceiling(self):
        for name in ("t1_double_static", "t2_double_dynamic", "t3_single_scurve"):
            with self.subTest(scene=name):
                scene = CONFIGS / f"ch4_eval_{name}.ini"
                cfg = training_config()
                self.assertEqual(cfg.getfloat("perception", "virtual_ceiling_prob"), 0.75)
                apply_evaluation_overrides(cfg, scene_config=scene)
                self.assertEqual(cfg.getfloat("perception", "virtual_ceiling_prob"), 0.0)
                self.assertEqual(cfg.getfloat("environment", "max_vertical_difference"), 2.0)

    def test_vertical_scale_mismatch_including_legacy_fallback(self):
        scene = CONFIGS / "ch4_eval_t1_double_static.ini"
        for scale in ("3.0", None):
            with self.subTest(scale=scale):
                cfg = training_config()
                if scale is None:
                    cfg.remove_option("environment", "max_vertical_difference")
                else:
                    cfg.set("environment", "max_vertical_difference", scale)
                with self.assertRaisesRegex(ValueError, "environment.max_vertical_difference"):
                    apply_evaluation_overrides(cfg, scene_config=scene)

    def test_scene_perception_options_and_missing_probability(self):
        with tempfile.TemporaryDirectory() as temp:
            scene = Path(temp) / "scene.ini"
            for options, expected in (
                ("", 0.0),
                ("[perception]\ncamera_id = scene_camera\n", 0.0),
                ("[perception]\ncamera_id = scene_camera\nvirtual_ceiling_prob = 0.5\n", 0.5),
            ):
                with self.subTest(options=options):
                    scene.write_text("[environment]\n" + options, encoding="utf-8")
                    cfg = training_config()
                    apply_evaluation_overrides(cfg, scene_config=scene)
                    self.assertEqual(cfg.getfloat("perception", "virtual_ceiling_prob"), expected)
                    if options:
                        self.assertEqual(cfg.get("perception", "camera_id"), "scene_camera")
