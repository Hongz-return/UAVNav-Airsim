"""Motion perception and episode resets without an AirSim connection."""

from configparser import ConfigParser
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from gym_env.envs.airsim_env import AirsimGymEnv
from gym_env.envs.perception import get_perception
from gym_env.envs.perception.base_perception import BasePerception
from gym_env.envs.perception.depth_motion_perception import DepthMotionPerception
from gym_env.envs.perception.depth_state_perception import DepthStatePerception
from scripts.training.train_parallel import METHOD_CONFIGS, ROOT


def make_config(height=3, width=5):
    cfg = ConfigParser()
    cfg.read_dict({'environment': {
        'screen_height': str(height), 'screen_width': str(width), 'max_depth_meters': '25',
    }})
    return cfg


class DepthMotionPerceptionTest(unittest.TestCase):
    def make_perception(self, cfg=None, cls=DepthMotionPerception):
        perception = cls(cfg if cfg is not None else make_config(), client=None)
        perception.set_state_feature_length(6)
        return perception

    def test_registry_shape_first_frame_and_state(self):
        self.assertIs(get_perception('depth_motion'), DepthMotionPerception)
        perception = self.make_perception()
        self.assertEqual(perception.motion_diff_max, 1.0)  # No [perception] section.
        state = np.array([-1, 0, 0.5, 1, 2, 0.25, 10])
        with patch.object(perception, '_get_depth', return_value=np.full((3, 5), 4, dtype=np.float32)):
            obs = perception.get_observation(state)
        self.assertEqual(obs['image'].shape, (3, 5, 2))
        self.assertEqual(obs['image'].dtype, np.uint8)
        np.testing.assert_array_equal(obs['image'][..., 1], 128)
        np.testing.assert_array_equal(obs['state'], [0, 0, 0.5, 1, 1, 0.25])
        self.assertEqual(obs['state'].dtype, np.float32)
        self.assertTrue(perception.get_observation_space().contains(obs))

    def test_known_meter_differences_and_clipping(self):
        cfg = make_config()
        cfg.read_dict({'perception': {'motion_diff_max': '2.0'}})
        perception = self.make_perception(cfg)
        first = np.full((3, 5), 5, dtype=np.float32)
        second = first + np.array([-3, -1, 0, 1, 3], dtype=np.float32)
        with patch.object(perception, '_get_depth', side_effect=[first, second]):
            perception.get_observation()
            obs = perception.get_observation()
        np.testing.assert_array_equal(obs['image'][..., 1], np.tile([1, 64, 128, 192, 255], (3, 1)))

    def test_reset_clears_previous_episode(self):
        perception = self.make_perception()
        with patch.object(perception, '_get_depth', side_effect=[
            np.full((3, 5), 2, dtype=np.float32), np.full((3, 5), 8, dtype=np.float32),
            np.full((3, 5), 4, dtype=np.float32),
        ]):
            perception.get_observation()
            self.assertTrue(np.all(perception.get_observation()['image'][..., 1] == 255))
            perception.reset()
            np.testing.assert_array_equal(perception.get_observation()['image'][..., 1], 128)

    def test_camera_warms_once_before_depth_and_failures_are_reported(self):
        client = Mock()
        scene = SimpleNamespace(width=1, height=1, image_data_uint8=b'image')
        depth = SimpleNamespace(width=5, height=3, image_data_float=[4.0] * 15)
        client.simGetImages.side_effect = [[scene, scene], [depth], [depth]]
        perception = DepthStatePerception(make_config(), client)
        with patch('gym_env.envs.perception.depth_state_perception.time.sleep'):
            np.testing.assert_array_equal(perception._get_depth(), np.full((3, 5), 4))
            np.testing.assert_array_equal(perception._get_depth(), np.full((3, 5), 4))
        self.assertEqual(client.simGetImages.call_count, 3)
        failed_client = Mock()
        failed_client.simGetImages.return_value = []
        failed = DepthStatePerception(make_config(), failed_client)
        with self.assertRaisesRegex(RuntimeError, 'warm up'):
            failed._get_depth()
        failed_client.simGetImages.side_effect = [[scene, scene]] + [[]] * 10
        with patch('gym_env.envs.perception.depth_state_perception.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'depth image'):
                failed._get_depth()

    def test_depth_channel_and_obstacle_distance_match_depth_state(self):
        legacy = self.make_perception(cls=DepthStatePerception)
        motion = self.make_perception()
        raw = np.random.default_rng(3).uniform(-2, 30, (7, 9)).astype(np.float32)
        raw[0, :3] = [np.nan, np.inf, -np.inf]
        state = np.linspace(-0.2, 1.2, 6, dtype=np.float32)
        for frame in (raw, np.full((7, 9), np.nan, dtype=np.float32), np.zeros((7, 9), dtype=np.float32)):
            with self.subTest(frame=frame[0, 0]), patch.object(legacy, '_get_depth', return_value=frame), patch.object(motion, '_get_depth', return_value=frame):
                expected = legacy.get_observation(state)
                actual = motion.get_observation(state)
                np.testing.assert_array_equal(actual['image'][..., :1], expected['image'])
                np.testing.assert_array_equal(actual['state'], expected['state'])
                self.assertEqual(motion.min_distance_to_obstacles, legacy.min_distance_to_obstacles)

    def test_motion_cleans_and_clips_before_resizing(self):
        perception = self.make_perception(make_config(height=2, width=2))
        first = np.array([[np.nan, -np.inf], [-5, 30]], dtype=np.float32)
        second = np.array([[np.inf, np.nan], [-3, 28]], dtype=np.float32)
        with patch.object(perception, '_get_depth', side_effect=[first, second]):
            perception.get_observation()
            np.testing.assert_array_equal(perception.get_observation()['image'][..., 1], 128)
        # Raw 2x2 -> 3x5 resizing must not spread NaN/Inf into the motion cache.
        perception = self.make_perception()
        with patch.object(perception, '_get_depth', side_effect=[first, np.full((2, 2), 5, dtype=np.float32)]):
            perception.get_observation()
            expected_previous = perception.resize_image(np.array([[25, 25], [0, 25]], dtype=np.float32))
            expected = np.rint(np.clip(5 - expected_previous, -1, 1) * 127 + 128).astype(np.uint8)
            np.testing.assert_array_equal(perception.get_observation()['image'][..., 1], expected)

    def test_depth_state_reset_is_a_no_op(self):
        perception = self.make_perception(cls=DepthStatePerception)
        self.assertIsNone(BasePerception.reset(perception))


class EnvironmentResetTest(unittest.TestCase):
    def test_motion_config_receives_dynamics_state_length(self):
        cfg = ConfigParser()
        cfg.read(ROOT / METHOD_CONFIGS['ma_motion'], encoding='utf-8')
        dynamics = Mock(state_feature_length=6, dt=0.1)
        env = AirsimGymEnv()
        with patch('gym_env.envs.airsim_env.MultirotorDynamicsAirsim', return_value=dynamics), patch('gym_env.envs.airsim_env.get_task', return_value=lambda cfg, env: None):
            env.set_config(cfg)
        self.assertEqual(env.observation_space['state'].shape, (6,))
        self.assertEqual(env.observation_space['image'].shape, (60, 90, 2))

    def test_perception_reset_precedes_observation_in_training_and_evaluation(self):
        for evaluation in (False, True):
            with self.subTest(evaluation=evaluation):
                env = AirsimGymEnv()
                calls = []
                env.dynamic_model = SimpleNamespace(
                    reset=Mock(), get_distance_to_goal_3d=lambda: 1.0,
                    set_start=Mock(), _set_goal_pose_single=Mock(),
                )
                env.perception = SimpleNamespace(reset=Mock(side_effect=lambda **_: calls.append('reset')))
                env.task = None
                env.episode_num = 0
                env._get_obs = lambda: calls.append('observation') or {'test': True}
                if evaluation:
                    env.set_evaluation_scenarios([{
                        'scenario_id': 'test', 'start_position': [0, 0, -1.5],
                        'goal_position': [10, 0, -1.5], 'yaw_rad': 0,
                    }])
                obs, info = env.reset(seed=0)
                self.assertEqual(calls, ['reset', 'observation'])
                env.perception.reset.assert_called_once_with(rng=env.np_random, disable_augmentation=evaluation)
                self.assertEqual(obs, {'test': True})
                self.assertEqual(info['scenario_id'], 'test' if evaluation else None)


if __name__ == '__main__':
    unittest.main()
