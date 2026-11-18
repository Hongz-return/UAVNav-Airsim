"""Regression checks for the isolated Chapter 4 experiment corrections."""
import math
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from gymnasium import spaces

from gym_env.envs.airsim_env import AirsimGymEnv
from gym_env.envs.tasks.base_task import BaseTask
from scripts.evaluation.evaluate_single import build_summary, evaluate
from gym_env.algorithms.mamba_sac import MambaSAC
from gym_env.networks.mamba_policy import MambaSACPolicy
from gym_env.utils import build_mamba_policy_kwargs
from stable_baselines3.common.logger import configure
from tests.test_ch4_policy_smoke import FakeNavigationEnv, fill_replay, make_model, read_config
from tools.runtime_support import MambaCheckpointCallback, build_runtime_env, load_model
from tools.training_logging import UAVLoggingCallback


class ExperimentCorrectionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_task_uses_real_multirotor_action_space(self):
        cfg = read_config('ma')
        dynamics = Mock()
        dynamics.client = Mock()
        dynamics.state_feature_length = 6
        dynamics.dt = .1
        dynamics.action_space = spaces.Box(
            low=np.array([.5, -2, -math.pi / 3], dtype=np.float32),
            high=np.array([4, 2, math.pi / 3], dtype=np.float32),
        )
        perception = Mock()
        with patch('gym_env.envs.airsim_env.MultirotorDynamicsAirsim', return_value=dynamics), \
             patch('gym_env.envs.airsim_env.get_perception', return_value=lambda *_: perception), \
             patch('gym_env.envs.airsim_env.get_task', return_value=BaseTask):
            env = AirsimGymEnv()
            env.set_config(cfg)
        np.testing.assert_allclose(env.task._action_delta_scale, [3.5, 4, 2 * math.pi / 3])

    def test_target_cnn_is_polyak_updated_once_per_training_step(self):
        model = make_model(read_config('ma'))
        fill_replay(model)
        target = [p.detach().clone() for p in model.policy.target_cnn.parameters()]
        model.train(gradient_steps=1, batch_size=4)
        for before, source, after in zip(target, model.policy.shared_cnn.parameters(),
                                         model.policy.target_cnn.parameters()):
            torch.testing.assert_close(after, before * (1 - model.tau) + source * model.tau)

    def test_reward_components_sum_to_actual_reward(self):
        cfg = read_config('ma')
        env = Mock()
        env.action_space = spaces.Box(
            low=np.array([.5, -2, -math.pi / 3], dtype=np.float32),
            high=np.array([4, 2, math.pi / 3], dtype=np.float32),
        )
        env.dynamic_model.goal_distance = 200.0
        env.previous_distance_from_des_point = 100.0
        env.min_distance_to_obstacles = 2.0
        task = BaseTask(cfg, env)
        task.get_distance_to_goal_3d = Mock(return_value=99.5)
        reward = task.compute_reward(False, np.array([1., 0., 0.]))
        self.assertAlmostEqual(sum(task.last_reward_components.values()), reward)
        self.assertAlmostEqual(task.last_reward_components['progress'], .05)
        task._cached_success = True
        reward = task.compute_reward(True, np.array([1., 0., 0.]))
        self.assertEqual(reward, 35.)
        self.assertEqual(task.last_reward_components['terminal'], 35.)
        self.assertAlmostEqual(sum(task.last_reward_components.values()), reward)

    def test_reward_components_capture_clipping_and_timeout_with_action_repeat(self):
        cfg = read_config('ma')
        stub = Mock()
        stub.action_space = spaces.Box(
            low=np.array([.5, -2, -math.pi / 3], dtype=np.float32),
            high=np.array([4, 2, math.pi / 3], dtype=np.float32),
        )
        stub.dynamic_model.goal_distance = 200.
        stub.previous_distance_from_des_point = 100.
        stub.min_distance_to_obstacles = 25.
        task = BaseTask(cfg, stub)
        task.get_distance_to_goal_3d = Mock(return_value=0.)
        reward = task.compute_reward(False, np.array([1., 0., 0.]))
        self.assertEqual(reward, 2.5)
        self.assertAlmostEqual(task.last_reward_components['clipping'], -7.485)
        self.assertAlmostEqual(sum(task.last_reward_components.values()), reward)

        env = AirsimGymEnv()
        env.action_repeat = 2
        env.step_num = env.total_step = 0
        env.max_episode_steps = 1
        env.cumulated_episode_reward = 0.
        env._episode_reward_components = {}
        env._execute_action = Mock(side_effect=lambda a: a)
        env._get_obs = Mock(return_value={})
        env.task = Mock()
        env.task.is_done.return_value = False
        def reward_for_step(done, action):
            env.task.last_reward_components = {'progress': .5, 'timeout': 0.}
            return .5
        env.task.compute_reward.side_effect = reward_for_step
        env.task.compute_timeout_reward.return_value = -10.
        env.task.get_info.return_value = {}
        env.dynamic_model = Mock(synchronous_steps=False)
        env.dynamic_model.get_position.return_value = [0., 0., 0.]
        env.dynamic_model.get_velocity.return_value = [0., 0., 0.]
        env.dynamic_model.last_velocity_world = [0., 0., 0.]
        env.dynamic_model.last_state_timestamp = 100
        env.min_distance_to_obstacles = 25.
        _, reward, terminated, truncated, info = env.step(np.array([1., 0., 0.]))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(reward, -9.)
        self.assertEqual(info['episode_reward_components']['timeout'], -10.)
        self.assertEqual(sum(info['episode_reward_components'].values()), reward)

    def test_full_checkpoint_preserves_replay_rng_and_closes_partial_episode(self):
        cfg = read_config('ma')
        model = make_model(cfg)
        fill_replay(model)
        image_shape = model.observation_space['image'].shape
        obs = {'image': np.zeros(image_shape, dtype=np.uint8), 'state': np.zeros(6, np.float32)}
        next_obs = {'image': np.full(image_shape, 23, dtype=np.uint8), 'state': np.ones(6, np.float32)}
        model.replay_buffer.add(obs, next_obs, np.array([1., 0., 0.]), 1., False, {})
        model._last_obs = {k: np.expand_dims(v, 0) for k, v in next_obs.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'full')
            model.save_full_checkpoint(path)
            checkpoint = torch.load(path + '.zip', map_location='cpu', weights_only=False)
            self.assertIn('random_state', checkpoint)
            self.assertIn('replay_buffer', checkpoint)
            loaded = load_model('SAC', path, model.env, cfg=cfg, tensorboard_log=directory)
            self.assertIsNone(loaded.replay_buffer.current_episode)
            np.testing.assert_array_equal(loaded.replay_buffer.episodes[-1].final_obs['image'], next_obs['image'])
            self.assertIsNotNone(loaded._pending_random_state)
            expected_state = loaded._pending_random_state
            loaded._restore_random_state(expected_state)
            expected = (random.random(), np.random.random(), torch.rand(1).item())
            loaded._restore_random_state(expected_state)
            actual = (random.random(), np.random.random(), torch.rand(1).item())
            np.testing.assert_allclose(actual, expected)

    def test_full_checkpoint_waits_for_episode_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            callback = MambaCheckpointCallback(10000, directory, verbose=0)
            callback.model = Mock()
            def write_checkpoint(path):
                Path(path + '.zip').write_bytes(b'full')
            callback.model.save_full_checkpoint.side_effect = write_checkpoint
            callback.n_calls = 20001
            callback.num_timesteps = 20001
            callback.locals = {'dones': np.array([False])}
            callback._on_step()
            callback.model.save_full_checkpoint.assert_not_called()
            callback.num_timesteps = 20010
            callback.locals = {'dones': np.array([True])}
            callback._on_step()
            self.assertEqual((Path(directory) / 'latest_full_checkpoint.zip').read_bytes(), b'full')

    def test_resume_applies_configured_learning_rate_and_preserves_optimizer_state(self):
        cfg = read_config('ma')
        model = make_model(cfg)
        fill_replay(model)
        model.train(gradient_steps=1, batch_size=4)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'full')
            model.save_full_checkpoint(path)
            checkpoint = torch.load(path + '.zip', map_location='cpu', weights_only=False)
            cfg.set('DRL', 'learning_rate', '0.0001')
            loaded = load_model('SAC', path, model.env, cfg=cfg, tensorboard_log=directory)
            self.assertAlmostEqual(loaded.lr_schedule(.5), .0001)
            for name, value in checkpoint['policy_state_dict'].items():
                torch.testing.assert_close(loaded.policy.state_dict()[name].cpu(), value)
            optimizers = [('actor_optimizer_state', loaded.policy.actor.optimizer),
                          ('critic_optimizer_state', loaded.policy.critic.optimizer),
                          ('ent_coef_optimizer_state', loaded.ent_coef_optimizer)]
            for field, optimizer in optimizers:
                actual = optimizer.state_dict()
                self.assertTrue(all(group['lr'] == .0001 for group in actual['param_groups']))
                for parameter, state in checkpoint[field]['state'].items():
                    for key, value in state.items():
                        restored = actual['state'][parameter][key]
                        if torch.is_tensor(restored):
                            restored = restored.cpu()
                        torch.testing.assert_close(restored, value)
            self.assertEqual(loaded.replay_buffer.size(), checkpoint['replay_buffer']['total_transitions'])
            self.assertIsNotNone(loaded._pending_random_state)
            loaded.set_logger(configure(folder=None, format_strings=[]))
            loaded.train(gradient_steps=1, batch_size=4)
            for _, optimizer in optimizers:
                self.assertTrue(all(group['lr'] == .0001 for group in optimizer.param_groups))

    def test_training_loop_writes_reward_components_and_full_checkpoint(self):
        cfg = read_config('ma')
        class ToyEnv(FakeNavigationEnv):
            def set_config(self, cfg):
                super().set_config(cfg)
                self.dynamic_model = Mock()
                self.dynamic_model.dt = .1
                self.dynamic_model.get_velocity.return_value = [0., 0., 0.]
                self.min_distance_to_obstacles = 4.
            def reset(self, *, seed=None, options=None):
                self.steps = 0
                return super().reset(seed=seed, options=options)
            def step(self, action):
                self.steps += 1
                obs = {'image': np.zeros(self.observation_space['image'].shape, np.uint8),
                       'state': np.zeros(6, np.float32)}
                done = self.steps == 4
                info = {'is_success': False, 'is_crash': done, 'step_num': self.steps,
                        'episode_reward_components': {'progress': 4., 'terminal': -1.} if done else {}}
                return obs, 1., done, False, info
        toy = ToyEnv()
        with patch('tools.runtime_support.gym.make', return_value=toy):
            env = build_runtime_env(cfg)
        model = MambaSAC(MambaSACPolicy, env, buffer_size=128, batch_size=2,
                         seq_len=4, learning_starts=2, train_freq=1, gradient_steps=1,
                         policy_kwargs=build_mamba_policy_kwargs(cfg), device='cpu', seed=7)
        model.set_logger(configure(folder=None, format_strings=[]))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = MambaCheckpointCallback(10000, directory, verbose=0)
            checkpoint.full_save_freq = 5
            model.learn(12, callback=[checkpoint, UAVLoggingCallback(log_freq=100)])
            payload = torch.load(Path(directory) / 'latest_full_checkpoint.zip',
                                 map_location='cpu', weights_only=False)
            self.assertEqual(payload['num_timesteps'], 12)
            self.assertEqual(payload['replay_buffer']['total_transitions'], 12)
            self.assertIsNone(payload['replay_buffer']['current_episode'])
            self.assertIn('random_state', payload)
            self.assertEqual(model.logger.name_to_value['reward_component_ma20/progress'], 4.)

    def test_evaluation_uses_terminal_wall_time_and_world_velocity(self):
        class Base:
            action_repeat = 1
            current_scenario = {}
            def __init__(self):
                self.dynamic_model = Mock(dt=.1)
                self.dynamic_model.get_position.return_value = [0, 0, 0]
                self.dynamic_model.goal_position = [2, 0, 0]
        class FakeVec:
            action_space = spaces.Box(low=-1., high=1., shape=(3,), dtype=np.float32)
            def __init__(self):
                self.envs = [Base()]
                self.episode = 0
            def reset(self):
                self.envs[0].episode_start_wall_time = 10.
                self.envs[0].episode_start_state_timestamp = 100
                return np.zeros((1, 1))
            def step(self, action):
                base = self.envs[0]
                start = base.episode_start_wall_time
                timestamp = base.episode_start_state_timestamp
                info = {'actual_action': action[0], 'velocity': [1., 3., .5],
                        'velocity_world': [1., 2., 3.], 'wall_monotonic_time': start + 2.,
                        'state_timestamp': timestamp + 200, 'terminal_position': [2, 0, 0],
                        'is_success': True, 'is_crash': False, 'step_num': 1,
                        'min_distance_to_obstacles': 3.}
                self.episode += 1
                base.episode_start_wall_time = start + 100.
                base.episode_start_state_timestamp = timestamp + 1000
                return np.zeros((1, 1)), np.array([1.]), np.array([True]), [info]
        model = Mock()
        model.predict.return_value = (np.zeros((1, 3)), None)
        results = evaluate(model, FakeVec(), 2)
        self.assertEqual(results['episode_times_sec'], [2., 2.])
        self.assertEqual(results['nominal_control_times_sec'], [.1, .1])
        self.assertEqual(results['state_timestamp_deltas'], [200, 200])
        np.testing.assert_allclose(results['max_speeds_mps'], [math.sqrt(14)] * 2)
        summary = build_summary(results)
        self.assertEqual(summary['avg_time_sec'], 2.)
        self.assertEqual(summary['avg_nominal_control_time_sec'], .1)


if __name__ == '__main__':
    unittest.main()
