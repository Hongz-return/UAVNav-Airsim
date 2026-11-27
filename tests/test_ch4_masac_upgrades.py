"""CPU regressions for cross-scene navigation and independent MA-SAC rollouts."""

import ast
from configparser import ConfigParser
from copy import deepcopy
from functools import partial
import math
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cosysairsim as airsim
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage

from gym_env.algorithms.mamba_sac import MambaSAC
from gym_env.envs.airsim_env import AirsimGymEnv
from gym_env.envs.dynamics.multirotor_airsim import MultirotorDynamicsAirsim, to_quaternion
from gym_env.envs.tasks.base_task import BaseTask
from gym_env.networks.mamba_policy import MambaSACPolicy
from gym_env.replay.sequence_buffer import SequenceReplayBuffer
from scripts.evaluation import evaluate_single
from scripts.training import train_single
from tools.runtime_support import (
    CheckpointMonitor,
    MambaCheckpointCallback,
    OffPolicyCheckpointCallback,
    build_parallel_runtime_env,
)
from tools.training_logging import BestModelCallback, UAVLoggingCallback


CONFIG_PATH = Path(__file__).resolve().parents[1] / 'configs' / 'config_mamba.ini'
MOCK_PORT = 42450
MODEL_OPTIONS = dict(
    device='cpu', buffer_size=128, batch_size=2, seq_len=2, learning_starts=3, train_freq=3,
    policy_kwargs=dict(backend='mambapy', inference_seq_len=4, cnn_output_dim=8, mamba_hidden_dim=16,
                       mamba_d_state=4, mamba_d_conv=2, mamba_expand=1, actor_hidden=[16], critic_hidden=[16]),
)


def read_config():
    cfg = ConfigParser()
    cfg.read(CONFIG_PATH, encoding='utf-8')
    return cfg


class ParallelNavigationEnv(gym.Env):
    """Lane identity and step number are visible in observations after auto-reset."""

    def __init__(self, lane=0):
        super().__init__()
        self.lane = lane
        self.horizon = lane + 2
        self.observation_space = spaces.Dict({
            'image': spaces.Box(0, 255, (40, 44, 1), dtype=np.uint8),
            'state': spaces.Box(0.0, 100.0, (3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    def _observe(self):
        return {
            'image': np.full(self.observation_space['image'].shape, 16 * self.lane + self.steps, np.uint8),
            'state': np.array([self.lane, self.steps, self.marker], np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.marker = self.np_random.random()
        return self._observe(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action for lane {self.lane}: {action}')
        self.steps += 1
        ended = self.steps == self.horizon
        terminated = ended and self.lane % 2 == 0
        truncated = ended and self.lane % 2 == 1
        info = {'is_success': terminated, 'is_crash': False, 'step_num': self.steps}
        return self._observe(), float(self.lane + 1), terminated, truncated, info


def make_monitor(lane):
    return CheckpointMonitor(ParallelNavigationEnv(lane))


def make_mock_runtime(cfg):
    return make_monitor(cfg.getint('airsim', 'rpc_port') - MOCK_PORT)


class AirsimGeneralizationTests(unittest.TestCase):
    def make_env(self, cfg):
        env = AirsimGymEnv()
        fake = ParallelNavigationEnv()
        dynamics = Mock(action_space=fake.action_space, state_feature_length=3, dt=0.1, synchronous_steps=False)
        dynamics.get_position.return_value = [0.0, 0.0, 1.0]
        dynamics.get_velocity.return_value = dynamics.last_velocity_world = [0.0, 0.0, 0.0]
        dynamics.get_distance_to_goal_3d.return_value = 8.0
        perception = Mock(min_distance_to_obstacles=6.0)
        perception.get_observation_space.return_value = fake.observation_space
        perception.get_observation.return_value = {
            key: np.zeros(space.shape, space.dtype) for key, space in fake.observation_space.spaces.items()
        }
        task = Mock(last_reward_components={})
        task.is_done.return_value = False
        task.compute_reward.return_value = 1.0
        task.get_info.side_effect = dict
        with (
            patch('gym_env.envs.airsim_env.MultirotorDynamicsAirsim', return_value=dynamics),
            patch('gym_env.envs.airsim_env.get_perception', return_value=lambda *_: perception),
            patch('gym_env.envs.airsim_env.get_task', return_value=lambda *_: task),
        ):
            env.set_config(cfg)
        return env, dynamics

    def test_symmetric_yaw_bounds_and_single_random_draw(self):
        client = Mock()
        client.armDisarm.return_value = True
        client.simGetVehiclePose.return_value = airsim.Pose()
        client.moveByVelocityAsync.return_value.get.return_value = True
        dynamics = MultirotorDynamicsAirsim(read_config(), client=client)
        upper_draw = float(np.nextafter(1.0, 0.0))
        for angle, draw, expected in ((0.6, 0.0, -0.6), (0.6, 0.25, -0.3), (0.6, 0.5, 0.0),
                                      (0.6, upper_draw, 0.6), (0.0, 0.25, 0.0)):
            with self.subTest(angle=angle, draw=draw):
                dynamics.set_start([0, 0, -1], random_angle=angle)
                rng = Mock()
                rng.random.return_value = draw
                with patch('gym_env.envs.dynamics.multirotor_airsim.to_quaternion', wraps=to_quaternion) as convert:
                    dynamics.reset(rng=rng)
                yaw = convert.call_args.args[2]
                self.assertAlmostEqual(yaw, expected)
                if angle:
                    self.assertGreaterEqual(yaw, -angle)
                    self.assertLess(yaw, angle)
                rng.random.assert_called_once_with()

    def test_fixed_physical_distance_scale_and_legacy_route_scale(self):
        for scale, goal_distance, expected in ((None, 20, 0.25), (0, 10, 0.5), (20, 10, 0.25),
                                               (20, 30, 0.25), (2, 20, 1.0)):
            with self.subTest(scale=scale, goal_distance=goal_distance):
                cfg = read_config()
                if scale is not None:
                    cfg.set('environment', 'distance_norm_scale', str(scale))
                dynamics = MultirotorDynamicsAirsim(cfg, client=Mock())
                dynamics.get_position = Mock(return_value=[0, 0, 1])
                dynamics.get_velocity = Mock(return_value=[1, 0, 0])
                dynamics.get_attitude = Mock(return_value=[0, 0, 0])
                dynamics.goal_position = [5, 0, 1]
                dynamics.goal_distance = goal_distance
                self.assertAlmostEqual(dynamics._get_state_feature()[0], expected)

    def test_progress_modes_and_legacy_reward_types(self):
        env = SimpleNamespace(dynamic_model=SimpleNamespace(goal_distance=40.0),
                              action_space=ParallelNavigationEnv().action_space)
        for reward_type, mode, expected in (('reward_v2', None, 1.0), ('reward_v2', 'relative', 1.0),
                                            ('reward_v2', ' MeTeRs ', 40.0), ('reward_final', 'relative', 40.0)):
            with self.subTest(reward_type=reward_type, mode=mode):
                cfg = read_config()
                cfg.set('options', 'reward_type', reward_type)
                if mode is not None:
                    cfg.set('reward', 'progress_mode', mode)
                task = BaseTask(cfg, env)
                self.assertEqual(task._compute_progress_reward(2.0), expected)
                self.assertEqual(task._compute_progress_reward(-2.0), -expected)
        cfg.set('reward', 'progress_mode', 'invalid')
        with self.assertRaisesRegex(ValueError, 'progress_mode'):
            BaseTask(cfg, env)

    def test_route_sampling_reverses_and_evaluation_takes_priority(self):
        cfg = read_config()
        routes = [([1, 1, -1], [3, 1, -2]), ([2, 3, -5], [7, 4, -2])]
        cfg.set('environment', 'training_routes', repr([
            {'start': routes[0][0], 'goal': routes[0][1]}, list(routes[1]),
        ]))
        cfg.set('environment', 'random_route_reverse', 'True')
        cfg.set('environment', 'start_random_angle', '0.63')
        env, dynamics = self.make_env(cfg)
        sampled = set()
        for seed in range(12):
            rng = np.random.default_rng(seed)
            index, reversed_route = int(rng.integers(2)), bool(rng.integers(2))
            start, goal = routes[index][::-1] if reversed_route else routes[index]
            dynamics.reset_mock()
            _, info = env.reset(seed=seed)
            self.assertEqual(info, {'route_index': index, 'route_reversed': reversed_route, 'scenario_id': None})
            self.assertIs(type(info['route_index']), int)
            self.assertIs(type(info['route_reversed']), bool)
            self.assertEqual(dynamics.mock_calls[:3], [
                unittest.mock.call.set_start(start, random_angle=0.63, fixed_yaw=None),
                unittest.mock.call._set_goal_pose_single(goal),
                unittest.mock.call.reset(rng=env.np_random),
            ])
            sampled.add((index, reversed_route))
        self.assertEqual(len(sampled), 4)

        start, goal = [19, 2, -4], [21, 2, -4]
        env.set_evaluation_scenarios([{'scenario_id': 'held-out', 'start_position': start,
                                       'goal_position': goal, 'yaw_rad': 1.2}])
        dynamics.reset_mock()
        _, info = env.reset(seed=7)
        self.assertEqual(info, {'scenario_id': 'held-out', 'route_index': None, 'route_reversed': False})
        dynamics.set_start.assert_called_once_with(start, random_angle=0.0, fixed_yaw=1.2)
        dynamics._set_goal_pose_single.assert_called_once_with(goal)
        self.assertEqual(env.np_random.bit_generator.state, np.random.default_rng(7).bit_generator.state)

    def test_route_validation_and_default_no_reversal(self):
        cfg = read_config()
        env = AirsimGymEnv()
        env.dynamic_model = Mock()
        for routes in ('[]', '{}', '[[[0, 1], [2, 3, 4]]]', "[{'start': [0, 1, 2]}]", '[[[True, 0, -1], [1, 1, -1]]]', '['):
            with self.subTest(routes=routes):
                cfg.set('environment', 'training_routes', routes)
                with self.assertRaisesRegex(ValueError, 'training.route'):
                    env._load_environment_config(cfg)
        cfg.set('environment', 'training_routes', '[[[0, 0, -1], [1, 1, -1]]]')
        env, dynamics = self.make_env(cfg)
        _, info = env.reset(seed=3)
        self.assertEqual(info, {'scenario_id': None, 'route_index': 0, 'route_reversed': False})
        dynamics.set_start.assert_called_with([0, 0, -1], random_angle=0.0, fixed_yaw=None)
        cfg.set('environment', 'training_routes', '  ')
        env, _ = self.make_env(cfg)
        self.assertEqual(env.reset(seed=3)[1], {'scenario_id': None, 'route_index': None, 'route_reversed': False})

    def test_collision_rpc_targets_named_vehicle_and_supports_unnamed_vehicle(self):
        for attrs, args in (({'vehicle_name': 'SimpleFlight'}, ('SimpleFlight',)), ({'vehicle_name': ''}, ()), ({}, ())):
            with self.subTest(attrs=attrs):
                client = Mock()
                client.simGetCollisionInfo.return_value = SimpleNamespace(has_collided=True)
                env = SimpleNamespace(client=client, dynamic_model=SimpleNamespace(**attrs),
                                      action_space=ParallelNavigationEnv().action_space)
                self.assertTrue(BaseTask(read_config(), env).is_crashed())
                client.simGetCollisionInfo.assert_called_once_with(*args)

    def test_airsim_log_rpc_is_opt_in(self):
        for setting, expected in ((None, 0), ('False', 0), ('True', 2)):
            with self.subTest(setting=setting):
                cfg = read_config()
                if setting is not None:
                    cfg.set('options', 'print_airsim_log', setting)
                env, dynamics = self.make_env(cfg)
                env.reset(seed=3)
                env.step_num = 8
                env.step(np.zeros(3, np.float32))
                dynamics.client.simPrintLogMessage.assert_not_called()
                env.step(np.zeros(3, np.float32))
                self.assertEqual(dynamics.client.simPrintLogMessage.call_count, expected)

    def test_align_start_yaw_to_goal_centers_reversed_and_diagonal_routes(self):
        client = Mock()
        client.armDisarm.return_value = True
        client.simGetVehiclePose.return_value = airsim.Pose()
        client.moveByVelocityAsync.return_value.get.return_value = True
        cfg = read_config()
        cfg.set('environment', 'align_start_yaw_to_goal', 'True')
        dynamics = MultirotorDynamicsAirsim(cfg, client=client)

        for start, goal, angle, draw, fixed_yaw, expected in (
            ([162.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.45, 0.5, None, math.pi),
            ([162.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.45, 0.0, None, math.pi - 0.45),
            ([162.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.45, 0.75, None, math.pi + 0.225),
            ([0.0, 0.0, 1.0], [162.0, 30.0, 1.0], 0.45, 0.5, None, math.atan2(30.0, 162.0)),
            ([162.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.45, 0.0, 1.2, 1.2),
        ):
            with self.subTest(start=start, goal=goal, draw=draw, fixed_yaw=fixed_yaw):
                dynamics.set_start(start, random_angle=angle, fixed_yaw=fixed_yaw)
                dynamics._set_goal_pose_single(goal)
                rng = Mock()
                rng.random.return_value = draw
                with patch('gym_env.envs.dynamics.multirotor_airsim.to_quaternion', wraps=to_quaternion) as convert:
                    dynamics.reset(rng=rng)
                yaw = convert.call_args.args[2]
                wrapped_diff = abs((yaw - expected + math.pi) % (2.0 * math.pi) - math.pi)
                self.assertLess(wrapped_diff, 1e-6)
                if fixed_yaw is None:
                    rng.random.assert_called_once_with()
                else:
                    rng.random.assert_not_called()

    def test_ch4_scene_configs_consistency_and_cross_scene_eval_override(self):
        configs_dir = Path(__file__).resolve().parents[1] / 'configs'
        names = (
            'ch4_train_trainenv.ini',
            'ch4_eval_t1_double_static.ini',
            'ch4_eval_t2_double_dynamic.ini',
            'ch4_eval_t3_single_scurve.ini',
        )

        def assert_in_workspace(point, ws_x, ws_y, ws_z):
            x, y, z_ned = point
            z_enu = -z_ned
            self.assertGreater(x, ws_x[0])
            self.assertLess(x, ws_x[1])
            self.assertGreater(y, ws_y[0])
            self.assertLess(y, ws_y[1])
            self.assertGreater(z_enu, ws_z[0])
            self.assertLess(z_enu, ws_z[1])

        loaded = {}
        for name in names:
            path = configs_dir / name
            self.assertTrue(path.is_file(), f'Missing config: {path}')
            cfg = ConfigParser()
            cfg.read(path, encoding='utf-8')
            loaded[name] = (path, cfg)
            self.assertTrue(cfg.getboolean('options', 'use_mamba'))
            self.assertEqual(cfg.getfloat('environment', 'max_depth_meters'), 25.0)
            self.assertEqual(cfg.getint('environment', 'screen_height'), 60)
            self.assertEqual(cfg.getint('environment', 'screen_width'), 90)
            self.assertEqual(cfg.getfloat('environment', 'distance_norm_scale'), 200.0)
            self.assertEqual(cfg.get('reward', 'progress_mode'), 'meters')
            self.assertAlmostEqual(cfg.getfloat('reward', 'progress_scale'), 0.15)

            ws_x = ast.literal_eval(cfg.get('environment', 'work_space_x'))
            ws_y = ast.literal_eval(cfg.get('environment', 'work_space_y'))
            ws_z = ast.literal_eval(cfg.get('environment', 'work_space_z'))
            assert_in_workspace(ast.literal_eval(cfg.get('environment', 'start_position')), ws_x, ws_y, ws_z)
            assert_in_workspace(ast.literal_eval(cfg.get('environment', 'goal_position')), ws_x, ws_y, ws_z)

        _, train_cfg = loaded['ch4_train_trainenv.ini']
        routes = AirsimGymEnv._parse_training_routes(train_cfg.get('environment', 'training_routes'))
        self.assertEqual(len(routes), 9)
        t_ws_x = ast.literal_eval(train_cfg.get('environment', 'work_space_x'))
        t_ws_y = ast.literal_eval(train_cfg.get('environment', 'work_space_y'))
        t_ws_z = ast.literal_eval(train_cfg.get('environment', 'work_space_z'))
        for start, goal in routes:
            assert_in_workspace(start, t_ws_x, t_ws_y, t_ws_z)
            assert_in_workspace(goal, t_ws_x, t_ws_y, t_ws_z)

        t1_path, _ = loaded['ch4_eval_t1_double_static.ini']
        eval_cfg = deepcopy(train_cfg)
        evaluate_single.apply_evaluation_overrides(eval_cfg, airsim_port=41591, scene_config=str(t1_path))
        self.assertEqual(ast.literal_eval(eval_cfg.get('environment', 'start_position')), [0.0, 0.0, 0.0])
        self.assertEqual(ast.literal_eval(eval_cfg.get('environment', 'goal_position')), [160.0, 0.0, 0.0])
        self.assertEqual(eval_cfg.get('environment', 'max_episode_steps'), '1800')
        self.assertFalse(eval_cfg.has_option('environment', 'training_routes'))
        self.assertEqual(eval_cfg.get('airsim', 'rpc_port'), '41591')

        bad_cfg = deepcopy(train_cfg)
        bad_cfg.set('environment', 'distance_norm_scale', '150.0')
        with self.assertRaisesRegex(ValueError, 'distance_norm_scale'):
            evaluate_single.apply_evaluation_overrides(bad_cfg, scene_config=str(t1_path))


class ParallelMambaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def make_dummy(self, width):
        env = VecTransposeImage(DummyVecEnv([partial(make_monitor, lane) for lane in range(width)]))
        self.addCleanup(env.close)
        return env

    def test_batch_actor_matches_independent_history_with_interleaved_resets(self):
        model = MambaSAC(MambaSACPolicy, self.make_dummy(1), seed=4, **MODEL_OPTIONS)
        actor = model.policy.actor.eval()
        with torch.no_grad():
            actor.mu.weight.uniform_(-0.3, 0.3)
        references = [deepcopy(actor) for _ in range(4)]
        rng = np.random.default_rng(17)

        def reference_action(reference, obs):
            tensors = {key: torch.as_tensor(value).unsqueeze(0) for key, value in obs.items()}
            with torch.no_grad():
                return reference(tensors, deterministic=True)[0].numpy()

        for step, reset_rows in enumerate(([], [], [1, 3], [], [2], [0], [1, 2, 3], [0, 1, 2, 3], [], [])):
            obs = {
                'image': rng.integers(0, 256, (4, *model.observation_space['image'].shape), dtype=np.uint8),
                'state': rng.random((4, 3), dtype=np.float32),
            }
            mask = np.isin(np.arange(4), reset_rows)
            if step == 2:
                actor.reset_history(mask)
                self.assertTrue(all(torch.count_nonzero(value[mask]).item() == 0
                                    for value in actor._history_buffer.values()))
                actual, _ = model.predict(obs, deterministic=True)
            else:
                actual, _ = model.predict(obs, episode_start=mask, deterministic=True)
            expected = []
            for lane, reference in enumerate(references):
                if mask[lane]:
                    reference.reset_history()
                expected.append(reference_action(reference, {key: value[lane] for key, value in obs.items()}))
            np.testing.assert_allclose(actual, np.array(expected), atol=1e-5, rtol=0)
            np.testing.assert_array_equal(actor._history_lens, [reference._history_len for reference in references])
            self.assertEqual(actor._history_lens.dtype, np.int64)

        single_obs = {key: value[0] for key, value in obs.items()}
        single_action, _ = model.predict(single_obs, deterministic=True)
        references[0].reset_history()
        np.testing.assert_allclose(single_action, reference_action(references[0], single_obs), atol=1e-5, rtol=0)
        np.testing.assert_array_equal(actor._history_lens, [1])
        model.predict(obs, episode_start=np.array([False, True, False, False]), deterministic=True)
        np.testing.assert_array_equal(actor._history_lens, [1, 1, 1, 1])
        model.predict(obs, deterministic=True)
        model.predict(obs, episode_start=True, deterministic=True)
        np.testing.assert_array_equal(actor._history_lens, [1, 1, 1, 1])

    def test_replay_keeps_four_episodes_separate_and_restores_old_and_new_state(self):
        env = ParallelNavigationEnv()
        args = (64, env.observation_space, env.action_space)
        replay = SequenceReplayBuffer(*args, seq_len=3, device='cpu', n_envs=4)

        def add(lane, step, *, done=False, timeout=False):
            obs = {'image': np.zeros(env.observation_space['image'].shape, np.uint8),
                   'state': np.array([lane, step, 0.0], np.float32)}
            next_obs = {**obs, 'state': np.array([lane, step + 1, 0.0], np.float32)}
            replay.add(obs, next_obs, np.zeros(3, np.float32), float(lane), done,
                       {'TimeLimit.truncated': timeout}, env_idx=lane)

        for lane in range(4):
            add(lane, 0)
        add(2, 1, timeout=True)
        add(0, 1, done=True)
        add(3, 1)
        with self.assertRaises(IndexError):
            add(4, 0)
        with self.assertRaises(IndexError):
            replay._finish_episode(-1)
        self.assertEqual(replay.size(), 7)
        snapshot = pickle.loads(pickle.dumps(replay.get_state()))
        self.assertIn('current_episode', snapshot)
        snapshot.pop('total_transitions')
        restored = SequenceReplayBuffer(*args, seq_len=3, device='cpu', n_envs=4)
        restored.set_state(snapshot)
        self.assertEqual(restored.size(), 7)
        self.assertEqual([ep.length if ep else None for ep in restored._active_episodes], [None, 1, None, 2])
        timeout, termination = restored.episodes
        self.assertEqual(timeout.terminateds, [False, False])
        np.testing.assert_array_equal(timeout.get_sequence(1, 3)['next_observations']['state'][0], [2, 2, 0])
        self.assertEqual(termination.terminateds, [False, True])
        self.assertEqual([obs['state'][0] for obs in termination.observations], [0, 0])
        self.assertEqual(restored.valid_starts, replay.valid_starts)
        restored._finish_episode(3)
        self.assertIsNone(restored._active_episodes[3])
        self.assertIsNotNone(restored._active_episodes[1])
        restored.set_state({'episodes': [], 'current_episode': snapshot['active_episodes'][1]})
        self.assertEqual(restored.size(), 1)
        self.assertIs(restored.current_episode, restored._active_episodes[0])
        self.assertEqual([ep for ep in restored._active_episodes[1:] if ep is not None], [])
        restored.current_episode = None
        self.assertIsNone(restored._active_episodes[0])

    def check_rollout_and_checkpoint(self, env, vector_steps):
        width = env.num_envs
        model = MambaSAC(MambaSACPolicy, env, seed=19, **MODEL_OPTIONS)
        model.set_logger(configure(folder=None, format_strings=[]))
        logging = UAVLoggingCallback(log_freq=2)
        train_steps = []
        train = model.train

        def train_and_record(gradient_steps, batch_size):
            train_steps.append(model.num_timesteps)
            train(gradient_steps, batch_size)

        total = vector_steps * width
        with patch.object(model, 'train', side_effect=train_and_record):
            model.learn(total, callback=logging)
        self.assertEqual(model.num_timesteps, total)
        self.assertEqual(model.replay_buffer.size(), total)
        self.assertEqual(train_steps, [((step + width - 1) // width) * width for step in range(6, total + 1, 3)])
        completed = list(model.replay_buffer.episodes)
        self.assertEqual(model._episode_num, sum(vector_steps // (lane + 2) for lane in range(width)))
        self.assertEqual(len(logging.episode_rewards), model._episode_num)
        self.assertTrue(np.isfinite(model.logger.name_to_value['train/critic_loss']))
        for episode in completed:
            lane = int(episode.observations[0]['state'][0])
            self.assertTrue(all(obs['state'][0] == lane for obs in episode.observations))
            self.assertEqual(episode.rewards, [float(lane + 1)] * (lane + 2))
            self.assertEqual(episode.terminateds[-1], lane % 2 == 0)
            if lane % 2:
                self.assertEqual(episode.final_obs['state'][1], lane + 2)
        self.assertEqual([ep.length for ep in model.replay_buffer._active_episodes],
                         [vector_steps % (lane + 2) for lane in range(width)])
        continuation_states = model._last_obs['state'].copy()
        old_env_rngs = model._capture_random_state()['environments']

        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'parallel')
            model.save_full_checkpoint(path)
            evaluation = self.make_dummy(1)
            loaded = MambaSAC.load(path, evaluation, **MODEL_OPTIONS)
            self.assertEqual(loaded.num_timesteps, total)
            self.assertEqual(loaded.replay_buffer.size(), total)
            self.assertIsNone(loaded.replay_buffer.current_episode)
            self.assertEqual(len(loaded.replay_buffer.episodes), len(completed) + width)
            for lane, episode in enumerate(list(loaded.replay_buffer.episodes)[-width:]):
                self.assertFalse(episode.terminateds[-1])
                np.testing.assert_array_equal(episode.final_obs['state'], continuation_states[lane])

            obs = evaluation.reset()
            expected, _ = model.predict(obs, episode_start=True, deterministic=True)
            action, _ = loaded.predict(obs, episode_start=True, deterministic=True)
            self.assertEqual(action.shape, (1, 3))
            np.testing.assert_array_equal(action, expected)
            obs, _, dones, _ = evaluation.step(action)
            action, _ = loaded.predict(obs, episode_start=dones, deterministic=True)
            _, _, dones, _ = evaluation.step(action)
            self.assertTrue(dones[0])

            resumed = MambaSAC.load(path, env, seed=99, **MODEL_OPTIONS)
            resumed.set_logger(configure(folder=None, format_strings=[]))
            resumed.learn(width, reset_num_timesteps=False)
            self.assertEqual(resumed.num_timesteps, total + width)
            for lane, episode in enumerate(resumed.replay_buffer._active_episodes):
                generator = np.random.default_rng()
                generator.bit_generator.state = old_env_rngs[lane]
                self.assertEqual(episode.length, 1)
                np.testing.assert_array_equal(episode.observations[0]['state'],
                                              np.array([lane, 0, generator.random()], dtype=np.float32))

    def test_dummy_vecenv_four_lanes_train_checkpoint_resume_and_single_evaluation(self):
        self.check_rollout_and_checkpoint(self.make_dummy(4), vector_steps=7)

    def test_spawned_vecenv_two_lanes_train_checkpoint_resume_and_single_evaluation(self):
        cfg = read_config()
        cfg.read_dict({'airsim': {'rpc_port': 41451}})
        with patch('tools.runtime_support._build_monitored_env', new=make_mock_runtime):
            env = build_parallel_runtime_env(cfg, [MOCK_PORT, MOCK_PORT + 1], base_seed=11)
        self.addCleanup(env.close)
        self.assertIsInstance(env.venv, SubprocVecEnv)
        self.assertFalse(hasattr(env, 'envs'))
        self.assertEqual(cfg.getint('airsim', 'rpc_port'), 41451)
        initial_states = env.reset()['state']
        np.testing.assert_array_equal(initial_states[:, 0], [0, 1])
        np.testing.assert_array_equal(initial_states[:, 2],
                                      np.array([np.random.default_rng(seed).random() for seed in (11, 12)], np.float32))
        self.check_rollout_and_checkpoint(env, vector_steps=5)

    def test_checkpoint_weight_cadence_uses_crossed_environment_steps(self):
        for width, start, calls, visited, expected in (
            (1, 11, 2, [12, 13, 14, 15], [14]),
            (4, 8, 0, [12, 16, 20, 24], [12, 16, 20]),
        ):
            with self.subTest(width=width):
                callback = MambaCheckpointCallback(save_freq=5, save_path='models', verbose=0)
                callback.model = Mock(n_envs=width)
                callback.n_calls = calls
                callback.num_timesteps = start
                callback._on_training_start()
                for step in visited:
                    callback.num_timesteps = step
                    callback._on_step()
                paths = [Path(call.args[0]).name for call in callback.model.save.call_args_list]
                self.assertEqual(paths, [f'checkpoint_{step}_steps' for step in expected])

    def test_cli_uses_single_override_or_parallel_factory(self):
        for ports in ([42450], [42450, 42451]):
            with self.subTest(ports=ports):
                argv = ['train_single', '-c', str(CONFIG_PATH), '--seed', '17', '--airsim-ports', *map(str, ports)]
                with (
                    patch.object(sys, 'argv', argv),
                    patch.object(train_single, 'build_runtime_env', side_effect=RuntimeError('single factory')) as single,
                    patch.object(train_single, 'build_parallel_runtime_env',
                                 side_effect=RuntimeError('parallel factory')) as parallel,
                ):
                    with self.assertRaisesRegex(RuntimeError, 'single factory' if len(ports) == 1 else 'parallel factory'):
                        train_single.main()
                if len(ports) == 1:
                    self.assertEqual(single.call_args.args[0].getint('airsim', 'rpc_port'), ports[0])
                    parallel.assert_not_called()
                else:
                    self.assertEqual(parallel.call_args.args[1], ports)
                    self.assertEqual(parallel.call_args.kwargs['base_seed'], 17)
                    single.assert_not_called()

    def test_parallel_callbacks_use_environment_timesteps(self):
        best_model = Mock(n_envs=4, num_timesteps=0, logger=Mock())
        best_model.get_env.return_value = Mock(num_envs=4)
        best_callback = BestModelCallback(save_path='models', check_freq=5, min_episodes=1, patience=10, verbose=0)
        best_callback.init_callback(best_model)
        best_callback._on_training_start()

        best_steps = [
            (4, np.array([True, False, False, False]), [{'is_success': True, 'episode': {'r': 10.0}}, {}, {}, {}], True, 0),
            (8, np.array([False] * 4), [{}, {}, {}, {}], True, 0),
            (12, np.array([False] * 4), [{}, {}, {}, {}], True, 5),
            (16, np.array([False] * 4), [{}, {}, {}, {}], False, 10),
        ]
        for step, dones, infos, expected_continue, expected_stall in best_steps:
            best_model.num_timesteps = step
            best_callback.update_locals({'dones': dones, 'infos': infos})
            self.assertIs(best_callback.on_step(), expected_continue)
            self.assertEqual(best_callback.steps_since_improvement, expected_stall)
        self.assertEqual([call.args[0] for call in best_model.save.call_args_list], ['models/best_model'])

        log_model = Mock(n_envs=4, num_timesteps=0, logger=Mock())
        log_model.get_env.return_value = Mock(num_envs=4)
        logging_callback = UAVLoggingCallback(log_freq=100)
        logging_callback.init_callback(log_model)
        logging_callback._on_training_start()
        log_model.num_timesteps = 996
        logging_callback.update_locals({
            'dones': np.array([True, False, False, False]),
            'infos': [{'is_success': True, 'is_crash': False, 'episode': {'r': 5.0, 'l': 3}, 'step_num': 3}, {}, {}, {}],
        })
        self.assertTrue(logging_callback.on_step())
        self.assertNotIn(
            'core/success_rate_recent',
            [call.args[0] for call in log_model.logger.record.call_args_list],
        )
        log_model.num_timesteps = 1000
        logging_callback.update_locals({'dones': np.array([False] * 4), 'infos': [{}, {}, {}, {}]})
        self.assertTrue(logging_callback.on_step())
        self.assertIn(('core/success_rate_recent', 1.0), [call.args for call in log_model.logger.record.call_args_list])

        off_model = Mock(n_envs=4, num_timesteps=8)
        off_model.get_env.return_value = Mock(num_envs=4)
        off_callback = OffPolicyCheckpointCallback(save_freq=5, save_path='models', verbose=0)
        off_callback.init_callback(off_model)
        off_callback._on_training_start()
        for step in (12, 16, 20, 24):
            off_model.num_timesteps = step
            self.assertTrue(off_callback.on_step())
        saved_names = [Path(call.args[0]).name for call in off_model.save.call_args_list]
        self.assertEqual(saved_names, ['checkpoint_12_steps', 'checkpoint_16_steps', 'checkpoint_20_steps'])

    def test_train_single_closes_env_on_failure(self):
        fake_env = Mock()
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / 'missing_checkpoint.zip'
            argv = ['train_single', '-c', str(CONFIG_PATH), '--resume', str(missing)]
            with (
                patch.object(sys, 'argv', argv),
                patch.object(train_single, 'build_runtime_env', return_value=fake_env),
            ):
                with self.assertRaises(FileNotFoundError):
                    train_single.main()
        fake_env.close.assert_called_once_with()
