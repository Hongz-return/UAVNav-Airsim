"""Regressions at the replay, resume and simulator boundaries."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cosysairsim as airsim
import numpy as np
import torch
from stable_baselines3.common.logger import configure

from gym_env.algorithms.mamba_sac import MambaSAC
from gym_env.envs.airsim_env import AirsimGymEnv
from gym_env.envs.dynamics.multirotor_airsim import MultirotorDynamicsAirsim
from gym_env.networks.mamba_policy import MambaSACPolicy
from gym_env.utils import build_mamba_policy_kwargs
from tests.test_ch4_policy_smoke import FakeNavigationEnv, fill_replay, make_model, read_config
from tools.runtime_support import build_runtime_env


class TrainingBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_compact_replay_and_auxiliary_targets_match_full_batch(self):
        for target in ('next_depth', 'depth_delta', 'terminal_risk'):
            with self.subTest(target=target):
                cfg = read_config('ma_motion_aux')
                cfg.set('aux', 'aux_target', target)
                cfg.set('mamba', 'n_stack', '2')
                model = make_model(cfg)
                self.addCleanup(model.env.close)
                fill_replay(model)
                # Include the true terminal and the final bootstrap frame of a timeout.
                options = {'failure_horizon': 2} if target == 'terminal_risk' else {}
                with patch('numpy.random.choice', return_value=np.array([0, 2, 3, 8])):
                    full = model.replay_buffer.sample(4, **options)
                    compact = model.replay_buffer.sample(4, compact=True, **options)
                self.assertNotIn('observations', compact)
                self.assertNotIn('next_observations', compact)
                self.assertEqual(compact['context_observations']['image'].dtype, torch.uint8)
                for key, value in compact.items():
                    if key == 'context_observations':
                        for field, tensor in value.items():
                            self.assertTrue(torch.equal(tensor.float(), full[key][field]))
                    else:
                        self.assertTrue(torch.equal(value, full[key]))
                contexts = torch.randn(4, model.seq_len, model.policy.mamba_hidden_dim)
                torch.testing.assert_close(model._compute_aux_loss(contexts, compact),
                                           model._compute_aux_loss(contexts, full), rtol=0, atol=0)
                with patch.object(model, '_window_indices', wraps=model._window_indices) as builds:
                    model.train(1, 4)
                self.assertEqual(builds.call_count, 2)

    def test_resumed_boundary_recreates_next_scenario_and_action(self):
        cfg = read_config('ma')

        class ScenarioEnv(FakeNavigationEnv):
            def __init__(self):
                self.scenarios = []
                self.actions = []

            def reset(self, *, seed=None, options=None):
                obs, info = super().reset(seed=seed, options=options)
                self.scenario = self.np_random.random(6, dtype=np.float32)
                self.scenarios.append(self.scenario.copy())
                obs['state'] = self.scenario
                self.observation = obs
                return obs, info

            def step(self, action):
                self.actions.append(action.copy())
                return self.observation, 1., True, False, {}

        def environment():
            toy = ScenarioEnv()
            with patch('tools.runtime_support.gym.make', return_value=toy):
                env = build_runtime_env(cfg)
            self.addCleanup(env.close)
            return toy, env

        options = dict(policy_kwargs=build_mamba_policy_kwargs(cfg), device='cpu', learning_starts=0,
                       batch_size=4, buffer_size=64)
        live, env = environment()
        model = MambaSAC(MambaSACPolicy, env, seed=7, **options)
        model.set_logger(configure(folder=None, format_strings=[]))
        model.learn(1)
        self.assertEqual(len(live.scenarios), 2)  # initial reset + one VecEnv auto-reset
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'boundary')
            model.save_full_checkpoint(path)
            model.learn(1, reset_num_timesteps=False)
            restored, fresh_env = environment()
            resumed = MambaSAC.load(path, fresh_env, seed=91, **options)
            resumed.set_logger(configure(folder=None, format_strings=[]))
            resumed.learn(1, reset_num_timesteps=False)
        np.testing.assert_array_equal(restored.scenarios, live.scenarios[1:])
        np.testing.assert_array_equal(restored.actions[0], live.actions[1])

    def test_synchronous_substep_reuses_pose_and_state(self):
        def run(synchronous):
            cfg = read_config('ma')
            cfg.read_dict({'airsim': {'synchronous_steps': str(synchronous)}})
            client = Mock()
            client.getSettingsString.return_value = '{"ClockType":"SteppableClock"}'
            dynamics = MultirotorDynamicsAirsim(cfg, client=client)
            dynamics.goal_position = [8., 1., 4.]
            dynamics.goal_distance = 10.
            env = AirsimGymEnv()
            env.dynamic_model = dynamics
            env.step_num = env.total_step = 0
            env.action_repeat = 2
            env.max_episode_steps = 10
            env.min_distance_to_obstacles = 5.
            env.cumulated_episode_reward = 0.
            env._episode_reward_components = {}
            env.task = Mock()
            env.task.is_done.return_value = False
            env.task.get_info.return_value = {}
            env.task.compute_reward.return_value = 1.
            env.task.last_reward_components = {}
            env._get_obs = dynamics._get_state_feature

            def advance(action):
                step = env.total_step + 1
                pose = airsim.Pose(airsim.Vector3r(step, 1, -2), airsim.Quaternionr())
                client.simGetVehiclePose.return_value = pose
                client.getMultirotorState.return_value = SimpleNamespace(
                    timestamp=step * 100,
                    kinematics_estimated=SimpleNamespace(linear_velocity=airsim.Vector3r(1, 2, -3),
                                                         angular_velocity=airsim.Vector3r(0, 0, .25)))
                return action

            env._execute_action = advance
            observation, reward, _, _, info = env.step(np.zeros(3, np.float32))
            return observation, reward, info, (client.simGetVehiclePose.call_count, client.getMultirotorState.call_count)

        reference, old_reward, old_info, asynchronous_calls = run(False)
        actual, reward, info, synchronous_calls = run(True)
        np.testing.assert_array_equal(reference, actual)
        self.assertEqual(reward, old_reward)
        for key in ('position', 'velocity', 'velocity_world', 'state_timestamp'):
            self.assertEqual(info[key], old_info[key])
        self.assertEqual(info['position'], [2, 1, 2])
        self.assertEqual(info['velocity_world'], [1, 2, 3])
        self.assertEqual(info['state_timestamp'], 200)
        self.assertEqual(synchronous_calls, (2, 2))
        self.assertEqual(asynchronous_calls, (12, 3))

    def test_continuing_and_restarting_a_partial_rollout(self):
        model = make_model(read_config('ma'))
        self.addCleanup(model.env.close)
        base = model.env.envs[0].unwrapped
        obs = {key: np.full(space.shape, .5, space.dtype)
               for key, space in base.observation_space.spaces.items()}
        with patch.object(base, 'step', return_value=(obs, 2., False, False, {})):
            model.learn(1)
            model.learn(1, reset_num_timesteps=False)
            self.assertEqual(model._rollout_episode_progress, (4., 2))
            self.assertEqual(model.replay_buffer.current_episode.length, 2)
            model.learn(1, reset_num_timesteps=True)
        self.assertEqual(model._rollout_episode_progress, (2., 1))
        finished = model.replay_buffer.episodes[-1]
        self.assertEqual(finished.length, 2)
        np.testing.assert_array_equal(finished.final_obs['state'], obs['state'])
        self.assertFalse(finished.terminateds[-1])
