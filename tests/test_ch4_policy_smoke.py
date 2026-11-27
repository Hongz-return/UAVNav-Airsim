"""CPU sequence-SAC integration, auxiliary supervision and checkpoint compatibility."""

from configparser import ConfigParser
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3.common.logger import configure

from gym_env.algorithms.mamba_sac import MambaSAC
from gym_env.envs.perception import get_perception
from gym_env.networks.mamba_policy import MambaSACPolicy
from gym_env.utils import build_mamba_policy_kwargs
from scripts.training.train_parallel import METHOD_CONFIGS, ROOT
from tools.runtime_support import build_runtime_env, create_model, load_model


METHODS = ('ma', 'ma_motion', 'ma_aux', 'ma_motion_aux')


def read_config(method):
    cfg = ConfigParser()
    cfg.read(ROOT / METHOD_CONFIGS[method], encoding='utf-8')
    return cfg


class FakeNavigationEnv(gym.Env):
    """Same configured image/state/action spaces, without constructing an RPC client."""

    def set_config(self, cfg):
        perception = get_perception(cfg.get('options', 'perception'))(cfg, client=None)
        perception.set_state_feature_length(6)
        self.observation_space = perception.get_observation_space()
        yaw = np.deg2rad(cfg.getfloat('multirotor', 'yaw_rate_max_deg'))
        self.action_space = spaces.Box(
            low=np.array([cfg.getfloat('multirotor', 'v_xy_min'), -2, -yaw], dtype=np.float32),
            high=np.array([cfg.getfloat('multirotor', 'v_xy_max'), 2, yaw], dtype=np.float32),
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {
            'image': np.zeros(self.observation_space['image'].shape, dtype=np.uint8),
            'state': np.zeros(6, dtype=np.float32),
        }, {}

    def step(self, action):
        raise AssertionError('Smoke tests write replay data directly; no rollout is needed')


def make_env(cfg):
    with patch('tools.runtime_support.gym.make', return_value=FakeNavigationEnv()):
        return build_runtime_env(cfg)


def make_model(cfg):
    model = MambaSAC(
        MambaSACPolicy, make_env(cfg), buffer_size=128, batch_size=4,
        seq_len=cfg.getint('mamba', 'seq_len'),
        policy_kwargs=build_mamba_policy_kwargs(cfg), device='cpu', seed=7,
    )
    model.set_logger(configure(folder=None, format_strings=[]))
    return model


def fill_replay(model):
    rng = np.random.default_rng(11)

    def observation():
        return {
            'image': rng.integers(0, 256, model.observation_space['image'].shape, dtype=np.uint8),
            'state': rng.random(model.observation_space['state'].shape, dtype=np.float32),
        }

    for truncated, length in ((False, 3), (True, 6)):
        obs = observation()
        for step in range(length):
            next_obs = observation()
            action = rng.uniform(model.action_space.low, model.action_space.high).astype(np.float32)
            last = step == length - 1
            model.replay_buffer.add(
                obs, next_obs, action, float(rng.normal()), done=last and not truncated,
                infos={'TimeLimit.truncated': last and truncated},
            )
            obs = next_obs


class Ch4PolicySmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def assert_finite_metrics(self, model):
        metrics = model.logger.name_to_value
        for key in ('train/actor_loss', 'train/critic_loss', 'train/ent_coef_loss'):
            self.assertIn(key, metrics)
        for key, value in metrics.items():
            with self.subTest(metric=key):
                self.assertTrue(np.isfinite(value), (key, value))

    def test_four_configs_train_and_roundtrip_through_runtime_loaders(self):
        baseline_keys = None
        for method in METHODS:
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                cfg = read_config(method)
                model = make_model(cfg)
                self.addCleanup(model.env.close)
                fill_replay(model)
                channels = 2 if 'motion' in method else 1
                self.assertEqual(model.observation_space['image'].shape, (channels, 60, 90))
                self.assertEqual(model.policy.shared_cnn.conv1[0].in_channels, channels)
                self.assertEqual(model.policy.actor.mamba._backend, 'mambapy')
                # Include terminal fallback frames, true truncation targets and padding in every update.
                with patch('gym_env.replay.sequence_buffer.np.random.choice', return_value=np.array([0, 2, 3, 8])):
                    batch = model.replay_buffer.sample(4)
                    self.assertEqual(batch['dones'][1, 0].item(), 1)
                    self.assertEqual(batch['dones'][3, 0].item(), 0)
                    self.assertEqual(batch['mask'][3, 1].item(), 0)
                    terminal = model.replay_buffer.episodes[0].get_sequence(2, 16)
                    truncated = model.replay_buffer.episodes[1].get_sequence(5, 16)
                    np.testing.assert_array_equal(terminal['next_observations']['image'][0], terminal['observations']['image'][0])
                    np.testing.assert_array_equal(truncated['next_observations']['image'][0], model.replay_buffer.episodes[1].final_obs['image'])
                    cnn_before = [p.detach().clone() for p in model.policy.shared_cnn.parameters()]
                    model.train(gradient_steps=2, batch_size=4)
                self.assert_finite_metrics(model)
                self.assertTrue(any(not torch.equal(before, after) for before, after in zip(cnn_before, model.policy.shared_cnn.parameters())))
                if model.policy.aux_weight > 0:
                    self.assertIn('train/aux_loss', model.logger.name_to_value)
                    auxiliary = list(model.policy.aux_predictor.parameters())
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in auxiliary))
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.policy.shared_cnn.parameters()))
                    actor_optimized = {id(p) for group in model.policy.actor.optimizer.param_groups for p in group['params']}
                    self.assertTrue(all(id(p) in actor_optimized for p in auxiliary))
                else:
                    self.assertFalse(hasattr(model.policy, 'aux_predictor'))
                    self.assertNotIn('train/aux_loss', model.logger.name_to_value)
                    if baseline_keys is None:
                        baseline_keys = set(model.policy.state_dict())
                    self.assertEqual(set(model.policy.state_dict()), baseline_keys)

                # Exercise the same history cache and config-aware load paths as evaluation/resume.
                obs = model.env.reset()
                expected_action, _ = model.predict(obs, deterministic=True, episode_start=np.array([True]))
                self.assertEqual(model.policy.actor._history_buffer['image'].shape, (1, 8, channels, 60, 90))
                model.num_timesteps = 17
                model._episode_num = 2
                path = str(Path(directory) / 'checkpoint')
                model.save_full_checkpoint(path)
                original_load = MambaSAC.load
                with patch('tools.runtime_support.MambaSAC.load', side_effect=lambda *args, **kwargs: original_load(*args, **kwargs, device='cpu')):
                    evaluated = load_model('SAC', path, model.env, cfg=cfg)
                    resumed = load_model('SAC', path, model.env, cfg=cfg, tensorboard_log=directory)
                self.assertEqual(evaluated.replay_buffer.size(), 0)
                self.assertEqual(resumed.replay_buffer.size(), 9)
                self.assertEqual(resumed.num_timesteps, 17)
                self.assertEqual(resumed._episode_num, 2)
                for loaded in (evaluated, resumed):
                    loaded.policy.set_training_mode(False)
                    for key, tensor in model.policy.state_dict().items():
                        self.assertTrue(torch.equal(tensor, loaded.policy.state_dict()[key]), key)
                    actual_action, _ = loaded.predict(obs, deterministic=True, episode_start=np.array([True]))
                    np.testing.assert_array_equal(actual_action, expected_action)
                resumed.set_logger(configure(folder=None, format_strings=[]))
                resumed.train(gradient_steps=1, batch_size=4)
                self.assert_finite_metrics(resumed)

    def test_zero_weight_preserves_weights_rng_and_context_interface(self):
        cfg = read_config('ma')
        legacy = make_model(cfg)
        self.addCleanup(legacy.env.close)
        rng_after_legacy = torch.get_rng_state().clone()
        cfg.read_dict({'aux': {'aux_weight': '0', 'aux_target': 'unused', 'aux_hidden': '1'}})
        disabled = make_model(cfg)
        self.addCleanup(disabled.env.close)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_after_legacy))
        self.assertFalse(hasattr(disabled.policy, 'aux_predictor'))
        for key, tensor in legacy.policy.state_dict().items():
            self.assertTrue(torch.equal(tensor, disabled.policy.state_dict()[key]), key)
        features = torch.randn(2, 4, legacy.policy.cnn_output_dim)
        states = torch.rand(2, 4, 6)
        actor = legacy.policy.actor
        torch.manual_seed(123)
        expected = actor.action_log_prob_sequence_from_features(features, states)
        torch.manual_seed(123)
        with patch.object(actor.mamba, 'forward', wraps=actor.mamba.forward) as forward:
            actual = actor.action_log_prob_sequence_from_features(features, states, return_context=True)
        self.assertEqual(forward.call_count, 1)
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        self.assertEqual(actual[2].shape, (2, 4, 128))

    def test_aux_targets_depth_only_masking_detach_and_empty_mask(self):
        for target, expected in (('next_depth', 0.01), ('depth_delta', 0.10)):
            with self.subTest(target=target):
                cfg = read_config('ma_motion_aux')
                cfg.set('aux', 'aux_target', target)
                model = make_model(cfg)
                self.addCleanup(model.env.close)
                for parameter in model.policy.aux_predictor.parameters():
                    torch.nn.init.zeros_(parameter)
                context = torch.randn(1, 4, 128, requires_grad=True)
                current = torch.full((1, 4, 2, 60, 90), 51.0, requires_grad=True)
                next_image = torch.tensor([102., 255., 153., 255.]).reshape(1, 4, 1, 1, 1).expand(1, 4, 2, 60, 90).clone()
                next_image[:, :, 1] = 0  # Motion channel must not enter the depth target.
                next_image.requires_grad_(True)
                normalized = torch.tensor([[[-1., 0., 1.]]]).expand(1, 4, 3)
                actions = model.policy.actor.scale_action(normalized).detach().requires_grad_(True)
                batch = {
                    'observations': {'image': current}, 'next_observations': {'image': next_image},
                    'actions': actions, 'mask': torch.tensor([[1., 1., 1., 0.]]),
                    'dones': torch.tensor([[0., 1., 0., 1.]]),
                }
                for n_stack in (1, 2):
                    with self.subTest(n_stack=n_stack):
                        model.policy.actor.n_stack = n_stack
                        if n_stack == 2:
                            batch['observations']['image'] = torch.stack([torch.zeros_like(current), current], dim=2)
                            batch['next_observations']['image'] = torch.stack([torch.zeros_like(next_image), next_image], dim=2)
                        with patch.object(model.policy.aux_predictor, 'forward', wraps=model.policy.aux_predictor.forward) as predict:
                            loss = model._compute_aux_loss(context, batch)
                        self.assertAlmostEqual(loss.item(), expected, places=6)
                        torch.testing.assert_close(predict.call_args.args[1], normalized)
                        self.assertFalse(predict.call_args.args[1].requires_grad)
                loss.backward()
                self.assertIsNone(current.grad)
                self.assertIsNone(next_image.grad)
                self.assertIsNone(actions.grad)
                batch['dones'].fill_(1)
                empty_loss = model._compute_aux_loss(context, batch)
                self.assertEqual(empty_loss.item(), 0.0)
                self.assertTrue(torch.isfinite(empty_loss))

    def test_aux_gradient_reaches_actor_and_cnn_and_critic_only_warns_once(self):
        cfg = read_config('ma_aux')
        model = make_model(cfg)
        self.addCleanup(model.env.close)
        fill_replay(model)
        batch = model.replay_buffer.sample(4)
        features = model.policy.extract_cnn_features(batch['observations'])
        _, _, context = model.policy.actor.action_log_prob_sequence_from_features(features, batch['observations']['state'], return_context=True)
        model._compute_aux_loss(context, batch).backward()
        for parameters in (model.policy.shared_cnn.parameters(), model.policy.actor.mamba.parameters()):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in parameters))

        cfg.set('mamba', 'cnn_update_mode', 'critic_only')
        with patch('builtins.print') as printed:
            frozen = make_model(cfg)
            self.addCleanup(frozen.env.close)
            fill_replay(frozen)
            frozen.train(gradient_steps=2, batch_size=4)
        warnings = [call for call in printed.call_args_list if '[Aux] Warning:' in str(call)]
        self.assertEqual(len(warnings), 1)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in frozen.policy.aux_predictor.parameters()))
        self.assertTrue(all(p.grad is None for p in frozen.policy.shared_cnn.parameters()))

    def test_zero_aux_loss_preserves_sac_initialization_and_updates(self):
        baseline = make_model(read_config('ma'))
        auxiliary = make_model(read_config('ma_aux'))
        for model in (baseline, auxiliary):
            self.addCleanup(model.env.close)
            fill_replay(model)
        for key, tensor in baseline.policy.state_dict().items():
            self.assertTrue(torch.equal(tensor, auxiliary.policy.state_dict()[key]), key)

        compute_aux = auxiliary._compute_aux_loss
        with patch.object(auxiliary, '_compute_aux_loss', side_effect=lambda context, batch: compute_aux(context, batch) * 0):
            for model in (baseline, auxiliary):
                torch.manual_seed(19)
                np.random.seed(19)
                model.train(gradient_steps=2, batch_size=4)
        for key, tensor in baseline.policy.state_dict().items():
            torch.testing.assert_close(tensor, auxiliary.policy.state_dict()[key], rtol=1e-6, atol=1e-8, msg=key)
        for tensor, actual in zip(baseline.policy.shared_cnn.parameters(), auxiliary.policy.shared_cnn.parameters()):
            self.assertTrue(torch.equal(tensor, actual))

    def test_actor_cnn_gradient_is_aux_only_and_clipped_once(self):
        model = make_model(read_config('ma_aux'))
        self.addCleanup(model.env.close)
        fill_replay(model)
        batch = model.replay_buffer.sample(4)
        auxiliary_ids = {id(p) for p in model.policy.aux_predictor.parameters()}
        original_clip = torch.nn.utils.clip_grad_norm_
        actor_updates = []

        def check_and_clip(parameters, max_norm):
            parameters = list(parameters)
            if auxiliary_ids.intersection(id(p) for p in parameters):
                actor_updates.append(True)
                self.assertEqual(len(parameters), len({id(p) for p in parameters}))
                # Independent aux-only forward at the same weights, before clipping or Adam.
                features = model.policy.extract_cnn_features(batch['context_observations'])
                contexts = []
                # Independent per-position reference, using the real short/full
                # history rather than a fresh unroll at the sampled start.
                for row in range(batch['mask'].shape[0]):
                    positions = []
                    for step in range(batch['mask'].shape[1]):
                        length = int(batch['history_lengths'][row, step])
                        end = model.policy.actor.inference_seq_len - 1 + step
                        _, _, value = model.policy.actor.forward_sequence_from_features(
                            features[row:row + 1, end - length + 1:end + 1],
                            batch['context_observations']['state'][row:row + 1, end - length + 1:end + 1],
                            return_context=True,
                        )
                        positions.append(value[0, -1])
                    contexts.append(torch.stack(positions))
                context = torch.stack(contexts)
                loss = model.policy.aux_weight * model._compute_aux_loss(context, batch)
                cnn_parameters = list(model.policy.shared_cnn.parameters())
                expected = torch.autograd.grad(loss, cnn_parameters)
                self.assertTrue(any(grad.abs().sum() > 0 for grad in expected))
                for parameter, gradient in zip(cnn_parameters, expected):
                    torch.testing.assert_close(parameter.grad, gradient)
            return original_clip(parameters, max_norm)

        with patch.object(model.replay_buffer, 'sample', return_value=batch), patch(
            'torch.nn.utils.clip_grad_norm_', side_effect=check_and_clip,
        ):
            model.train(gradient_steps=1, batch_size=4)
        self.assertEqual(len(actor_updates), 1)

    def test_create_model_uses_aux_config(self):
        for method in METHODS:
            with self.subTest(method=method), patch('tools.runtime_support.MambaSAC') as constructor:
                cfg = read_config(method)
                env = make_env(cfg)
                self.addCleanup(env.close)
                create_model('SAC', env, cfg, log_path=None)
                self.assertEqual(constructor.call_args.kwargs['policy_kwargs'], build_mamba_policy_kwargs(cfg))


if __name__ == '__main__':
    unittest.main()
