"""The registered loss control changes only the per-position error function."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from stable_baselines3.common.logger import configure

from gym_env.algorithms.mamba_sac import MambaSAC
from gym_env.utils import build_mamba_policy_kwargs
from tests.test_ch4_policy_smoke import fill_replay, make_env, make_model, read_config
from tools.runtime_support import create_model, load_model


class CriticLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_huber_values_and_gradients_are_bit_identical(self):
        model = make_model(read_config('ma'))
        self.addCleanup(model.env.close)
        self.assertEqual(model.critic_loss, 'huber')
        error = torch.tensor([-35., -1.001, -1., -.5, 0., .5, 1., 1.001, 35.], requires_grad=True)
        legacy_error = error.detach().clone().requires_grad_(True)
        actual = model._critic_position_error(error)
        legacy = torch.where(legacy_error.abs() <= 1.0, 0.5 * legacy_error.pow(2), legacy_error.abs() - 0.5)
        self.assertTrue(torch.equal(actual, legacy))
        actual.sum().backward()
        legacy.sum().backward()
        self.assertTrue(torch.equal(error.grad, legacy_error.grad))

    def test_default_full_update_matches_legacy_huber_bit_for_bit(self):
        cfg = read_config('ma')
        current, legacy = make_model(cfg), make_model(cfg)
        for model in (current, legacy):
            self.addCleanup(model.env.close)
            fill_replay(model)
        state = current._capture_random_state()
        current.train(1, 4)
        legacy._restore_random_state(state)
        with patch.object(legacy, '_critic_position_error', side_effect=lambda error:
                          torch.where(error.abs() <= 1.0, 0.5 * error.pow(2), error.abs() - 0.5)):
            legacy.train(1, 4)
        for key, value in current.policy.state_dict().items():
            self.assertTrue(torch.equal(value, legacy.policy.state_dict()[key]), key)
        self.assertTrue(torch.equal(current.log_ent_coef, legacy.log_ent_coef))
        self.assertEqual(current.logger.name_to_value, legacy.logger.name_to_value)

    def test_mse_training_uses_half_square_and_unchanged_masked_reduction(self):
        cfg = read_config('ma')
        cfg.set('DRL', 'critic_loss', 'mse')
        env = make_env(cfg)
        self.addCleanup(env.close)
        model = create_model('SAC', env, cfg, log_path=None, seed=7)
        model.set_logger(configure(folder=None, format_strings=[]))
        self.assertEqual(model.critic_loss, 'mse')
        fill_replay(model)
        batch = model.replay_buffer.sample(4, history_len=8)
        batch['mask'][0] = 0
        captured = []

        def error_function(error):
            captured.append(error.detach().clone())
            result = model_error(error)
            self.assertTrue(torch.equal(result, .5 * error.pow(2)))
            return result

        model_error = model._critic_position_error
        with patch.object(model.replay_buffer, 'sample', return_value=batch), \
                patch.object(model, '_critic_position_error', side_effect=error_function):
            model.train(1, 4)
        mask = batch['mask']
        expected = ((mask * (.5 * captured[0].pow(2))).sum()
                    + (mask * (.5 * captured[1].pow(2))).sum()) / (2 * mask.sum().clamp(min=1.0))
        self.assertEqual(model.logger.name_to_value['train/critic_loss'], expected.item())
        for parameter in model.policy.parameters():
            self.assertTrue(torch.isfinite(parameter).all())
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_mse_gradients_and_all_masked_batch(self):
        model = make_model(read_config('ma'))
        self.addCleanup(model.env.close)
        model.critic_loss = 'mse'
        errors = torch.tensor([[35., -25., .5], [-35., 1., 0.]], requires_grad=True)
        mask = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
        loss = (mask * model._critic_position_error(errors)).sum() / mask.sum().clamp(min=1.)
        loss.backward()
        self.assertTrue(torch.equal(errors.grad, mask * errors.detach() / mask.sum()))
        self.assertTrue(torch.isfinite(errors.grad).all())
        empty_loss = (torch.zeros_like(mask) * model._critic_position_error(errors)).sum()
        self.assertEqual(empty_loss.item(), 0.)

    def test_checkpoints_preserve_mse_and_old_checkpoints_default_to_huber(self):
        cfg = read_config('ma')
        model = make_model(cfg)
        self.addCleanup(model.env.close)
        model.critic_loss = 'mse'
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'model')
            model.save(path)
            restored = MambaSAC.load(path, env=model.env, load_buffer=False,
                                     policy_kwargs=build_mamba_policy_kwargs(cfg), device='cpu')
            self.assertEqual(restored.critic_loss, 'mse')
            cfg.set('DRL', 'critic_loss', 'mse')
            self.assertEqual(load_model('SAC', path + '.zip', model.env, cfg).critic_loss, 'mse')
            payload = torch.load(path + '.zip', weights_only=False)
            del payload['critic_loss']
            torch.save(payload, path + '.zip')
            restored = MambaSAC.load(path, env=model.env, load_buffer=False,
                                     policy_kwargs=build_mamba_policy_kwargs(cfg), device='cpu')
            self.assertEqual(restored.critic_loss, 'huber')

    def test_invalid_loss_is_rejected(self):
        cfg = read_config('ma')
        cfg.set('DRL', 'critic_loss', 'squared')
        env = make_env(cfg)
        self.addCleanup(env.close)
        with self.assertRaisesRegex(ValueError, 'critic_loss'):
            create_model('SAC', env, cfg, log_path=None)


if __name__ == '__main__':
    unittest.main()
