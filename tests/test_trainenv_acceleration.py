"""Regression checks for the optimized MA-SAC and TrainEnv runtime."""
from functools import partial
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cosysairsim as airsim
import numpy as np
import torch
from mambapy.mamba import Mamba, MambaConfig
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from gym_env.algorithms.mamba_sac import MambaSAC
from gym_env.envs.perception.depth_state_perception import DepthStatePerception
from gym_env.networks import MambaExtractor, TimeDistributedCNN
from gym_env.networks.mamba_extractor import MambaBlock
from gym_env.networks.mamba_policy import MambaSACPolicy
from tests.test_ch4_masac_upgrades import MODEL_OPTIONS, make_monitor
from tests.test_ch4_policy_smoke import fill_replay, make_model, read_config
from tools.trainenv_instances import trainenv_settings


class ObservedVecEnv(DummyVecEnv):
    pending = False

    def step_async(self, actions):
        assert not self.pending
        super().step_async(actions)
        self.pending = True

    def step_wait(self):
        assert self.pending
        try:
            return super().step_wait()
        finally:
            self.pending = False


class StopAfterThreeBatches(BaseCallback):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.steps = []
        self.ended = False

    def _on_step(self):
        assert not self.env.pending
        self.steps.append(self.num_timesteps)
        return len(self.steps) < 3

    def _on_training_end(self):
        self.ended = True


class TrainEnvAccelerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_left_padding_and_gradients_match_independent_histories(self):
        torch.manual_seed(9)
        block = MambaBlock(7, 11, d_state=4, d_conv=4, backend="mambapy")
        for length, histories in ((1, [1, 1]), (8, [1, 2, 5, 8]), (32, [1, 3, 9, 32])):
            with self.subTest(length=length):
                sequence = torch.randn(len(histories), length, 7, requires_grad=True)
                result = block.forward_masked_last(sequence, torch.tensor(histories))
                reference = torch.cat([
                    block(sequence[row:row + 1, -history:], return_full_seq=False)
                    for row, history in enumerate(histories)
                ])
                torch.testing.assert_close(result, reference, atol=1e-5, rtol=0)
                cotangent = torch.randn_like(result)
                inputs = (sequence, *block.parameters())
                actual = torch.autograd.grad(result, inputs, grad_outputs=cotangent)
                expected = torch.autograd.grad(reference, inputs, grad_outputs=cotangent)
                for left, right in zip(actual, expected, strict=True):
                    torch.testing.assert_close(left, right, atol=1e-5, rtol=0)

    def test_unsupported_history_configurations_keep_legacy_forward(self):
        block = MambaBlock(7, 11, d_state=4, backend="mambapy")
        self.assertFalse(block.supports_batched_last(0))
        self.assertFalse(block.supports_batched_last(33))
        for options in ({"n_layers": 2}, {"bias": True}, {"inner_layernorms": True}):
            config = {"d_model": 7, "n_layers": 1, "d_state": 4, **options}
            block.mamba = Mamba(MambaConfig(**config))
            self.assertFalse(block.supports_batched_last(8))
            self.assertEqual(block(torch.zeros(2, 8, 7)).shape, (2, 8, 11))
        for backend in ("lstm", "transformer"):
            block = MambaBlock(8, 16, backend=backend)
            self.assertFalse(block.supports_batched_last(8))
        self.assertEqual(MambaExtractor.__module__, "gym_env.networks.mamba_extractor")
        self.assertEqual(TimeDistributedCNN.__module__, "gym_env.networks.mamba_extractor")

    def test_one_gradient_update_preserves_auxiliary_heads(self):
        for target in ("next_depth", "depth_delta", "terminal_risk"):
            with self.subTest(target=target):
                cfg = read_config("ma_motion_aux")
                cfg.set("aux", "aux_target", target)
                legacy, fast = make_model(cfg), make_model(cfg)
                fast.fast_history = True
                for model in (legacy, fast):
                    self.addCleanup(model.env.close)
                    fill_replay(model)
                    np.random.seed(23)
                    torch.manual_seed(23)
                    model.train(1, 4)
                for key, weights in fast.policy.state_dict().items():
                    torch.testing.assert_close(weights, legacy.policy.state_dict()[key], atol=1e-5, rtol=0)
                for parameter, reference in zip(fast.policy.parameters(), legacy.policy.parameters(), strict=True):
                    if reference.grad is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        torch.testing.assert_close(parameter.grad, reference.grad, atol=1e-5, rtol=0)
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in fast.policy.aux_predictor.parameters()))

    def make_overlapped_model(self):
        observed = ObservedVecEnv([partial(make_monitor, lane) for lane in range(4)])
        env = VecTransposeImage(observed)
        self.addCleanup(env.close)
        model = MambaSAC(MambaSACPolicy, env, seed=3, overlap_updates=True, **MODEL_OPTIONS)
        model.set_logger(configure(folder=None, format_strings=[]))
        return model, observed

    def test_overlap_uses_committed_replay_and_preserves_counts_and_early_stop(self):
        model, env = self.make_overlapped_model()
        stop = StopAfterThreeBatches(env)
        updates = []

        def record_update(*_):
            updates.append((model.replay_buffer.size(), env.pending))

        with patch.object(model, "train", side_effect=record_update):
            model.learn(20, callback=stop)
        self.assertEqual(updates, [(8, False), (8, True), (8, True)])
        self.assertEqual(stop.steps, [4, 8, 12])
        self.assertTrue(stop.ended)
        self.assertEqual(model.replay_buffer.size(), 12)
        terminal, timeout = model.replay_buffer.episodes
        self.assertTrue(terminal.terminateds[-1])
        self.assertFalse(timeout.terminateds[-1])
        np.testing.assert_array_equal(timeout.final_obs["state"][:2], [1, 3])

    def test_overlap_drains_sent_step_on_update_failure(self):
        model, env = self.make_overlapped_model()
        fill_replay(model)
        model.learning_starts = 0
        with self.assertRaisesRegex(ValueError, "divisible"):
            model.learn(5)
        with patch.object(model, "train", side_effect=RuntimeError("update failed")):
            with self.assertRaisesRegex(RuntimeError, "update failed"):
                model.learn(4)
        self.assertFalse(env.pending)

    def test_depth_only_settings_keep_camera_parameters(self):
        template = {
            "ViewMode": "SpringArmChase",
            "CameraDefaults": {"CaptureSettings": [
                {"ImageType": 1, "Width": 1920, "Height": 1080, "FOV_Degrees": 90},
                {"ImageType": 0, "Width": 1920, "Height": 1080},
            ]},
            "Vehicles": {"Drone": {"Cameras": {"0": {"Pitch": -6, "CaptureSettings": [
                {"ImageType": 1, "FOV_Degrees": 72},
            ]}}}},
        }
        settings = trainenv_settings(template)
        self.assertEqual(settings["ViewMode"], "NoDisplay")
        self.assertEqual(settings["CameraDefaults"]["CaptureSettings"], [
            {"ImageType": 1, "Width": 320, "Height": 180, "FOV_Degrees": 90},
        ])
        self.assertEqual(settings["Vehicles"]["Drone"]["Cameras"]["0"], {
            "Pitch": -6, "CaptureSettings": [{"ImageType": 1, "Width": 320, "Height": 180, "FOV_Degrees": 72}],
        })
        self.assertEqual(template["CameraDefaults"]["CaptureSettings"][0]["Width"], 1920)

    def test_depth_only_warmup_never_requests_scene(self):
        cfg = read_config("ma")
        cfg.read_dict({"airsim": {"depth_only": True}})
        client = Mock()
        client.simGetImages.side_effect = [
            [SimpleNamespace(width=1, height=1, image_data_uint8=b"png")],
            [SimpleNamespace(width=1, height=1, image_data_float=[1.5])],
        ]
        perception = DepthStatePerception(cfg, client)
        with patch("gym_env.envs.perception.depth_state_perception.time.sleep"):
            self.assertEqual(perception._get_depth().item(), 1.5)
        for call in client.simGetImages.call_args_list:
            self.assertEqual([request.image_type for request in call.args[0]], [airsim.ImageType.DepthPlanar])
