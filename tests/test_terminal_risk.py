"""Verify risk targets, censored timeouts and an actual actor-head update."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium import spaces

from gym_env.replay.sequence_buffer import SequenceReplayBuffer
from tests.test_ch4_policy_smoke import make_model, read_config


class TerminalRiskTest(unittest.TestCase):
    def test_action_repeat_terminal_component_survives_checkpoint(self):
        space = spaces.Box(-1, 1, (1,), np.float32)
        replay = SequenceReplayBuffer(16, spaces.Dict({'state': space}), space, seq_len=1, device='cpu')
        zero = np.zeros(1, np.float32)
        obs = {'state': zero}
        for terminal in (-25., -20., 35.):
            replay.add(obs, obs, zero, terminal + .5, True,
                       {'reward_components': {'progress': .5, 'terminal': terminal}})
        restored = SequenceReplayBuffer(16, replay.observation_space, space, seq_len=1, device='cpu')
        restored.set_state(replay.get_state())
        with patch('gym_env.replay.sequence_buffer.np.random.choice', return_value=np.arange(3)):
            batch = restored.sample(3, failure_horizon=2)
        np.testing.assert_array_equal(batch['failure_targets'][:, 0], [[1, 0], [0, 1], [0, 0]])
        np.testing.assert_array_equal(batch['rewards'][:, 0], [-24.5, -19.5, 35.5])

    def test_horizon_labels_and_optimizer(self):
        cfg = read_config("ma")
        cfg["aux"] = {"aux_weight": "0.1", "aux_target": "terminal_risk", "aux_hidden": "16"}
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        model = make_model(cfg)
        self.addCleanup(model.env.close)

        rng = np.random.default_rng(39)
        obs = {
            "image": rng.integers(0, 256, model.observation_space["image"].shape, dtype=np.uint8),
            "state": rng.random(6, dtype=np.float32),
        }
        for end_reward, timeout in [(-25.0, False), (-20.0, False), (35.0, False), (-10.0, True)]:
            for step in range(5):
                final = step == 4
                model.replay_buffer.add(obs, obs, model.action_space.sample(),
                                        end_reward if final else 0.0, done=final and not timeout,
                                        infos={"TimeLimit.truncated": final and timeout})
        with patch("gym_env.replay.sequence_buffer.np.random.choice", return_value=np.arange(0, 20, 5)):
            batch = model.replay_buffer.sample(4, failure_horizon=2)
            np.testing.assert_array_equal(batch["failure_targets"][0, :5, 0], [0, 0, 0, 1, 1])
            np.testing.assert_array_equal(batch["failure_targets"][1, :5, 1], [0, 0, 0, 1, 1])
            self.assertEqual(float(batch["failure_targets"][2:].sum()), 0)
            np.testing.assert_array_equal(batch["failure_mask"][3, :5], [1, 1, 1, 1, 0])
            self.assertEqual(float(batch["failure_mask"][:, 5:].sum()), 0)

        head = model.policy.aux_predictor[-1]
        previous = head.weight.detach().clone()
        model.train(gradient_steps=1, batch_size=4)
        self.assertFalse(torch.equal(previous, head.weight))
        self.assertTrue(np.isfinite(model.logger.name_to_value["train/aux_loss"]))


if __name__ == "__main__":
    unittest.main()
