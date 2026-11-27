"""Training windows must reproduce actual episode-local rollout histories."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium import spaces

from gym_env.replay.sequence_buffer import SequenceReplayBuffer
from tests.test_ch4_policy_smoke import make_model, read_config


class HistoryReplayTest(unittest.TestCase):
    def test_eviction_and_compaction_preserve_live_start_order(self):
        space = spaces.Dict({'state': spaces.Box(0, 255, (1,), np.float32)})
        replay = SequenceReplayBuffer(12, space, spaces.Box(-1, 1, (1,), np.float32),
                                      seq_len=3, device='cpu')
        for i in range(32):
            value = np.array([i], np.float32)
            replay.add({'state': value}, {'state': value + 1}, value, 0., False,
                       {'TimeLimit.truncated': i % 4 == 3})
            if i not in (15, 31):
                continue
            # At 16 transitions the stale prefix remains. At 32 it compacts.
            with patch('gym_env.replay.sequence_buffer.np.random.choice', return_value=np.arange(12)):
                batch = replay.sample(12)
            np.testing.assert_array_equal(batch['observations']['state'][:, 0, 0], np.arange(i-11, i+1))
            np.testing.assert_array_equal(batch['observations']['state'][-1, :, 0], [i, i, i])
            np.testing.assert_array_equal(batch['next_observations']['state'][-1, :, 0], [i+1] * 3)

    def test_real_prefix_next_frame_short_history_and_padding(self):
        observations = spaces.Dict({'image': spaces.Box(0, 255, (1, 2, 2), np.uint8),
                                    'state': spaces.Box(0, 255, (1,), np.float32)})
        replay = SequenceReplayBuffer(64, observations, spaces.Box(-1, 1, (3,), np.float32),
                                      seq_len=4, device='cpu')

        def obs(value):
            return {'image': np.full((1, 2, 2), value, np.uint8),
                    'state': np.asarray([value], np.float32)}

        for base, length, truncated in [(0, 10, False), (100, 3, True)]:
            for index in range(length):
                last = index == length - 1
                replay.add(obs(base + index), obs(base + index + 1), np.zeros(3), float(index),
                           last and not truncated, {'TimeLimit.truncated': last and truncated})
        selected = [(0, 0), (0, 6), (0, 9), (1, 0), (1, 2)]
        choices = np.asarray([replay.valid_starts.index(pair) for pair in selected])
        for history in [1, 4, 8]:
            with self.subTest(history=history), patch('gym_env.replay.sequence_buffer.np.random.choice', return_value=choices):
                batch = replay.sample(len(selected), history_len=history)
            self.assertEqual(batch['context_observations']['state'].shape, (5, history + 4, 1))
            self.assertEqual(batch['history_lengths'].dtype, torch.long)
            for row, (episode_id, start) in enumerate(selected):
                episode = replay.episodes[episode_id]
                for step in range(min(4, episode.length - start)):
                    position = start + step
                    length = min(history, position + 1)
                    self.assertEqual(batch['history_lengths'][row, step], length)
                    end = history - 1 + step
                    actual = batch['context_observations']['state'][row, end - length + 1:end + 1, 0]
                    expected = [entry['state'][0] for entry in episode.observations[max(0, position - history + 1):position + 1]]
                    np.testing.assert_array_equal(actual.numpy(), expected)
                    next_length = min(history, length + 1)
                    next_window = batch['context_observations']['state'][row, end - next_length + 2:end + 2, 0]
                    expected_next = (expected + [batch['next_observations']['state'][row, step, 0].item()])[-history:]
                    np.testing.assert_array_equal(next_window.numpy(), expected_next)
            self.assertEqual(batch['dones'][2, 0], 1)
            self.assertEqual(batch['dones'][4, 0], 0)
            self.assertEqual(batch['next_observations']['state'][4, 0, 0], 103)
            self.assertEqual(batch['mask'][4].sum(), 1)
            self.assertEqual(batch['mask'].sum(), 4 + 4 + 1 + 3 + 1)

        restored = SequenceReplayBuffer(64, observations, replay.action_space, seq_len=4, device='cpu')
        restored.set_state(replay.get_state())
        with patch('gym_env.replay.sequence_buffer.np.random.choice', return_value=choices):
            new_batch = restored.sample(len(selected), history_len=8)
        torch.testing.assert_close(new_batch['context_observations']['image'], batch['context_observations']['image'])


class HistoryLearnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def prepare(self, method='ma', history=8, stack=1):
        cfg = read_config(method)
        cfg.set('mamba', 'inference_seq_len', str(history))
        cfg.set('mamba', 'n_stack', str(stack))
        model = make_model(cfg)
        self.addCleanup(model.env.close)
        rng = np.random.default_rng(246)
        episodes = []
        for length, truncated in [(17, False), (5, True)]:
            sequence = [{'image': rng.integers(0, 256, model.observation_space['image'].shape, dtype=np.uint8),
                         'state': rng.uniform(.05, .95, model.observation_space['state'].shape).astype(np.float32)}
                        for _ in range(length + 1)]
            for index in range(length):
                action = rng.uniform(model.action_space.low, model.action_space.high).astype(np.float32)
                last = index == length - 1
                model.replay_buffer.add(sequence[index], sequence[index + 1], action,
                    35. if last and not truncated else -.1, last and not truncated,
                    {'TimeLimit.truncated': last and truncated})
            episodes.append((sequence, truncated))
        selected = [(0, 0), (0, 8), (0, 16), (1, 0), (1, 4)]
        choices = np.asarray([model.replay_buffer.valid_starts.index(pair) for pair in selected])
        with patch('gym_env.replay.sequence_buffer.np.random.choice', return_value=choices):
            batch = model.replay_buffer.sample(len(selected), history_len=history)
        return model, episodes, selected, batch

    @staticmethod
    def tensors(observations):
        return {key: torch.as_tensor(np.stack([entry[key] for entry in observations])).unsqueeze(0).float()
                for key in observations[0]}

    def test_actor_current_and_next_match_online_rollout(self):
        for method, stack, history in [('ma', 1, 8), ('ma_motion', 1, 8), ('ma', 1, 1), ('ma', 2, 4), ('ma_motion', 2, 4)]:
            with self.subTest(method=method, stack=stack, history=history):
                model, episodes, selected, batch = self.prepare(method, history, stack)
                actor = model.policy.actor
                online = {}
                with torch.no_grad():
                    for episode_id, (sequence, truncated) in enumerate(episodes):
                        actor.reset_history()
                        for index, observation in enumerate(sequence if truncated else sequence[:-1]):
                            actor({key: torch.as_tensor(value).unsqueeze(0).float() for key, value in observation.items()},
                                  deterministic=True, use_history=True)
                            distribution = actor.action_dist.distribution
                            online[episode_id, index] = (distribution.loc.clone()[0], distribution.scale.log().clone()[0])
                    cached_history = {key: value.clone() for key, value in actor._history_buffer.items()} if actor._history_buffer is not None else None
                    features = model.policy.extract_cnn_features(batch['context_observations'])
                    current = model._actor_history_parameters(features, batch['context_observations']['state'], batch['history_lengths'])
                    following = model._actor_history_parameters(features, batch['context_observations']['state'], batch['history_lengths'], next_step=True)
                for row, (episode_id, start) in enumerate(selected):
                    for step in range(int(batch['mask'][row].sum())):
                        for actual, expected in zip(current, online[episode_id, start + step]):
                            torch.testing.assert_close(actual[row, step], expected, rtol=2e-5, atol=3e-6)
                        if not batch['dones'][row, step]:
                            for actual, expected in zip(following, online[episode_id, start + step + 1]):
                                torch.testing.assert_close(actual[row, step], expected, rtol=2e-5, atol=3e-6)
                if cached_history is not None:
                    for key in cached_history:
                        self.assertTrue(torch.equal(cached_history[key], actor._history_buffer[key]))

    def test_critic_and_td_targets_use_shifted_real_histories(self):
        model, episodes, selected, batch = self.prepare()
        context = batch['context_observations']
        with torch.no_grad():
            features = model.policy.extract_cnn_features(context)
            target_features = model.policy.extract_cnn_features(context, use_target=True)
            q = model._critic_history_values(model.policy.critic, features, context['state'], batch['actions'], batch['history_lengths'])
            mean, _ = model._actor_history_parameters(features, context['state'], batch['history_lengths'], next_step=True)
            actions = model.policy.actor.scale_action(mean.tanh())
            next_q = model._critic_history_values(model.policy.critic_target, target_features, context['state'], actions, batch['history_lengths'], next_step=True)
        for row, (episode_id, start) in enumerate(selected):
            sequence, _ = episodes[episode_id]
            episode = model.replay_buffer.episodes[episode_id]
            for step in range(int(batch['mask'][row].sum())):
                position = start + step
                begin = max(0, position - 7)
                obs = self.tensors(sequence[begin:position + 1])
                past_actions = torch.as_tensor(np.stack(episode.actions[begin:position + 1])).unsqueeze(0)
                with torch.no_grad():
                    expected = model.policy.critic.forward_sequence(obs, past_actions)
                for actual, reference in zip(q, expected):
                    torch.testing.assert_close(actual[row, step], reference[0, -1], rtol=2e-5, atol=3e-6)
                if not batch['dones'][row, step]:
                    position += 1
                    begin = max(0, position - 7)
                    obs = self.tensors(sequence[begin:position + 1])
                    prior = torch.as_tensor(np.stack(episode.actions[begin:position])).unsqueeze(0) if position > begin else torch.empty(1, 0, 3)
                    candidate = torch.cat([prior, actions[row, step].view(1, 1, 3)], dim=1)
                    with torch.no_grad():
                        expected = model.policy.critic_target.forward_sequence(obs, candidate)
                    for actual, reference in zip(next_q, expected):
                        torch.testing.assert_close(actual[row, step], reference[0, -1], rtol=2e-5, atol=3e-6)
        with patch.object(model, '_sample_history_actions', return_value=(actions, torch.zeros_like(batch['rewards']))) as sampler:
            targets = model._compute_td_targets(batch, ent_coef=.1)
        self.assertTrue(sampler.call_args.kwargs['next_step'])
        expected = batch['rewards'] + .99 * (1 - batch['dones']) * torch.minimum(*next_q)
        torch.testing.assert_close(targets, expected)
        self.assertEqual(targets[2, 0], 35)
        self.assertEqual(batch['dones'][4, 0], 0)
        self.assertNotEqual(targets[4, 0], batch['rewards'][4, 0])

    def test_window_assembly_preserves_gradients_and_masks_padding(self):
        model, _, _, batch = self.prepare(history=4)
        states = batch['context_observations']['state']
        features = model.policy.extract_cnn_features(batch['context_observations']).detach().requires_grad_(True)
        mean, _ = model._actor_history_parameters(features, states, batch['history_lengths'])
        loss = (mean.square().sum(-1) * batch['mask']).sum()
        parameters = [model.policy.actor.mu.weight, next(model.policy.actor.mamba.parameters())]
        actual = torch.autograd.grad(loss, [features, *parameters])
        reference_features = features.detach().clone().requires_grad_(True)
        rows = []
        for row in range(len(batch['mask'])):
            values = []
            for step in range(batch['mask'].shape[1]):
                length = int(batch['history_lengths'][row, step])
                end = 3 + step
                mu, _ = model.policy.actor.forward_sequence_from_features(
                    reference_features[row:row + 1, end - length + 1:end + 1],
                    states[row:row + 1, end - length + 1:end + 1])
                values.append(mu[0, -1])
            rows.append(torch.stack(values))
        reference_loss = (torch.stack(rows).square().sum(-1) * batch['mask']).sum()
        expected = torch.autograd.grad(reference_loss, [reference_features, *parameters])
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=3e-5, atol=2e-6)
        self.assertEqual(actual[0][0, :3].abs().sum(), 0)
        self.assertGreater(actual[0][1, :3].abs().sum(), 0)
        self.assertEqual(actual[0][:, -1].abs().sum(), 0)

        actions = batch['actions'].clone().requires_grad_(True)
        values = model._critic_history_values(model.policy.critic, features.detach(), states, actions, batch['history_lengths'])
        action_grad, = torch.autograd.grad((values[0] * batch['mask']).sum(), actions)
        self.assertGreater(action_grad.abs().sum(), 0)
        self.assertEqual((action_grad * (1 - batch['mask']).unsqueeze(-1)).abs().sum(), 0)


if __name__ == '__main__':
    unittest.main()
