"""
Sequence Replay Buffer for Mamba-based RL.
Stores complete episodes and samples continuous trajectory segments.
"""
from bisect import bisect_left

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from gymnasium import spaces
from collections import deque


class EpisodeStorage:
    """Storage for a single episode."""

    def __init__(self):
        self.observations: List[Dict[str, np.ndarray]] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.terminateds: List[bool] = []  # True termination (crash/success/outside)
        self.final_obs: Optional[Dict[str, np.ndarray]] = None  # For truncated episodes
        self.length = 0
        self.terminal_reward: Optional[float] = None

    def add(self, obs: Dict[str, np.ndarray], action: np.ndarray,
            reward: float, terminated: bool):
        """
        Add transition.

        Args:
            terminated: True if episode ended due to task (crash/success/outside).
                       False if episode ended due to truncation (timeout).
                       This affects whether Critic can bootstrap.
        """
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminateds.append(terminated)
        self.length += 1

    def set_final_obs(self, final_obs: Dict[str, np.ndarray]):
        """Set the final observation for truncated episodes (needed for bootstrap)."""
        self.final_obs = final_obs

    def get_state(self) -> Dict:
        data = dict(observations=self.observations, actions=self.actions, rewards=self.rewards,
                    terminateds=self.terminateds, final_obs=self.final_obs, length=self.length)
        if self.terminal_reward is not None:
            data['terminal_reward'] = self.terminal_reward
        return data

    @classmethod
    def from_state(cls, data: Dict):
        episode = cls()
        episode.observations = data['observations']
        episode.actions = data['actions']
        episode.rewards = data['rewards']
        episode.terminateds = data['terminateds']
        episode.final_obs = data['final_obs']
        episode.length = data['length']
        episode.terminal_reward = data.get('terminal_reward')
        return episode

    def read_context(self, start: int, end: int, history: int, right_pad: int):
        """Read H-1 prefix frames, the segment and its bootstrap frame."""
        first = max(0, start - history + 1)
        frames = self.observations[first:end]
        if end < self.length:
            next_frame = self.observations[end]
        else:
            next_frame = self.final_obs if self.final_obs is not None else frames[-1]

        # Left-pad a short history with the first frame. Beyond a real segment,
        # context repeats its bootstrap frame, including at a timeout.
        padding = history - 1 - (start - first)
        frames = [frames[0]] * padding + frames + [next_frame] * (1 + right_pad)
        return {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}

    def get_sequence(self, start_idx: int, seq_len: int, history_len: Optional[int] = None) -> Dict:
        """Read one episode-local segment, including its final bootstrap frame."""
        history = 1 if history_len is None else history_len
        if history < 1:
            raise ValueError('history_len must be positive')
        end = min(start_idx + seq_len, self.length)
        length = end - start_idx
        context = self.read_context(start_idx, end, history, 0 if history_len is None else seq_len - length)
        current = slice(history - 1, history - 1 + length)
        following = slice(history, history + length)
        result = {
            'observations': {k: v[current].copy() for k, v in context.items()},
            'next_observations': {k: v[following].copy() for k, v in context.items()},
            'actions': np.stack(self.actions[start_idx:end]),
            'rewards': np.asarray(self.rewards[start_idx:end]),
            'dones': np.asarray(self.terminateds[start_idx:end]),
            'seq_len': length,
        }
        if history_len is not None:
            result['context_observations'] = context
            steps = np.minimum(np.arange(seq_len) + start_idx + 1, end)
            result['history_lengths'] = np.minimum(steps, history)
        return result


class SequenceReplayBuffer:
    """
    Replay buffer that stores complete episodes and samples continuous sequences.

    Key features:
    - Stores complete episodes (not individual transitions)
    - Samples continuous trajectory segments of fixed length
    - Handles episode boundaries correctly
    - Supports Dict observation spaces (image + state)
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        seq_len: int = 16,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size  # Max transitions (not episodes)
        self.observation_space = observation_space
        self.action_space = action_space
        self.seq_len = seq_len

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Episode storage
        self.episodes: deque = deque()
        self.current_episode: Optional[EpisodeStorage] = None
        self.total_transitions = 0

        # Index for efficient sampling: (episode_idx, start_idx) pairs
        # episode_idx is a monotonically increasing global counter.
        self.valid_starts: List[Tuple[int, int]] = []
        self._next_ep_idx = 0        # Global Episode counter (never resets)
        self._min_valid_ep_idx = 0   # Lowest ep_idx still alive in deque

    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        done: bool,
        infos: Optional[Dict] = None,
    ):
        """
        Add a transition to the buffer.

        Args:
            obs: Current observation (stored)
            next_obs: Next observation (stored for truncated episodes to enable bootstrap)
            action: Action taken
            reward: Reward received
            done: True ONLY for true terminations (crash/success/outside).
                  False for truncations (timeout) - allows Critic to bootstrap.
            infos: Must contain 'TimeLimit.truncated' for truncated episodes.

        Episode boundary handling:
        - True termination (done=True): episode ends, no bootstrap (next_obs not needed)
        - Truncation (TimeLimit.truncated=True): episode ends, next_obs stored for bootstrap
        """
        # Start new episode if needed
        if self.current_episode is None:
            self.current_episode = EpisodeStorage()

        # Check if this is a truncation (timeout) - episode ends but allows bootstrap
        truncated = infos.get('TimeLimit.truncated', False) if infos else False

        # Add transition
        # done=True only for true terminations, False for truncations (allows bootstrap)
        self.current_episode.add(obs, action, reward, done)
        self.total_transitions += 1

        # End episode on ANY done condition (termination OR truncation)
        # This prevents multiple episodes from being concatenated
        episode_ended = done or truncated
        if episode_ended:
            # With action_repeat the transition reward may include earlier dense rewards.
            terminal = (infos or {}).get('reward_components', {}).get('terminal')
            if done and terminal is not None and terminal != reward:
                self.current_episode.terminal_reward = float(terminal)
            # For truncated episodes, store next_obs for proper bootstrap
            if truncated and not done:
                self.current_episode.set_final_obs(next_obs)
            self._finish_episode()

        # Safety: force finish if current episode is too long (prevents memory issues)
        # This shouldn't happen with proper TimeLimit wrapper, but adds robustness
        max_episode_len = min(self.buffer_size // 2, 10000)  # Cap at half buffer or 10k steps
        if self.current_episode is not None and self.current_episode.length >= max_episode_len:
            print(f"Warning: Episode exceeded {max_episode_len} steps, forcing finish")
            self._finish_episode()

        # Remove old episodes if over capacity
        while self.total_transitions > self.buffer_size and len(self.episodes) > 1:
            removed = self.episodes.popleft()
            self.total_transitions -= removed.length
            self._min_valid_ep_idx += 1

    def _finish_episode(self):
        """Finish current episode and add to buffer."""
        if self.current_episode is not None and self.current_episode.length > 0:
            ep_idx = self._next_ep_idx
            self._next_ep_idx += 1
            self.episodes.append(self.current_episode)
            # Register valid starting positions with global ep_idx
            for start in range(self.current_episode.length):
                self.valid_starts.append((ep_idx, start))
        self.current_episode = None

    def sample(self, batch_size: int, history_len: int = 8, *,
               compact: bool = False, failure_horizon: Optional[int] = None,
               failure_rewards: Tuple[float, float] = (-25.0, -20.0)) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of sequences. Training can request compact=True to
        transfer one native-dtype context and keep grouping lengths on CPU.
        The default returns independent float32 observation batches.

        Returns:
            Dict with:
            - observations: Dict[str, Tensor] with shape (batch, seq_len, ...)
            - actions: (batch, seq_len, action_dim)
            - rewards: (batch, seq_len)
            - dones: (batch, seq_len)
            - next_observations: Dict[str, Tensor] with shape (batch, seq_len, ...)
            - mask: (batch, seq_len) - 1 for valid, 0 for padded
            - context_observations: (batch, history_len + seq_len, ...)
            - history_lengths: (batch, seq_len), actual episode-local history
            - failure_targets, failure_mask: optional finite-horizon labels
        """
        if failure_horizon is not None and failure_horizon < 1:
            raise ValueError("failure_horizon must be positive")
        if history_len < 1:
            raise ValueError('history_len must be positive')
        # Completed episodes append starts in order. Evictions leave a stale
        # prefix. Keep the original population order and NumPy RNG draws.
        first_live = bisect_left(self.valid_starts, (self._min_valid_ep_idx,))
        if len(self.valid_starts) > 2 * self.total_transitions:
            del self.valid_starts[:first_live]
            first_live = 0
        population = len(self.valid_starts) - first_live
        if not population:
            raise ValueError("No valid data in buffer")
        indices = np.random.choice(population, batch_size, replace=population < batch_size)

        shape = (batch_size, self.seq_len)

        def observation_batch(steps, native=False):
            return {key: np.empty((batch_size, steps, *space.shape), space.dtype if native else np.float32)
                    for key, space in self.observation_space.spaces.items()}

        if not compact:
            observations = observation_batch(self.seq_len)
            next_observations = observation_batch(self.seq_len)
        context = observation_batch(history_len + self.seq_len, native=compact)
        actions = np.zeros((*shape, *self.action_space.shape), np.float32)
        rewards = np.zeros(shape, np.float32)
        dones = np.ones(shape, np.float32)
        mask = np.zeros(shape, np.float32)
        lengths = np.empty(shape, np.int64)
        if failure_horizon is not None:
            failure_targets = np.zeros((*shape, len(failure_rewards)), np.float32)
            failure_mask = np.zeros(shape, np.float32)

        for row, index in enumerate(indices):
            ep_idx, start_idx = self.valid_starts[first_live + index]
            episode = self.episodes[ep_idx - self._min_valid_ep_idx]
            end = min(start_idx + self.seq_len, episode.length)
            actual_len = end - start_idx
            valid = slice(0, actual_len)
            frames = episode.read_context(start_idx, end, history_len, self.seq_len - actual_len)
            for key, sequence in frames.items():
                context[key][row] = sequence
                if compact:
                    continue
                current = sequence[history_len - 1:history_len - 1 + actual_len]
                observations[key][row, valid] = current
                observations[key][row, actual_len:] = current[-1]
                next_observations[key][row] = sequence[history_len:]
            steps = np.minimum(np.arange(self.seq_len) + start_idx + 1, end)
            lengths[row] = np.minimum(steps, history_len)
            actions[row, valid] = episode.actions[start_idx:end]
            rewards[row, valid] = episode.rewards[start_idx:end]
            dones[row, valid] = episode.terminateds[start_idx:end]
            mask[row, valid] = 1
            if failure_horizon is not None:
                remaining = episode.length - (start_idx + np.arange(actual_len))
                ended = bool(episode.terminateds[-1])
                terminal = episode.terminal_reward
                if terminal is None:
                    terminal = episode.rewards[-1]
                types = np.asarray(failure_rewards) == terminal
                failure_targets[row, valid] = (remaining <= failure_horizon)[:, None] & types & ended
                # A timeout censors horizons beyond the observed prefix.
                failure_mask[row, valid] = ended | (remaining >= failure_horizon)

        def tensors(arrays):
            return {key: torch.as_tensor(value, device=self.device) for key, value in arrays.items()}

        result = tensors({'actions': actions, 'rewards': rewards, 'dones': dones,
                          'mask': mask})
        result['history_lengths'] = torch.as_tensor(lengths, device='cpu' if compact else self.device)
        result['context_observations'] = tensors(context)
        if not compact:
            result.update(observations=tensors(observations), next_observations=tensors(next_observations))
        if failure_horizon is not None:
            result.update(tensors({'failure_targets': failure_targets, 'failure_mask': failure_mask}))
        return result

    def size(self) -> int:
        """Return total number of transitions."""
        return self.total_transitions

    def __len__(self) -> int:
        return self.total_transitions

    def get_state(self) -> Dict:
        """
        Get buffer state for checkpointing.

        Returns:
            Dict containing all buffer data and indices for serialization.
        """
        return {
            'episodes': [episode.get_state() for episode in self.episodes],
            'current_episode': self.current_episode.get_state() if self.current_episode is not None else None,
            'total_transitions': self.total_transitions,
            'valid_starts': self.valid_starts,
            'next_ep_idx': self._next_ep_idx,
            'min_valid_ep_idx': self._min_valid_ep_idx,
        }

    def set_state(self, state: Dict) -> None:
        """
        Restore buffer state from checkpoint.

        Args:
            state: Dict from get_state()
        """
        self.episodes = deque(EpisodeStorage.from_state(data) for data in state['episodes'])
        current = state['current_episode']
        self.current_episode = EpisodeStorage.from_state(current) if current is not None else None
        self.valid_starts = []

        # Restore counters/indices (with backward compatibility for old checkpoints)
        if 'total_transitions' in state:
            self.total_transitions = int(state['total_transitions'])
        else:
            total = sum(ep.length for ep in self.episodes)
            if self.current_episode is not None:
                total += self.current_episode.length
            self.total_transitions = total

        self._next_ep_idx = int(state.get('next_ep_idx', len(self.episodes)))
        self._min_valid_ep_idx = int(state.get('min_valid_ep_idx', 0))

        saved_valid_starts = state.get('valid_starts')
        if saved_valid_starts is not None:
            self.valid_starts = [(int(ep_idx), int(start_idx)) for ep_idx, start_idx in saved_valid_starts]

        # Rebuild valid starts index
        if not self.valid_starts:
            self._rebuild_valid_starts()
        else:
            self.valid_starts = [
                (ep_idx, start_idx)
                for ep_idx, start_idx in self.valid_starts
                if ep_idx >= self._min_valid_ep_idx
            ]

        min_next_ep_idx = self._min_valid_ep_idx + len(self.episodes)
        if self._next_ep_idx < min_next_ep_idx:
            self._next_ep_idx = min_next_ep_idx

    def _rebuild_valid_starts(self) -> None:
        """Rebuild valid start indices from restored episodes."""
        self.valid_starts = []

        # Keep indices contiguous when loading old checkpoints that had no index metadata.
        self._min_valid_ep_idx = 0
        self._next_ep_idx = len(self.episodes)

        for ep_idx, episode in enumerate(self.episodes):
            for start_idx in range(episode.length):
                self.valid_starts.append((ep_idx, start_idx))
