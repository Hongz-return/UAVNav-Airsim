"""
Sequence Replay Buffer for Mamba-based RL.
Stores complete episodes and samples continuous trajectory segments.
"""
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

    def get_sequence(self, start_idx: int, seq_len: int) -> Dict:
        """Get a sequence starting from start_idx."""
        end_idx = min(start_idx + seq_len, self.length)
        actual_len = end_idx - start_idx

        # Stack observations
        obs_seq = {
            key: np.stack([self.observations[i][key] for i in range(start_idx, end_idx)])
            for key in self.observations[0].keys()
        }

        # Stack actions, rewards, terminateds
        actions = np.stack(self.actions[start_idx:end_idx])
        rewards = np.array(self.rewards[start_idx:end_idx])
        terminateds = np.array(self.terminateds[start_idx:end_idx])

        # Next observations (shifted by 1)
        # For the last step, use final_obs if available (truncated episode), else use last obs
        def get_next_obs(i, key):
            if i + 1 < self.length:
                return self.observations[i + 1][key]
            elif self.final_obs is not None:
                # Truncated episode: use stored final_obs for bootstrap
                return self.final_obs[key]
            else:
                # Terminated episode: use last obs (won't be used due to done=True)
                return self.observations[self.length - 1][key]

        next_obs_seq = {
            key: np.stack([get_next_obs(i, key) for i in range(start_idx, end_idx)])
            for key in self.observations[0].keys()
        }

        return {
            'observations': obs_seq,
            'actions': actions,
            'rewards': rewards,
            'dones': terminateds,  # Only true terminations, not truncations
            'next_observations': next_obs_seq,
            'seq_len': actual_len,
        }


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
        self.n_envs = n_envs

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

    def _compact_valid_starts(self):
        """Remove stale entries (lazy compaction)."""
        self.valid_starts = [
            (ei, si) for ei, si in self.valid_starts
            if ei >= self._min_valid_ep_idx
        ]

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of sequences.

        Returns:
            Dict with:
            - observations: Dict[str, Tensor] with shape (batch, seq_len, ...)
            - actions: (batch, seq_len, action_dim)
            - rewards: (batch, seq_len)
            - dones: (batch, seq_len)
            - next_observations: Dict[str, Tensor] with shape (batch, seq_len, ...)
            - mask: (batch, seq_len) - 1 for valid, 0 for padded
        """
        if len(self.valid_starts) == 0:
            raise ValueError("No valid data in buffer")

        # Lazy compaction: purge stale entries periodically
        if self._min_valid_ep_idx > 0 and len(self.valid_starts) > 2 * self.total_transitions:
            self._compact_valid_starts()

        # Filter valid entries on-the-fly during sampling
        live_starts = [
            (ei, si) for ei, si in self.valid_starts
            if ei >= self._min_valid_ep_idx
        ]
        if len(live_starts) == 0:
            raise ValueError("No valid data in buffer after filtering")

        replace = len(live_starts) < batch_size
        indices = np.random.choice(len(live_starts), batch_size, replace=replace)

        batch_obs = {key: [] for key in self.observation_space.spaces.keys()}
        batch_next_obs = {key: [] for key in self.observation_space.spaces.keys()}
        batch_actions = []
        batch_rewards = []
        batch_dones = []
        batch_masks = []

        for idx in indices:
            ep_idx, start_idx = live_starts[idx]
            # Map global ep_idx to deque position
            deque_pos = ep_idx - self._min_valid_ep_idx
            episode = self.episodes[deque_pos]

            seq_data = episode.get_sequence(start_idx, self.seq_len)
            actual_len = seq_data['seq_len']

            # Pad if necessary
            pad_len = self.seq_len - actual_len

            for key in batch_obs.keys():
                obs_seq = seq_data['observations'][key]
                next_obs_seq = seq_data['next_observations'][key]

                if pad_len > 0:
                    # Pad with last observation
                    pad_shape = (pad_len,) + obs_seq.shape[1:]
                    obs_pad = np.broadcast_to(obs_seq[-1:], pad_shape)
                    next_obs_pad = np.broadcast_to(next_obs_seq[-1:], pad_shape)
                    obs_seq = np.concatenate([obs_seq, obs_pad], axis=0)
                    next_obs_seq = np.concatenate([next_obs_seq, next_obs_pad], axis=0)

                batch_obs[key].append(obs_seq)
                batch_next_obs[key].append(next_obs_seq)

            # Pad actions, rewards, terminateds
            actions = seq_data['actions']
            rewards = seq_data['rewards']
            terminateds = seq_data['dones']  # Only true terminations

            if pad_len > 0:
                actions = np.concatenate([actions, np.zeros((pad_len,) + actions.shape[1:])], axis=0)
                rewards = np.concatenate([rewards, np.zeros(pad_len)], axis=0)
                # Padded positions are treated as terminated (no bootstrap)
                terminateds = np.concatenate([terminateds, np.ones(pad_len, dtype=bool)], axis=0)

            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_dones.append(terminateds)

            # Mask: 1 for valid, 0 for padded
            mask = np.concatenate([np.ones(actual_len), np.zeros(pad_len)])
            batch_masks.append(mask)

        # Stack and convert to tensors
        result = {
            'observations': {
                key: torch.tensor(np.stack(batch_obs[key]), dtype=torch.float32, device=self.device)
                for key in batch_obs.keys()
            },
            'next_observations': {
                key: torch.tensor(np.stack(batch_next_obs[key]), dtype=torch.float32, device=self.device)
                for key in batch_next_obs.keys()
            },
            'actions': torch.tensor(np.stack(batch_actions), dtype=torch.float32, device=self.device),
            'rewards': torch.tensor(np.stack(batch_rewards), dtype=torch.float32, device=self.device),
            'dones': torch.tensor(np.stack(batch_dones), dtype=torch.float32, device=self.device),
            'mask': torch.tensor(np.stack(batch_masks), dtype=torch.float32, device=self.device),
        }

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
        # Serialize episodes
        episodes_data = []
        for ep in self.episodes:
            ep_data = {
                'observations': ep.observations,
                'actions': ep.actions,
                'rewards': ep.rewards,
                'terminateds': ep.terminateds,
                'final_obs': ep.final_obs,
                'length': ep.length,
            }
            episodes_data.append(ep_data)

        # Serialize current episode if exists
        current_ep_data = None
        if self.current_episode is not None:
            current_ep_data = {
                'observations': self.current_episode.observations,
                'actions': self.current_episode.actions,
                'rewards': self.current_episode.rewards,
                'terminateds': self.current_episode.terminateds,
                'final_obs': self.current_episode.final_obs,
                'length': self.current_episode.length,
            }

        return {
            'episodes': episodes_data,
            'current_episode': current_ep_data,
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
        # Clear current buffer
        self.episodes.clear()
        self.valid_starts.clear()
        self.current_episode = None
        self.total_transitions = 0

        # Restore episodes
        for ep_data in state['episodes']:
            ep = EpisodeStorage()
            ep.observations = ep_data['observations']
            ep.actions = ep_data['actions']
            ep.rewards = ep_data['rewards']
            ep.terminateds = ep_data['terminateds']
            ep.final_obs = ep_data['final_obs']
            ep.length = ep_data['length']
            self.episodes.append(ep)

        # Restore current episode
        if state['current_episode'] is not None:
            ep_data = state['current_episode']
            self.current_episode = EpisodeStorage()
            self.current_episode.observations = ep_data['observations']
            self.current_episode.actions = ep_data['actions']
            self.current_episode.rewards = ep_data['rewards']
            self.current_episode.terminateds = ep_data['terminateds']
            self.current_episode.final_obs = ep_data['final_obs']
            self.current_episode.length = ep_data['length']

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
