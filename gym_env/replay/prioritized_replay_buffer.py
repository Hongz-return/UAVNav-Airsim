"""
Prioritized Experience Replay Buffer for Stable-Baselines3.
Supports both regular and Dict observation spaces.
"""

import threading
from typing import Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize

from .sum_tree import SumTree


class PERMixin:
    """Mixin class that adds PER functionality to any ReplayBuffer."""

    def _init_per(self, buffer_size, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000, eps=1e-6):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.eps = eps
        self.tree = SumTree(buffer_size)
        self.max_priority = 1.0
        self._n_calls = 0
        self._lock = threading.Lock()

    @property
    def beta(self):
        fraction = min(self._n_calls / self.beta_frames, 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def _per_add(self, data_idx):
        priority = self.max_priority ** self.alpha
        self.tree.update(data_idx, priority)

    def _per_sample_indices(self, batch_size):
        self._n_calls += 1
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)
        total = self.tree.total
        segment = total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            s = np.random.uniform(low, high)
            idx, priority = self.tree.get(s)
            idx = min(idx, self.size() - 1)
            indices[i] = idx
            priorities[i] = priority
        probs = priorities / (total + 1e-10)
        weights = (self.size() * probs) ** (-self.beta)
        weights = weights / weights.max()
        return indices, weights

    def update_priorities(self, indices, td_errors):
        with self._lock:
            for idx, td_error in zip(indices, td_errors):
                priority = (np.abs(td_error) + self.eps) ** self.alpha
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, priority)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()


class PrioritizedReplayBuffer(PERMixin, ReplayBuffer):
    """PER buffer for non-Dict observation spaces."""

    def __init__(self, buffer_size, observation_space, action_space, device="auto",
                 n_envs=1, optimize_memory_usage=False, handle_timeout_termination=True,
                 alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000, eps=1e-6):
        ReplayBuffer.__init__(self, buffer_size=buffer_size, observation_space=observation_space,
                              action_space=action_space, device=device, n_envs=n_envs,
                              optimize_memory_usage=optimize_memory_usage,
                              handle_timeout_termination=handle_timeout_termination)
        self._init_per(buffer_size, alpha, beta_start, beta_end, beta_frames, eps)

    def add(self, obs, next_obs, action, reward, done, infos):
        with self._lock:
            data_idx = self.pos
            super().add(obs, next_obs, action, reward, done, infos)
            self._per_add(data_idx)

    def sample(self, batch_size, env=None):
        with self._lock:
            indices, weights = self._per_sample_indices(batch_size)
            samples = self._get_samples(indices, env)
            weights_tensor = th.tensor(weights, dtype=th.float32, device=self.device)
            return samples, weights_tensor, indices


class PrioritizedDictReplayBuffer(PERMixin, DictReplayBuffer):
    """PER buffer for Dict observation spaces."""

    def __init__(self, buffer_size, observation_space, action_space, device="auto",
                 n_envs=1, optimize_memory_usage=False, handle_timeout_termination=True,
                 alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000, eps=1e-6):
        DictReplayBuffer.__init__(self, buffer_size=buffer_size, observation_space=observation_space,
                                  action_space=action_space, device=device, n_envs=n_envs,
                                  optimize_memory_usage=optimize_memory_usage,
                                  handle_timeout_termination=handle_timeout_termination)
        self._init_per(buffer_size, alpha, beta_start, beta_end, beta_frames, eps)

    def add(self, obs, next_obs, action, reward, done, infos):
        with self._lock:
            data_idx = self.pos
            super().add(obs, next_obs, action, reward, done, infos)
            self._per_add(data_idx)

    def sample(self, batch_size, env=None):
        with self._lock:
            indices, weights = self._per_sample_indices(batch_size)
            samples = self._get_samples(indices, env)
            weights_tensor = th.tensor(weights, dtype=th.float32, device=self.device)
            return samples, weights_tensor, indices


class UniformReplayBuffer(ReplayBuffer):
    """Uniform buffer for non-Dict spaces with PER-compatible interface."""

    def __init__(self, buffer_size, observation_space, action_space, device="auto",
                 n_envs=1, optimize_memory_usage=False, handle_timeout_termination=True,
                 alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000, eps=1e-6):
        super().__init__(buffer_size=buffer_size, observation_space=observation_space,
                         action_space=action_space, device=device, n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)
        self._lock = threading.Lock()

    def add(self, obs, next_obs, action, reward, done, infos):
        with self._lock:
            super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size, env=None):
        with self._lock:
            samples = super().sample(batch_size, env)
            weights = th.ones(batch_size, dtype=th.float32, device=self.device)
            return samples, weights, None

    def update_priorities(self, indices, td_errors):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()


class UniformDictReplayBuffer(DictReplayBuffer):
    """Uniform buffer for Dict spaces with PER-compatible interface."""

    def __init__(self, buffer_size, observation_space, action_space, device="auto",
                 n_envs=1, optimize_memory_usage=False, handle_timeout_termination=True,
                 alpha=0.6, beta_start=0.4, beta_end=1.0, beta_frames=100000, eps=1e-6):
        super().__init__(buffer_size=buffer_size, observation_space=observation_space,
                         action_space=action_space, device=device, n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage,
                         handle_timeout_termination=handle_timeout_termination)
        self._lock = threading.Lock()

    def add(self, obs, next_obs, action, reward, done, infos):
        with self._lock:
            super().add(obs, next_obs, action, reward, done, infos)

    def sample(self, batch_size, env=None):
        with self._lock:
            samples = super().sample(batch_size, env)
            weights = th.ones(batch_size, dtype=th.float32, device=self.device)
            return samples, weights, None

    def update_priorities(self, indices, td_errors):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()


def get_buffer_class(observation_space, use_per):
    """Get correct buffer class based on observation space and PER setting."""
    is_dict = isinstance(observation_space, spaces.Dict)
    if use_per:
        return PrioritizedDictReplayBuffer if is_dict else PrioritizedReplayBuffer
    else:
        return UniformDictReplayBuffer if is_dict else UniformReplayBuffer
