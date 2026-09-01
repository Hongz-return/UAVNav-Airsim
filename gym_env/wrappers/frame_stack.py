"""
Frame stacking wrapper for temporal observations.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import warnings


class DictFrameStack(gym.Wrapper):
    """
    Frame stacking for Dict observation space.
    Stacks the last n_stack frames for temporal modeling.

    Args:
        env: Gymnasium environment with Dict observation space
        n_stack: Number of frames to stack
    """

    def __init__(self, env, n_stack: int = 4):
        super().__init__(env)
        self.n_stack = n_stack

        # Create stacked observation space
        old_space = env.observation_space
        assert isinstance(old_space, spaces.Dict), "Only Dict obs space supported"

        new_spaces = {}
        for key, space in old_space.spaces.items():
            if key == "image":
                # Stack images: (H, W, C) -> (n_stack, H, W, C)
                low = np.repeat(space.low[np.newaxis, ...], n_stack, axis=0)
                high = np.repeat(space.high[np.newaxis, ...], n_stack, axis=0)
                new_spaces[key] = spaces.Box(low=low, high=high, dtype=space.dtype)
            elif key == "state":
                # Stack states: (D,) -> (n_stack, D)
                low = np.repeat(space.low[np.newaxis, ...], n_stack, axis=0)
                high = np.repeat(space.high[np.newaxis, ...], n_stack, axis=0)
                new_spaces[key] = spaces.Box(low=low, high=high, dtype=space.dtype)
            else:
                # Keep other spaces unchanged
                warnings.warn(f"DictFrameStack: Unknown key '{key}'. Passing it through without stacking.")
                new_spaces[key] = space

        self.observation_space = spaces.Dict(new_spaces)

        # Frame buffers
        self.image_buffer = deque(maxlen=n_stack)
        self.state_buffer = deque(maxlen=n_stack)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Fill buffers with initial observation
        for _ in range(self.n_stack):
            self.image_buffer.append(obs["image"])
            self.state_buffer.append(obs["state"])

        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add new frame to buffers
        self.image_buffer.append(obs["image"])
        self.state_buffer.append(obs["state"])

        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _get_stacked_obs(self):
        return {
            "image": np.stack(list(self.image_buffer), axis=0),  # (n_stack, H, W, C)
            "state": np.stack(list(self.state_buffer), axis=0),  # (n_stack, D)
        }
