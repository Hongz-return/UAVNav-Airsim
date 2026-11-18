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
        channels_first: Emit each HWC image as CHW for the sequence policy.
    """

    def __init__(self, env, n_stack: int = 4, *, channels_first: bool = False):
        super().__init__(env)
        self.n_stack = n_stack
        self.channels_first = channels_first

        # Create stacked observation space
        old_space = env.observation_space
        assert isinstance(old_space, spaces.Dict), "Only Dict obs space supported"

        new_spaces = {}
        for key, space in old_space.spaces.items():
            if key in ("image", "state"):
                low, high = space.low, space.high
                if key == "image" and channels_first:
                    low, high = np.moveaxis(low, -1, 0), np.moveaxis(high, -1, 0)
                new_spaces[key] = spaces.Box(
                    low=np.repeat(low[None], n_stack, axis=0),
                    high=np.repeat(high[None], n_stack, axis=0),
                    dtype=space.dtype,
                )
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
        images = np.stack(self.image_buffer, axis=0)
        return {
            "image": np.moveaxis(images, -1, 1) if self.channels_first else images,
            "state": np.stack(list(self.state_buffer), axis=0),  # (n_stack, D)
        }
