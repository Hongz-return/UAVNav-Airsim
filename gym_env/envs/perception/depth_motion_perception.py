"""Depth image with an explicit frame-difference channel in meters."""

import numpy as np
from gymnasium import spaces

from .depth_state_perception import DepthStatePerception


class DepthMotionPerception(DepthStatePerception):
    """Return Dict(image: (H, W, 2) uint8, state: normalized float32)."""

    def __init__(self, cfg, client):
        super().__init__(cfg, client)
        self.motion_diff_max = cfg.getfloat('perception', 'motion_diff_max', fallback=1.0)
        if self.motion_diff_max <= 0:
            raise ValueError("motion_diff_max must be positive")
        self._previous_depth = None

    def reset(self) -> None:
        self._previous_depth = None

    def get_observation_space(self) -> spaces.Space:
        observation_space = super().get_observation_space()
        observation_space['image'] = spaces.Box(
            low=0, high=255,
            shape=(self.screen_height, self.screen_width, 2), dtype=np.uint8,
        )
        return observation_space

    def get_observation(self, state_feature: np.ndarray = None) -> dict:
        depth_meter = self._get_depth()
        # Keep the depth channel and obstacle distance identical to depth_state.
        depth_uint8 = self.preprocess_depth(self.resize_image(depth_meter))

        depth_clean = np.where(np.isfinite(depth_meter), depth_meter, self.max_depth_meters)
        depth_clean = np.clip(depth_clean, 0, self.max_depth_meters)
        depth_resized = self.resize_image(depth_clean)
        if self._previous_depth is None:
            motion = np.full(depth_uint8.shape, 128, dtype=np.uint8)
        else:
            diff = depth_resized - self._previous_depth
            motion = np.rint(np.clip(diff / self.motion_diff_max, -1, 1) * 127 + 128).astype(np.uint8)
        self._previous_depth = depth_resized.copy()

        state_norm = np.zeros((self.state_feature_length,), dtype=np.float32)
        if state_feature is not None:
            state_norm = np.asarray(state_feature, dtype=np.float32)
            state_norm = np.clip(state_norm[:self.state_feature_length], 0.0, 1.0)

        return {'image': np.stack([depth_uint8, motion], axis=-1), 'state': state_norm}
