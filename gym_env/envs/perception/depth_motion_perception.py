"""Depth image with an explicit frame-difference channel in meters."""

import numpy as np

from .depth_state_perception import DepthStatePerception


class DepthMotionPerception(DepthStatePerception):
    """Return Dict(image: (H, W, 2) uint8, state: normalized float32)."""

    image_channels = 2

    def __init__(self, cfg, client):
        super().__init__(cfg, client)
        self.motion_diff_max = cfg.getfloat('perception', 'motion_diff_max', fallback=1.0)
        if self.motion_diff_max <= 0:
            raise ValueError("motion_diff_max must be positive")
        self._previous_depth = None

    def reset(self, *, rng=None, disable_augmentation=False) -> None:
        super().reset(rng=rng, disable_augmentation=disable_augmentation)
        self._previous_depth = None

    def _image_channels(self, depth_meter, depth_uint8):
        # Motion requires cleaning in metres before resizing.
        depth_clean = np.where(np.isfinite(depth_meter), depth_meter, self.max_depth_meters)
        depth_clean = np.clip(depth_clean, 0, self.max_depth_meters)
        depth_resized = self.resize_image(depth_clean)
        if self._previous_depth is None:
            motion = np.full(depth_uint8.shape, 128, dtype=np.uint8)
        else:
            diff = depth_resized - self._previous_depth
            motion = np.rint(np.clip(diff / self.motion_diff_max, -1, 1) * 127 + 128).astype(np.uint8)
        self._previous_depth = depth_resized
        return np.stack([depth_uint8, motion], axis=-1)
