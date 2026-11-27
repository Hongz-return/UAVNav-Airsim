"""Depth image + state vector perception (Dict observation)."""

import math
import time

import numpy as np
import cosysairsim as airsim
from gymnasium import spaces

from .base_perception import BasePerception


class DepthStatePerception(BasePerception):
    """
    Perception that returns a Dict:
      - image: depth (H, W, 1) uint8 (0-255) from DepthPlanar
      - state: normalized state vector float32 (0-1)
    """

    image_channels = 1

    def __init__(self, cfg, client):
        super().__init__(cfg, client)
        # Will be set from dynamic model
        self.state_feature_length = 0
        self._camera_warmed = False
        self._depth_only = cfg.getboolean("airsim", "depth_only", fallback=False)
        self._virtual_ceiling = None
        self._load_virtual_ceiling_config(cfg)

    def _load_virtual_ceiling_config(self, cfg):
        self.virtual_ceiling_prob = cfg.getfloat("perception", "virtual_ceiling_prob", fallback=0.0)
        self._ceiling_height_min = cfg.getfloat("perception", "virtual_ceiling_height_min", fallback=1.5)
        self._ceiling_height_max = cfg.getfloat("perception", "virtual_ceiling_height_max", fallback=3.0)
        self._ceiling_arch_drop_max = cfg.getfloat("perception", "virtual_ceiling_arch_drop_max", fallback=0.6)
        hfov_deg = cfg.getfloat("perception", "virtual_ceiling_hfov_deg", fallback=90.0)
        aspect_ratio = cfg.getfloat("perception", "virtual_ceiling_aspect_ratio", fallback=16.0 / 9.0)
        if not 0 <= self.virtual_ceiling_prob <= 1:
            raise ValueError("perception.virtual_ceiling_prob must be in [0, 1]")
        if not math.isfinite(self._ceiling_height_min) or self._ceiling_height_min <= 0:
            raise ValueError("perception.virtual_ceiling_height_min must be finite and positive")
        if not math.isfinite(self._ceiling_height_max) or self._ceiling_height_max < self._ceiling_height_min:
            raise ValueError("perception.virtual_ceiling_height_max must be finite and at least the minimum height")
        if not math.isfinite(self._ceiling_arch_drop_max) or self._ceiling_arch_drop_max < 0:
            raise ValueError("perception.virtual_ceiling_arch_drop_max must be finite and non-negative")
        if not 0 < hfov_deg < 179:
            raise ValueError("perception.virtual_ceiling_hfov_deg must be in (0, 179)")
        if not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
            raise ValueError("perception.virtual_ceiling_aspect_ratio must be finite and positive")

        if self.virtual_ceiling_prob > 0:
            tan_half_vfov = math.tan(math.radians(hfov_deg) / 2.0) / aspect_ratio
            rows = (np.arange(self.screen_height) + 0.5) / self.screen_height
            cols = (np.arange(self.screen_width) + 0.5) / self.screen_width
            v = (0.5 - rows) * 2.0 * tan_half_vfov
            self._ceiling_v = v[v > 0, None]
            self._ceiling_u_squared = ((cols - 0.5) * 2.0)[None, :] ** 2

    def reset(self, *, rng=None, disable_augmentation=False) -> None:
        self._virtual_ceiling = None
        if disable_augmentation or self.virtual_ceiling_prob == 0.0:
            return

        rng = np.random.default_rng() if rng is None else rng
        if rng.random() < self.virtual_ceiling_prob:
            self._virtual_ceiling = (
                rng.uniform(self._ceiling_height_min, self._ceiling_height_max),
                rng.uniform(0.0, self._ceiling_arch_drop_max),
            )

    def set_state_feature_length(self, length: int):
        """Set state feature length from dynamic model."""
        self.state_feature_length = length

    def get_observation_space(self) -> spaces.Space:
        """Return observation space as Dict(image, state)."""
        return spaces.Dict({
            "image": spaces.Box(0, 255, (self.screen_height, self.screen_width, self.image_channels), np.uint8),
            "state": spaces.Box(0, 1, (self.state_feature_length,), np.float32),
        })

    def get_observation(
        self, state_feature: np.ndarray = None, *, dz_above_start: float | None = None
    ) -> dict:
        depth_meter = self._get_depth()
        depth_resized = self.resize_image(depth_meter)
        if self._virtual_ceiling is not None and dz_above_start is not None:
            self._apply_virtual_ceiling(depth_resized, dz_above_start)
        depth_uint8 = self.preprocess_depth(depth_resized)
        image = self._image_channels(depth_meter, depth_uint8)
        if state_feature is None:
            state = np.zeros(self.state_feature_length, dtype=np.float32)
        else:
            state = np.asarray(state_feature, dtype=np.float32)[:self.state_feature_length]
            state = np.clip(state, 0.0, 1.0)
        return {"image": image, "state": state}

    def _apply_virtual_ceiling(self, depth_resized, dz_above_start):
        height, arch_drop = self._virtual_ceiling
        h_crown = max(0.35, height - dz_above_start)
        h_local = np.maximum(0.35, h_crown - arch_drop * self._ceiling_u_squared)
        # DepthPlanar measures forward distance along the camera's optical axis.
        ceiling_depth = h_local / self._ceiling_v
        upper_depth = depth_resized[:self._ceiling_v.shape[0]]
        depth_for_comparison = np.where(np.isfinite(upper_depth), upper_depth, self.max_depth_meters)
        upper_depth[:] = np.minimum(depth_for_comparison, ceiling_depth)

    def _image_channels(self, depth_meter, depth_uint8):
        return depth_uint8[..., None]

    def _get_depth(self):
        """Fetch depth (meters) from AirSim using DepthPlanar."""
        max_retries = 10
        if not self._camera_warmed:
            # A valid-sized first depth frame can contain uninitialized render values.
            requests = [airsim.ImageRequest(self.camera_id, airsim.ImageType.DepthPlanar, False, True)]
            if not self._depth_only:
                requests.insert(0, airsim.ImageRequest(self.camera_id, airsim.ImageType.Scene, False, True))
            for _ in range(max_retries):
                warmed = self.client.simGetImages(requests)
                if len(warmed) == len(requests) and all(
                    image.width > 0 and image.height > 0 and len(image.image_data_uint8) > 0
                    for image in warmed
                ):
                    time.sleep(0.25)
                    self._camera_warmed = True
                    break
            else:
                raise RuntimeError("Failed to warm up the AirSim camera")
        for attempt in range(max_retries):
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.camera_id, airsim.ImageType.DepthPlanar, True)
            ])
            if responses and responses[0].width > 0 and responses[0].height > 0:
                break
            print(f"get_image_fail... retry {attempt + 1}/{max_retries}")
        else:
            raise RuntimeError("Failed to get a depth image from AirSim")

        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float,
            responses[0].width,
            responses[0].height
        )
        # DepthPlanar already in meters
        return depth_img
