"""Depth image + state vector perception (Dict observation)."""

import numpy as np
import cv2
import cosysairsim as airsim
from gymnasium import spaces

from .base_perception import BasePerception


class DepthStatePerception(BasePerception):
    """
    Perception that returns a Dict:
      - image: depth (H, W, 1) uint8 (0-255) from DepthPlanar
      - state: normalized state vector float32 (0-1)
    """

    def __init__(self, cfg, client):
        super().__init__(cfg, client)
        # Will be set from dynamic model
        self.state_feature_length = 0

    def set_state_feature_length(self, length: int):
        """Set state feature length from dynamic model."""
        self.state_feature_length = length

    def get_observation_space(self) -> spaces.Space:
        """Return observation space as Dict(image, state)."""
        image_shape = (self.screen_height, self.screen_width, 1)
        return spaces.Dict({
            "image": spaces.Box(
                low=np.zeros(image_shape, dtype=np.uint8),
                high=np.full(image_shape, 255, dtype=np.uint8),
                dtype=np.uint8
            ),
            "state": spaces.Box(
                low=np.zeros((self.state_feature_length,), dtype=np.float32),
                high=np.ones((self.state_feature_length,), dtype=np.float32),
                dtype=np.float32
            ),
        })

    def get_observation(self, state_feature: np.ndarray = None) -> dict:
        """Get depth image (uint8) and normalized state vector."""
        depth_meter = self._get_depth()
        depth_resized = self.resize_image(depth_meter)
        depth_uint8 = self.preprocess_depth(depth_resized)

        # HWC -> (H, W, 1)
        depth_ch = np.expand_dims(depth_uint8, axis=-1)

        # State is already normalized to [0, 1] from dynamics model
        state_norm = np.zeros((self.state_feature_length,), dtype=np.float32)
        if state_feature is not None:
            state_norm = np.asarray(state_feature, dtype=np.float32)
            state_norm = np.clip(state_norm[:self.state_feature_length], 0.0, 1.0)

        return {
            "image": depth_ch,
            "state": state_norm,
        }

    def _get_depth(self):
        """Fetch depth (meters) from AirSim using DepthPlanar."""
        max_retries = 10
        for attempt in range(max_retries):
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.camera_id, airsim.ImageType.DepthPlanar, True)
            ])
            if responses[0].width > 0:
                break
            print(f"get_image_fail... retry {attempt + 1}/{max_retries}")
        else:
            # Return fallback depth (max range) if all retries fail
            print("WARNING: Failed to get depth image, using fallback")
            return np.full((self.screen_height, self.screen_width),
                          self.max_depth_meters, dtype=np.float32)

        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float,
            responses[0].width,
            responses[0].height
        )
        # DepthPlanar already in meters
        return depth_img
