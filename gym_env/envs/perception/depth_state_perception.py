"""Depth image + state vector perception (Dict observation)."""

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

    def set_state_feature_length(self, length: int):
        """Set state feature length from dynamic model."""
        self.state_feature_length = length

    def get_observation_space(self) -> spaces.Space:
        """Return observation space as Dict(image, state)."""
        return spaces.Dict({
            "image": spaces.Box(0, 255, (self.screen_height, self.screen_width, self.image_channels), np.uint8),
            "state": spaces.Box(0, 1, (self.state_feature_length,), np.float32),
        })

    def get_observation(self, state_feature: np.ndarray = None) -> dict:
        depth_meter = self._get_depth()
        depth_uint8 = self.preprocess_depth(self.resize_image(depth_meter))
        image = self._image_channels(depth_meter, depth_uint8)
        if state_feature is None:
            state = np.zeros(self.state_feature_length, dtype=np.float32)
        else:
            state = np.asarray(state_feature, dtype=np.float32)[:self.state_feature_length]
            state = np.clip(state, 0.0, 1.0)
        return {"image": image, "state": state}

    def _image_channels(self, depth_meter, depth_uint8):
        return depth_uint8[..., None]

    def _get_depth(self):
        """Fetch depth (meters) from AirSim using DepthPlanar."""
        max_retries = 10
        if not self._camera_warmed:
            # A valid-sized first depth frame can contain uninitialized render values.
            for _ in range(max_retries):
                warmed = self.client.simGetImages([
                    airsim.ImageRequest(self.camera_id, airsim.ImageType.Scene, False, True),
                    airsim.ImageRequest(self.camera_id, airsim.ImageType.DepthPlanar, False, True),
                ])
                if len(warmed) == 2 and all(
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
