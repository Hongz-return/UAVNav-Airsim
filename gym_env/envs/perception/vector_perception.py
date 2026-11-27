"""Vector-only perception (state features, no depth image)."""

import numpy as np
from gymnasium import spaces

from .base_perception import BasePerception


class VectorPerception(BasePerception):
    """
    Perception that returns only the normalized state vector.
    No depth image is used — suitable for MLP-SAC baseline.

    Observation: flat float32 vector of state features (0-1).
    min_distance_to_obstacles is set to inf (no visual info).
    """

    def __init__(self, cfg, client):
        super().__init__(cfg, client)
        self.state_feature_length = 0

    def set_state_feature_length(self, length: int):
        """Set state feature length from dynamic model."""
        self.state_feature_length = length

    def get_observation_space(self) -> spaces.Space:
        """Return flat Box observation space for state vector."""
        return spaces.Box(
            low=np.zeros((self.state_feature_length,), dtype=np.float32),
            high=np.ones((self.state_feature_length,), dtype=np.float32),
            dtype=np.float32,
        )

    def get_observation(
        self, state_feature: np.ndarray = None, *, dz_above_start: float | None = None
    ) -> np.ndarray:
        """Get normalized state vector only (no depth image query)."""
        # No depth image -> no obstacle distance info
        self.min_distance_to_obstacles = float('inf')

        state_norm = np.zeros((self.state_feature_length,), dtype=np.float32)
        if state_feature is not None:
            state_norm = np.asarray(state_feature, dtype=np.float32)
            state_norm = np.clip(state_norm[:self.state_feature_length], 0.0, 1.0)

        return state_norm
