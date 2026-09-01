"""Base class for perception modules."""

from abc import ABC, abstractmethod
import configparser
from configparser import ConfigParser
from gymnasium import spaces
import numpy as np


class BasePerception(ABC):
    """
    Abstract base class for perception modules.

    Each perception module handles:
    - Image/sensor acquisition from AirSim
    - Preprocessing (resize, normalize, etc.)
    - Observation space definition
    """

    def __init__(self, cfg: ConfigParser, client):
        """
        Initialize perception module.

        Args:
            cfg: ConfigParser with perception parameters
            client: AirSim client for image acquisition
        """
        self.cfg = cfg
        self.client = client
        self._load_config(cfg)

        # Will be set after processing
        self.min_distance_to_obstacles = float('inf')

    def _load_config(self, cfg: ConfigParser):
        """Load perception parameters from config."""
        self.screen_height = cfg.getint('environment', 'screen_height')
        self.screen_width = cfg.getint('environment', 'screen_width')
        self.max_depth_meters = cfg.getfloat('environment', 'max_depth_meters')

        # Camera ID (default "0")
        try:
            self.camera_id = cfg.get('perception', 'camera_id')
        except (configparser.NoSectionError, configparser.NoOptionError):
            self.camera_id = "0"

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """
        Define the observation space for this perception type.

        Returns:
            gymnasium.spaces.Space
        """
        pass

    @abstractmethod
    def get_observation(self, state_feature: np.ndarray = None) -> np.ndarray:
        """
        Get observation from sensors.

        Args:
            state_feature: Optional state features to concatenate

        Returns:
            Observation array matching observation_space
        """
        pass

    def preprocess_depth(self, depth_raw: np.ndarray) -> np.ndarray:
        """
        Standard depth preprocessing with NaN/Inf handling.

        Args:
            depth_raw: Raw depth image (meters)

        Returns:
            Processed depth image (0-255 uint8)
        """
        # Clean NaN/Inf values - replace with max depth
        depth_clean = np.where(
            np.isfinite(depth_raw), depth_raw, self.max_depth_meters
        )

        # Clip to valid range [0, max_depth]
        depth_clipped = np.clip(depth_clean, 0, self.max_depth_meters)

        # Normalize to 0-255
        depth_scaled = depth_clipped / self.max_depth_meters * 255

        # Update min distance for reward computation using robust statistics
        # Use 1st percentile instead of min() to avoid noise/invalid depth spikes
        valid_depth = depth_clipped[(depth_clipped > 0.1) & (depth_clipped < self.max_depth_meters)]
        if len(valid_depth) > 0:
            self.min_distance_to_obstacles = float(np.percentile(valid_depth, 1))
        else:
            self.min_distance_to_obstacles = self.max_depth_meters

        return depth_scaled.astype(np.uint8)

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions."""
        import cv2
        return cv2.resize(image, (self.screen_width, self.screen_height))
