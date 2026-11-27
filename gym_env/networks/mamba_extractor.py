"""Original feature extractor and import paths for historical checkpoints."""

from typing import Dict

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .temporal import CausalTransformerBlock, MambaBlock


class TimeDistributedCNN(nn.Module):
    """
    Shared CNN that processes each frame independently.
    Input: (B, n_stack, C, H, W)
    Output: (B, n_stack, cnn_feat_dim)
    """

    def __init__(self, n_channels: int, height: int, width: int, output_dim: int = 64):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, height, width)
            cnn_flat_dim = self.cnn(sample).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_flat_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, n_stack, C, H, W)
        Returns:
            features: (B, n_stack, output_dim)
        """
        B, T, C, H, W = images.shape

        # Reshape for batch processing: (B * T, C, H, W)
        images_flat = images.view(B * T, C, H, W)

        # CNN forward
        features_flat = self.fc(self.cnn(images_flat))  # (B * T, output_dim)

        # Reshape back: (B, T, output_dim)
        features = features_flat.view(B, T, -1)

        return features


class MambaExtractor(BaseFeaturesExtractor):
    """
    Mamba-based feature extractor for SAC.

    Architecture:
        1. Shared CNN: extract features from each frame
        2. Concatenate with state sequence
        3. Independent Mamba blocks for Actor and Critic
        4. Output context for decision heads

    Note: This extractor outputs features that will be used by both Actor and Critic.
          For truly independent Mamba, use MambaSACPolicy instead.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 64,
        mamba_hidden_dim: int = 256,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
    ):
        # Calculate output dimension
        features_dim = mamba_hidden_dim
        super().__init__(observation_space, features_dim=features_dim)

        # Parse observation space
        image_shape = observation_space["image"].shape  # (n_stack, C, H, W) after VecTransposeImage
        state_shape = observation_space["state"].shape  # (n_stack, state_dim)

        self.n_stack = image_shape[0]

        # Detect image format (after VecTransposeImage: (n_stack, C, H, W))
        if image_shape[1] in [1, 3, 4] and image_shape[1] < image_shape[2]:
            n_channels, height, width = image_shape[1], image_shape[2], image_shape[3]
            self.channels_first = True
        else:
            height, width, n_channels = image_shape[1], image_shape[2], image_shape[3]
            self.channels_first = False

        state_dim = state_shape[1]

        print(f"  MambaExtractor Configuration:")
        print(f"    n_stack={self.n_stack}, image=({n_channels}, {height}, {width}), state_dim={state_dim}")

        # 1. Shared CNN (Time Distributed)
        self.cnn = TimeDistributedCNN(n_channels, height, width, cnn_output_dim)

        # 2. Combined feature dimension: CNN output + state
        combined_dim = cnn_output_dim + state_dim
        print(f"    Combined feature dim: {cnn_output_dim} (CNN) + {state_dim} (state) = {combined_dim}")

        # 3. Mamba block for sequence modeling
        self.mamba = MambaBlock(
            input_dim=combined_dim,
            hidden_dim=mamba_hidden_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            observations: Dict with:
                - "image": (B, n_stack, C, H, W) or (B, n_stack, H, W, C)
                - "state": (B, n_stack, state_dim)

        Returns:
            context: (B, mamba_hidden_dim)
        """
        images = observations["image"].float()  # (B, n_stack, ...)
        states = observations["state"].float()  # (B, n_stack, state_dim)

        # Normalize images
        images = images / 255.0

        # Handle channel order
        if not self.channels_first:
            # (B, T, H, W, C) -> (B, T, C, H, W)
            images = images.permute(0, 1, 4, 2, 3)

        # 1. CNN feature extraction (time distributed)
        cnn_features = self.cnn(images)  # (B, n_stack, cnn_output_dim)

        # 2. Concatenate with state sequence
        combined = torch.cat([cnn_features, states], dim=-1)  # (B, n_stack, combined_dim)

        # 3. Mamba sequence modeling
        context = self.mamba(combined)  # (B, mamba_hidden_dim)

        return context
