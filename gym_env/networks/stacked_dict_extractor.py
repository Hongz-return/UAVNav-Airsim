"""Feature extractor for Dict observations with stacked depth images."""

from typing import Dict

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .mamba_policy import CoordAttention2d


class FrameStackedDictExtractor(BaseFeaturesExtractor):
    """
    CNN-based extractor for Dict observations after ``DictFrameStack``.

    Expected stacked image observation shapes:
    - image: (n_stack, C, H, W), or
    - image: (n_stack, H, W, C)
    - state: (n_stack, state_dim)

    The baseline keeps all temporal information in the input, but does not use
    any recurrent/sequence block. Frames are concatenated along channel
    dimension and processed by a plain CNN.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        cnn_output_dim: int = 64,
        cnn_attention: str = "none",
        cnn_attention_reduction: int = 8,
        cnn_attention_stage: str = "auto",
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        image_shape = observation_space["image"].shape
        state_shape = observation_space["state"].shape

        if len(image_shape) != 4:
            raise ValueError(
                "FrameStackedDictExtractor expects stacked image observations with shape "
                f"(n_stack, C, H, W), got {image_shape}"
            )
        if len(state_shape) != 2:
            raise ValueError(
                "FrameStackedDictExtractor expects stacked state observations with shape "
                f"(n_stack, state_dim), got {state_shape}"
            )

        n_stack, n_channels, height, width, image_layout = self._parse_image_shape(image_shape)
        self.n_stack = n_stack
        self.n_channels = n_channels
        self.height = height
        self.width = width
        self.image_layout = image_layout
        self.state_flat_dim = state_shape[0] * state_shape[1]
        self.cnn_attention = cnn_attention.strip().lower()
        self.cnn_attention_stage = self._resolve_attention_stage(cnn_attention_stage)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_stack * n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        if self.cnn_attention == "coordatt":
            self.attention = CoordAttention2d(64, reduction=cnn_attention_reduction)
        elif self.cnn_attention == "none":
            self.attention = nn.Identity()
        else:
            raise ValueError(f"Unsupported CNN attention type: {cnn_attention}")

        with torch.no_grad():
            sample = torch.zeros(1, n_stack * n_channels, height, width)
            cnn_flat_dim = self._forward_cnn(sample).shape[1]

        self.cnn_head = nn.Sequential(
            nn.Linear(cnn_flat_dim, cnn_output_dim),
            nn.ReLU(),
        )

        self._features_dim = cnn_output_dim + self.state_flat_dim

    def _resolve_attention_stage(self, attention_stage: str) -> str:
        if self.cnn_attention == "none":
            return "none"
        stage = attention_stage.strip().lower()
        if stage == "auto":
            return "conv2"
        if stage not in {"conv2", "conv3"}:
            raise ValueError(f"Unsupported CNN attention stage: {attention_stage}")
        return stage

    def _forward_cnn(self, images: torch.Tensor) -> torch.Tensor:
        features = images
        for layer_index, layer in enumerate(self.cnn):
            features = layer(features)
            if layer_index == 3 and self.cnn_attention_stage == "conv2":
                features = self.attention(features)
            elif layer_index == 5 and self.cnn_attention_stage == "conv3":
                features = self.attention(features)
        return features

    @staticmethod
    def _parse_image_shape(image_shape):
        """Infer whether stacked images are in NCHW or NHWC layout."""
        if len(image_shape) != 4:
            raise ValueError(
                "FrameStackedDictExtractor expects a stacked image shape with 4 dims, "
                f"got {image_shape}"
            )

        n_stack = image_shape[0]

        # Preferred layouts:
        # - (n_stack, C, H, W)
        # - (n_stack, H, W, C)
        if image_shape[1] in (1, 3, 4):
            n_channels, height, width = image_shape[1], image_shape[2], image_shape[3]
            return n_stack, n_channels, height, width, "nchw"

        if image_shape[-1] in (1, 3, 4):
            height, width, n_channels = image_shape[1], image_shape[2], image_shape[3]
            return n_stack, n_channels, height, width, "nhwc"

        raise ValueError(
            "Could not infer stacked image layout. Expected channels in axis 1 or axis -1, "
            f"got {image_shape}"
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        images = observations["image"].float() / 255.0
        states = observations["state"].float()

        if images.ndim != 5:
            raise ValueError(
                "FrameStackedDictExtractor expects batched stacked images with 5 dims, "
                f"got {tuple(images.shape)}"
            )

        if self.image_layout == "nhwc":
            images = images.permute(0, 1, 4, 2, 3).contiguous()

        batch_size, n_stack, n_channels, height, width = images.shape
        images = images.reshape(batch_size, n_stack * n_channels, height, width)

        image_features = self.cnn_head(self._forward_cnn(images))
        state_features = states.reshape(batch_size, -1)
        return torch.cat([image_features, state_features], dim=1)
