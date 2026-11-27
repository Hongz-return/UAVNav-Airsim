"""Spatial encoding and coordinate attention for depth observations."""

import torch
import torch.nn as nn


class HardSigmoid(nn.Module):
    """Batch-size agnostic hard sigmoid used by lightweight attention blocks."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x + 3.0, min=0.0, max=6.0) / 6.0


class HardSwish(nn.Module):
    """Hard-swish activation without batch-dependent normalization."""

    def __init__(self):
        super().__init__()
        self.hsigmoid = HardSigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.hsigmoid(x)


class CoordAttention2d(nn.Module):
    """Coordinate attention without batch-dependent normalization."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduction = max(1, reduction)
        hidden_dim = max(8, channels // reduction)

        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            HardSwish(),
        )
        self.attn_h = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        self.attn_w = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, height, width = x.shape

        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)

        mixed = torch.cat([x_h, x_w], dim=2)
        mixed = self.shared(mixed)

        x_h, x_w = torch.split(mixed, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.attn_h(x_h))
        a_w = torch.sigmoid(self.attn_w(x_w))
        return identity * a_h * a_w


class SequenceCNN(nn.Module):
    """
    CNN that processes sequences of stacked frames.
    Input: (B, T, n_stack, C, H, W) - T timesteps, each with n_stack frames
    Output: (B, T, cnn_output_dim)
    """

    def __init__(
        self,
        n_channels: int,
        height: int,
        width: int,
        n_stack: int,
        output_dim: int = 64,
        attention_type: str = "none",
        attention_reduction: int = 8,
        attention_stage: str = "auto",
    ):
        super().__init__()
        self.n_stack = n_stack
        self.attention_type = attention_type.lower()
        self.attention_stage = self._resolve_attention_stage(attention_stage)

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels * n_stack, 32, kernel_size=8, stride=4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.attention = self._build_attention(attention_reduction)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            sample = torch.zeros(1, n_channels * n_stack, height, width)
            cnn_flat_dim = self._forward_cnn(sample).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(cnn_flat_dim, output_dim),
            nn.ReLU(),
        )

    def _resolve_attention_stage(self, attention_stage: str) -> str:
        stage = attention_stage.lower()
        if self.attention_type == "none":
            return "none"
        if stage == "auto":
            return "conv2" if self.attention_type == "coordatt" else "conv3"
        if stage not in {"conv2", "conv3"}:
            raise ValueError(f"Unsupported CNN attention stage: {attention_stage}")
        return stage

    def _build_attention(self, attention_reduction: int) -> nn.Module:
        if self.attention_type == "none":
            return nn.Identity()
        if self.attention_type == "coordatt":
            return CoordAttention2d(64, reduction=attention_reduction)
        raise ValueError(f"Unsupported CNN attention type: {self.attention_type}")

    def _apply_attention(self, features: torch.Tensor, stage: str) -> torch.Tensor:
        if self.attention_stage == stage:
            return self.attention(features)
        return features

    def _forward_cnn(self, images: torch.Tensor) -> torch.Tensor:
        features = self.conv1(images)
        features = self.conv2(features)
        features = self._apply_attention(features, "conv2")
        features = self.conv3(features)
        features = self._apply_attention(features, "conv3")
        return self.flatten(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Various shapes depending on n_stack and context:
                - (B, T, n_stack, C, H, W): sequence with frame stacking
                - (B, n_stack, C, H, W): single step with frame stacking
                - (B, T, C, H, W): sequence without frame stacking (n_stack=1)
                - (B, C, H, W): single step without frame stacking (n_stack=1)
        Returns:
            features: (B, T, output_dim) or (B, output_dim)
        """
        # Fold the optional stack into channels, preserving batch/time axes.
        frame_dims = 4 if self.n_stack > 1 else 3
        leading = images.shape[:-frame_dims]
        channels, height, width = images.shape[-3:]
        flat = images.reshape(-1, self.n_stack * channels, height, width)
        features = self.fc(self._forward_cnn(flat))
        return features.reshape(*leading, features.shape[-1])
