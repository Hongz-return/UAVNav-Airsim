"""
Mamba-based feature extractor for temporal RL.
Architecture: CNN (shared) -> Mamba (independent) -> MLP Head
"""
import math
import torch
import torch.nn as nn
from typing import Dict, Tuple
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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


class CausalTransformerBlock(nn.Module):
    """
    Causal Transformer for sequence modeling (baseline comparison with Mamba).
    Uses causal mask to prevent attending to future tokens.

    Input: (B, T, input_dim)
    Output: (B, T, hidden_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.short_horizon_max_period = 32.0

        # Project input to hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer encoder with causal mask
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_norm = nn.LayerNorm(hidden_dim)

    def _build_short_horizon_positional_encoding(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Fixed sinusoidal positional encoding tuned for short control horizons.

        Compared with a learned absolute table, this removes a source of overfitting
        and keeps the transformer focused on short-range temporal ordering.
        """
        if seq_len <= 0:
            return torch.zeros(1, 0, self.hidden_dim, device=device, dtype=dtype)

        half_dim = self.hidden_dim // 2
        if half_dim == 0:
            return torch.zeros(1, seq_len, self.hidden_dim, device=device, dtype=dtype)

        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        if half_dim == 1:
            periods = torch.ones(1, device=device, dtype=torch.float32)
        else:
            periods = torch.exp(
                torch.linspace(
                    0.0,
                    math.log(self.short_horizon_max_period),
                    steps=half_dim,
                    device=device,
                    dtype=torch.float32,
                )
            )

        angles = positions.unsqueeze(1) / periods.unsqueeze(0)
        pos_encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if pos_encoding.shape[1] < self.hidden_dim:
            pad = torch.zeros(seq_len, self.hidden_dim - pos_encoding.shape[1], device=device, dtype=torch.float32)
            pos_encoding = torch.cat([pos_encoding, pad], dim=1)

        return pos_encoding.unsqueeze(0).to(dtype=dtype)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask (upper triangular = -inf)."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, sequence: torch.Tensor, return_full_seq: bool = True) -> torch.Tensor:
        """
        Args:
            sequence: (B, T, input_dim)
            return_full_seq: if True, return (B, T, hidden_dim); else return (B, hidden_dim)
        """
        B, T, _ = sequence.shape

        # Project to hidden dim
        x = self.input_proj(sequence)  # (B, T, hidden_dim)

        # Add fixed short-horizon positional encoding.
        x = x + self._build_short_horizon_positional_encoding(T, sequence.device, x.dtype)

        # Causal mask
        causal_mask = self._generate_causal_mask(T, sequence.device)

        # Transformer forward
        out = self.transformer(x, mask=causal_mask)  # (B, T, hidden_dim)
        out = self.output_norm(out)

        if return_full_seq:
            return out
        else:
            return out[:, -1]


class MambaBlock(nn.Module):
    """
    Sequence modeling block with multiple backend options.
    Input: (B, T, input_dim)
    Output: (B, T, hidden_dim) - full sequence output for temporal learning

    Backends:
    - "auto": Try mamba_ssm -> mambapy -> LSTM (fallback)
    - "mamba_ssm": CUDA-optimized Mamba (Linux only)
    - "mambapy": Pure PyTorch Mamba (Windows compatible)
    - "lstm": LSTM baseline
    - "transformer": Causal Transformer baseline

    Args:
        force_lstm: Shortcut for backend="lstm"
        force_transformer: Shortcut for backend="transformer"
        backend: Explicit backend selection (overrides force_* flags)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        force_lstm: bool = False,
        force_transformer: bool = False,
        backend: str = "auto",
        # Transformer-specific params
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()
        self._backend = None

        # Determine target backend
        if backend != "auto":
            target_backend = backend
        elif force_transformer:
            target_backend = "transformer"
        elif force_lstm:
            target_backend = "lstm"
        else:
            target_backend = "auto"

        # === Transformer backend ===
        if target_backend == "transformer":
            print(f"    Using Transformer (Causal): input={input_dim}, hidden={hidden_dim}, heads={transformer_heads}, layers={transformer_layers}")
            self.transformer = CausalTransformerBlock(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_heads=transformer_heads,
                n_layers=transformer_layers,
            )
            self._backend = "transformer"
            return

        # === LSTM backend ===
        if target_backend == "lstm":
            print(f"    Using LSTM (baseline): input={input_dim}, hidden={hidden_dim}")
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self._backend = "lstm"
            return

        # === Auto or explicit Mamba ===
        # Try 1: CUDA-optimized mamba-ssm (Linux)
        if target_backend in ["auto", "mamba_ssm"]:
            try:
                from mamba_ssm import Mamba
                self.mamba = Mamba(
                    d_model=input_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.proj = nn.Linear(input_dim, hidden_dim)
                self._backend = "mamba_ssm"
                print(f"    Using mamba-ssm (CUDA): input={input_dim}, hidden={hidden_dim}")
                return
            except ImportError:
                if target_backend == "mamba_ssm":
                    raise ImportError("mamba_ssm not installed, cannot use mamba_ssm backend")

        # Try 2: mambapy - Pure PyTorch Mamba (Windows compatible)
        if target_backend in ["auto", "mambapy"] and self._backend is None:
            try:
                from mambapy.mamba import MambaConfig, Mamba as MambaPy
                config = MambaConfig(
                    d_model=input_dim,
                    n_layers=1,  # Single layer for RL
                    d_state=d_state,
                    d_conv=d_conv,
                    expand_factor=expand,
                    pscan=True,  # Use parallel scan
                )
                self.mamba = MambaPy(config)
                self.proj = nn.Linear(input_dim, hidden_dim)
                self._backend = "mambapy"
                print(f"    Using mambapy (Pure PyTorch): input={input_dim}, hidden={hidden_dim}")
                return
            except Exception as e:
                if target_backend == "mambapy":
                    raise ImportError(f"mambapy failed: {e}")
                print(f"    Warning: mambapy failed: {e}")

        # Fallback: LSTM
        if self._backend is None:
            print("    Warning: Falling back to LSTM")
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self._backend = "lstm"

    @property
    def backend(self) -> str:
        """Return current backend name."""
        return self._backend

    def forward(self, sequence: torch.Tensor, return_full_seq: bool = True) -> torch.Tensor:
        """
        Args:
            sequence: (B, T, input_dim)
            return_full_seq: if True, return (B, T, hidden_dim); else return (B, hidden_dim)
        Returns:
            output: (B, T, hidden_dim) or (B, hidden_dim)
        """
        if self._backend == "transformer":
            return self.transformer(sequence, return_full_seq=return_full_seq)
        elif self._backend == "mamba_ssm":
            out = self.mamba(sequence)  # (B, T, input_dim)
            out = self.proj(out)  # (B, T, hidden_dim)
        elif self._backend == "mambapy":
            out = self.mamba(sequence)  # (B, T, input_dim)
            out = self.proj(out)  # (B, T, hidden_dim)
        else:
            # LSTM fallback
            out, _ = self.lstm(sequence)  # (B, T, hidden_dim)

        if return_full_seq:
            return out  # (B, T, hidden_dim)
        else:
            return out[:, -1]  # (B, hidden_dim) - last timestep only


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
