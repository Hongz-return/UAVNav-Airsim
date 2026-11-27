"""Sequence backends used by the actor and critics."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def supports_batched_last(self, length: int) -> bool:
        if self.backend != "mambapy" or not 1 <= length <= 32 or len(self.mamba.layers) != 1:
            return False
        config = self.mamba.layers[0].mixer.config
        return not (config.bias or config.inner_layernorms or config.use_cuda)

    def forward_masked_last(self, sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Evaluate the final state of a supported block over left-aligned padding.

        Each right-aligned suffix starts at zero state and zero convolution history.
        The caller groups unpadded sequences for other backends/configurations.
        """
        length = sequence.shape[1]
        if not self.supports_batched_last(length):
            raise ValueError("This temporal block requires unpadded sequences")
        valid = (torch.arange(length, device=sequence.device)
                 >= (length - lengths[:, None])).unsqueeze(-1)
        layer = self.mamba.layers[0]
        mixer = layer.mixer
        config = mixer.config

        normalized = layer.norm(sequence).masked_fill(~valid, 0)
        inputs, gate = mixer.in_proj(normalized).chunk(2, dim=-1)
        inputs = F.silu(mixer.conv1d(inputs.transpose(1, 2))[:, :, :length].transpose(1, 2))
        delta, write, read = mixer.x_proj(inputs).split(
            [config.dt_rank, config.d_state, config.d_state], dim=-1,
        )
        delta = F.softplus(mixer.dt_proj(delta)).masked_fill(~valid, 0)

        # h_L = sum_i exp(A * sum_{j>i} delta_j) * delta_i * B_i * x_i.
        # Only h_L projected onto C_L is needed. The exclusive suffix sum
        # avoids subtracting nearly equal cumulative times.
        suffix_time = F.pad(delta[:, 1:].flip(1).cumsum(dim=1).flip(1), (0, 0, 0, 1))
        decay = torch.exp(suffix_time.unsqueeze(-1) * -mixer.A_log.float().exp())
        readout = (write * read[:, -1:]).to(decay.dtype)
        weighted_read = torch.bmm(
            decay.reshape(-1, config.d_inner, config.d_state),
            readout.reshape(-1, config.d_state, 1),
        ).reshape_as(inputs)
        recurrent = (weighted_read * (delta * inputs)).sum(dim=1)
        last = recurrent + mixer.D.float() * inputs[:, -1]
        return self.proj(sequence[:, -1] + mixer.out_proj(last * F.silu(gate[:, -1])))

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
