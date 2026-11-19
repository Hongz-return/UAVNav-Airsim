"""
Mamba-SAC Policy with independent Actor/Critic Mamba blocks.
Supports both single-step inference and sequence-based training.
Architecture: CNN (shared) -> Mamba (independent) -> MLP Head
"""

import torch
import torch.nn as nn
from typing import Dict, List, Type, Optional, Any, Union, Tuple
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.sac.policies import SACPolicy, LOG_STD_MIN, LOG_STD_MAX

from .mamba_extractor import TimeDistributedCNN, MambaBlock


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


class NextDepthPredictor(nn.Module):
    """Predict pooled next depth (or depth delta) from Actor context and action."""

    def __init__(
        self, context_dim: int, action_dim: int, out_h: int = 15,
        out_w: int = 22, hidden: int = 256, target: str = "next_depth",
    ):
        super().__init__()
        if target not in {"next_depth", "depth_delta"}:
            raise ValueError(f"Unsupported auxiliary target: {target}")
        self.out_h = out_h
        self.out_w = out_w
        self.net = nn.Sequential(
            nn.Linear(context_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_h * out_w),
            nn.Sigmoid() if target == "next_depth" else nn.Tanh(),
        )

    def forward(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        prediction = self.net(torch.cat([context, actions], dim=-1))
        return prediction.reshape(*context.shape[:-1], self.out_h, self.out_w)


class _ObservationEncoder(nn.Module):
    """Shared input layout for the actor and both critics."""

    def set_cnn(self, cnn: SequenceCNN):
        self._cnn = cnn

    def _process_obs(self, observations: Dict[str, torch.Tensor], is_sequence: bool = False):
        features = self._cnn(observations["image"].float() / 255.0)
        return self._combine_features(features, observations["state"], is_sequence)

    def _combine_features(self, cnn_features, states, is_sequence=False):
        if self.n_stack > 1:
            states = states[..., -1, :]
        combined = self.input_norm(torch.cat([cnn_features, states.float()], dim=-1))
        return combined if is_sequence else combined.unsqueeze(1)


class MambaActor(_ObservationEncoder):
    """
    Actor network with Mamba for sequence processing.
    Supports both single-step inference and sequence-based training.

    Inference with history buffer:
    - Maintains a rolling history of observations (length = inference_seq_len)
    - Each step: add new obs to history, process full sequence, take last output
    - This allows Mamba to leverage temporal context during deployment
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        cnn_output_dim: int = 64,
        mamba_hidden_dim: int = 256,
        actor_hidden: List[int] = [256, 128],
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        force_lstm: bool = False,
        inference_seq_len: int = 8,
        backend: str = "auto",
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()

        self.action_dim = action_space.shape[0]
        self.inference_seq_len = inference_seq_len

        # Action scaling: SquashedDiagGaussian outputs [-1, 1], scale to action_space
        self.register_buffer("action_low", torch.tensor(action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_space.high, dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)

        # Parse observation space
        image_shape = observation_space["image"].shape
        state_shape = observation_space["state"].shape

        # Handle both n_stack > 1 and n_stack = 1 cases
        if len(image_shape) == 4:
            # (n_stack, C, H, W) - with frame stacking
            self.n_stack = image_shape[0]
        elif len(image_shape) == 3:
            # (C, H, W) - no frame stacking
            self.n_stack = 1
        else:
            raise ValueError(f"Unexpected image shape: {image_shape}")

        # State dim: (n_stack, state_dim) or (state_dim,)
        if len(state_shape) == 2:
            state_dim = state_shape[1]
        else:
            state_dim = state_shape[0]

        self.cnn_output_dim = cnn_output_dim

        # CNN for image features
        self._cnn = None  # Will be set by set_cnn()

        # LayerNorm after concat(visual, state) - unify scale, reduce training variance
        combined_dim = cnn_output_dim + state_dim
        self.input_norm = nn.LayerNorm(combined_dim)

        # Sequence modeling block (Mamba/LSTM/Transformer)
        self.mamba = MambaBlock(
            input_dim=combined_dim,
            hidden_dim=mamba_hidden_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            force_lstm=force_lstm,
            backend=backend,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )

        self.context_norm = nn.LayerNorm(mamba_hidden_dim)

        # Actor head
        head_layers = []
        prev_dim = mamba_hidden_dim
        for hidden_dim in actor_hidden:
            head_layers.append(nn.Linear(prev_dim, hidden_dim))
            head_layers.append(nn.LayerNorm(hidden_dim))
            head_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.head = nn.Sequential(*head_layers)
        self.mu = nn.Linear(prev_dim, self.action_dim)
        self.log_std = nn.Linear(prev_dim, self.action_dim)

        # Initialize output layers
        # mu: small init so initial actions are near center of action space
        nn.init.uniform_(self.mu.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mu.bias, -3e-3, 3e-3)
        # log_std: init to -1 for moderate exploration (std ≈ 0.37)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
        nn.init.constant_(self.log_std.bias, -1.0)  # Initial std ≈ 0.37 for exploration

        self.action_dist = SquashedDiagGaussianDistribution(self.action_dim)

        # History buffer for inference (will be initialized on first use)
        self._history_buffer = None
        self._history_len = 0

    def reset_history(self):
        """Reset history buffer at episode start."""
        self._history_buffer = None
        self._history_len = 0

    def set_inference_seq_len(self, seq_len: int) -> None:
        """Cap/expand the inference history window used at rollout time.

        Used for closed-loop history-length ablations. Does not change trained
        weights. seq_len=1 is equivalent to disabling the history buffer.
        """
        seq_len = int(seq_len)
        if seq_len < 1:
            raise ValueError(f"inference_seq_len must be >= 1, got {seq_len}")
        self.inference_seq_len = seq_len
        self.reset_history()

    def _init_history_buffer(self, obs: Dict[str, torch.Tensor]):
        """Initialize history buffer with first observation."""
        self._history_buffer = {
            key: value.new_zeros((value.shape[0], self.inference_seq_len, *value.shape[1:]))
            for key, value in obs.items()
        }
        self._history_len = 0

    def _add_to_history(self, obs: Dict[str, torch.Tensor]):
        """Add observation to history buffer (FIFO)."""
        if self._history_buffer is None:
            self._init_history_buffer(obs)

        if self._history_len < self.inference_seq_len:
            # Buffer not full, just append
            self._history_buffer["image"][:, self._history_len] = obs["image"]
            self._history_buffer["state"][:, self._history_len] = obs["state"]
            self._history_len += 1
        else:
            # Buffer full, shift left and add new obs at end
            self._history_buffer["image"] = torch.roll(self._history_buffer["image"], -1, dims=1)
            self._history_buffer["state"] = torch.roll(self._history_buffer["state"], -1, dims=1)
            self._history_buffer["image"][:, -1] = obs["image"]
            self._history_buffer["state"][:, -1] = obs["state"]

    def _get_history_obs(self) -> Dict[str, torch.Tensor]:
        """Get current history as sequence observation."""
        # Return only the filled portion
        return {
            "image": self._history_buffer["image"][:, :self._history_len],
            "state": self._history_buffer["state"][:, :self._history_len],
        }

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale action from [-1, 1] to actual action space."""
        return action * self.action_scale + self.action_bias

    def unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Unscale action from actual action space to [-1, 1]."""
        return (action - self.action_bias) / self.action_scale

    def forward_sequence_from_features(
        self, cnn_features: torch.Tensor, states: torch.Tensor, return_context: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Process sequence using pre-computed CNN features (for feature caching).

        Args:
            cnn_features: (B, T, cnn_output_dim) - pre-computed CNN features
            states: (B, T, n_stack, state_dim) - raw state observations

        Returns:
            mean: (B, T, action_dim)
            log_std: (B, T, action_dim)
            context: (B, T, hidden_dim), only when return_context=True
        """
        combined = self._combine_features(cnn_features, states, is_sequence=True)

        # Mamba processes full sequence
        context = self.mamba(combined, return_full_seq=True)  # (B, T, hidden_dim)
        context = self.context_norm(context)

        # Actor head for each timestep
        B, T, _ = context.shape
        context_flat = context.view(B * T, -1)
        features = self.head(context_flat)
        mean = self.mu(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        if return_context:
            return mean.view(B, T, -1), log_std.view(B, T, -1), context
        return mean.view(B, T, -1), log_std.view(B, T, -1)

    def action_log_prob_sequence_from_features(
        self, cnn_features: torch.Tensor, states: torch.Tensor, return_context: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Get actions and log probs using pre-computed CNN features.

        Args:
            cnn_features: (B, T, cnn_output_dim) - pre-computed CNN features
            states: (B, T, n_stack, state_dim) - raw state observations

        Returns:
            actions: (B, T, action_dim) - SCALED to actual action space
            log_probs: (B, T)
            context: (B, T, hidden_dim), only when return_context=True
        """
        if return_context:
            mean, log_std, context = self.forward_sequence_from_features(cnn_features, states, return_context=True)
        else:
            mean, log_std = self.forward_sequence_from_features(cnn_features, states)
        B, T, _ = mean.shape

        mean_flat = mean.view(B * T, -1)
        log_std_flat = log_std.view(B * T, -1)

        self.action_dist.proba_distribution(mean_flat, log_std_flat)
        actions_flat = self.action_dist.get_actions(deterministic=False)
        log_probs_flat = self.action_dist.log_prob(actions_flat)

        actions_scaled = self.scale_action(actions_flat)

        if return_context:
            return actions_scaled.view(B, T, -1), log_probs_flat.view(B, T), context
        return actions_scaled.view(B, T, -1), log_probs_flat.view(B, T)

    def forward(self, observations: Dict[str, torch.Tensor], deterministic: bool = False, use_history: bool = True) -> torch.Tensor:
        """
        Single step forward for inference. Returns scaled actions.

        Args:
            observations: Current observation {image: (B, n_stack, ...), state: (B, n_stack, ...)}
            deterministic: Whether to use deterministic action
            use_history: If True, use history buffer for temporal context (recommended for deployment)
                        If False, process single observation only (faster but loses temporal info)
        """
        if use_history and self.inference_seq_len > 1:
            # Add current obs to history
            self._add_to_history(observations)

            # Get full history sequence
            history_obs = self._get_history_obs()  # (B, T, n_stack, ...)

            # Process as sequence
            combined = self._process_obs(history_obs, is_sequence=True)  # (B, T, combined_dim)
            context = self.mamba(combined, return_full_seq=True)  # (B, T, hidden_dim)
            context = self.context_norm(context)

            # Take last timestep's context
            context = context[:, -1, :]  # (B, hidden_dim)
        else:
            # Original single-step processing (no history)
            combined = self._process_obs(observations, is_sequence=False)  # (B, 1, combined_dim)
            context = self.mamba(combined, return_full_seq=False)  # (B, hidden_dim)
            context = self.context_norm(context)

        features = self.head(context)
        mean = self.mu(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.action_dist.proba_distribution(mean, log_std)
        actions = self.action_dist.get_actions(deterministic=deterministic)  # [-1, 1]
        return self.scale_action(actions)  # Scale to actual action space

    def action_log_prob(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step action with log prob. Returns scaled actions."""
        combined = self._process_obs(observations, is_sequence=False)

        context = self.mamba(combined, return_full_seq=False)
        context = self.context_norm(context)

        features = self.head(context)
        mean = self.mu(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.action_dist.proba_distribution(mean, log_std)
        actions = self.action_dist.get_actions(deterministic=False)  # [-1, 1]
        log_prob = self.action_dist.log_prob(actions)
        return self.scale_action(actions), log_prob  # Return scaled actions


class MambaCritic(_ObservationEncoder):
    """
    Critic network with Mamba for sequence processing.
    Supports both single-step inference and sequence-based training.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        cnn_output_dim: int = 64,
        mamba_hidden_dim: int = 256,
        critic_hidden: List[int] = [256, 256],
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        n_critics: int = 2,
        force_lstm: bool = False,
        backend: str = "auto",
        transformer_heads: int = 4,
        transformer_layers: int = 2,
    ):
        super().__init__()

        self.action_dim = action_space.shape[0]

        # Action normalization: scale actions to [-1, 1] for critic input
        self.register_buffer("action_low", torch.tensor(action_space.low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_space.high, dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)

        # Parse observation space
        image_shape = observation_space["image"].shape
        state_shape = observation_space["state"].shape

        # Handle both n_stack > 1 and n_stack = 1 cases
        if len(image_shape) == 4:
            # (n_stack, C, H, W) - with frame stacking
            self.n_stack = image_shape[0]
        elif len(image_shape) == 3:
            # (C, H, W) - no frame stacking
            self.n_stack = 1
        else:
            raise ValueError(f"Unexpected image shape: {image_shape}")

        # State dim: (n_stack, state_dim) or (state_dim,)
        if len(state_shape) == 2:
            state_dim = state_shape[1]
        else:
            state_dim = state_shape[0]

        self.cnn_output_dim = cnn_output_dim

        self._cnn = None

        # LayerNorm after concat(visual, state) - unify scale, reduce training variance
        combined_dim = cnn_output_dim + state_dim
        self.input_norm = nn.LayerNorm(combined_dim)

        # Sequence modeling block (Mamba/LSTM/Transformer)
        self.mamba = MambaBlock(
            input_dim=combined_dim,
            hidden_dim=mamba_hidden_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            force_lstm=force_lstm,
            backend=backend,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )

        self.context_norm = nn.LayerNorm(mamba_hidden_dim)

        # Twin Q-networks
        self.q_networks = nn.ModuleList()
        for _ in range(n_critics):
            q_layers = []
            prev_dim = mamba_hidden_dim + self.action_dim
            for hidden_dim in critic_hidden:
                q_layers.append(nn.Linear(prev_dim, hidden_dim))
                q_layers.append(nn.LayerNorm(hidden_dim))
                q_layers.append(nn.ReLU())
                prev_dim = hidden_dim

            output_layer = nn.Linear(prev_dim, 1)
            nn.init.uniform_(output_layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(output_layer.bias, -3e-3, 3e-3)
            q_layers.append(output_layer)
            self.q_networks.append(nn.Sequential(*q_layers))

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action from actual action space to [-1, 1] for critic input."""
        return (action - self.action_bias) / self.action_scale

    def forward_sequence_from_features(self, cnn_features: torch.Tensor, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process sequence using pre-computed CNN features (for feature caching).

        Args:
            cnn_features: (B, T, cnn_output_dim) - pre-computed CNN features
            states: (B, T, n_stack, state_dim) - raw state observations
            actions: (B, T, action_dim) - SCALED actions

        Returns:
            q1, q2: (B, T) Q-values for each timestep
        """
        combined = self._combine_features(cnn_features, states, is_sequence=True)

        # Mamba processes full sequence
        context = self.mamba(combined, return_full_seq=True)  # (B, T, hidden_dim)
        context = self.context_norm(context)

        B, T, _ = context.shape

        # Normalize actions to [-1, 1] for consistent input scale
        actions_norm = self.normalize_action(actions)

        # Concatenate with normalized actions
        q_input = torch.cat([context, actions_norm], dim=-1)
        q_input_flat = q_input.view(B * T, -1)

        # Q-values
        q_values = []
        for q_net in self.q_networks:
            q = q_net(q_input_flat)
            q = torch.clamp(q, -1000.0, 1000.0)
            q_values.append(q.view(B, T))

        return tuple(q_values)

    def forward_sequence(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence.

        Args:
            observations: Dict with 'image' (B, T, n_stack, C, H, W), 'state' (B, T, n_stack, state_dim)
            actions: (B, T, action_dim) - SCALED actions (actual physical values)

        Returns:
            q1, q2: (B, T) Q-values for each timestep
        """
        combined = self._process_obs(observations, is_sequence=True)  # (B, T, combined_dim)

        # Mamba processes full sequence
        context = self.mamba(combined, return_full_seq=True)  # (B, T, hidden_dim)
        context = self.context_norm(context)

        B, T, _ = context.shape

        # Normalize actions to [-1, 1] for consistent input scale
        actions_norm = self.normalize_action(actions)

        # Concatenate with normalized actions
        q_input = torch.cat([context, actions_norm], dim=-1)  # (B, T, hidden_dim + action_dim)
        q_input_flat = q_input.view(B * T, -1)

        # Q-values
        q_values = []
        for q_net in self.q_networks:
            q = q_net(q_input_flat)  # (B*T, 1)
            q = torch.clamp(q, -1000.0, 1000.0)
            q_values.append(q.view(B, T))

        return tuple(q_values)

    def forward(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Single step forward."""
        combined = self._process_obs(observations, is_sequence=False)  # (B, 1, combined_dim)

        context = self.mamba(combined, return_full_seq=False)  # (B, hidden_dim)
        context = self.context_norm(context)

        # Normalize actions to [-1, 1] for consistent input scale
        actions_norm = self.normalize_action(actions)

        q_input = torch.cat([context, actions_norm], dim=-1)

        q_values = []
        for q_net in self.q_networks:
            q = q_net(q_input)
            q = torch.clamp(q, -1000.0, 1000.0)
            q_values.append(q)

        return tuple(q_values)


class MambaSACPolicy(SACPolicy):
    """
    SAC Policy with Mamba for sequence-based temporal learning.

    Args:
        force_lstm: Use LSTM instead of Mamba (for baseline comparison)
        backend: Sequence model backend ("auto", "mamba_ssm", "mambapy", "lstm", "transformer")
        cnn_update_mode: "critic_only" or "both"
            - critic_only: Only Critic optimizer updates CNN (original design)
            - both: Both Actor and Critic update CNN (may improve Actor learning)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        cnn_output_dim: int = 64,
        cnn_attention: str = "none",
        cnn_attention_reduction: int = 8,
        cnn_attention_stage: str = "auto",
        mamba_hidden_dim: int = 128,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        actor_hidden: List[int] = [128, 64],
        critic_hidden: List[int] = [128, 128],
        force_lstm: bool = False,
        inference_seq_len: int = 8,
        cnn_update_mode: str = "both",  # "critic_only" or "both"
        backend: str = "auto",  # "auto", "mamba_ssm", "mambapy", "lstm", "transformer"
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        aux_weight: float = 0.0,
        aux_target: str = "next_depth",
        aux_out_h: int = 15,
        aux_out_w: int = 22,
        aux_hidden: int = 256,
    ):
        self.cnn_output_dim = cnn_output_dim
        self.cnn_attention = cnn_attention
        self.cnn_attention_reduction = cnn_attention_reduction
        self.cnn_attention_stage = cnn_attention_stage
        self.mamba_hidden_dim = mamba_hidden_dim
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.actor_hidden = actor_hidden
        self.critic_hidden = critic_hidden
        self.n_critics = n_critics
        self.force_lstm = force_lstm
        self.inference_seq_len = inference_seq_len
        self.cnn_update_mode = cnn_update_mode
        self.backend = backend
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.aux_weight = aux_weight
        self.aux_target = aux_target
        self.aux_out_h = aux_out_h
        self.aux_out_w = aux_out_w
        self.aux_hidden = aux_hidden

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=False,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Build Actor, Critic, and shared CNN."""
        # Determine backend name for logging
        if self.backend == "transformer":
            backend_name = "Transformer"
        elif self.backend == "lstm" or self.force_lstm:
            backend_name = "LSTM"
        else:
            backend_name = f"Mamba({self.backend})"
        print(f"  Building MambaSACPolicy (Sequence-based, backend={backend_name}):")

        # Parse observation space
        # With frame stacking (n_stack > 1): (n_stack, C, H, W) after VecTransposeImage
        # Without frame stacking (n_stack = 1): (C, H, W) after VecTransposeImage
        image_shape = self.observation_space["image"].shape

        if len(image_shape) == 4:
            # (n_stack, C, H, W) - with frame stacking
            n_stack = image_shape[0]
            n_channels, height, width = image_shape[1], image_shape[2], image_shape[3]
        elif len(image_shape) == 3:
            # (C, H, W) - no frame stacking, treat as n_stack=1
            n_stack = 1
            n_channels, height, width = image_shape[0], image_shape[1], image_shape[2]
        else:
            raise ValueError(f"Unexpected image shape: {image_shape}")

        # Shared CNN (for actor and critic)
        self.shared_cnn = SequenceCNN(
            n_channels,
            height,
            width,
            n_stack,
            self.cnn_output_dim,
            attention_type=self.cnn_attention,
            attention_reduction=self.cnn_attention_reduction,
            attention_stage=self.cnn_attention_stage,
        )
        attention_desc = "none"
        if self.shared_cnn.attention_type != "none":
            attention_desc = (
                f"{self.shared_cnn.attention_type.upper()}@"
                f"{self.shared_cnn.attention_stage.upper()}"
            )
        print(
            f"    Shared CNN: ({n_stack} x {n_channels}ch, {height}x{width}) -> "
            f"{self.cnn_output_dim}, attention={attention_desc}"
        )

        # Target CNN (independent, updated via Polyak)
        # This prevents target Q from drifting with encoder gradients
        self.target_cnn = SequenceCNN(
            n_channels,
            height,
            width,
            n_stack,
            self.cnn_output_dim,
            attention_type=self.cnn_attention,
            attention_reduction=self.cnn_attention_reduction,
            attention_stage=self.cnn_attention_stage,
        )
        self.target_cnn.load_state_dict(self.shared_cnn.state_dict())
        self.target_cnn.requires_grad_(False)
        print(f"    Target CNN: independent copy (Polyak updated)")

        # Actor
        print(f"    Actor: {backend_name}(hidden={self.mamba_hidden_dim}) -> MLP{self.actor_hidden}")
        print(f"    Inference history length: {self.inference_seq_len}")
        self.actor = MambaActor(
            self.observation_space,
            self.action_space,
            cnn_output_dim=self.cnn_output_dim,
            mamba_hidden_dim=self.mamba_hidden_dim,
            actor_hidden=self.actor_hidden,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            force_lstm=self.force_lstm,
            inference_seq_len=self.inference_seq_len,
            backend=self.backend,
            transformer_heads=self.transformer_heads,
            transformer_layers=self.transformer_layers,
        )
        self.actor.set_cnn(self.shared_cnn)

        # set_cnn registers the shared encoder as a submodule. Keep the original
        # optimizer parameter order so existing Adam checkpoints resume exactly.
        actor_params = list(self.actor.parameters())
        print(f"    CNN update mode: {self.cnn_update_mode} (actor encoder gradients are auxiliary-only)")

        # Critic
        print(f"    Critic: {backend_name}(hidden={self.mamba_hidden_dim}) -> MLP{self.critic_hidden}")
        self.critic = MambaCritic(
            self.observation_space,
            self.action_space,
            cnn_output_dim=self.cnn_output_dim,
            mamba_hidden_dim=self.mamba_hidden_dim,
            critic_hidden=self.critic_hidden,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            n_critics=self.n_critics,
            force_lstm=self.force_lstm,
            backend=self.backend,
            transformer_heads=self.transformer_heads,
            transformer_layers=self.transformer_layers,
        )
        self.critic.set_cnn(self.shared_cnn)

        # Critic target (uses independent target_cnn for stable target Q)
        self.critic_target = MambaCritic(
            self.observation_space,
            self.action_space,
            cnn_output_dim=self.cnn_output_dim,
            mamba_hidden_dim=self.mamba_hidden_dim,
            critic_hidden=self.critic_hidden,
            mamba_d_state=self.mamba_d_state,
            mamba_d_conv=self.mamba_d_conv,
            mamba_expand=self.mamba_expand,
            n_critics=self.n_critics,
            force_lstm=self.force_lstm,
            backend=self.backend,
            transformer_heads=self.transformer_heads,
            transformer_layers=self.transformer_layers,
        )
        self.critic_target.set_cnn(self.target_cnn)  # Use independent target CNN
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        # The critic loss updates the encoder in both supported modes.
        critic_params = list(self.critic.parameters())
        self.critic.optimizer = self.optimizer_class(
            critic_params,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Build the optional head last so it cannot change seeded SAC initialization.
        if self.aux_weight > 0:
            if self.aux_target == "terminal_risk":
                self.aux_predictor = nn.Sequential(
                    nn.Linear(self.mamba_hidden_dim, self.aux_hidden),
                    nn.SiLU(),
                    nn.Linear(self.aux_hidden, 2),
                )
            else:
                self.aux_predictor = NextDepthPredictor(
                    self.mamba_hidden_dim, self.action_space.shape[0],
                    self.aux_out_h, self.aux_out_w, self.aux_hidden, self.aux_target,
                )
            actor_params += list(self.aux_predictor.parameters())
            if self.cnn_update_mode == "critic_only":
                print("    [Aux] Warning: critic_only freezes the CNN during the Actor auxiliary update")

        self.actor.optimizer = self.optimizer_class(
            actor_params,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Note: Entropy coefficient (log_ent_coef) is initialized in MambaSAC._setup_model()
        # to avoid duplicate initialization. Policy only needs target_entropy for reference.
        self.target_entropy = float(-self.action_space.shape[0])

    def _predict(self, observation: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        return self.actor(observation, deterministic=deterministic)

    def set_training_mode(self, mode: bool) -> None:
        self.actor.train(mode)
        self.critic.train(mode)
        if self.aux_weight > 0:
            self.aux_predictor.train(mode)
        self.training = mode

    def extract_cnn_features(self, observations: Dict[str, torch.Tensor], use_target: bool = False) -> torch.Tensor:
        """
        Extract CNN features from observations (for feature caching optimization).

        Args:
            observations: Dict with 'image'
                - n_stack > 1: (B, T, n_stack, C, H, W)
                - n_stack = 1: (B, T, C, H, W)
            use_target: If True, use target_cnn; else use shared_cnn

        Returns:
            cnn_features: (B, T, cnn_output_dim)
        """
        images = observations["image"].float() / 255.0

        cnn = self.target_cnn if use_target else self.shared_cnn
        return cnn(images)
