"""Temporal actor and critics with a shared observation layout."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.sac.policies import LOG_STD_MIN, LOG_STD_MAX

from .temporal import MambaBlock
from .vision import SequenceCNN


def _mlp_layers(input_dim, hidden_dims):
    layers = []
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()))
        input_dim = hidden_dim
    return layers, input_dim


class _ObservationEncoder(nn.Module):
    """Shared input layout for the actor and both critics."""

    def __init__(
        self, observation_space, action_space, cnn_output_dim, mamba_hidden_dim,
        mamba_d_state, mamba_d_conv, mamba_expand, force_lstm, backend,
        transformer_heads, transformer_layers,
    ):
        super().__init__()
        self.action_dim = action_space.shape[0]

        # Both heads use the same physical action bounds and normalized scale.
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

    def masked_last_context(self, cnn_features, states, lengths):
        combined = self._combine_features(cnn_features, states, is_sequence=True)
        return self.context_norm(self.mamba.forward_masked_last(combined, lengths))


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
        super().__init__(
            observation_space, action_space, cnn_output_dim, mamba_hidden_dim,
            mamba_d_state, mamba_d_conv, mamba_expand, force_lstm, backend,
            transformer_heads, transformer_layers,
        )
        self.inference_seq_len = inference_seq_len

        # Actor head
        head_layers, prev_dim = _mlp_layers(mamba_hidden_dim, actor_hidden)
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
        self._history_lens: Optional[np.ndarray] = None

    @property
    def _history_len(self) -> int:
        return 0 if self._history_lens is None else int(self._history_lens[0])

    def reset_history(self, env_indices=None):
        """Reset history for all environments or just the selected rows."""
        if env_indices is None:
            self._history_buffer = None
            self._history_lens = None
        elif self._history_buffer is not None:
            indices = np.asarray(env_indices)
            if indices.size:
                rows = torch.as_tensor(indices, device=self._history_buffer["image"].device)
                for history in self._history_buffer.values():
                    history[rows] = 0
                self._history_lens[indices] = 0

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
        batch_size = obs["image"].shape[0]
        self._history_buffer = {
            key: value.new_zeros((batch_size, self.inference_seq_len, *value.shape[1:]))
            for key, value in obs.items()
        }
        self._history_lens = np.zeros(batch_size, dtype=np.int64)

    def _add_to_history(self, obs: Dict[str, torch.Tensor]):
        """Add observation to history buffer (FIFO)."""
        if self._history_buffer is None or self._history_lens.size != obs["image"].shape[0]:
            self._init_history_buffer(obs)

        for index, length in enumerate(self._history_lens):
            if length < self.inference_seq_len:
                for key, history in self._history_buffer.items():
                    history[index, length] = obs[key][index]
                self._history_lens[index] += 1
            else:
                for key, history in self._history_buffer.items():
                    history[index] = torch.roll(history[index], -1, dims=0)
                    history[index, -1] = obs[key][index]

    def _get_history_obs(self) -> Dict[str, torch.Tensor]:
        """Get current history as sequence observation."""
        # Return only the filled portion
        return {
            key: history[:, :self._history_len] for key, history in self._history_buffer.items()
        }

    def _history_context(self, history_obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        combined = self._process_obs(history_obs, is_sequence=True)
        context = self.mamba(combined, return_full_seq=True)
        return self.context_norm(context)[:, -1, :]

    def scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Scale action from [-1, 1] to actual action space."""
        return action * self.action_scale + self.action_bias

    def unscale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Unscale action from actual action space to [-1, 1]."""
        return (action - self.action_bias) / self.action_scale

    def distribution_parameters(self, context):
        features = self.head(context)
        return self.mu(features), self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)

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
        mean, log_std = self.distribution_parameters(context.view(B * T, -1))

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
            self._add_to_history(observations)
            if np.all(self._history_lens == self._history_len):
                context = self._history_context(self._get_history_obs())
            else:
                context = None
                for length in np.unique(self._history_lens):
                    rows = torch.as_tensor(
                        np.flatnonzero(self._history_lens == length), device=observations["image"].device
                    )
                    history_obs = {
                        key: history[:, :length].index_select(0, rows)
                        for key, history in self._history_buffer.items()
                    }
                    group_context = self._history_context(history_obs)
                    if context is None:
                        context = group_context.new_empty((self._history_lens.size, group_context.shape[-1]))
                    context[rows] = group_context
        else:
            # Original single-step processing (no history)
            combined = self._process_obs(observations, is_sequence=False)  # (B, 1, combined_dim)
            context = self.mamba(combined, return_full_seq=False)  # (B, hidden_dim)
            context = self.context_norm(context)

        mean, log_std = self.distribution_parameters(context)

        self.action_dist.proba_distribution(mean, log_std)
        actions = self.action_dist.get_actions(deterministic=deterministic)  # [-1, 1]
        return self.scale_action(actions)  # Scale to actual action space

    def action_log_prob(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single step action with log prob. Returns scaled actions."""
        combined = self._process_obs(observations, is_sequence=False)

        context = self.mamba(combined, return_full_seq=False)
        context = self.context_norm(context)

        mean, log_std = self.distribution_parameters(context)

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
        super().__init__(
            observation_space, action_space, cnn_output_dim, mamba_hidden_dim,
            mamba_d_state, mamba_d_conv, mamba_expand, force_lstm, backend,
            transformer_heads, transformer_layers,
        )

        # Twin Q-networks
        self.q_networks = nn.ModuleList()
        for _ in range(n_critics):
            q_layers, prev_dim = _mlp_layers(mamba_hidden_dim + self.action_dim, critic_hidden)
            output_layer = nn.Linear(prev_dim, 1)
            nn.init.uniform_(output_layer.weight, -3e-3, 3e-3)
            nn.init.uniform_(output_layer.bias, -3e-3, 3e-3)
            q_layers.append(output_layer)
            self.q_networks.append(nn.Sequential(*q_layers))

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        """Normalize action from actual action space to [-1, 1] for critic input."""
        return (action - self.action_bias) / self.action_scale

    def values_from_context(self, context, actions):
        inputs = torch.cat([context, self.normalize_action(actions)], dim=-1)
        return tuple(q_net(inputs).clamp(-1000.0, 1000.0) for q_net in self.q_networks)

    def _sequence_values(self, combined, actions):
        context = self.context_norm(self.mamba(combined, return_full_seq=True))
        batch_size, steps, hidden_dim = context.shape
        values = self.values_from_context(
            context.view(batch_size * steps, hidden_dim), actions.reshape(batch_size * steps, -1),
        )
        return tuple(value.view(batch_size, steps) for value in values)

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
        return self._sequence_values(combined, actions)

    def forward_sequence(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a sequence.

        Args:
            observations: Dict with 'image' (B, T, n_stack, C, H, W), 'state' (B, T, n_stack, state_dim)
            actions: (B, T, action_dim) - SCALED actions (actual physical values)

        Returns:
            q1, q2: (B, T) Q-values for each timestep
        """
        combined = self._process_obs(observations, is_sequence=True)
        return self._sequence_values(combined, actions)

    def forward(self, observations: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Single step forward."""
        combined = self._process_obs(observations, is_sequence=False)
        context = self.context_norm(self.mamba(combined, return_full_seq=False))
        return self.values_from_context(context, actions)
