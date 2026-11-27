"""SB3 adapter and optimizer construction for the MA-SAC policy."""

from typing import Any, Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.sac.policies import SACPolicy

# Keep these import paths available for existing callers and pickled policies.
from .actor_critic import MambaActor, MambaCritic, _ObservationEncoder
from .temporal import MambaBlock
from .vision import CoordAttention2d, HardSigmoid, HardSwish, SequenceCNN


def __getattr__(name):
    if name == "TimeDistributedCNN":
        from .mamba_extractor import TimeDistributedCNN

        return TimeDistributedCNN
    raise AttributeError(name)


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
