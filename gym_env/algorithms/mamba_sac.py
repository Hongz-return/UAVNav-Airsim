"""
Mamba-SAC: SAC with Sequence Replay Buffer for proper temporal learning.
"""
import numpy as np
import random
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Type, Union, Any
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise

from gym_env.algorithms.mamba_history import HistoryWindows
from gym_env.replay.sequence_buffer import SequenceReplayBuffer
from gym_env.networks.mamba_policy import MambaSACPolicy
from tools.training_logging import record_mamba_rollout_metrics, record_mamba_training_metrics


class MambaSAC(BaseAlgorithm):
    """
    Soft Actor-Critic with Sequence Replay Buffer for Mamba networks.
    
    Key differences from standard SAC:
    - Uses SequenceReplayBuffer to sample continuous trajectories
    - Mamba processes full sequences, not single transitions
    - TD targets computed across sequence with proper masking
    """
    
    policy_aliases = {"MambaSACPolicy": MambaSACPolicy}
    
    def __init__(
        self,
        policy: Union[str, Type[MambaSACPolicy]],
        env: GymEnv,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        batch_size: int = 32,
        seq_len: int = 16,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        critic_loss: str = "huber",
        failure_horizon: int = 30,
        failure_rewards: Tuple[float, float] = (-25.0, -20.0),
        fast_history: bool = False,
        overlap_updates: bool = False,
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            support_multi_env=True,
            supported_action_spaces=(spaces.Box,),
        )
        
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self._rollout_episode_progress = (0.0, 0) if self.n_envs == 1 else [(0.0, 0)] * self.n_envs
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        if critic_loss not in {"huber", "mse"}:
            raise ValueError("critic_loss must be 'huber' or 'mse'")
        self.critic_loss = critic_loss
        self.failure_horizon = failure_horizon
        self.failure_rewards = failure_rewards
        self.fast_history = fast_history
        self.overlap_updates = overlap_updates
        
        # Q-value clipping bound to prevent explosion during training.
        # With gamma=0.99, reward in [-2,2], max theoretical Q is ~200.
        # We set it slightly higher to account for terminal rewards (+10).
        self.q_value_bound = getattr(self, 'q_value_bound', 300.0)
        
        self.ent_coef_init = ent_coef
        self.target_entropy_init = target_entropy
        self.demo_action_anchor = None
        self.demo_terminal_altitude_weight = 0.0
        self.demo_terminal_precision_weight = 0.0
        self.demo_replay_fraction = 0.0
        
        if _init_setup_model:
            self._setup_model()
    
    def _setup_model(self) -> None:
        """Setup networks and buffer."""
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        
        # Create replay buffer
        self.replay_buffer = SequenceReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            seq_len=self.seq_len,
            device=self.device,
            n_envs=self.n_envs,
        )
        
        # Create policy
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        
        # Setup entropy coefficient
        if self.target_entropy_init == "auto":
            self.target_entropy = float(-np.prod(self.action_space.shape))
        else:
            self.target_entropy = float(self.target_entropy_init)
        
        if self.ent_coef_init == "auto":
            # Initialize to 0.1 (log(0.1) ~= -2.3) for reasonable exploration
            init_value = torch.log(torch.tensor([0.1], device=self.device))
            self.log_ent_coef = init_value.clone().requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            self.log_ent_coef = None
            self.ent_coef = float(self.ent_coef_init)
    
    @property
    def ent_coef(self) -> float:
        if self.log_ent_coef is not None:
            return torch.exp(self.log_ent_coef).item()
        return self._ent_coef
    
    @ent_coef.setter
    def ent_coef(self, value: float):
        self._ent_coef = value

    @staticmethod
    def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: Union[torch.device, str]) -> None:
        """Move optimizer internal tensor states to the target device."""
        target_device = torch.device(device)
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(target_device)

    def _compute_aux_loss(self, context: torch.Tensor, batch: Dict) -> torch.Tensor:
        """Auxiliary prediction loss at valid, observable training positions."""
        if self.policy.aux_target == "terminal_risk":
            logits = self.policy.aux_predictor(context)
            errors = F.binary_cross_entropy_with_logits(
                logits, batch["failure_targets"], reduction="none",
                pos_weight=logits.new_tensor([5.0, 8.0]),
            ).mean(dim=-1)
            known = batch["failure_mask"]
            return (known * errors).sum() / known.sum().clamp_min(1.0)

        # A terminal fallback frame cannot supervise next-depth prediction.
        with torch.no_grad():
            def pool_depth(images):
                # (B,T,C,H,W), or use the newest frame of (B,T,stack,C,H,W).
                depth = images[:, :, -1, 0] if self.policy.actor.n_stack > 1 else images[:, :, 0]
                B, T, H, W = depth.shape
                pooled = F.adaptive_avg_pool2d(
                    depth.reshape(B * T, 1, H, W).float() / 255.0,
                    (self.policy.aux_out_h, self.policy.aux_out_w),
                )
                return pooled.reshape(B, T, self.policy.aux_out_h, self.policy.aux_out_w)

            if 'next_observations' in batch:
                following = batch['next_observations']['image']
                current = batch['observations']['image']
            else:
                frames = batch['context_observations']['image']
                steps = batch['mask'].shape[1]
                following = frames[:, -steps:]
                current = frames[:, -steps - 1:-1]
            target = pool_depth(following)
            if self.policy.aux_target == 'depth_delta':
                target = target - pool_depth(current)
            actions = self.policy.actor.unscale_action(batch['actions']).detach()
            aux_mask = batch['mask'] * (1 - batch['dones'])

        prediction = self.policy.aux_predictor(context, actions)
        mse_per_step = (prediction - target).pow(2).mean(dim=(-2, -1))
        return (aux_mask * mse_per_step).sum() / aux_mask.sum().clamp(min=1.0)
    
    def _window_indices(self, history_lengths, device, next_step=False):
        """Build reusable gather indices from the replay sampler's CPU metadata."""
        return HistoryWindows(history_lengths, self.policy.actor.inference_seq_len, device, next_step)

    def _actor_history_parameters(self, features, states, history_lengths,
                                  next_step=False, return_context=False, plan=None):
        """Evaluate the actor at the last position of each real history window."""
        batch_size, steps = history_lengths.shape
        actor = self.policy.actor
        plan = plan if plan is not None else self._window_indices(history_lengths, features.device, next_step)
        if self.fast_history and actor.mamba.supports_batched_last(plan.history):
            window_features, window_states = plan.gather(features, states)
            context = actor.masked_last_context(window_features, window_states, plan.device_lengths)
            means, log_stds = actor.distribution_parameters(context)
            output = (means.reshape(batch_size, steps, -1), log_stds.reshape(batch_size, steps, -1))
            return (*output, context.reshape(batch_size, steps, -1)) if return_context else output

        means = features.new_zeros(batch_size * steps, actor.action_dim)
        log_stds = torch.zeros_like(means)
        contexts = features.new_zeros(batch_size * steps, self.policy.mamba_hidden_dim) if return_context else None
        for positions, window_features, window_states in plan.gather_groups(features, states):
            mean, log_std, context = actor.forward_sequence_from_features(
                window_features, window_states, return_context=True,
            )
            means = means.index_copy(0, positions, mean[:, -1])
            log_stds = log_stds.index_copy(0, positions, log_std[:, -1])
            if return_context:
                contexts = contexts.index_copy(0, positions, context[:, -1])
        output = (means.reshape(batch_size, steps, -1), log_stds.reshape(batch_size, steps, -1))
        if return_context:
            return (*output, contexts.reshape(batch_size, steps, -1))
        return output

    def _sample_history_actions(self, features, states, history_lengths,
                                next_step=False, return_context=False, plan=None):
        parameters = self._actor_history_parameters(
            features, states, history_lengths, next_step, return_context, plan,
        )
        means, log_stds = parameters[:2]
        batch_size, steps, action_dim = means.shape
        actor = self.policy.actor
        actor.action_dist.proba_distribution(means.reshape(-1, action_dim), log_stds.reshape(-1, action_dim))
        normalized_actions = actor.action_dist.get_actions(deterministic=False)
        log_probs = actor.action_dist.log_prob(normalized_actions).reshape(batch_size, steps)
        actions = actor.scale_action(normalized_actions).reshape(batch_size, steps, action_dim)
        if return_context:
            return actions, log_probs, parameters[2]
        return actions, log_probs

    def _critic_history_values(self, critic, features, states, actions, history_lengths, next_step=False, plan=None):
        batch_size, steps = history_lengths.shape
        flat_actions = actions.reshape(batch_size * steps, -1)
        plan = plan if plan is not None else self._window_indices(history_lengths, features.device, next_step)
        if self.fast_history and critic.mamba.supports_batched_last(plan.history):
            window_features, window_states = plan.gather(features, states)
            context = critic.masked_last_context(window_features, window_states, plan.device_lengths)
            return tuple(value.reshape(batch_size, steps)
                         for value in critic.values_from_context(context, flat_actions))

        values = [features.new_zeros(batch_size * steps) for _ in critic.q_networks]
        for positions, window_features, window_states in plan.gather_groups(features, states):
            # The critic's temporal block encodes observations only. Actions
            # enter its per-position Q head; only the last action/output is used.
            window_actions = flat_actions.index_select(0, positions).unsqueeze(1).expand(-1, window_features.shape[1], -1)
            outputs = critic.forward_sequence_from_features(window_features, window_states, window_actions)
            values = [value.index_copy(0, positions, output[:, -1]) for value, output in zip(values, outputs)]
        return tuple(value.reshape(batch_size, steps) for value in values)

    def _compute_td_targets(self, batch, ent_coef, features=None):
        with torch.no_grad():
            context = batch['context_observations']
            if features is None:
                features = self.policy.extract_cnn_features(context)
            target_features = self.policy.extract_cnn_features(context, use_target=True)
            plan = self._window_indices(batch['history_lengths'], features.device, next_step=True)
            next_actions, next_log_probs = self._sample_history_actions(
                features, context['state'], batch['history_lengths'], next_step=True, plan=plan,
            )
            next_values = self._critic_history_values(
                self.policy.critic_target, target_features, context['state'], next_actions,
                batch['history_lengths'], next_step=True, plan=plan,
            )
            next_q = torch.min(*next_values).clamp(-self.q_value_bound, self.q_value_bound)
            return batch['rewards'] + self.gamma * (1 - batch['dones']) * (
                next_q - ent_coef * next_log_probs.clamp(-20, 20)
            )

    def _critic_position_error(self, error: torch.Tensor) -> torch.Tensor:
        """Keep Huber's quadratic coefficient and the caller's masked reduction."""
        if self.critic_loss == "mse":
            return 0.5 * error.pow(2)
        return torch.where(error.abs() <= 1.0, 0.5 * error.pow(2), error.abs() - 0.5)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Train policy using sequence batches with feature caching optimization.
        
        Action scaling convention:
        - Buffer stores SCALED actions (actual physical values: v_xy, v_z, yaw_rate)
        - Critic receives SCALED actions
        - Actor outputs SCALED actions, log_prob computed on unscaled [-1,1] internally
        
        Optimizations:
        - CNN features cached for the joined history/current/next observations
        - Each loss position uses the same sliding history as online inference
        - Critic parameters frozen while retaining actor action gradients
        - Efficient masked loss computation
        """
        self._update_learning_rate([self.policy.actor.optimizer, self.policy.critic.optimizer])
        
        metric_rows = []
        use_aux = self.policy.aux_weight > 0
        sample_options = {}
        if use_aux and self.policy.aux_target == "terminal_risk":
            sample_options.update(failure_horizon=self.failure_horizon, failure_rewards=self.failure_rewards)
        if self.demo_action_anchor is not None:
            sample_options["include_episode_indices"] = True
            if self.demo_replay_fraction > 0:
                sample_options["demo_prefix"] = (self.demo_action_anchor[1], self.demo_replay_fraction)

        aux_updates_encoder = use_aux and self.policy.cnn_update_mode == "both"
        critic_clip = [p for group in self.policy.critic.optimizer.param_groups for p in group['params']]
        if self.policy.cnn_update_mode == "critic_only":
            # Historical critic_only runs counted this gradient twice in clipping.
            # Preserve their update scale and optimizer/checkpoint semantics.
            critic_clip += list(self.policy.shared_cnn.parameters())
        actor_clip = [p for group in self.policy.actor.optimizer.param_groups for p in group['params']]

        for _ in range(gradient_steps):
            # Sample sequence batch
            batch = self.replay_buffer.sample(
                batch_size, history_len=self.policy.actor.inference_seq_len, compact=True, **sample_options
            )
            
            metrics = self._train_batch(batch, use_aux, aux_updates_encoder, critic_clip, actor_clip)
            metric_rows.append(torch.stack(list(metrics.values())))
            self._n_updates += 1

        if metric_rows:
            columns = torch.stack(metric_rows).T.cpu().tolist()
            summary = dict(zip(metrics, columns))
            if use_aux:
                self.logger.record('train/aux_loss', np.mean(summary.pop('aux_loss')))
            if "demo_action_loss" in summary:
                self.logger.record("train/demo_action_loss", np.mean(summary.pop("demo_action_loss")))
            summary.setdefault('ent_coef_losses', [])
            record_mamba_training_metrics(logger=self.logger, ent_coef=self.ent_coef, **summary)
            self.logger.record('train/n_updates', self._n_updates)

    def _train_batch(self, batch, use_aux, aux_updates_encoder, critic_clip, actor_clip):
        """Apply one critic, actor, entropy and target update to already sampled data."""
        actions = batch['actions']  # (B, T, action_dim) - SCALED (actual values from env)
        mask = batch['mask']  # (B, T)
        context_obs = batch['context_observations']
        history_lengths = batch['history_lengths']
        plan = self._window_indices(history_lengths, mask.device)

        # Defensive fix for resumed checkpoints: keep entropy tensor/optimizer on train device.
        if self.log_ent_coef is not None and self.log_ent_coef.device != mask.device:
            self.log_ent_coef.data.copy_(self.log_ent_coef.data.to(mask.device))
            self._move_optimizer_state_to_device(self.ent_coef_optimizer, mask.device)

        ent_coef = self.log_ent_coef.detach().exp() if self.log_ent_coef is not None else self.ent_coef
        mask_sum = mask.sum().clamp(min=1.0)  # Avoid division by zero

        # ========== Critic Update ==========
        # Both paths see the same online weights before the critic update.
        # The target critic still encodes its input with its own Polyak CNN.
        critic_features = self.policy.extract_cnn_features(context_obs)
        td_targets = self._compute_td_targets(batch, ent_coef, critic_features.detach())
        q1_values, q2_values = self._critic_history_values(
            self.policy.critic, critic_features, context_obs['state'], actions, history_lengths, plan=plan,
        )

        # Clip Q values to prevent explosion during training
        q1_values = q1_values.clamp(-self.q_value_bound, self.q_value_bound)
        q2_values = q2_values.clamp(-self.q_value_bound, self.q_value_bound)

        # Only the per-position function changes; retain the exact reduction.
        q1_diff = q1_values - td_targets
        q2_diff = q2_values - td_targets
        q1_huber = self._critic_position_error(q1_diff)
        q2_huber = self._critic_position_error(q2_diff)
        critic_loss = ((mask * q1_huber).sum() + (mask * q2_huber).sum()) / (2 * mask_sum)

        self.policy.critic.optimizer.zero_grad()
        critic_loss.backward()

        critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_clip, 1.0)
        self.policy.critic.optimizer.step()

        # Both modules register the same encoder. SAC actor gradients stop at
        # its features. Only the auxiliary loss may update it in "both" mode.
        self.policy.critic.requires_grad_(False)
        try:
            self.policy.shared_cnn.requires_grad_(aux_updates_encoder)
            obs_features_cached = self.policy.extract_cnn_features(context_obs)
            if aux_updates_encoder:
                aux_cnn_features = obs_features_cached
                obs_features_cached = obs_features_cached.detach().requires_grad_(True)

            # Actor samples new actions using cached features
            if use_aux or self.demo_action_anchor is not None:
                actions_pi, log_probs, context = self._sample_history_actions(
                    obs_features_cached, context_obs['state'], history_lengths, return_context=True, plan=plan
                )
            else:
                actions_pi, log_probs = self._sample_history_actions(
                    obs_features_cached, context_obs['state'], history_lengths, plan=plan
                )
            # Clip log_probs to prevent extreme values
            log_probs = log_probs.clamp(-20, 20)

            # Critic parameters are frozen. Post-update observation context is
            # shared by both Q heads. Only the path from Q to actions is needed.
            q1_pi, q2_pi = self._critic_history_values(
                self.policy.critic, obs_features_cached.detach(), context_obs['state'], actions_pi,
                history_lengths, plan=plan,
            )
            min_q_pi = torch.min(q1_pi, q2_pi).clamp(-300, 300)

            actor_loss = (mask * (ent_coef * log_probs - min_q_pi)).sum() / mask_sum
            if self.demo_action_anchor is not None:
                demo_loss = self._demo_action_loss(context, actions, mask, batch["episode_indices"],
                                                   batch["steps_remaining"])
                actor_loss = actor_loss + demo_loss
            if use_aux:
                aux_loss = self._compute_aux_loss(context, batch)
                actor_loss = actor_loss + self.policy.aux_weight * aux_loss
                if aux_updates_encoder:
                    aux_feature_grad, = torch.autograd.grad(
                        self.policy.aux_weight * aux_loss, obs_features_cached, retain_graph=True,
                    )

            self.policy.actor.optimizer.zero_grad()
            actor_loss.backward()
            if aux_updates_encoder:
                aux_cnn_features.backward(aux_feature_grad)

            actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor_clip, 1.0)
            self.policy.actor.optimizer.step()
        finally:
            self.policy.critic.requires_grad_(True)

        # ========== Entropy Coefficient Update ==========
        if self.log_ent_coef is not None:
            # log_probs already detached implicitly since actor update is done
            ent_coef_loss = -(self.log_ent_coef * (mask * (log_probs.detach() + self.target_entropy))).sum() / mask_sum

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

        # ========== Target Network Update ==========
        # critic_target owns target_cnn, so its parameters are updated here too.
        polyak_update(self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau)

        # Reduce diagnostics on-device. Transfer the scalar rows together
        # after all gradient steps, independently of the training graph.
        with torch.no_grad():
            q_mean = (q1_values + q2_values) / 2

            def masked_mean(values):
                return (values * mask).sum().double() / mask_sum

            metrics = {
                'actor_losses': actor_loss, 'critic_losses': critic_loss,
                'q_values_list': masked_mean(q_mean), 'q_max_list': q_mean.max(),
                'q_min_list': q_mean.min(), 'td_errors_list': masked_mean((q1_diff.abs() + q2_diff.abs()) / 2),
                'log_probs_list': masked_mean(log_probs),
                'action_magnitudes': masked_mean(actions_pi.abs().mean(dim=-1)),
                'critic_grad_norms': critic_grad_norm, 'actor_grad_norms': actor_grad_norm,
                'mask_sums': mask_sum,
            }
            if use_aux:
                metrics['aux_loss'] = aux_loss
            if self.demo_action_anchor is not None:
                metrics["demo_action_loss"] = demo_loss
            if self.log_ent_coef is not None:
                metrics['ent_coef_losses'] = ent_coef_loss
        return metrics

    def _demo_action_loss(self, context, actions, mask, episode_indices, steps_remaining):
        """Anchor only the deterministic action, leaving SAC entropy adaptive."""
        roofed_end, demo_end, roofed_weight, corridor_weight = self.demo_action_anchor
        roofed = episode_indices < roofed_end
        corridor = (episode_indices >= roofed_end) & (episode_indices < demo_end)
        row_weight = roofed * roofed_weight + corridor * corridor_weight
        actor = self.policy.actor
        means, _ = actor.distribution_parameters(context)
        squared_error = (means.tanh() - actor.unscale_action(actions)).square()
        demo_steps = ((roofed | corridor)[:, None] * mask).sum().clamp_min(1)
        loss = (squared_error.mean(dim=-1) * mask * row_weight[:, None]).sum() / demo_steps
        if self.demo_terminal_altitude_weight > 0:
            finish = mask * corridor[:, None] * (steps_remaining <= 40)
            finish_loss = (squared_error[..., 1] * finish).sum() / finish.sum().clamp_min(1)
            loss = loss + self.demo_terminal_altitude_weight * finish_loss
        precision_weight = getattr(self, "demo_terminal_precision_weight", 0.0)
        if precision_weight > 0:
            finish_all = mask * (roofed | corridor)[:, None] * (steps_remaining <= 50)
            finish_all_loss = (squared_error.mean(dim=-1) * finish_all).sum() / finish_all.sum().clamp_min(1)
            loss = loss + precision_weight * finish_all_loss
        return loss

    def _updates_due(self):
        previous = max(self.num_timesteps, self.learning_starts)
        following = self.num_timesteps + self.n_envs
        return max(0, following // self.train_freq - previous // self.train_freq)

    def _replay_ready(self):
        return bool(self.replay_buffer.episodes) and self.replay_buffer.size() >= self.batch_size * self.seq_len

    def _can_overlap_updates(self):
        if not self._replay_ready():
            return False
        replay = self.replay_buffer
        # If adding the batch can evict episodes, their maximum length bounds
        # the drop in replay size. Keep marginal cases on the synchronous path.
        max_episode = min(replay.buffer_size // 2, 10000)
        return (replay.size() + self.n_envs <= replay.buffer_size
                or self.batch_size * self.seq_len <= replay.buffer_size - max_episode)

    def _step_and_update(self, action, updates):
        """Use committed replay while other processes advance the next vector batch."""
        if not (self.n_envs > 1 and self.overlap_updates and updates and self._can_overlap_updates()):
            return self.env.step(action), updates

        self.env.step_async(action)
        try:
            for _ in range(updates):
                self.train(self.gradient_steps, self.batch_size)
        except BaseException as exc:
            try:
                self.env.step_wait()
            except Exception as wait_error:
                exc.add_note(f"AirSim step also failed while unwinding: {wait_error!r}")
            raise
        return self.env.step_wait(), 0

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "MambaSAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        """Collect experience and train."""
        if self.overlap_updates and self.n_envs > 1 and total_timesteps % self.n_envs:
            raise ValueError("Overlapped training requires a budget divisible by the number of environments")

        pending = getattr(self, '_pending_random_state', None)
        if pending is not None:
            self._restore_random_state(pending)
            if self.n_envs == 1 and hasattr(self.env, "envs"):
                monitor = self.env.envs[0]
                if hasattr(monitor, 'pending_reset_origin') and 'environment' in pending:
                    monitor.pending_reset_origin = pending.get('episode_reset_origin', {
                        'seed': None, 'rng': pending['environment'],
                    })
            self._last_obs = None
        starting_episode = reset_num_timesteps or self._last_obs is None
        if starting_episode and self._last_obs is not None:
            # A fresh learn() reset truncates unfinished segments from this process.
            for env_idx, episode in enumerate(self.replay_buffer._active_episodes):
                if episode is not None:
                    episode.set_final_obs({key: value[env_idx].copy() for key, value in self._last_obs.items()})
                    self.replay_buffer._finish_episode(env_idx)
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        
        self._pending_random_state = None
        obs = self._last_obs  # _setup_learn performed the one required reset.
        if starting_episode:
            self.policy.actor.reset_history()
        callback.on_training_start(locals(), globals())

        # Episode tracking for logging
        episode_rewards = []
        episode_lengths = []
        if starting_episode:
            episode_progress = [(0.0, 0)] * self.n_envs
        elif self.n_envs == 1:
            episode_progress = [self._rollout_episode_progress]
        else:
            episode_progress = self._rollout_episode_progress.copy()
        
        while self.num_timesteps < total_timesteps:
            # Select action
            if self.num_timesteps < self.learning_starts:
                action = np.stack([self.action_space.sample() for _ in range(self.n_envs)])
            else:
                action, _ = self.predict(obs, deterministic=False, use_history=True)

            # An overlapped update samples completed episodes from the previous
            # committed batch. A batch that makes replay ready trains after insertion.
            result, remaining_updates = self._step_and_update(action, self._updates_due())
            next_obs, rewards, dones, infos = result
            previous_episode_num = self._episode_num
            for env_idx, (reward, done, info) in enumerate(zip(rewards, dones, infos, strict=True)):
                terminated = done and not info.get('TimeLimit.truncated', False)
                # VecEnv auto-resets. A timeout must bootstrap from the true terminal observation.
                actual_next_obs = info.get('terminal_observation') if done else None
                if not isinstance(actual_next_obs, dict):
                    actual_next_obs = {key: value[env_idx] for key, value in next_obs.items()}
                self.replay_buffer.add(
                    obs={key: value[env_idx] for key, value in obs.items()},
                    next_obs=actual_next_obs,
                    action=info.get("stage3_guided_action", action[env_idx]),
                    reward=float(reward),
                    done=terminated,
                    infos=info,
                    env_idx=env_idx,
                )

                episode_reward, episode_length = episode_progress[env_idx]
                episode_reward += reward
                episode_length += 1
                if done:
                    self._episode_num += 1
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    episode_progress[env_idx] = (0.0, 0)
                else:
                    episode_progress[env_idx] = (episode_reward, episode_length)

            obs = next_obs
            self._last_obs = obs
            self._last_episode_starts = dones.copy()
            self.num_timesteps += self.n_envs
            done_indices = np.flatnonzero(dones)
            if done_indices.size == self.n_envs:
                self.policy.actor.reset_history()
            elif done_indices.size:
                self.policy.actor.reset_history(done_indices)

            if remaining_updates and self._replay_ready():
                for _ in range(remaining_updates):
                    self.train(self.gradient_steps, self.batch_size)
            
            # Callback
            callback.update_locals(locals())
            if callback.on_step() is False:
                break
            
            # Logging - only when episode ends and at log_interval
            if (done_indices.size and log_interval is not None
                    and previous_episode_num // log_interval < self._episode_num // log_interval):
                record_mamba_rollout_metrics(
                    logger=self.logger,
                    episode_num=self._episode_num,
                    total_timesteps=self.num_timesteps,
                    buffer_size=self.replay_buffer.size(),
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                )
                self.logger.dump(step=self.num_timesteps)
        
        self._rollout_episode_progress = episode_progress[0] if self.n_envs == 1 else episode_progress
        callback.on_training_end()
        return self
    
    def predict(
        self,
        observation: Dict[str, np.ndarray],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
        use_history: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Get action from policy with history-aware inference.
        
        Args:
            observation: Current observation
            state: Not used (kept for API compatibility)
            episode_start: Boolean flag or mask of environments starting a new episode
            deterministic: Whether to use deterministic action
            use_history: Whether to use the actor's inference history buffer
        
        Returns:
            action: Selected action
            state: None (kept for API compatibility)
        """
        with torch.no_grad():
            # Check if observation is already batched (from VecEnv)
            first_key = next(iter(observation))
            is_batched = observation[first_key].ndim > len(self.observation_space[first_key].shape)
            obs_tensor = {
                key: torch.as_tensor(val, dtype=torch.float32, device=self.device)
                for key, val in observation.items()
            }
            if not is_batched:
                obs_tensor = {key: val.unsqueeze(0) for key, val in obs_tensor.items()}

            actor = self.policy.actor
            if episode_start is not None:
                mask = np.asarray(episode_start, dtype=bool)
                if mask.any():
                    if mask.all():
                        actor.reset_history()
                    elif actor._history_lens is None or actor._history_lens.size == obs_tensor['image'].shape[0]:
                        actor.reset_history(np.flatnonzero(mask))

            action = actor(
                obs_tensor,
                deterministic=deterministic,
                use_history=use_history,
            )
            action = action.cpu().numpy()
            
            if not is_batched:
                action = action[0]
        
        return action, state
    
    def save(self, path: str, include_buffer: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Save path (without extension)
            include_buffer: If True, save replay buffer for true resume (larger file)
        """
        checkpoint = {
            # Network weights
            'policy_state_dict': self.policy.state_dict(),
            # Optimizer states for proper resume
            'actor_optimizer_state': self.policy.actor.optimizer.state_dict(),
            'critic_optimizer_state': self.policy.critic.optimizer.state_dict(),
            # Entropy coefficient
            'log_ent_coef': self.log_ent_coef,
            'ent_coef_optimizer_state': self.ent_coef_optimizer.state_dict() if self.log_ent_coef is not None else None,
            # Training progress
            'num_timesteps': self.num_timesteps,
            '_episode_num': self._episode_num,
            '_n_updates': self._n_updates,
            # Config for validation
            'seq_len': self.seq_len,
            'batch_size': self.batch_size,
            'critic_loss': self.critic_loss,
        }
        
        # Optionally save replay buffer (for true checkpoint resume)
        if self.policy.aux_weight > 0 and self.policy.aux_target == "terminal_risk":
            checkpoint["failure_horizon"] = self.failure_horizon
            checkpoint["failure_rewards"] = self.failure_rewards
        if include_buffer:
            checkpoint['replay_buffer'] = self.replay_buffer.get_state()
            checkpoint['random_state'] = self._capture_random_state()
            if isinstance(self._last_obs, dict):
                continuations = [
                    {key: value[env_idx].copy() for key, value in self._last_obs.items()}
                    if episode is not None else None
                    for env_idx, episode in enumerate(self.replay_buffer._active_episodes)
                ]
                if any(obs is not None for obs in continuations):
                    checkpoint['continuation_observations'] = continuations
                    if continuations[0] is not None:
                        checkpoint['continuation_observation'] = continuations[0]
        
        torch.save(checkpoint, path + '.zip')

    def _capture_random_state(self) -> dict:
        state = {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'action_space': self.action_space.np_random.bit_generator.state,
        }
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
        if self.n_envs == 1 and hasattr(self.env, "envs"):
            monitor = self.env.envs[0]
            state['environment'] = monitor.unwrapped.np_random.bit_generator.state
            origin = getattr(monitor, 'reset_origin', None)
            if origin is not None and np.any(self._last_episode_starts) and self.replay_buffer.current_episode is None:
                state['episode_reset_origin'] = origin
        elif self.env is not None:
            state['environments'] = [rng.bit_generator.state for rng in self.env.get_attr('np_random')]
        return state

    def _restore_random_state(self, state: dict) -> None:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'].cpu())
        self.action_space.np_random.bit_generator.state = state['action_space']
        if 'torch_cuda' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in state['torch_cuda']])
        if 'environment' in state and self.n_envs == 1 and hasattr(self.env, "envs"):
            self.env.envs[0].unwrapped.np_random.bit_generator.state = state['environment']
        elif self.env is not None:
            env_states = state.get('environments', [state['environment']] if 'environment' in state else [])
            if len(env_states) == self.n_envs:
                generators = self.env.get_attr('np_random')
                for env_idx, (rng, rng_state) in enumerate(zip(generators, env_states, strict=True)):
                    rng.bit_generator.state = rng_state
                    self.env.set_attr('np_random', rng, indices=env_idx)
                # A queued seed would overwrite the restored generator on the next reset.
                self.env.unwrapped._reset_seeds()
    
    def save_full_checkpoint(self, path: str) -> None:
        """Save full checkpoint including replay buffer for true resume."""
        self.save(path, include_buffer=True)
    
    @classmethod
    def load(cls, path: str, env: GymEnv, load_buffer: bool = True, **kwargs):
        """
        Load model from checkpoint.
        
        Args:
            path: Checkpoint path
            env: Environment
            load_buffer: If True and buffer exists in checkpoint, restore it
            **kwargs: Additional arguments for model creation
        
        Returns:
            Loaded MambaSAC model
        """
        if not path.endswith('.zip'):
            path = path + '.zip'
        
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch versions do not support weights_only.
            checkpoint = torch.load(path, map_location="cpu")
        
        kwargs.setdefault('critic_loss', checkpoint.get('critic_loss', 'huber'))
        for setting in ("failure_horizon", "failure_rewards"):
            if setting in checkpoint:
                kwargs.setdefault(setting, checkpoint[setting])
        model = cls(
            policy=MambaSACPolicy,
            env=env,
            _init_setup_model=True,
            **kwargs,
        )
        
        # Load network weights
        model.policy.load_state_dict(checkpoint['policy_state_dict'])
        model.policy = model.policy.to(model.device)

        # Load optimizer states for proper resume
        if 'actor_optimizer_state' in checkpoint:
            model.policy.actor.optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
            model._move_optimizer_state_to_device(model.policy.actor.optimizer, model.device)
        if 'critic_optimizer_state' in checkpoint:
            model.policy.critic.optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
            model._move_optimizer_state_to_device(model.policy.critic.optimizer, model.device)

        # Load entropy coefficient
        if checkpoint.get('log_ent_coef') is not None and model.log_ent_coef is not None:
            saved_log_ent_coef = checkpoint['log_ent_coef'].detach().to(model.device)
            model.log_ent_coef.data.copy_(saved_log_ent_coef)
        if checkpoint.get('ent_coef_optimizer_state') is not None and model.log_ent_coef is not None:
            model.ent_coef_optimizer.load_state_dict(checkpoint['ent_coef_optimizer_state'])
            model._move_optimizer_state_to_device(model.ent_coef_optimizer, model.device)
        
        # Load training progress
        model.num_timesteps = checkpoint.get('num_timesteps', 0)
        model._episode_num = checkpoint.get('_episode_num', 0)
        model._n_updates = checkpoint.get('_n_updates', 0)
        
        # Load replay buffer if available and requested
        if load_buffer and 'replay_buffer' in checkpoint:
            continuations = checkpoint.get('continuation_observations', [checkpoint.get('continuation_observation')])
            # The new simulator sessions reset. Seal saved segments with their own bootstrap observations.
            model.replay_buffer.set_state(checkpoint['replay_buffer'], continuation_observations=continuations)
            model._pending_random_state = checkpoint.get('random_state')
            print(f"  Replay buffer restored: {model.replay_buffer.size()} transitions")
        else:
            print("  Replay buffer NOT restored (will start fresh)")
        
        print(f"  Resumed from timestep {model.num_timesteps}, episode {model._episode_num}")
        
        return model
    
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["replay_buffer"]
    
    def set_env(self, env: GymEnv) -> None:
        """Set environment."""
        super().set_env(env)
