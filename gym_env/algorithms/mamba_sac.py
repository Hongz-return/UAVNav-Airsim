"""
Mamba-SAC: SAC with Sequence Replay Buffer for proper temporal learning.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Type, Union, Any
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, Schedule, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise

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
            support_multi_env=False,
            supported_action_spaces=(spaces.Box,),
        )
        
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        
        # Q-value clipping bound to prevent explosion during training.
        # With gamma=0.99, reward in [-2,2], max theoretical Q is ~200.
        # We set it slightly higher to account for terminal rewards (+10).
        self.q_value_bound = getattr(self, 'q_value_bound', 300.0)
        
        self.ent_coef_init = ent_coef
        self.target_entropy_init = target_entropy
        
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
    
    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Train policy using sequence batches with feature caching optimization.
        
        Action scaling convention:
        - Buffer stores SCALED actions (actual physical values: v_xy, v_z, yaw_rate)
        - Critic receives SCALED actions
        - Actor outputs SCALED actions, log_prob computed on unscaled [-1,1] internally
        
        Optimizations:
        - Feature caching: CNN features computed once per obs/next_obs, reused across actor/critic
        - Critic frozen during actor update (no redundant grad computation)
        - Detached Q values for actor loss
        - Efficient masked loss computation
        """
        self._update_learning_rate([self.policy.actor.optimizer, self.policy.critic.optimizer])
        
        actor_losses = []
        critic_losses = []
        ent_coef_losses = []
        
        # Enhanced diagnostics
        q_values_list = []
        q_max_list = []
        q_min_list = []
        log_probs_list = []
        action_magnitudes = []
        actor_grad_norms = []
        critic_grad_norms = []
        td_errors_list = []
        mask_sums = []
        
        for _ in range(gradient_steps):
            # Sample sequence batch
            batch = self.replay_buffer.sample(batch_size)
            
            obs = batch['observations']  # Dict: {key: (B, T, ...)}
            next_obs = batch['next_observations']
            actions = batch['actions']  # (B, T, action_dim) - SCALED (actual values from env)
            rewards = batch['rewards']  # (B, T)
            dones = batch['dones']  # (B, T)
            mask = batch['mask']  # (B, T)

            # Defensive fix for resumed checkpoints: keep entropy tensor/optimizer on train device.
            if self.log_ent_coef is not None and self.log_ent_coef.device != mask.device:
                self.log_ent_coef.data.copy_(self.log_ent_coef.data.to(mask.device))
                self._move_optimizer_state_to_device(self.ent_coef_optimizer, mask.device)
            
            ent_coef = self.ent_coef
            mask_sum = mask.sum().clamp(min=1.0)  # Avoid division by zero
            
            # ========== Feature Caching: Pre-compute CNN features ==========
            # This avoids redundant CNN forward passes during actor/critic updates
            # obs_features: used by critic (with grad) and actor (no grad on CNN)
            # next_obs_features: used by actor for next_actions (no grad)
            # next_obs_target_features: used by critic_target (no grad, uses target_cnn)
            
            # ========== Critic Update ==========
            with torch.no_grad():
                # Pre-compute features for target Q computation
                next_obs_features = self.policy.extract_cnn_features(next_obs, use_target=False)
                next_obs_target_features = self.policy.extract_cnn_features(next_obs, use_target=True)
                
                # Actor outputs scaled actions using cached features
                next_actions, next_log_probs = self.policy.actor.action_log_prob_sequence_from_features(
                    next_obs_features, next_obs['state']
                )
                # Critic target uses target_cnn features
                q1_next, q2_next = self.policy.critic_target.forward_sequence_from_features(
                    next_obs_target_features, next_obs['state'], next_actions
                )
                target_q_values = torch.min(q1_next, q2_next)
                
                # Clip target Q values to prevent explosion
                target_q_values = target_q_values.clamp(-self.q_value_bound, self.q_value_bound)
                next_log_probs = next_log_probs.clamp(-20, 20)  # Prevent extreme log probs
                
                td_targets = rewards + self.gamma * (1 - dones) * (target_q_values - ent_coef * next_log_probs)
            
            # Critic evaluates buffer actions (need grad for CNN update)
            # Note: We compute obs_features WITH grad here since critic optimizer updates CNN
            q1_values, q2_values = self.policy.critic.forward_sequence(obs, actions)
            
            # Clip Q values to prevent explosion during training
            q1_values = q1_values.clamp(-self.q_value_bound, self.q_value_bound)
            q2_values = q2_values.clamp(-self.q_value_bound, self.q_value_bound)
            
            # Use Huber loss instead of MSE for robustness to outliers
            # Huber loss is quadratic for small errors, linear for large errors
            q1_diff = q1_values - td_targets
            q2_diff = q2_values - td_targets
            # Huber loss with delta=1.0
            q1_huber = torch.where(q1_diff.abs() <= 1.0, 0.5 * q1_diff.pow(2), q1_diff.abs() - 0.5)
            q2_huber = torch.where(q2_diff.abs() <= 1.0, 0.5 * q2_diff.pow(2), q2_diff.abs() - 0.5)
            critic_loss = ((mask * q1_huber).sum() + (mask * q2_huber).sum()) / (2 * mask_sum)
            
            self.policy.critic.optimizer.zero_grad()
            critic_loss.backward()
            
            # Gradient clipping depends on cnn_update_mode
            cnn_update_mode = getattr(self.policy, 'cnn_update_mode', 'both')
            if cnn_update_mode == "both":
                # CNN is updated by Actor, so Critic only clips its own params
                critic_params = list(self.policy.critic.parameters())
            else:
                # critic_only: Critic updates CNN, so clip both
                critic_params = list(self.policy.critic.parameters()) + list(self.policy.shared_cnn.parameters())
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
            self.policy.critic.optimizer.step()
            critic_losses.append(critic_loss.item())
            
            # Collect critic diagnostics
            critic_grad_norms.append(critic_grad_norm.item())
            q_mean = (q1_values + q2_values) / 2
            q_values_list.append((q_mean * mask).sum().item() / mask_sum.item())
            q_max_list.append(q_mean.max().item())
            q_min_list.append(q_mean.min().item())
            td_errors_list.append(((q1_diff.abs() + q2_diff.abs()) / 2 * mask).sum().item() / mask_sum.item())
            
            # ========== Actor Update ==========
            # Freeze critic to avoid unnecessary gradient computation
            for param in self.policy.critic.parameters():
                param.requires_grad = False
            
            # CNN gradient flow depends on cnn_update_mode
            cnn_update_mode = getattr(self.policy, 'cnn_update_mode', 'both')
            if cnn_update_mode == "critic_only":
                # Freeze shared CNN - it's updated by critic optimizer only
                for param in self.policy.shared_cnn.parameters():
                    param.requires_grad = False
                # Pre-compute obs features (no grad since CNN is frozen)
                with torch.no_grad():
                    obs_features_cached = self.policy.extract_cnn_features(obs, use_target=False)
            else:
                # cnn_update_mode == "both": Actor also updates CNN
                # Compute features WITH gradient for CNN update
                obs_features_cached = self.policy.extract_cnn_features(obs, use_target=False)
            
            # Actor samples new actions using cached features
            actions_pi, log_probs = self.policy.actor.action_log_prob_sequence_from_features(
                obs_features_cached, obs['state']
            )
            # Clip log_probs to prevent extreme values
            log_probs = log_probs.clamp(-20, 20)
            
            # Critic evaluates actor's scaled actions using cached features
            q1_pi, q2_pi = self.policy.critic.forward_sequence_from_features(
                obs_features_cached, obs['state'], actions_pi
            )
            min_q_pi = torch.min(q1_pi, q2_pi).clamp(-300, 300)
            
            actor_loss = (mask * (ent_coef * log_probs - min_q_pi)).sum() / mask_sum
            
            self.policy.actor.optimizer.zero_grad()
            actor_loss.backward()
            
            # Gradient clipping: include CNN if Actor updates it
            if cnn_update_mode == "both":
                actor_params = list(self.policy.actor.parameters()) + list(self.policy.shared_cnn.parameters())
            else:
                actor_params = list(self.policy.actor.parameters())
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            self.policy.actor.optimizer.step()
            actor_losses.append(actor_loss.item())
            
            # Collect actor diagnostics
            actor_grad_norms.append(actor_grad_norm.item())
            log_probs_list.append((log_probs * mask).sum().item() / mask_sum.item())
            action_magnitudes.append((actions_pi.abs().mean(dim=-1) * mask).sum().item() / mask_sum.item())
            mask_sums.append(mask_sum.item())
            
            # Unfreeze critic and shared CNN
            for param in self.policy.critic.parameters():
                param.requires_grad = True
            if cnn_update_mode == "critic_only":
                for param in self.policy.shared_cnn.parameters():
                    param.requires_grad = True
            
            # ========== Entropy Coefficient Update ==========
            if self.log_ent_coef is not None:
                # log_probs already detached implicitly since actor update is done
                ent_coef_loss = -(self.log_ent_coef * (mask * (log_probs.detach() + self.target_entropy))).sum() / mask_sum
                
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                ent_coef_losses.append(ent_coef_loss.item())
            
            # ========== Target Network Update ==========
            # Update both critic_target AND target_cnn with Polyak averaging
            polyak_update(self.policy.critic.parameters(), self.policy.critic_target.parameters(), self.tau)
            polyak_update(self.policy.shared_cnn.parameters(), self.policy.target_cnn.parameters(), self.tau)
        
        record_mamba_training_metrics(
            logger=self.logger,
            ent_coef=self.ent_coef,
            actor_losses=actor_losses,
            critic_losses=critic_losses,
            ent_coef_losses=ent_coef_losses,
            q_values_list=q_values_list,
            q_max_list=q_max_list,
            q_min_list=q_min_list,
            td_errors_list=td_errors_list,
            log_probs_list=log_probs_list,
            action_magnitudes=action_magnitudes,
            actor_grad_norms=actor_grad_norms,
            critic_grad_norms=critic_grad_norms,
            mask_sums=mask_sums,
        )

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
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        
        callback.on_training_start(locals(), globals())
        
        obs = self.env.reset()
        
        # Reset actor history at episode start
        self.policy.actor.reset_history()
        
        # Episode tracking for logging
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0.0
        current_episode_length = 0
        
        while self.num_timesteps < total_timesteps:
            # Select action
            if self.num_timesteps < self.learning_starts:
                action = np.array([self.action_space.sample()])
            else:
                with torch.no_grad():
                    # obs from VecEnv is already batched: {key: (1, ...)}
                    obs_tensor = {
                        key: torch.tensor(val, dtype=torch.float32, device=self.device)
                        for key, val in obs.items()
                    }
                    # Training: use_history=True for temporal context during rollout
                    action = self.policy.actor(obs_tensor, deterministic=False, use_history=True)
                    action = action.cpu().numpy()
                    # SAC uses entropy regularization for exploration, no extra noise needed
            
            # Step environment (VecEnv returns 4 values: obs, reward, done, info)
            next_obs, rewards, dones, infos = self.env.step(action)
            
            reward = rewards[0]
            done = dones[0]
            info = infos[0]
            
            # Check if this is a true termination or just truncation (timeout)
            # Truncation allows Critic to bootstrap, termination does not
            truncated = info.get('TimeLimit.truncated', False)
            terminated = done and not truncated
            
            # CRITICAL: VecEnv auto-resets on done, so next_obs is the NEW episode's first obs
            # The TRUE terminal observation is stored in info['terminal_observation']
            # We must use terminal_observation for proper TD bootstrap on truncation
            if done and 'terminal_observation' in info:
                # Use the actual terminal observation, not the reset observation
                terminal_obs = info['terminal_observation']
                # terminal_observation is a dict for Dict obs space
                if isinstance(terminal_obs, dict):
                    actual_next_obs = {k: v for k, v in terminal_obs.items()}
                else:
                    actual_next_obs = {k: v[0] for k, v in next_obs.items()}
            else:
                actual_next_obs = {k: v[0] for k, v in next_obs.items()}
            
            # Store the policy output action exactly as the environment receives it.
            # The executed action equals action[0] in the current direct-control setup.
            # Critic, actor, and replay buffer all use the same action representation.
            # Reward and transition are therefore aligned on that action space.
            stored_action = action[0]
            
            # Store transition (unbatch obs for buffer)
            self.replay_buffer.add(
                obs={k: v[0] for k, v in obs.items()},
                next_obs=actual_next_obs,
                action=stored_action,
                reward=float(reward),
                done=terminated,  # Only true terminations, not truncations
                infos=info,
            )
            
            obs = next_obs
            self.num_timesteps += 1
            current_episode_reward += reward
            current_episode_length += 1
            
            # Reset handled automatically by VecEnv
            if done:
                self._episode_num += 1
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                current_episode_reward = 0.0
                current_episode_length = 0
                # Reset actor history for new episode
                self.policy.actor.reset_history()
            
            # Train
            if self.num_timesteps > self.learning_starts and self.num_timesteps % self.train_freq == 0:
                if self.replay_buffer.size() >= self.batch_size * self.seq_len:
                    self.train(self.gradient_steps, self.batch_size)
            
            # Callback
            callback.update_locals(locals())
            if callback.on_step() is False:
                break
            
            # Logging - only when episode ends and at log_interval
            if done and log_interval is not None and self._episode_num % log_interval == 0:
                record_mamba_rollout_metrics(
                    logger=self.logger,
                    episode_num=self._episode_num,
                    total_timesteps=self.num_timesteps,
                    buffer_size=self.replay_buffer.size(),
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                )
                self.logger.dump(step=self.num_timesteps)
        
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
            episode_start: If True, reset history buffer for new episode
            deterministic: Whether to use deterministic action
            use_history: Whether to use the actor's inference history buffer
        
        Returns:
            action: Selected action
            state: None (kept for API compatibility)
        """
        # Reset history at episode start
        if episode_start is not None and np.any(episode_start):
            self.policy.actor.reset_history()
        
        with torch.no_grad():
            # Check if observation is already batched (from VecEnv)
            first_key = list(observation.keys())[0]
            is_batched = len(observation[first_key].shape) > len(self.observation_space[first_key].shape)
            
            if is_batched:
                obs_tensor = {
                    key: torch.tensor(val, dtype=torch.float32, device=self.device)
                    for key, val in observation.items()
                }
            else:
                obs_tensor = {
                    key: torch.tensor(val, dtype=torch.float32, device=self.device).unsqueeze(0)
                    for key, val in observation.items()
                }
            
            action = self.policy.actor(
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
            # Config for validation
            'seq_len': self.seq_len,
            'batch_size': self.batch_size,
        }
        
        # Optionally save replay buffer (for true checkpoint resume)
        if include_buffer:
            checkpoint['replay_buffer'] = self.replay_buffer.get_state()
        
        torch.save(checkpoint, path + '.zip')
    
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
        
        # Load replay buffer if available and requested
        if load_buffer and 'replay_buffer' in checkpoint:
            model.replay_buffer.set_state(checkpoint['replay_buffer'])
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
