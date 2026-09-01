"""
SAC and TD3 variants with Prioritized Experience Replay support.

These subclasses override the train() method to:
1. Use importance sampling weights in loss computation
2. Update priorities based on TD-errors after each batch
"""

from typing import Optional, Union, Dict, Any
import torch as th
from torch.nn import functional as F

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.type_aliases import GymEnv, Schedule
from stable_baselines3.common.noise import ActionNoise
from tools.training_logging import record_sac_training_metrics, record_td3_training_metrics

# Max gradient norm for clipping (prevents gradient explosion)
MAX_GRAD_NORM = 0.5


class SAC_PER(SAC):
    """SAC with Prioritized Experience Replay support."""

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        """Train with PER: apply IS weights and update priorities."""
        # Switch to train mode
        self.policy.set_training_mode(True)
        
        # Update learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            # Sample with priorities
            replay_data, weights, indices = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            # Convert weights to tensor if not already
            if not isinstance(weights, th.Tensor):
                weights = th.tensor(weights, device=self.device, dtype=th.float32)

            # Entropy coefficient
            ent_coef = self._get_ent_coef()
            ent_coefs.append(ent_coef.item())

            # Entropy loss (no change needed)
            if self.ent_coef_optimizer is not None:
                with th.no_grad():
                    actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Target Q-values
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q1, next_q2 = self.critic_target(replay_data.next_observations, next_actions)
                next_q = th.min(next_q1, next_q2) - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q
                # Clamp target Q to prevent explosion
                target_q = th.clamp(target_q, -1000.0, 1000.0)

            # Current Q-values
            current_q1, current_q2 = self.critic(replay_data.observations, replay_data.actions)
            
            # TD-errors for priority update
            with th.no_grad():
                td_errors = (th.abs(current_q1 - target_q) + th.abs(current_q2 - target_q)) / 2
                td_errors_np = td_errors.squeeze().cpu().numpy()

            # Weighted critic loss (apply IS weights)
            critic_loss_1 = (weights * F.mse_loss(current_q1, target_q, reduction='none').squeeze()).mean()
            critic_loss_2 = (weights * F.mse_loss(current_q2, target_q, reduction='none').squeeze()).mean()
            critic_loss = critic_loss_1 + critic_loss_2
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            # Gradient clipping to prevent explosion
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.critic.optimizer.step()

            # Actor loss (no IS weights needed)
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            q1_pi, q2_pi = self.critic(replay_data.observations, actions_pi)
            min_q_pi = th.min(q1_pi, q2_pi)
            actor_loss = (ent_coef * log_prob - min_q_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            # Gradient clipping to prevent explosion
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
            self.actor.optimizer.step()

            # Update target networks
            if self._n_updates % self.target_update_interval == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Update priorities in buffer
            if indices is not None:
                self.replay_buffer.update_priorities(indices, td_errors_np)

            self._n_updates += 1

        record_sac_training_metrics(
            logger=self.logger,
            ent_coefs=ent_coefs,
            actor_losses=actor_losses,
            critic_losses=critic_losses,
            ent_coef_losses=ent_coef_losses,
        )

    def _get_ent_coef(self):
        """Get current entropy coefficient."""
        if self.ent_coef_optimizer is not None:
            return th.exp(self.log_ent_coef.detach())
        return self.ent_coef_tensor


class TD3_PER(TD3):
    """TD3 with Prioritized Experience Replay support."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Train with PER: apply IS weights and update priorities."""
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            self._n_updates += 1
            
            # Sample with priorities
            replay_data, weights, indices = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            
            if not isinstance(weights, th.Tensor):
                weights = th.tensor(weights, device=self.device, dtype=th.float32)

            with th.no_grad():
                # Target policy smoothing
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Target Q-values
                next_q1, next_q2 = self.critic_target(replay_data.next_observations, next_actions)
                next_q = th.min(next_q1, next_q2)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q
                # Clamp target Q to prevent explosion
                target_q = th.clamp(target_q, -1000.0, 1000.0)

            # Current Q-values
            current_q1, current_q2 = self.critic(replay_data.observations, replay_data.actions)
            
            # TD-errors for priority update
            with th.no_grad():
                td_errors = (th.abs(current_q1 - target_q) + th.abs(current_q2 - target_q)) / 2
                td_errors_np = td_errors.squeeze().cpu().numpy()

            # Weighted critic loss
            critic_loss_1 = (weights * F.mse_loss(current_q1, target_q, reduction='none').squeeze()).mean()
            critic_loss_2 = (weights * F.mse_loss(current_q2, target_q, reduction='none').squeeze()).mean()
            critic_loss = critic_loss_1 + critic_loss_2
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM)
            self.critic.optimizer.step()

            # Delayed actor update
            if self._n_updates % self.policy_delay == 0:
                actor_loss = -self.critic.q1_forward(replay_data.observations, 
                                                      self.actor(replay_data.observations)).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)
                self.actor.optimizer.step()

                # Update target networks
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            # Update priorities
            if indices is not None:
                self.replay_buffer.update_priorities(indices, td_errors_np)

        record_td3_training_metrics(
            logger=self.logger,
            critic_losses=critic_losses,
            actor_losses=actor_losses,
        )
