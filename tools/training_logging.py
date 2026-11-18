"""TensorBoard metric writers and training callbacks."""

from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def infer_model_label(cfg, algo: str) -> str:
    """Pick a stable TensorBoard series name from the active config."""
    if (
        algo == "SAC"
        and not cfg.getboolean("options", "use_mamba", fallback=False)
        and cfg.get("vision", "cnn_attention", fallback="none").strip().lower()
        == "coordatt"
    ):
        return "CoordAtt_SAC"

    if algo == "SAC" and cfg.getboolean("options", "use_mamba", fallback=False):
        backend = cfg.get("mamba", "mamba_backend", fallback="auto").strip().lower()
        attention = cfg.get("mamba", "cnn_attention", fallback="none").strip().lower()
        if backend == "lstm":
            return "LSTM"
        if backend == "transformer":
            return "Transformer"
        if attention == "coordatt":
            return "Mamba_CoordAtt"
        return "Mamba"

    return algo


def record_sac_training_metrics(logger, ent_coefs, actor_losses, critic_losses, ent_coef_losses) -> None:
    """Record SAC-family loss metrics from one train() pass."""
    logger.record("train/ent_coef", np.mean(ent_coefs))
    logger.record("train/actor_loss", np.mean(actor_losses))
    logger.record("train/critic_loss", np.mean(critic_losses))
    if ent_coef_losses:
        logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))


def record_td3_training_metrics(logger, critic_losses, actor_losses) -> None:
    """Record TD3-family loss metrics from one train() pass."""
    logger.record("train/critic_loss", np.mean(critic_losses))
    if actor_losses:
        logger.record("train/actor_loss", np.mean(actor_losses))


def record_mamba_training_metrics(
    logger,
    ent_coef,
    actor_losses,
    critic_losses,
    ent_coef_losses,
    q_values_list,
    q_max_list,
    q_min_list,
    td_errors_list,
    log_probs_list,
    action_magnitudes,
    actor_grad_norms,
    critic_grad_norms,
    mask_sums,
) -> None:
    """Record the custom diagnostics emitted by MambaSAC training."""
    logger.record("train/actor_loss", np.mean(actor_losses))
    logger.record("train/critic_loss", np.mean(critic_losses))
    logger.record("train/ent_coef", ent_coef)
    if ent_coef_losses:
        logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    logger.record("q/mean", np.mean(q_values_list))
    logger.record("q/max", np.mean(q_max_list))
    logger.record("q/min", np.mean(q_min_list))
    logger.record("q/td_error", np.mean(td_errors_list))

    logger.record("policy/mean_log_prob", np.mean(log_probs_list))
    logger.record("policy/action_magnitude", np.mean(action_magnitudes))

    logger.record("grad/actor_norm", np.mean(actor_grad_norms))
    logger.record("grad/critic_norm", np.mean(critic_grad_norms))

    logger.record("seq/mask_sum", np.mean(mask_sums))


def record_mamba_rollout_metrics(logger, episode_num, total_timesteps, buffer_size, episode_rewards, episode_lengths, episode_successes=None, success_episode_lengths=None) -> None:
    """Record the episodic rollout summary emitted by MambaSAC."""
    recent_rewards = episode_rewards[-100:]
    recent_lengths = episode_lengths[-100:]

    logger.record("time/episodes", episode_num)
    logger.record("time/total_timesteps", total_timesteps)
    logger.record("rollout/buffer_size", buffer_size)
    logger.record("rollout/ep_rew_mean", np.mean(recent_rewards))
    logger.record("rollout/ep_len_mean", np.mean(recent_lengths))
    if episode_successes:
        logger.record("rollout/success_rate", np.mean(episode_successes[-100:]))
    if success_episode_lengths:
        logger.record("rollout/success_ep_len_mean", np.mean(success_episode_lengths[-100:]))


class UAVLoggingCallback(BaseCallback):
    """Log UAV state and episode-level statistics for all training algorithms."""

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_crashes = []
        self.episode_reward_components = {}
        self.success_episode_steps = []
        self.success_episode_times = []
        self.last_success_episode_steps = None
        self.best_episode_reward = float("-inf")
        self.episode_yaw_rate_abs_means = []
        self.episode_vertical_speed_abs_means = []
        self.episode_min_obstacle_dists = []
        self.current_episode_yaw_rates = []
        self.current_episode_vertical_speeds = []
        self.current_episode_obstacle_dists = []

    def _get_base_env(self):
        env = self.training_env.envs[0]
        while hasattr(env, "env"):
            env = env.env
        return env

    def _get_control_dt(self, env) -> Optional[float]:
        dynamic_model = getattr(env, "dynamic_model", None)
        dt = getattr(dynamic_model, "dt", None)
        if dt is None:
            return None
        action_repeat = getattr(env, "action_repeat", 1)
        return float(action_repeat * dt)

    def _finalize_episode_stability(self) -> None:
        if self.current_episode_yaw_rates:
            self.episode_yaw_rate_abs_means.append(float(np.mean(self.current_episode_yaw_rates)))
        if self.current_episode_vertical_speeds:
            self.episode_vertical_speed_abs_means.append(float(np.mean(self.current_episode_vertical_speeds)))
        if self.current_episode_obstacle_dists:
            self.episode_min_obstacle_dists.append(float(min(self.current_episode_obstacle_dists)))

        self.current_episode_yaw_rates.clear()
        self.current_episode_vertical_speeds.clear()
        self.current_episode_obstacle_dists.clear()

    def _on_step(self) -> bool:
        env = self._get_base_env()
        control_dt = self._get_control_dt(env)

        if hasattr(env, "dynamic_model"):
            dm = env.dynamic_model
            vel = dm.get_velocity()
            self.current_episode_yaw_rates.append(abs(float(vel[2])))
            self.current_episode_vertical_speeds.append(abs(float(vel[1])))
            if hasattr(env, "min_distance_to_obstacles"):
                self.current_episode_obstacle_dists.append(float(env.min_distance_to_obstacles))

        if self.n_calls % self.log_freq == 0:
            if hasattr(env, "dynamic_model"):
                dm = env.dynamic_model
                pos = dm.get_position()
                self.logger.record("uav/pos_x", pos[0])
                self.logger.record("uav/pos_y", pos[1])
                self.logger.record("uav/pos_z", pos[2])

                vel = dm.get_velocity()
                self.logger.record("uav/vel_forward", vel[0])
                self.logger.record("uav/vel_up", vel[1])
                self.logger.record("uav/yaw_rate", vel[2])

                dist = dm.get_distance_to_goal_3d() if dm.navigation_3d else dm.get_distance_to_goal_2d()
                self.logger.record("uav/distance_to_goal", dist)

                if hasattr(env, "min_distance_to_obstacles"):
                    self.logger.record("uav/min_obstacle_dist", env.min_distance_to_obstacles)
                if hasattr(env, "step_num"):
                    self.logger.record("uav/step_in_episode", env.step_num)
                if hasattr(env, "cumulated_episode_reward"):
                    self.logger.record("uav/cumulated_reward", env.cumulated_episode_reward)

        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx]
                    for name, value in info.get('episode_reward_components', {}).items():
                        history = self.episode_reward_components.setdefault(name, [])
                        history.append(float(value))
                        self.logger.record(f'reward_component/{name}', float(value))
                        self.logger.record(f'reward_component_ma20/{name}', np.mean(history[-20:]))
                    if "episode" in info:
                        episode_reward = float(info["episode"]["r"])
                        episode_length = int(info["episode"].get("l", info.get("step_num", 0)))
                        self.episode_rewards.append(episode_reward)
                        self.episode_lengths.append(episode_length)
                        self.best_episode_reward = max(self.best_episode_reward, episode_reward)
                        self.logger.record("episode/return_raw", episode_reward)
                        self.logger.record("episode/return_ma20", np.mean(self.episode_rewards[-20:]))
                        self.logger.record(
                            "episode/reward_per_step",
                            episode_reward / max(episode_length, 1),
                        )
                        self.logger.record("episode/length_raw", episode_length)
                    if "is_success" in info:
                        is_success = float(info["is_success"])
                        self.episode_successes.append(is_success)
                        self.logger.record("episode/success_raw", is_success)
                        self.logger.record("episode/success_rate_ma20", np.mean(self.episode_successes[-20:]))
                        self.logger.record("episode/success_rate_ma50", np.mean(self.episode_successes[-50:]))
                        if is_success and "step_num" in info:
                            success_steps = int(info["step_num"])
                            self.success_episode_steps.append(success_steps)
                            self.last_success_episode_steps = success_steps
                            if control_dt is not None:
                                self.success_episode_times.append(success_steps * control_dt)
                    if "is_crash" in info:
                        is_crash = float(info["is_crash"])
                        self.episode_crashes.append(is_crash)
                        self.logger.record("episode/collision_raw", is_crash)
                    self._finalize_episode_stability()

        if self.success_episode_steps:
            self.logger.record("rollout/success_ep_len_mean", np.mean(self.success_episode_steps[-100:]))

        if self.n_calls % 1000 == 0 and self.episode_successes:
            self.logger.record("episode/success_rate_recent", np.mean(self.episode_successes[-100:]))
            self.logger.record("episode/crash_rate_recent", np.mean(self.episode_crashes[-100:]))
            if self.last_success_episode_steps is not None:
                self.logger.record("episode/success_steps_last", self.last_success_episode_steps)
            if self.success_episode_steps:
                self.logger.record("episode/success_steps_mean_recent", np.mean(self.success_episode_steps[-100:]))

            self.logger.record("core/success_rate_recent", np.mean(self.episode_successes[-100:]))
            self.logger.record("core/collision_rate_recent", np.mean(self.episode_crashes[-100:]))
            if self.last_success_episode_steps is not None:
                self.logger.record("core/success_steps_last", self.last_success_episode_steps)
            if self.success_episode_steps:
                self.logger.record("core/success_steps_mean_recent", np.mean(self.success_episode_steps[-100:]))
            if self.success_episode_times:
                self.logger.record("core/success_time_mean_recent_sec", np.mean(self.success_episode_times[-100:]))
            if self.episode_rewards:
                self.logger.record("core/mean_reward_recent", np.mean(self.episode_rewards[-100:]))
                self.logger.record("core/best_reward", self.best_episode_reward)
            if self.episode_yaw_rate_abs_means:
                self.logger.record("core/stability/yaw_rate_abs_mean_recent", np.mean(self.episode_yaw_rate_abs_means[-100:]))
            if self.episode_vertical_speed_abs_means:
                self.logger.record(
                    "core/stability/vertical_speed_abs_mean_recent",
                    np.mean(self.episode_vertical_speed_abs_means[-100:]),
                )
            if self.episode_min_obstacle_dists:
                self.logger.record(
                    "core/stability/min_obstacle_dist_mean_recent",
                    np.mean(self.episode_min_obstacle_dists[-100:]),
                )

        return True


class BestModelCallback(BaseCallback):
    """Save the best model based on recent success rate and reward."""

    def __init__(
        self,
        save_path: str,
        check_freq: int = 5000,
        min_episodes: int = 20,
        patience: int = 50000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.min_episodes = min_episodes
        self.patience = patience

        self.best_success_rate = 0.0
        self.best_mean_reward = float("-inf")
        self.episode_successes = []
        self.episode_rewards = []
        self.steps_since_improvement = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx]
                    if "is_success" in info:
                        self.episode_successes.append(float(info["is_success"]))
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])

        if self.n_calls % self.check_freq == 0 and len(self.episode_successes) >= self.min_episodes:
            recent_success = np.mean(self.episode_successes[-50:])
            recent_reward = np.mean(self.episode_rewards[-50:]) if self.episode_rewards else float("-inf")

            improved = False
            if recent_success > self.best_success_rate:
                self.best_success_rate = recent_success
                improved = True
            elif recent_success == self.best_success_rate and recent_reward > self.best_mean_reward:
                improved = True

            if improved:
                self.best_mean_reward = recent_reward
                self.steps_since_improvement = 0
                save_file = f"{self.save_path}/best_model"
                self.model.save(save_file)

                if self.verbose > 0:
                    print(
                        f"[Best Model] Step {self.n_calls}: success_rate={recent_success:.2%}, "
                        f"mean_reward={recent_reward:.1f} -> Saved to {save_file}"
                    )
            else:
                self.steps_since_improvement += self.check_freq

            self.logger.record("best/success_rate", self.best_success_rate)
            self.logger.record("best/mean_reward", self.best_mean_reward)
            self.logger.record("best/steps_since_improvement", self.steps_since_improvement)

            if self.patience > 0 and self.steps_since_improvement >= self.patience:
                if self.verbose > 0:
                    print(f"[Early Stop] No improvement for {self.patience} steps. Stopping training.")
                return False

        return True
