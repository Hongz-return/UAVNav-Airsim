"""Shared runtime helpers for training and evaluation entrypoints."""

import ast
import os
import pickle
from configparser import ConfigParser
from typing import Optional

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import gym_env  # Registers the custom env id used by both train and evaluate.
from gym_env.algorithms import MambaSAC
from gym_env.networks import FrameStackedDictExtractor, MambaSACPolicy
from gym_env.replay import SAC_PER, TD3_PER, get_buffer_class
from gym_env.utils import build_mamba_policy_kwargs, get_backend_display_name
from gym_env.wrappers import DictFrameStack


def get_replay_buffer_path(model_path: str) -> str:
    """Companion replay-buffer path for a model or checkpoint path."""
    root, ext = os.path.splitext(model_path)
    if ext == ".zip":
        return root + "_replay_buffer.pkl"
    return model_path + "_replay_buffer.pkl"


def save_replay_buffer_file(model, replay_buffer_path: str) -> bool:
    """Persist replay buffer using SB3 helpers when available, else pickle."""
    if not hasattr(model, "replay_buffer") or model.replay_buffer is None:
        return False

    if hasattr(model, "save_replay_buffer"):
        try:
            model.save_replay_buffer(replay_buffer_path)
            return True
        except Exception as exc:
            print(f"[Checkpoint] Warning: save_replay_buffer failed ({exc}), falling back to pickle.")

    try:
        with open(replay_buffer_path, "wb") as fh:
            pickle.dump(model.replay_buffer, fh)
        return True
    except Exception as exc:
        print(f"[Checkpoint] Warning: could not save replay buffer to {replay_buffer_path}: {exc}")
        return False


def load_replay_buffer_file(model, replay_buffer_path: str) -> bool:
    """Restore replay buffer from disk for standard off-policy algorithms."""
    if not os.path.exists(replay_buffer_path):
        return False

    if hasattr(model, "load_replay_buffer"):
        try:
            model.load_replay_buffer(replay_buffer_path)
            return True
        except Exception as exc:
            print(f"[Resume] Warning: load_replay_buffer failed ({exc}), falling back to pickle.")

    try:
        with open(replay_buffer_path, "rb") as fh:
            model.replay_buffer = pickle.load(fh)
        if hasattr(model.replay_buffer, "device"):
            model.replay_buffer.device = model.device
        return True
    except Exception as exc:
        print(f"[Resume] Warning: could not restore replay buffer from {replay_buffer_path}: {exc}")
        return False


def apply_config_overrides(
    cfg: ConfigParser,
    env_name_override: Optional[str] = None,
    algo_override: Optional[str] = None,
) -> None:
    """Apply CLI overrides while preserving current model/backend compatibility rules."""
    if env_name_override is not None:
        cfg.set("options", "env_name", env_name_override)

    if algo_override is not None:
        cfg.set("options", "algo", algo_override)
        if algo_override != "SAC":
            # Sequence backends are only used by the custom SAC variant.
            cfg.set("options", "use_mamba", "False")


def use_mamba_backend(cfg: ConfigParser) -> bool:
    """Return whether the current config uses the custom sequence SAC backend."""
    return cfg.getboolean("options", "use_mamba", fallback=False)


def get_frame_stack_count(cfg: ConfigParser) -> int:
    """Read frame-stack count from the config according to the active backend."""
    if use_mamba_backend(cfg):
        return cfg.getint("mamba", "n_stack", fallback=1)
    return cfg.getint("options", "n_stack", fallback=1)


def build_runtime_env(cfg: ConfigParser):
    """Create the gym env and apply the shared runtime wrappers used by train/eval."""
    env = gym.make("airsim-env-v0")
    env.unwrapped.set_config(cfg)

    n_stack = get_frame_stack_count(cfg)
    if n_stack > 1:
        env = DictFrameStack(env, n_stack=n_stack)
        print(f"Frame stacking: {n_stack} frames")
    else:
        print("Frame stacking: disabled (n_stack=1)")

    env = Monitor(env)
    monitored_env = env
    env = DummyVecEnv([lambda e=monitored_env: e])
    env = VecTransposeImage(env)
    return env


def get_per_config(cfg: ConfigParser, total_timesteps: int) -> dict:
    """Get PER configuration from the config file."""
    use_per = cfg.getboolean("DRL", "use_per", fallback=False)
    if not use_per:
        return {"use_per": False}

    return {
        "use_per": True,
        "alpha": cfg.getfloat("DRL", "per_alpha", fallback=0.6),
        "beta_start": cfg.getfloat("DRL", "per_beta_start", fallback=0.4),
        "beta_end": cfg.getfloat("DRL", "per_beta_end", fallback=1.0),
        "beta_frames": total_timesteps,
        "eps": cfg.getfloat("DRL", "per_eps", fallback=1e-6),
    }


def get_standard_model_class(algo: str, cfg: ConfigParser):
    """Resolve the standard SB3/PER class for the requested algorithm."""
    if algo == "SAC":
        return SAC_PER if cfg.getboolean("DRL", "use_per", fallback=False) else SAC
    if algo == "TD3":
        return TD3_PER if cfg.getboolean("DRL", "use_per", fallback=False) else TD3
    if algo == "PPO":
        return PPO
    raise ValueError(f"Unknown algorithm: {algo}")


def create_model(algo: str, env, cfg: ConfigParser, log_path: str, seed: int = 0, total_timesteps: int = 100000):
    """Create an SB3 model based on the config, with optional PER and Mamba support."""
    policy_name = cfg.get("options", "policy_name")
    perception_type = cfg.get("options", "perception")
    use_mamba = use_mamba_backend(cfg)
    n_stack = get_frame_stack_count(cfg)

    if perception_type in {"depth_state", "depth_motion"}:
        policy_base = "MultiInputPolicy"
    else:
        policy_base = "MlpPolicy" if policy_name == "mlp" else "CnnPolicy"

    activation_fn = th.nn.Tanh if cfg.get("options", "activation_function") == "tanh" else th.nn.ReLU
    net_arch = ast.literal_eval(cfg.get("options", "net_arch"))
    policy_kwargs = dict(activation_fn=activation_fn, net_arch=net_arch)

    if perception_type in {"depth_state", "depth_motion"} and not use_mamba and n_stack > 1:
        policy_kwargs["features_extractor_class"] = FrameStackedDictExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "cnn_output_dim": cfg.getint("vision", "cnn_output_dim", fallback=64),
            "cnn_attention": cfg.get("vision", "cnn_attention", fallback="none"),
            "cnn_attention_reduction": cfg.getint(
                "vision", "cnn_attention_reduction", fallback=8
            ),
            "cnn_attention_stage": cfg.get(
                "vision", "cnn_attention_stage", fallback="auto"
            ),
        }
        attention = policy_kwargs["features_extractor_kwargs"]["cnn_attention"]
        print(
            f"Using FrameStackedDictExtractor for stacked Dict observations "
            f"(n_stack={n_stack}, attention={attention})"
        )

    lr = cfg.getfloat("DRL", "learning_rate")

    if use_mamba and algo == "SAC":
        mamba_policy_kwargs = build_mamba_policy_kwargs(cfg)
        seq_len = cfg.getint("mamba", "seq_len", fallback=16)
        cnn_update_mode = mamba_policy_kwargs["cnn_update_mode"]
        inference_seq_len = mamba_policy_kwargs["inference_seq_len"]

        backend_name = get_backend_display_name(mamba_policy_kwargs)
        print(f"MambaSAC with SequenceReplayBuffer (backend={backend_name}):")
        print(f"  n_stack={n_stack}, train_seq_len={seq_len}, inference_seq_len={inference_seq_len}")
        print(f"  hidden={mamba_policy_kwargs['mamba_hidden_dim']}")
        print(f"  actor={mamba_policy_kwargs['actor_hidden']}, critic={mamba_policy_kwargs['critic_hidden']}")
        print(f"  cnn_update_mode={cnn_update_mode}")

        return MambaSAC(
            policy=MambaSACPolicy,
            env=env,
            learning_rate=lr,
            buffer_size=cfg.getint("DRL", "buffer_size"),
            learning_starts=cfg.getint("DRL", "learning_starts"),
            batch_size=cfg.getint("DRL", "batch_size"),
            seq_len=seq_len,
            tau=0.005,
            gamma=cfg.getfloat("DRL", "gamma"),
            train_freq=cfg.getint("DRL", "train_freq"),
            gradient_steps=cfg.getint("DRL", "gradient_steps"),
            tensorboard_log=log_path,
            policy_kwargs=mamba_policy_kwargs,
            seed=seed,
            verbose=1,
        )

    per_cfg = get_per_config(cfg, total_timesteps)
    use_per = per_cfg["use_per"]

    if algo == "PPO":
        if use_per:
            print("Warning: PER is not applicable to PPO (on-policy). Ignoring.")
        return PPO(policy_base, env, learning_rate=lr, policy_kwargs=policy_kwargs, tensorboard_log=log_path, seed=seed, verbose=1)

    if algo in {"SAC", "TD3"}:
        n_actions = env.action_space.shape[-1]
        if algo == "TD3":
            noise_sigma = cfg.getfloat("DRL", "action_noise_sigma") * np.ones(n_actions)
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma)
        else:
            action_noise = None

        buffer_class = get_buffer_class(env.observation_space, use_per)
        buffer_kwargs = {
            "alpha": per_cfg.get("alpha", 0.6),
            "beta_start": per_cfg.get("beta_start", 0.4),
            "beta_end": per_cfg.get("beta_end", 1.0),
            "beta_frames": per_cfg.get("beta_frames", total_timesteps),
            "eps": per_cfg.get("eps", 1e-6),
        }

        if use_per:
            algo_class = SAC_PER if algo == "SAC" else TD3_PER
            print(f"PER enabled: alpha={buffer_kwargs['alpha']}, beta={buffer_kwargs['beta_start']}->{buffer_kwargs['beta_end']}")
            common_kwargs = dict(
                policy_kwargs=policy_kwargs,
                buffer_size=cfg.getint("DRL", "buffer_size"),
                gamma=cfg.getfloat("DRL", "gamma"),
                learning_starts=cfg.getint("DRL", "learning_starts"),
                learning_rate=lr,
                batch_size=cfg.getint("DRL", "batch_size"),
                train_freq=(cfg.getint("DRL", "train_freq"), "step"),
                gradient_steps=cfg.getint("DRL", "gradient_steps"),
                tensorboard_log=log_path,
                seed=seed,
                verbose=1,
                replay_buffer_class=buffer_class,
                replay_buffer_kwargs=buffer_kwargs,
            )
        else:
            algo_class = SAC if algo == "SAC" else TD3
            common_kwargs = dict(
                policy_kwargs=policy_kwargs,
                buffer_size=cfg.getint("DRL", "buffer_size"),
                gamma=cfg.getfloat("DRL", "gamma"),
                learning_starts=cfg.getint("DRL", "learning_starts"),
                learning_rate=lr,
                batch_size=cfg.getint("DRL", "batch_size"),
                train_freq=(cfg.getint("DRL", "train_freq"), "step"),
                gradient_steps=cfg.getint("DRL", "gradient_steps"),
                tensorboard_log=log_path,
                seed=seed,
                verbose=1,
            )

        return algo_class(policy_base, env, action_noise=action_noise, **common_kwargs)

    raise ValueError(f"Unknown algorithm: {algo}")


def load_model(algo: str, model_file: str, env, cfg: Optional[ConfigParser] = None, tensorboard_log: Optional[str] = None):
    """Load a trained model for evaluation or resume, honoring the active backend."""
    use_mamba = use_mamba_backend(cfg) if cfg else False

    if use_mamba and algo == "SAC":
        policy_kwargs = build_mamba_policy_kwargs(cfg)
        backend_name = get_backend_display_name(policy_kwargs)
        print(f"Loading MambaSAC with backend={backend_name}, inference_seq_len={policy_kwargs['inference_seq_len']}")
        if tensorboard_log is None:
            return MambaSAC.load(
                model_file,
                env=env,
                policy_kwargs=policy_kwargs,
                load_buffer=False,
            )

        return MambaSAC.load(
            model_file,
            env=env,
            policy_kwargs=policy_kwargs,
            seq_len=cfg.getint("mamba", "seq_len", fallback=16),
            buffer_size=cfg.getint("DRL", "buffer_size"),
            learning_starts=cfg.getint("DRL", "learning_starts"),
            batch_size=cfg.getint("DRL", "batch_size"),
            gamma=cfg.getfloat("DRL", "gamma"),
            train_freq=cfg.getint("DRL", "train_freq"),
            gradient_steps=cfg.getint("DRL", "gradient_steps"),
            tensorboard_log=tensorboard_log,
        )

    model_cls = get_standard_model_class(algo, cfg) if cfg else {"SAC": SAC, "TD3": TD3, "PPO": PPO}[algo]
    load_kwargs = {"env": env}
    if tensorboard_log is not None:
        load_kwargs["tensorboard_log"] = tensorboard_log
    return model_cls.load(model_file, **load_kwargs)


class MambaCheckpointCallback(BaseCallback):
    """Periodic weights-only checkpoint; the final run stores the resumable buffer."""

    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = f"{self.save_path}/checkpoint_{self.num_timesteps}_steps"
            self.model.save(path, include_buffer=False)
            if self.verbose > 0:
                print(f"[Checkpoint] Saved weights-only checkpoint to {path}")
        return True


class OffPolicyCheckpointCallback(BaseCallback):
    """Periodic weights-only checkpoint; avoid duplicating the replay buffer."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "checkpoint", verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 0:
                print(f"[Checkpoint] Saved weights-only model to {path}.zip")
        return True


def build_checkpoint_callback(algo: str, use_mamba: bool, model_path: str, save_freq: int = 10000):
    """Create the appropriate checkpoint callback for the active algorithm/backend."""
    if use_mamba and algo == "SAC":
        return MambaCheckpointCallback(save_freq=save_freq, save_path=model_path, verbose=1)
    if algo in {"SAC", "TD3"}:
        return OffPolicyCheckpointCallback(save_freq=save_freq, save_path=model_path, name_prefix="checkpoint", verbose=1)
    return CheckpointCallback(save_freq=save_freq, save_path=model_path, name_prefix="checkpoint")
