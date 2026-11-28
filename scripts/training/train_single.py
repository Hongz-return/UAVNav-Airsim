"""
UAV Navigation DRL Training Script
Usage: python -m scripts.training.train_single -c configs/config_sac_framestack4.ini
"""
import argparse
import datetime
import hashlib
import json
import os
import re

from configparser import ConfigParser
from tools.training_logging import (
    BestModelCallback,
    TrainingProgressCallback,
    UAVLoggingCallback,
    infer_model_label,
)
from tools.runtime_support import (
    apply_config_overrides,
    build_checkpoint_callback,
    build_parallel_runtime_env,
    build_runtime_env,
    create_model,
    get_replay_buffer_path,
    load_model,
    load_replay_buffer_file,
    save_replay_buffer_file,
    use_mamba_backend,
)
from tools.experiment_paths import TRAIN_RUNS_ROOT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train UAV navigation model")
    parser.add_argument('-c', '--config', default='configs/config_sac_framestack4.ini',
                        help='Path to config file')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Override total_timesteps from config')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Override base log directory (default: Logs/TrainData/Runs/manual/<env>)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to a checkpoint under Logs/TrainData/Runs/')
    parser.add_argument('--restore-replay', action='store_true',
                        help='Restore replay buffer from a full checkpoint even when writing to a new --run-dir')
    parser.add_argument('--algo', type=str, default=None, choices=['SAC', 'TD3', 'PPO'],
                        help='Override algorithm from config.ini')
    parser.add_argument('--env-name', type=str, default=None,
                        help='Override environment/task name from config.ini, e.g. Tunnel or Obs')
    parser.add_argument('--airsim-ports', nargs='+', type=int, default=None,
                        help='RPC ports of independent AirSim instances')
    parser.add_argument('--checkpoint-every', type=int, default=10000,
                        help='Weights-only checkpoint interval in environment transitions')
    parser.add_argument('--progress-every', type=int, default=0,
                        help='Console progress interval in environment transitions (0 disables)')
    args = parser.parse_args(argv)
    if args.checkpoint_every < 1 or args.progress_every < 0:
        parser.error('checkpoint interval must be positive and progress interval nonnegative')
    return args


def reject_unsafe_in_place_resume(checkpoint_path: str) -> None:
    """Refuse a resume that would overwrite newer checkpoints in the same run."""
    match = re.fullmatch(r"checkpoint_(\d+)_steps\.zip", os.path.basename(checkpoint_path))
    if not match:
        return

    selected_step = int(match.group(1))
    models_dir = os.path.dirname(checkpoint_path)
    newer_steps = []
    for filename in os.listdir(models_dir):
        candidate = re.fullmatch(r"checkpoint_(\d+)_steps\.zip", filename)
        if candidate and int(candidate.group(1)) > selected_step:
            newer_steps.append(int(candidate.group(1)))
    if newer_steps:
        raise RuntimeError(
            f"Refusing in-place resume from step {selected_step}: "
            f"newer checkpoint step {max(newer_steps)} already exists in {models_dir}. "
            "Copy the selected checkpoint into a new run directory before resuming."
        )


def main(argv=None):
    args = parse_args(argv)
    # Load config
    cfg = ConfigParser()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print(f"Current working directory: {os.getcwd()}")
        return

    cfg.read(args.config)

    # Verify config was loaded correctly
    if not cfg.has_section('options'):
        print(f"Error: Config file '{args.config}' is invalid or empty")
        print(f"Make sure the file exists and has [options] section")
        return

    apply_config_overrides(cfg, env_name_override=args.env_name, algo_override=args.algo)
    run_training(args, cfg)


def run_training(args, cfg, run_dir=None, simulator_job=None, *, extra_callbacks=()):
    env_name = cfg.get('options', 'env_name')
    algo = cfg.get('options', 'algo')
    dynamic_name = cfg.get('options', 'dynamic_name')
    policy_name = cfg.get('options', 'policy_name')
    total_timesteps = cfg.getint('options', 'total_timesteps')
    # Override total_timesteps if provided via command line
    if args.timesteps is not None:
        total_timesteps = args.timesteps

    print(f"Config: {args.config}")
    print(f"Environment: {env_name}, Algorithm: {algo}, Timesteps: {total_timesteps}")
    stage3 = cfg.has_section("stage3")
    stage3_v2 = cfg.has_section("stage3_v2")
    if stage3 and (run_dir is None or not args.resume):
        raise ValueError("Launch Stage 3 with train_trainenv, a source checkpoint, and a new run directory")
    env_kwargs = {"training": True} if (stage3 or stage3_v2) else {}

    # Create environment
    if args.airsim_ports and len(args.airsim_ports) > 1:
        env = build_parallel_runtime_env(cfg, args.airsim_ports, base_seed=args.seed, **env_kwargs)
    else:
        if args.airsim_ports:
            cfg.read_dict({"airsim": {"rpc_port": args.airsim_ports[0]}})
        env = build_runtime_env(cfg, **env_kwargs)
    try:
        if simulator_job is not None and env.num_envs > 1:
            for worker in env.venv.processes:
                simulator_job.add(worker)
        use_mamba = use_mamba_backend(cfg)

        checkpoint_path = None
        if args.resume:
            checkpoint_path = args.resume
            if not checkpoint_path.endswith('.zip'):
                checkpoint_path += '.zip'
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        if checkpoint_path and run_dir is None:
            reject_unsafe_in_place_resume(checkpoint_path)
            log_dir = os.path.dirname(os.path.dirname(checkpoint_path))
            log_path = f'{log_dir}/tb_logs'
            model_path = f'{log_dir}/models'
        else:
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')

            # Use custom logdir if provided, otherwise use default
            if args.logdir:
                base_dir = args.logdir
            else:
                base_dir = str(TRAIN_RUNS_ROOT / 'manual' / env_name)

            log_dir = (str(run_dir) if run_dir is not None
                       else f'{base_dir}/{timestamp}_{dynamic_name}_{policy_name}_{algo}')
            if checkpoint_path and os.path.realpath(log_dir) == os.path.realpath(os.path.dirname(os.path.dirname(checkpoint_path))):
                raise ValueError("Fine-tuning must write to a new run directory")
            log_path = f'{log_dir}/tb_logs'
            model_path = f'{log_dir}/models'
            config_path = f'{log_dir}/config'
            config_file_for_run = os.path.join(config_path, 'config.ini')

            os.makedirs(log_path, exist_ok=True)
            os.makedirs(model_path, exist_ok=True)
            os.makedirs(config_path, exist_ok=True)

            # Save config
            with open(config_file_for_run, 'w', encoding='utf-8') as f:
                cfg.write(f)

            resume_details = {}
            if checkpoint_path:
                with open(checkpoint_path, "rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                resume_details = {
                    "resume_checkpoint": os.path.abspath(checkpoint_path),
                    "resume_checkpoint_sha256": digest,
                    "resume_replay": bool(getattr(args, "restore_replay", False)),
                }
            with open(os.path.join(log_dir, 'run.json'), 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                        "config_source": os.path.abspath(args.config),
                        "config_snapshot": os.path.abspath(config_file_for_run),
                        "environment": env_name,
                        "algorithm": algo,
                        "seed": args.seed,
                        "airsim_ports": args.airsim_ports,
                        "total_timesteps": total_timesteps,
                        "checkpoint_every": args.checkpoint_every,
                        "progress_every": args.progress_every,
                        "tensorboard_dir": os.path.abspath(log_path),
                        "logging": "tensorboard_and_console" if args.progress_every else "tensorboard_only",
                        **resume_details,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
                f.write("\n")

        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            print(f"Log directory: {log_dir}")
            resume_options = (
                {"load_replay": getattr(args, "restore_replay", False), "seed": args.seed}
                if run_dir is not None else {}
            )
            model = load_model(algo, checkpoint_path, env, cfg=cfg, tensorboard_log=log_path, **resume_options)
            if run_dir is None and algo in {'SAC', 'TD3'} and not (use_mamba and algo == 'SAC'):
                replay_buffer_path = get_replay_buffer_path(checkpoint_path)
                if load_replay_buffer_file(model, replay_buffer_path):
                    print(f"Replay buffer restored from: {replay_buffer_path}")
                else:
                    print("Warning: replay buffer checkpoint not found or failed to load; resume will continue with a fresh buffer.")
            print("Model loaded successfully. Continuing training...")
        else:
            print(f"Random seed: {args.seed}")
            model = create_model(algo, env, cfg, log_path, seed=args.seed, total_timesteps=total_timesteps)

        checkpoint_callback = build_checkpoint_callback(
            algo=algo,
            use_mamba=use_mamba,
            model_path=model_path,
            save_freq=args.checkpoint_every,
        )

        # UAV state logging callback
        uav_logging_callback = UAVLoggingCallback(log_freq=100)

        # Best model saving and early stopping callback
        best_model_callback = BestModelCallback(
            save_path=model_path,
            check_freq=5000,       # Check every 5000 steps
            min_episodes=20,       # Need at least 20 episodes before checking
            patience=0,            # Disable early stopping (0 = disable)
            verbose=1
        )

        callbacks = [checkpoint_callback, uav_logging_callback, best_model_callback]
        callbacks.extend(extra_callbacks)
        if stage3 or stage3_v2:
            from tools.ue_benchmark.ch4_stage3_training import Stage3CourseMetrics

            callbacks.append(Stage3CourseMetrics(os.path.join(log_dir, "course_episodes.jsonl")))
        if args.progress_every:
            callbacks.append(TrainingProgressCallback(args.progress_every))

        # Train
        reset_num_timesteps = not bool(args.resume)
        tb_log_name = infer_model_label(cfg, algo)
        if args.resume:
            already_trained = int(getattr(model, 'num_timesteps', 0))
            if args.timesteps is not None:
                # In resume mode, CLI timesteps means "additional training budget"
                learn_timesteps = args.timesteps
                print(
                    f"Resume mode: additional training {learn_timesteps} steps "
                    f"(from {already_trained} to {already_trained + learn_timesteps})"
                )
            else:
                # Without CLI override, treat config total_timesteps as final target
                learn_timesteps = max(total_timesteps - already_trained, 0)
                print(
                    f"Resume mode: target total={total_timesteps}, current={already_trained}, "
                    f"remaining={learn_timesteps}"
                )
        else:
            learn_timesteps = total_timesteps

        if learn_timesteps > 0:
            print(f"Starting training on {model.device} ({model.n_envs} environment(s))...", flush=True)
            model.learn(
                learn_timesteps,
                callback=callbacks,
                reset_num_timesteps=reset_num_timesteps,
                tb_log_name=tb_log_name,
                log_interval=1,
            )
        else:
            print("No remaining timesteps, skip training.")

        # Save final model
        if use_mamba and algo == 'SAC':
            model.save_full_checkpoint(f'{model_path}/model_sb3')
        else:
            model.save(f'{model_path}/model_sb3')
        if algo in {'SAC', 'TD3'} and not (use_mamba and algo == 'SAC'):
            replay_buffer_path = get_replay_buffer_path(f'{model_path}/model_sb3')
            if save_replay_buffer_file(model, replay_buffer_path):
                print(f"Replay buffer saved to: {replay_buffer_path}")
        print(f"Training finished. Model saved to: {model_path}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
