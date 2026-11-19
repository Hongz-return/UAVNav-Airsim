"""
UAV navigation evaluation script.

Usage:
    python -m scripts.evaluation.evaluate_single -m Logs/TrainData/Runs/<run_dir> -n 30 --save
"""

import argparse
import ast
import csv
import datetime
import inspect
import json
import os
import time
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from tools.runtime_support import apply_config_overrides, build_runtime_env, load_model
from tools.experiment_paths import EVALUATION_RESULTS_ROOT, evaluation_parent_for_run

DEFAULT_EPISODES = 100


def parse_args():
    """Parse CLI args for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained UAV navigation model")
    parser.add_argument(
        "-m",
        "--model_path",
        required=True,
        help="Path to a training run directory under Logs/TrainData/Runs",
    )
    parser.add_argument(
        "-n",
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Number of evaluation episodes (default: {DEFAULT_EPISODES})",
    )
    parser.add_argument("--save", action="store_true", help="Save evaluation results")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EVALUATION_RESULTS_ROOT,
        help="Root for evaluation results (default: Logs/EvaData/Results)",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Use models/checkpoint_<step>_steps.zip from the run directory, e.g. --checkpoint-step 90000",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Use best_model.zip instead of the final model",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Deprecated: evaluation already uses the final model (model_sb3.zip) by default",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="Override the environment/task name from config.ini, e.g. Tunnel or Obs",
    )
    parser.add_argument(
        "--airsim-ip",
        default=None,
        help="Override the AirSim server IP from the frozen training config",
    )
    parser.add_argument(
        "--airsim-port",
        type=int,
        default=None,
        help="Override the AirSim RPC port from the frozen training config",
    )
    parser.add_argument(
        "--save-tag",
        "--save_tag",
        dest="save_tag",
        type=str,
        default=None,
        help="Optional suffix for the save directory, e.g. eval_100_Tunnel",
    )
    parser.add_argument(
        "--scenario-seed",
        type=int,
        default=2026,
        help="Seed for the reproducible evaluation scenario list (default: 2026)",
    )
    parser.add_argument(
        "--yaw-range-deg",
        type=float,
        default=10.0,
        help="Uniform initial-yaw range in degrees (default: +/-10)",
    )
    parser.add_argument(
        "--scenario-manifest",
        help="Optional JSON scenario manifest; overrides generated scenarios",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help=(
            "MambaSAC only: disable the inference history buffer (Trunc ablation); "
            "each step uses only the current observation"
        ),
    )
    parser.add_argument(
        "--inference-seq-len",
        type=int,
        default=None,
        help=(
            "MambaSAC only: override actor inference history length at eval time "
            "(e.g. 2/4/6/8). seq_len=1 is equivalent to --no-history."
        ),
    )
    return parser.parse_args()


def apply_evaluation_overrides(cfg, env_name=None, airsim_ip=None, airsim_port=None):
    """Apply runtime-only evaluation overrides in memory."""
    apply_config_overrides(cfg, env_name_override=env_name)
    if airsim_ip is not None:
        cfg.set("airsim", "ip", airsim_ip)
    if airsim_port is not None:
        cfg.set("airsim", "rpc_port", str(airsim_port))


def get_base_env(env):
    """Unwrap SB3/Gym wrappers to the AirSim environment."""
    while hasattr(env, "envs"):
        env = env.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def build_evaluation_scenarios(
    cfg,
    episodes: int,
    seed: int,
    yaw_range_deg: float,
    manifest_path=None,
):
    """Load or deterministically generate the ordered evaluation scenarios."""
    if episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if yaw_range_deg < 0:
        raise ValueError("--yaw-range-deg must be non-negative")

    if manifest_path:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        scenarios = payload.get("scenarios", payload) if isinstance(payload, dict) else payload
        if len(scenarios) < episodes:
            raise ValueError(f"Scenario manifest has {len(scenarios)} rows, need {episodes}")
        return scenarios[:episodes], os.path.abspath(manifest_path)

    start = [float(value) for value in ast.literal_eval(cfg.get("environment", "start_position"))]
    goal = [float(value) for value in ast.literal_eval(cfg.get("environment", "goal_position"))]
    rng = np.random.default_rng(seed)
    yaws_deg = rng.uniform(-yaw_range_deg, yaw_range_deg, size=episodes)
    scenarios = [
        {
            "scenario_id": f"seed_{seed}_episode_{index:03d}",
            "start_position": start,
            "goal_position": goal,
            "yaw_deg": float(yaw_deg),
            "yaw_rad": float(np.deg2rad(yaw_deg)),
        }
        for index, yaw_deg in enumerate(yaws_deg)
    ]
    return scenarios, None


def mean_or_none(values):
    """Return float mean or None when the list is empty."""
    finite_values = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite_values)) if finite_values else None


def format_metric(value, fmt: str = ".2f", suffix: str = "") -> str:
    """Format optional scalar metrics for console output."""
    if value is None:
        return "N/A"
    return f"{value:{fmt}}{suffix}"


def build_summary(results):
    """Aggregate per-episode metrics into paper-friendly summary statistics."""
    successes = results["successes"]
    success_indices = [i for i, flag in enumerate(successes) if float(flag) == 1.0]

    def select(values):
        return [values[i] for i in success_indices]

    summary = {
        "episodes": len(successes),
        "success_rate_pct": float(np.mean(successes) * 100.0) if successes else 0.0,
        "collision_rate_pct": float(np.mean(results["crashes"]) * 100.0) if results["crashes"] else 0.0,
        "timeout_rate_pct": float(np.mean(results["timeouts"]) * 100.0) if results["timeouts"] else 0.0,
        "out_of_bounds_rate_pct": float(np.mean(results["out_of_bounds"]) * 100.0) if results["out_of_bounds"] else 0.0,
        "avg_reward": float(np.mean(results["rewards"])) if results["rewards"] else 0.0,
        "reward_std": float(np.std(results["rewards"])) if results["rewards"] else 0.0,
        "spl": float(np.mean(results["spls"])) if results["spls"] else 0.0,
        "avg_path_length_m": mean_or_none(results["path_lengths"]),
        "avg_path_length_success_m": mean_or_none(select(results["path_lengths"])),
        "avg_steps": mean_or_none(results["episode_steps"]),
        "avg_steps_success": mean_or_none(select(results["episode_steps"])),
        "avg_time_sec": mean_or_none(results["episode_times_sec"]),
        "avg_time_success_sec": mean_or_none(select(results["episode_times_sec"])),
        "avg_nominal_control_time_sec": mean_or_none(results["nominal_control_times_sec"]),
        "avg_state_timestamp_delta": mean_or_none(results["state_timestamp_deltas"]),
        "avg_speed_mps": mean_or_none(results["avg_speeds_mps"]),
        "avg_speed_success_mps": mean_or_none(select(results["avg_speeds_mps"])),
        "max_speed_mps": mean_or_none(results["max_speeds_mps"]),
        "avg_min_obstacle_dist_m": mean_or_none(results["min_obstacle_dists"]),
        "smoothness": mean_or_none(results["smoothness"]),
        "control_effort": mean_or_none(results["control_effort"]),
        "mean_yaw_rate": mean_or_none(results["mean_yaw_rate"]),
        "latency_ms": mean_or_none(results["inference_times_ms"]),
    }
    return summary


def save_summary_files(summary, eval_dir):
    """Save scalar summary metrics in JSON and CSV formats."""
    json_path = os.path.join(eval_dir, "summary_metrics.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)

    csv_path = os.path.join(eval_dir, "summary_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, "" if value is None else value])


def get_terminal_status(is_success: bool, is_crash: bool, is_outside: bool) -> str:
    """Convert termination flags to a single episode status label."""
    if is_success:
        return "success"
    if is_crash:
        return "collision"
    if is_outside:
        return "out_of_bounds"
    return "timeout"


def save_trajectory_csvs(results, eval_dir):
    """Export each episode trajectory to a standalone CSV file."""
    outdir = os.path.join(eval_dir, "trajectories_csv")
    os.makedirs(outdir, exist_ok=True)

    max_action_dim = 0
    for actions in results["step_actions"]:
        for action in actions:
            max_action_dim = max(max_action_dim, len(action))

    for ep_idx, trajectory in enumerate(results["trajectories"], start=1):
        status = results["terminal_statuses"][ep_idx - 1]
        csv_path = os.path.join(outdir, f"episode_{ep_idx:03d}_{status}.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "episode",
                "point_index",
                "step_index",
                "x",
                "y",
                "z",
                "obstacle_dist_m",
                "yaw_rate_rad_s",
                "speed_mps",
                "is_terminal",
                "terminal_type",
            ] + [f"action_{dim}" for dim in range(max_action_dim)]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for point_index, point in enumerate(trajectory):
                metric_index = point_index - 1
                row = {
                    "episode": ep_idx,
                    "point_index": point_index,
                    "step_index": "" if metric_index < 0 else metric_index + 1,
                    "x": point[0] if len(point) > 0 else "",
                    "y": point[1] if len(point) > 1 else "",
                    "z": point[2] if len(point) > 2 else "",
                    "obstacle_dist_m": "",
                    "yaw_rate_rad_s": "",
                    "speed_mps": "",
                    "is_terminal": int(point_index == len(trajectory) - 1),
                    "terminal_type": status if point_index == len(trajectory) - 1 else "",
                }

                if metric_index >= 0:
                    if metric_index < len(results["step_obstacle_dists"][ep_idx - 1]):
                        row["obstacle_dist_m"] = results["step_obstacle_dists"][ep_idx - 1][metric_index]
                    if metric_index < len(results["step_yaw_rates"][ep_idx - 1]):
                        row["yaw_rate_rad_s"] = results["step_yaw_rates"][ep_idx - 1][metric_index]
                    if metric_index < len(results["step_speeds_mps"][ep_idx - 1]):
                        row["speed_mps"] = results["step_speeds_mps"][ep_idx - 1][metric_index]
                    if metric_index < len(results["step_actions"][ep_idx - 1]):
                        action = results["step_actions"][ep_idx - 1][metric_index]
                        for dim in range(max_action_dim):
                            row[f"action_{dim}"] = action[dim] if dim < len(action) else ""

                writer.writerow(row)


def save_terminal_event_files(results, eval_dir):
    """Export terminal points in one master CSV plus per-event CSV files."""
    rows = []
    for ep_idx, trajectory in enumerate(results["trajectories"], start=1):
        start_point = results["start_positions"][ep_idx - 1]
        goal_point = results["goal_positions"][ep_idx - 1]
        terminal_point = results["terminal_positions"][ep_idx - 1]
        status = results["terminal_statuses"][ep_idx - 1]
        rows.append(
            {
                "episode": ep_idx,
                "terminal_type": status,
                "start_x": start_point[0],
                "start_y": start_point[1],
                "start_z": start_point[2],
                "goal_x": goal_point[0],
                "goal_y": goal_point[1],
                "goal_z": goal_point[2],
                "terminal_x": terminal_point[0],
                "terminal_y": terminal_point[1],
                "terminal_z": terminal_point[2],
                "reward": results["rewards"][ep_idx - 1],
                "path_length_m": results["path_lengths"][ep_idx - 1],
                "steps": results["episode_steps"][ep_idx - 1],
                "time_sec": results["episode_times_sec"][ep_idx - 1],
                "avg_speed_mps": results["avg_speeds_mps"][ep_idx - 1],
                "spl": results["spls"][ep_idx - 1],
                "min_obstacle_dist_m": results["min_obstacle_dists"][ep_idx - 1],
            }
        )

    fieldnames = list(rows[0].keys()) if rows else [
        "episode",
        "terminal_type",
        "start_x",
        "start_y",
        "start_z",
        "goal_x",
        "goal_y",
        "goal_z",
        "terminal_x",
        "terminal_y",
        "terminal_z",
        "reward",
        "path_length_m",
        "steps",
        "time_sec",
        "avg_speed_mps",
        "spl",
        "min_obstacle_dist_m",
    ]

    master_csv = os.path.join(eval_dir, "terminal_events.csv")
    with open(master_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_mapping = {
        "success": "success_points.csv",
        "collision": "collision_points.csv",
        "timeout": "timeout_points.csv",
        "out_of_bounds": "out_of_bounds_points.csv",
    }
    for terminal_type, filename in split_mapping.items():
        with open(os.path.join(eval_dir, filename), "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows([row for row in rows if row["terminal_type"] == terminal_type])


def evaluate(model, env, num_episodes: int, use_history: bool = True):
    """Run evaluation and collect per-episode and per-step statistics."""
    results = {
        "rewards": [],
        "successes": [],
        "crashes": [],
        "timeouts": [],
        "out_of_bounds": [],
        "path_lengths": [],
        "spls": [],
        "smoothness": [],
        "inference_times_ms": [],
        "control_effort": [],
        "mean_yaw_rate": [],
        "episode_steps": [],
        "episode_times_sec": [],
        "nominal_control_times_sec": [],
        "state_timestamp_starts": [],
        "state_timestamp_ends": [],
        "state_timestamp_deltas": [],
        "avg_speeds_mps": [],
        "max_speeds_mps": [],
        "min_obstacle_dists": [],
        "start_positions": [],
        "goal_positions": [],
        "terminal_positions": [],
        "terminal_statuses": [],
        "trajectories": [],
        "step_obstacle_dists": [],
        "step_yaw_rates": [],
        "step_speeds_mps": [],
        "step_actions": [],
        "scenario_ids": [],
        "scenario_yaws_deg": [],
    }

    base_env = get_base_env(env)
    action_dim = env.action_space.shape[-1]
    control_dt = float(getattr(base_env, "action_repeat", 1) * getattr(base_env.dynamic_model, "dt", 0.0))
    print(f"  [INFO] Direct control evaluation (action_dim={action_dim})")
    print(f"  [INFO] Nominal action duration: {control_dt:.3f}s")

    # DummyVecEnv automatically resets after a terminal step. Reset once here,
    # then consume that auto-reset observation at the start of each next episode.
    base_env.record_evaluation_timing = True
    obs = env.reset()
    for ep in range(num_episodes):
        episode_start_wall = base_env.episode_start_wall_time
        start_timestamp = base_env.episode_start_state_timestamp
        episode_scenario = dict(getattr(base_env, "current_scenario", None) or {})
        episode_reward = 0.0
        trajectory = []
        actions = []
        inference_times = []
        obstacle_dists = []
        yaw_rates = []
        speeds = []

        start_pos = np.zeros(3, dtype=np.float32)
        goal_pos = np.zeros(3, dtype=np.float32)
        optimal_distance = 1.0
        try:
            start_pos = np.array(base_env.dynamic_model.get_position())
            goal_pos = np.array(base_env.dynamic_model.goal_position)
            optimal_distance = float(np.linalg.norm(goal_pos - start_pos))
            trajectory.append(start_pos.tolist())
        except Exception:
            trajectory.append(start_pos.tolist())

        episode_start = np.array([True])

        while True:
            t0 = time.perf_counter()
            predict_kwargs = {
                "deterministic": True,
                "episode_start": episode_start,
            }
            if not use_history:
                predict_kwargs["use_history"] = False
            try:
                action, _ = model.predict(obs, **predict_kwargs)
            except TypeError as exc:
                if not use_history:
                    raise RuntimeError(
                        "--no-history / use_history=False requires MambaSAC.predict"
                    ) from exc
                raise
            t1 = time.perf_counter()

            inference_times.append((t1 - t0) * 1000.0)
            episode_start = np.array([False])

            obs, rewards, dones, infos = env.step(action)
            step_info = infos[0] if infos else {}
            executed_action = np.asarray(step_info.get("actual_action", action[0]), dtype=np.float32)
            actions.append(executed_action.copy())
            episode_reward += rewards[0]

            obs_dist = float(step_info.get("min_distance_to_obstacles", float("inf")))
            if not np.isfinite(obs_dist):
                try:
                    obs_dist = float(base_env.min_distance_to_obstacles)
                except Exception:
                    obs_dist = float("inf")
            obstacle_dists.append(obs_dist)

            velocity = step_info.get("velocity")
            if velocity is not None and len(velocity) >= 3:
                yaw_rate = abs(float(velocity[2]))
                world_velocity = step_info.get('velocity_world')
                if world_velocity is None:
                    raise RuntimeError('Evaluation requires measured three-dimensional world velocity')
                speed = float(np.linalg.norm(np.asarray(world_velocity, dtype=np.float32)))
            else:
                try:
                    vel = base_env.dynamic_model.get_velocity()
                    yaw_rate = abs(vel[2])
                    speed = float(np.linalg.norm(np.asarray(base_env.dynamic_model.last_velocity_world, dtype=np.float32)))
                except Exception:
                    yaw_rate = 0.0
                    speed = 0.0
            yaw_rates.append(yaw_rate)
            speeds.append(speed)

            if dones[0]:
                terminal_position = step_info.get("terminal_position", step_info.get("position"))
                if terminal_position is not None:
                    trajectory.append(list(terminal_position))
                break

            position = step_info.get("position")
            if position is not None:
                trajectory.append(list(position))
            else:
                try:
                    cur_pos = np.array(base_env.dynamic_model.get_position())
                    trajectory.append(cur_pos.tolist())
                except Exception:
                    pass

        info = infos[0] if infos else {}
        is_success = info.get("is_success", False)
        is_crash = info.get("is_crash", False)
        is_outside = info.get("is_not_in_workspace", False)
        is_timeout = not is_success and not is_crash and not is_outside

        path_length = 0.0
        if len(trajectory) > 1:
            traj_np = np.array(trajectory)
            path_length = float(np.sum(np.linalg.norm(traj_np[1:] - traj_np[:-1], axis=1)))

        spl = 0.0
        if is_success:
            denom = max(optimal_distance, path_length)
            spl = optimal_distance / denom if denom > 0 else 0.0

        acts_np = np.asarray(actions)
        smoothness = 0.0
        if len(actions) > 1:
            smoothness = float(np.mean(np.linalg.norm(acts_np[1:] - acts_np[:-1], axis=1) ** 2))

        control_effort = 0.0
        if actions:
            control_effort = float(np.sum(np.linalg.norm(acts_np, axis=1) ** 2))

        episode_steps = int(info.get("step_num", len(actions)))
        nominal_time_sec = float(episode_steps * control_dt)
        episode_time_sec = float(info['wall_monotonic_time'] - episode_start_wall)
        if episode_time_sec <= 0:
            raise RuntimeError('Evaluation wall time must be positive')
        end_timestamp = int(info['state_timestamp'])
        timestamp_delta = end_timestamp - int(start_timestamp)
        if timestamp_delta < 0:
            raise RuntimeError('AirSim state timestamp moved backwards during an episode')
        avg_speed = float(path_length / episode_time_sec) if episode_time_sec > 0 else 0.0
        max_speed = float(np.max(speeds)) if speeds else 0.0
        min_obstacle_dist = float(np.min(obstacle_dists)) if obstacle_dists else float("inf")
        mean_yaw = float(np.mean(yaw_rates)) if yaw_rates else 0.0
        avg_inference_time = float(np.mean(inference_times)) if inference_times else 0.0

        results["rewards"].append(episode_reward)
        results["successes"].append(float(is_success))
        results["crashes"].append(float(is_crash))
        results["timeouts"].append(float(is_timeout))
        results["out_of_bounds"].append(float(is_outside))
        results["path_lengths"].append(path_length)
        results["spls"].append(spl)
        results["smoothness"].append(smoothness)
        results["inference_times_ms"].append(avg_inference_time)
        results["control_effort"].append(control_effort)
        results["mean_yaw_rate"].append(mean_yaw)
        results["episode_steps"].append(episode_steps)
        results["episode_times_sec"].append(episode_time_sec)
        results["nominal_control_times_sec"].append(nominal_time_sec)
        results["state_timestamp_starts"].append(int(start_timestamp))
        results["state_timestamp_ends"].append(end_timestamp)
        results["state_timestamp_deltas"].append(timestamp_delta)
        results["avg_speeds_mps"].append(avg_speed)
        results["max_speeds_mps"].append(max_speed)
        results["min_obstacle_dists"].append(min_obstacle_dist)
        results["start_positions"].append(start_pos.tolist())
        results["goal_positions"].append(goal_pos.tolist())
        results["terminal_positions"].append(np.asarray(trajectory[-1], dtype=np.float32).tolist())
        results["terminal_statuses"].append(get_terminal_status(is_success, is_crash, is_outside))
        results["trajectories"].append(trajectory)
        results["step_obstacle_dists"].append(obstacle_dists)
        results["step_yaw_rates"].append(yaw_rates)
        results["step_speeds_mps"].append(speeds)
        results["step_actions"].append([a.tolist() for a in actions])
        results["scenario_ids"].append(episode_scenario.get("scenario_id"))
        results["scenario_yaws_deg"].append(episode_scenario.get("yaw_deg"))

        status = "[Success]" if is_success else ("[Crash]" if is_crash else ("[OOB]" if is_outside else "[Timeout]"))
        print(
            f"Ep {ep + 1}/{num_episodes}: {status} | SPL: {spl:.3f} | Len: {path_length:.1f}m | "
            f"Steps: {episode_steps:d} | Time: {episode_time_sec:.1f}s | Speed: {avg_speed:.2f}m/s | "
            f"Smooth: {smoothness:.4f} | Yaw: {mean_yaw:.4f} | Effort: {control_effort:.1f} | "
            f"Latency: {avg_inference_time:.1f}ms"
        )

    return results


def main():
    args = parse_args()

    if not args.model_path:
        raise ValueError("Please provide --model_path (-m), e.g. logs/Tunnel/<run_name>")

    config_file = os.path.join(args.model_path, "config", "config.ini")
    final_model_file = os.path.join(args.model_path, "models", "model_sb3.zip")
    best_model_file = os.path.join(args.model_path, "models", "best_model.zip")
    checkpoint_file = None
    if args.checkpoint_step is not None:
        checkpoint_file = os.path.join(args.model_path, "models", f"checkpoint_{args.checkpoint_step}_steps.zip")

    if args.final:
        print("[WARN] --final is deprecated; final model is already the default.")

    if checkpoint_file is not None:
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Requested checkpoint not found: {checkpoint_file}")
        model_file = checkpoint_file
        print(f"Using checkpoint: checkpoint_{args.checkpoint_step}_steps.zip")
    elif args.best:
        if not os.path.exists(best_model_file):
            raise FileNotFoundError(f"Best model not found: {best_model_file}")
        model_file = best_model_file
        print("Using best model (best_model.zip)")
    elif os.path.exists(final_model_file):
        model_file = final_model_file
        print("Using final model")
    else:
        model_file = None

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config not found: {config_file}")

    if not model_file or not os.path.exists(model_file):
        models_dir = os.path.join(args.model_path, "models")
        checkpoints = [f for f in os.listdir(models_dir) if f.startswith("checkpoint") and f.endswith(".zip")]
        if not checkpoints:
            raise FileNotFoundError(f"No model found in: {models_dir}")
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        model_file = os.path.join(models_dir, checkpoints[-1])
        print(f"Using checkpoint: {checkpoints[-1]}")

    cfg = ConfigParser()
    cfg.read(config_file)
    if cfg.getboolean("options", "use_pid_residual", fallback=False):
        raise ValueError(
            "This checkpoint was trained with legacy PID control and is incompatible with the current NoPID runtime."
        )

    env_name_before = cfg.get("options", "env_name")
    if args.env_name:
        print(f"Overriding environment: {env_name_before} -> {args.env_name}")
    apply_evaluation_overrides(
        cfg,
        env_name=args.env_name,
        airsim_ip=args.airsim_ip,
        airsim_port=args.airsim_port,
    )
    env_name = cfg.get("options", "env_name")
    algo = cfg.get("options", "algo")
    scenarios, scenario_manifest_source = build_evaluation_scenarios(
        cfg,
        args.episodes,
        args.scenario_seed,
        args.yaw_range_deg,
        args.scenario_manifest,
    )

    print(f"Model: {args.model_path}")
    print(f"Environment: {env_name}, Algorithm: {algo}")
    print(f"Episodes: {args.episodes}")
    if args.no_history and args.inference_seq_len is not None and args.inference_seq_len > 1:
        raise ValueError("Cannot combine --no-history with --inference-seq-len > 1")
    effective_seq_len = 1 if args.no_history else args.inference_seq_len
    use_history = effective_seq_len is None or effective_seq_len > 1
    if not use_history:
        history_mode, history_note = "trunc", "DISABLED (Trunc / L=1)"
    elif effective_seq_len is None:
        history_mode, history_note = "full", "enabled (Full)"
    else:
        history_mode, history_note = f"L{effective_seq_len}", f"override L={effective_seq_len}"
    print(f"History: {history_note}")
    print("-" * 70)

    env = build_runtime_env(cfg)
    np.random.seed(args.scenario_seed)
    env.seed(args.scenario_seed)
    base_env = get_base_env(env)
    # DummyVecEnv performs one automatic reset after the final terminal step.
    base_env.set_evaluation_scenarios(scenarios + [scenarios[-1]])
    model = load_model(algo, model_file, env, cfg=cfg)
    if hasattr(model, "policy"):
        model.policy.set_training_mode(False)

    if args.inference_seq_len is not None:
        actor = getattr(getattr(model, "policy", None), "actor", None)
        if actor is None or not hasattr(actor, "set_inference_seq_len"):
            raise ValueError(
                f"--inference-seq-len requires Mamba actor with set_inference_seq_len; "
                f"got {type(model).__name__}"
            )
        actor.set_inference_seq_len(args.inference_seq_len)
        print(f"Actor inference_seq_len set to {actor.inference_seq_len}")

    if (not use_history) and "use_history" not in inspect.signature(model.predict).parameters:
        raise ValueError(
            f"--no-history / inference-seq-len=1 requires MambaSAC.predict; got {type(model).__name__}"
        )
    results = evaluate(model, env, args.episodes, use_history=use_history)
    summary = build_summary(results)

    print("=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)

    n = len(results["successes"])
    n_success = sum(1 for s in results["successes"] if s == 1.0)
    n_crash = sum(1 for c in results["crashes"] if c == 1.0)
    n_timeout = sum(1 for t in results["timeouts"] if t == 1.0)
    n_oob = sum(1 for o in results["out_of_bounds"] if o == 1.0)

    print(f"\n  Outcome Breakdown ({n} episodes):")
    print(f"    Success Rate:      {summary['success_rate_pct']:6.1f}%  ({n_success}/{n})")
    print(f"    Collision Rate:    {summary['collision_rate_pct']:6.1f}%  ({n_crash}/{n})")
    print(f"    Timeout Rate:      {summary['timeout_rate_pct']:6.1f}%  ({n_timeout}/{n})")
    print(f"    Out-of-Bounds:     {summary['out_of_bounds_rate_pct']:6.1f}%  ({n_oob}/{n})")

    print("\n  Navigation Efficiency:")
    print(f"    SPL:               {summary['spl']:.3f}")
    print(f"    Avg Path Length:   {format_metric(summary['avg_path_length_m'], '.1f', ' m')}  (all episodes)")
    print(f"    Success Path Len:  {format_metric(summary['avg_path_length_success_m'], '.1f', ' m')}  (success only)")
    print(f"    Avg Steps:         {format_metric(summary['avg_steps'], '.1f')}")
    print(f"    Success Steps:     {format_metric(summary['avg_steps_success'], '.1f')}")
    print(f"    Avg Wall Time:     {format_metric(summary['avg_time_sec'], '.2f', ' s')}")
    print(f"    Avg Nominal Time:  {format_metric(summary['avg_nominal_control_time_sec'], '.2f', ' s')}")
    print(f"    Success Time:      {format_metric(summary['avg_time_success_sec'], '.2f', ' s')}")
    print(f"    Avg Speed (wall):  {format_metric(summary['avg_speed_mps'], '.2f', ' m/s')}")
    print(f"    Success Speed:     {format_metric(summary['avg_speed_success_mps'], '.2f', ' m/s')}")
    print(f"    Max Speed:         {format_metric(summary['max_speed_mps'], '.2f', ' m/s')}")
    print(f"    Min Obs Dist:      {format_metric(summary['avg_min_obstacle_dist_m'], '.2f', ' m')}")

    print("\n  Control Quality:")
    print(f"    Act Smoothness:    {format_metric(summary['smoothness'], '.4f')}  (MSE, lower=better)")
    print(f"    Control Effort:    {format_metric(summary['control_effort'], '.1f')}  (sum(||a||^2), lower=less energy)")
    print(f"    Mean |Yaw Rate|:   {format_metric(summary['mean_yaw_rate'], '.4f')}  (rad/s, lower=smoother)")

    print("\n  Inference Performance:")
    print(f"    Mean Latency:      {format_metric(summary['latency_ms'], '.2f', ' ms/step')}")
    print(f"    Max Latency:       {np.max(results['inference_times_ms']):.2f} ms/step")

    print("\n  Reward:")
    print(f"    Average Reward:    {summary['avg_reward']:.2f} +/- {summary['reward_std']:.2f}")
    print("=" * 70)

    if args.save:
        save_tag = args.save_tag or args.env_name
        history_tag = f"history_{history_mode}"
        if history_tag not in str(save_tag):
            save_tag = f"{save_tag}_{history_tag}" if save_tag else history_tag

        if args.best:
            model_type = "best"
        elif args.checkpoint_step is not None:
            model_type = f"checkpoint_{args.checkpoint_step}"
        else:
            model_type = "final"

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        eval_dir_name = f"eval_{args.episodes}_{model_type}"
        if save_tag:
            eval_dir_name = f"{eval_dir_name}_{save_tag}"
        eval_dir_name = f"{eval_dir_name}_{timestamp}"

        eval_dir = evaluation_parent_for_run(args.model_path, args.output_root) / eval_dir_name
        eval_dir.mkdir(parents=True, exist_ok=False)

        evaluation_manifest = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_run": str(Path(args.model_path).resolve()),
            "model_file": str(Path(model_file).resolve()),
            "result_dir": str(eval_dir.resolve()),
            "model_type": model_type,
            "episodes": args.episodes,
            "environment": env_name,
            "algorithm": algo,
            "use_history": use_history,
            "history_mode": history_mode,
            "inference_seq_len": effective_seq_len,
            "scenario_seed": args.scenario_seed,
            "yaw_range_deg": args.yaw_range_deg,
            "scenario_manifest_source": scenario_manifest_source,
            "time_metric": "wall time from post-reset observation to terminal step; excludes auto-reset",
            "nominal_time_metric": "number of actions times commanded action duration",
            "state_timestamp_metric": "raw AirSim MultirotorState timestamp; units and clock semantics unverified",
            "speed_metric": "3D path distance / wall time; max is sampled 3D world velocity",
        }
        with open(
            os.path.join(eval_dir, "evaluation_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(evaluation_manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        with open(
            os.path.join(eval_dir, "scenario_manifest.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump({"scenarios": scenarios}, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        np.save(
            f"{eval_dir}/results.npy",
            {
                "rewards": results["rewards"],
                "successes": results["successes"],
                "crashes": results["crashes"],
                "timeouts": results["timeouts"],
                "path_lengths": results["path_lengths"],
                "spls": results["spls"],
                "smoothness": results["smoothness"],
                "inference_times_ms": results["inference_times_ms"],
                "out_of_bounds": results["out_of_bounds"],
                "control_effort": results["control_effort"],
                "mean_yaw_rate": results["mean_yaw_rate"],
                "episode_steps": results["episode_steps"],
                "episode_times_sec": results["episode_times_sec"],
                "nominal_control_times_sec": results["nominal_control_times_sec"],
                "state_timestamp_starts": results["state_timestamp_starts"],
                "state_timestamp_ends": results["state_timestamp_ends"],
                "state_timestamp_deltas": results["state_timestamp_deltas"],
                "avg_speeds_mps": results["avg_speeds_mps"],
                "max_speeds_mps": results["max_speeds_mps"],
                "min_obstacle_dists": results["min_obstacle_dists"],
                "start_positions": results["start_positions"],
                "goal_positions": results["goal_positions"],
                "terminal_positions": results["terminal_positions"],
                "terminal_statuses": results["terminal_statuses"],
                "scenario_ids": results["scenario_ids"],
                "scenario_yaws_deg": results["scenario_yaws_deg"],
            },
        )

        np.save(f"{eval_dir}/trajectories.npy", np.array(results["trajectories"], dtype=object))

        np.save(
            f"{eval_dir}/step_data.npy",
            {
                "step_obstacle_dists": results["step_obstacle_dists"],
                "step_yaw_rates": results["step_yaw_rates"],
                "step_speeds_mps": results["step_speeds_mps"],
                "step_actions": results["step_actions"],
            },
        )
        save_summary_files(summary, eval_dir)
        save_trajectory_csvs(results, eval_dir)
        save_terminal_event_files(results, eval_dir)

        print(f"\nResults saved to: {eval_dir}")
        print("  results.npy      - per-episode scalar metrics")
        print("  trajectories.npy - per-episode 3D trajectories")
        print("  trajectories_csv - one CSV per episode trajectory")
        print("  step_data.npy    - per-step time-series (obstacle dist, yaw rate, speed, actions)")
        print("  terminal_events.csv + *_points.csv - terminal markers for plotting")
        print("  summary_metrics.json/.csv - paper-ready aggregate metrics")
        print("  evaluation_manifest.json - evaluated model and history protocol")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
