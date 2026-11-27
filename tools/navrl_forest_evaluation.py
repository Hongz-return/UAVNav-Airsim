"""Evaluate a MA-SAC checkpoint on a dedicated NavRL forest simulator."""

from configparser import ConfigParser
from copy import deepcopy
from itertools import pairwise
import math
from pathlib import Path

import numpy as np
import torch

from tools.runtime_support import build_runtime_env, load_model, use_mamba_backend


# The evaluation core is the central 40 m square. AirSim's x origin is
# at world x = -24 m, and its horizontal y coordinate uses the world sign.
CORE_BOUNDS = ((4.0, 44.0), (-20.0, 20.0))


def _core_segment_fraction(start, end):
    """Clip a horizontal segment to the core and return its length fraction."""
    enter, leave = 0.0, 1.0
    for origin, destination, (lower, upper) in zip(start, end, CORE_BOUNDS):
        delta = destination - origin
        if delta == 0:
            if not lower <= origin <= upper:
                return 0.0
            continue
        near, far = sorted(((lower - origin) / delta, (upper - origin) / delta))
        enter, leave = max(enter, near), min(leave, far)
        if enter >= leave:
            return 0.0
    return leave - enter


def _path_metrics(positions, crossing_axis, goal):
    """Measure the sampled horizontal trajectory and its directed core crossing."""
    if type(crossing_axis) is not int or crossing_axis not in (0, 1):
        raise ValueError("NavRL crossing_axis must be 0 (x) or 1 (y).")
    if any(not math.isfinite(coordinate) for point in positions for coordinate in point):
        raise ValueError("NavRL trajectory contains a non-finite position.")

    tangent_axis = 1 - crossing_axis
    lower, upper = CORE_BOUNDS[tangent_axis]
    center = sum(CORE_BOUNDS[crossing_axis]) / 2.0
    direction = 1 if goal[crossing_axis] > center else -1
    last_side = (positions[0][crossing_axis] - center) * direction
    horizontal_path = core_path = 0.0
    crossed_core = False
    for start, end in pairwise(positions):
        segment_length = math.dist(start, end)
        horizontal_path += segment_length
        core_path += segment_length * _core_segment_fraction(start, end)
        end_side = (end[crossing_axis] - center) * direction
        if not crossed_core and last_side < 0 < end_side:
            start_side = (start[crossing_axis] - center) * direction
            crossing = -start_side / (end_side - start_side)
            tangent = start[tangent_axis] + crossing * (end[tangent_axis] - start[tangent_axis])
            crossed_core = lower < tangent < upper
        # A sample exactly on the centerline does not yet prove a crossing.
        if end_side != 0:
            last_side = end_side

    return {
        "horizontal_path_m": horizontal_path,
        "core_path_fraction": core_path / horizontal_path if horizontal_path else 0.0,
        "crossed_core": crossed_core,
    }


def _evaluate_scenario(model, gym_env, transpose, scenario, *, seed=None):
    scenario_id = scenario["scenario_id"]
    obs, info = gym_env.reset(seed=seed)
    if info["scenario_id"] != scenario_id:
        raise RuntimeError(f"AirSim reset to {info['scenario_id']!r}, expected {scenario_id!r}.")

    path = None
    if "crossing_axis" in scenario:
        # The synchronous forest reset already reads and validates this pose.
        # Use the settled position and avoid another trip to the simulator.
        reset_state = gym_env.unwrapped.dynamic_model.last_reset_state
        path = [tuple(reset_state["position_ned_m"][:2])]

    episode_start = np.ones(1, dtype=bool)
    collision = False
    # The raw Gym wrapper has the same frame stack as training and ends at the
    # terminal observation without DummyVecEnv's implicit next reset.
    for steps in range(1, gym_env.unwrapped.max_episode_steps + 1):
        action, _ = model.predict(
            transpose(obs), episode_start=episode_start, deterministic=True,
        )
        episode_start[0] = False
        obs, _, terminated, truncated, info = gym_env.step(action)
        collision |= bool(info["is_crash"] or info.get("collision_event", False))
        if path is not None:
            path.append(tuple(info["position"][:2]))
        if terminated or truncated:
            break
    else:
        raise RuntimeError(f"Evaluation episode {scenario_id!r} exceeded its step limit.")

    goal_reached = bool(info["is_success"])
    if collision:
        terminal_type = "collision"
    elif info["is_not_in_workspace"]:
        terminal_type = "out_of_bounds"
    elif goal_reached:
        terminal_type = "success"
    elif truncated:
        terminal_type = "timeout"
    else:
        raise RuntimeError(f"Evaluation episode {scenario_id!r} has no terminal outcome.")
    outcome = {
        "scenario_id": scenario_id,
        "terminal_type": terminal_type,
        "goal_reached": goal_reached,
        "collision": collision,
        "collision_free_success": terminal_type == "success",
        "steps": steps,
    }
    if path is not None:
        outcome.update(_path_metrics(path, scenario["crossing_axis"], scenario["goal_position"]))
    return outcome


def evaluate_checkpoint(
    checkpoint: Path, cfg: ConfigParser, scenarios: list[dict], *, seed: int,
) -> list[dict]:
    """Run the ordered cases once each against the dedicated RPC port in cfg."""
    checkpoint = Path(checkpoint)
    if checkpoint.suffix != ".zip":
        checkpoint = Path(str(checkpoint) + ".zip")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    port = cfg.getint("airsim", "rpc_port", fallback=0)
    if not (1 <= port <= 65535 and cfg.has_section("navrl_forest")):
        raise ValueError("Forest evaluation requires [navrl_forest] and an explicit AirSim RPC port.")
    if cfg.get("options", "algo") != "SAC" or not use_mamba_backend(cfg):
        raise ValueError("Forest evaluation requires the MA-SAC backend.")
    scenario_ids = [scenario["scenario_id"] for scenario in scenarios]
    if not scenario_ids or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Evaluation scenarios must have unique IDs and cannot be empty.")

    eval_cfg = deepcopy(cfg)
    env = build_runtime_env(eval_cfg)
    # build_runtime_env returns VecTransposeImage(DummyVecEnv([monitored_env])).
    gym_env = env.venv.envs[0]
    try:
        gym_env.unwrapped.set_evaluation_scenarios(scenarios)
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        # Loading an independent policy constructs Torch modules. Preserve the
        # training RNG state across that initialization and the evaluation.
        with torch.random.fork_rng(devices=devices):
            model = load_model("SAC", str(checkpoint), env, cfg=eval_cfg, load_replay=False)
            model.policy.set_training_mode(False)
            return [
                _evaluate_scenario(
                    model, gym_env, env.transpose_observations, scenario,
                    seed=seed if index == 0 else None,
                )
                for index, scenario in enumerate(scenarios)
            ]
    finally:
        try:
            env.close()
        finally:
            gym_env.unwrapped.client.client.close()
