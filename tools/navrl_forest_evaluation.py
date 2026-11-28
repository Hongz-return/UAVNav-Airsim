"""Evaluate a MA-SAC checkpoint on a dedicated NavRL forest simulator."""

from configparser import ConfigParser
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from itertools import pairwise
import json
import logging
import math
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from zlib import crc32

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


def _evaluate_scenario(model, gym_env, transpose, scenario, *, seed=None, on_reset=None):
    scenario_id = scenario["scenario_id"]
    obs, info = gym_env.reset(seed=seed)
    if info["scenario_id"] != scenario_id:
        raise RuntimeError(f"AirSim reset to {info['scenario_id']!r}, expected {scenario_id!r}.")
    if on_reset is not None:
        on_reset(scenario_id)

    path = None
    if "crossing_axis" in scenario:
        # The synchronous forest reset already reads and validates this pose.
        # Use the settled position and avoid another trip to the simulator.
        reset_state = gym_env.unwrapped.dynamic_model.last_reset_state
        path = [tuple(reset_state["position_ned_m"][:2])]
        goal_ned = scenario["goal_position"]
        goal_up = (goal_ned[0], goal_ned[1], -goal_ned[2])
        closest = math.dist(reset_state["position_ned_m"], goal_ned)

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
            closest = min(closest, math.dist(info["position"], goal_up))
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
        outcome["closest_goal_distance_m"] = closest
    return outcome


def evaluate_checkpoint(
    checkpoint: Path, cfg: ConfigParser, scenarios: list[dict], *, seed: int, on_outcome=None, ports=None,
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

    ports = list(ports) if ports is not None else [port]
    if not ports or len(set(ports)) != len(ports) or any(not 1 <= p <= 65535 for p in ports):
        raise ValueError("Evaluation ports must be distinct valid RPC ports.")
    work = Queue()
    for index, scenario in enumerate(scenarios):
        case_seed = seed if index == 0 else None
        if len(ports) > 1:
            case_seed = (seed + crc32(scenario["scenario_id"].encode())) % (2 ** 32)
        work.put((scenario, case_seed))
    outcomes = {}
    save_lock = Lock()
    model_lock = Lock()

    def save(row):
        with save_lock:
            outcomes[row["scenario_id"]] = row
            if on_outcome is not None:
                on_outcome(row)

    def fly(port_number):
        eval_cfg = deepcopy(cfg)
        eval_cfg.set("airsim", "rpc_port", str(port_number))
        marker = "active_scenario" if len(ports) == 1 else f"active_scenario_{port_number}"
        _fly_queue(checkpoint, eval_cfg, work, save, model_lock, marker, port_number)

    if len(ports) == 1:
        fly(ports[0])
    else:
        with ThreadPoolExecutor(max_workers=len(ports)) as pool:
            list(pool.map(fly, ports))
    return [outcomes[key] for key in scenario_ids]


def _fly_queue(checkpoint, eval_cfg, work, save, model_lock, marker, port_number):
    round_dir = checkpoint.parent
    publish_goals = (round_dir / "scenario_manifest.json").is_file()
    radius = eval_cfg.getfloat("environment", "accept_radius", fallback=0.5)
    def publish(scenario_id):
        if publish_goals:
            try:
                pending = round_dir / (marker + ".tmp")
                pending.write_text(json.dumps({"scenario_id": scenario_id, "accept_radius_m": radius}), encoding="utf-8")
                pending.replace(round_dir / (marker + ".json"))
            except OSError as exc:
                logging.getLogger(__name__).warning("Cannot publish evaluation goal: %s", exc)

    env = build_runtime_env(eval_cfg)
    # build_runtime_env returns VecTransposeImage(DummyVecEnv([monitored_env])).
    gym_env = env.venv.envs[0]
    try:
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        # Loading an independent policy constructs Torch modules. Preserve the
        # training RNG state across that initialization and the evaluation.
        with model_lock, torch.random.fork_rng(devices=devices):
            model = load_model("SAC", str(checkpoint), env, cfg=eval_cfg, load_replay=False)
            model.policy.set_training_mode(False)
        while True:
            try:
                scenario, case_seed = work.get_nowait()
            except Empty:
                break
            publish(None)
            gym_env.unwrapped.set_evaluation_scenarios([scenario])
            row = _evaluate_scenario(model, gym_env, env.transpose_observations, scenario,
                                     seed=case_seed, on_reset=publish)
            row["evaluation_port"] = port_number
            save(row)
    finally:
        publish(None)
        try:
            env.close()
        finally:
            gym_env.unwrapped.client.client.close()
