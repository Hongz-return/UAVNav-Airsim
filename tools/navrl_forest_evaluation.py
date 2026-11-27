"""Evaluate a MA-SAC checkpoint on a dedicated NavRL forest simulator."""

from configparser import ConfigParser
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from tools.runtime_support import build_runtime_env, load_model, use_mamba_backend


def _evaluate_scenario(model, gym_env, transpose, scenario, *, seed=None):
    scenario_id = scenario["scenario_id"]
    obs, info = gym_env.reset(seed=seed)
    if info["scenario_id"] != scenario_id:
        raise RuntimeError(f"AirSim reset to {info['scenario_id']!r}, expected {scenario_id!r}.")

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
    return {
        "scenario_id": scenario_id,
        "terminal_type": terminal_type,
        "goal_reached": goal_reached,
        "collision": collision,
        "collision_free_success": terminal_type == "success",
        "steps": steps,
    }


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
