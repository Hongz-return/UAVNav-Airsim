"""Coordinates and runtime configuration for the three UE5.9 forest levels."""

from copy import deepcopy
import math


EDGE_M = 24.0
ORIGIN_AGL_M = 1.5
VEHICLE_NAME = "SimpleFlight"
CAMERA_FORWARD_M = 0.65


def sample_training_route(rng):
    """Draw two independent edge points and return AirSim NED poses and yaw."""
    points = []
    for _ in range(2):
        side = int(rng.integers(4))
        edge = EDGE_M if side % 2 == 0 else -EDGE_M
        along = float(rng.uniform(-EDGE_M, EDGE_M))
        x, y = (edge, along) if side < 2 else (along, edge)
        points.append((x + EDGE_M, y, ORIGIN_AGL_M - float(rng.uniform(0.5, 2.5))))
    start, goal = points
    return start, goal, math.atan2(goal[1] - start[1], goal[0] - start[0])


def forest_config(base, stage):
    """Apply the open-arena geometry without changing the MA-SAC policy contract."""
    if (base.get("options", "algo") != "SAC" or base.get("options", "perception") != "depth_state"
            or not base.getboolean("options", "use_mamba")
            or not base.getboolean("options", "navigation_3d")):
        raise ValueError("The forest curriculum requires 3D depth-state MA-SAC.")
    if base.has_section("stage3") or base.has_section("stage3_v2") or base.has_option("scene", "runtime_script"):
        raise ValueError("Choose the base MA-SAC config without tunnel or sequence-scene overrides.")
    cfg = deepcopy(base)
    cfg.remove_option("environment", "training_routes")
    cfg.read_dict({
        "options": {"env_name": "Obs", "project_name": "Ch4_NavRL_Forest_MASAC"},
        "environment": {
            "start_position": [0.0, 0.0, 0.0],
            "goal_position": [2 * EDGE_M, 0.0, 0.0],
            "start_random_angle": 0.0,
            "work_space_x": [-1.0, 49.0],
            "work_space_y": [-25.0, 25.0],
            "work_space_z": [-1.3, 2.5],
            "accept_radius": 0.5,
        },
        "perception": {"virtual_ceiling_prob": 0.0},
        "airsim": {
            "ip": "127.0.0.1",
            "vehicle_name": VEHICLE_NAME,
            "synchronous_steps": True,
            "depth_only": True,
            "cache_step_state": True,
            "timeout_value": base.getfloat("airsim", "timeout_value", fallback=15.0),
        },
        "navrl_forest": {
            "stage": stage.number,
            "map_path": stage.map_path,
            "static_count": stage.static_count,
            "dynamic_count": stage.dynamic_count,
        },
    })
    return cfg
