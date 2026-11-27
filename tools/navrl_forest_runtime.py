"""Coordinates and runtime configuration for the UE5.9 forest levels."""

import ast
from copy import deepcopy
import math


EDGE_M = 24.0
ORIGIN_AGL_M = 1.5
ARENA_HALF_SIZE_M = 25.0
VEHICLE_NAME = "SimpleFlight"
CAMERA_FORWARD_M = 0.65


def parse_route_portals(value):
    """Read the portal layout saved in the forest training configuration."""
    try:
        edge, offsets, jitter = ast.literal_eval(value)
        if (not isinstance(offsets, (list, tuple)) or not offsets
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (edge, jitter, *offsets))
                or not 0 < edge < ARENA_HALF_SIZE_M
                or not 0 < jitter < ARENA_HALF_SIZE_M - max(map(abs, offsets))):
            raise ValueError
    except (SyntaxError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("navrl_forest.route_portals must contain a valid edge, tangent offsets and jitter") from error
    return edge, tuple(offsets), jitter


def sample_training_route(rng, route_portals=None):
    """Return seeded AirSim NED endpoints and initial yaw for one episode."""
    points = []
    if route_portals is None:
        # Keep V1's independent draws and their order for replaying old runs.
        for _ in range(2):
            side = int(rng.integers(4))
            edge = EDGE_M if side % 2 == 0 else -EDGE_M
            along = float(rng.uniform(-EDGE_M, EDGE_M))
            x, y = (edge, along) if side < 2 else (along, edge)
            points.append((x + EDGE_M, y, ORIGIN_AGL_M - float(rng.uniform(0.5, 2.5))))
    else:
        distance, offsets, jitter = route_portals
        side = int(rng.integers(4))
        edge = distance if side % 2 == 0 else -distance
        for normal in (edge, -edge):
            center = offsets[int(rng.integers(len(offsets)))]
            along = center + float(rng.uniform(-jitter, jitter))
            x, y = (normal, along) if side < 2 else (along, normal)
            points.append((x + EDGE_M, y, ORIGIN_AGL_M - float(rng.uniform(0.5, 2.5))))
    start, goal = points
    return start, goal, math.atan2(goal[1] - start[1], goal[0] - start[0])


def forest_config(base, stage, *, route_portals=None):
    """Apply the open-arena geometry without changing the MA-SAC policy contract."""
    if (base.get("options", "algo") != "SAC" or base.get("options", "perception") != "depth_state"
            or not base.getboolean("options", "use_mamba")
            or not base.getboolean("options", "navigation_3d")):
        raise ValueError("The forest curriculum requires 3D depth-state MA-SAC.")
    if base.has_section("stage3") or base.has_section("stage3_v2") or base.has_option("scene", "runtime_script"):
        raise ValueError("Choose the base MA-SAC config without tunnel or sequence-scene overrides.")
    route_config = None if route_portals is None else repr(route_portals)
    route_edge = EDGE_M if route_config is None else parse_route_portals(route_config)[0]
    cfg = deepcopy(base)
    cfg.remove_option("environment", "training_routes")
    cfg.read_dict({
        "options": {"env_name": "Obs", "project_name": "Ch4_NavRL_Forest_MASAC"},
        "environment": {
            "start_position": [EDGE_M - route_edge, 0.0, 0.0],
            "goal_position": [EDGE_M + route_edge, 0.0, 0.0],
            "start_random_angle": 0.0,
            "work_space_x": [EDGE_M - ARENA_HALF_SIZE_M, EDGE_M + ARENA_HALF_SIZE_M],
            "work_space_y": [-ARENA_HALF_SIZE_M, ARENA_HALF_SIZE_M],
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
    if route_config is None:
        cfg.remove_option("navrl_forest", "route_portals")
    else:
        cfg.set("navrl_forest", "route_portals", route_config)
    return cfg
