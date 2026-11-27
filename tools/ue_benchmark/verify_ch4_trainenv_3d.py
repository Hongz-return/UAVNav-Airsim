"""Commission the packaged course with genuine AirSim flights and collision probes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path

import cosysairsim as airsim
import numpy as np

from gym_env.envs.airsim_env import AirsimGymEnv
from gym_env.envs.dynamics.multirotor_airsim import quaternion_to_euler, to_quaternion
from tools.trainenv_instances import DEFAULT_3D_EXE, DEFAULT_EDITOR, HOST, TRAINENV_3D_MAP, launch_trainenv
from tools.ue_benchmark.ch4_trainenv_3d_course import (
    COURSE_FILE, obstacles, position_agl, read_json, sha256, structural_boxes, to_ned, write_json,
)
from tools.ue_benchmark.ch4_trainenv_3d_runtime import (
    REPORT, SETTINGS_TEMPLATE, SPAWN, TRAIN_CONFIG, VEHICLE, DepthAudit,
    assert_reset_depth_consistent, collision_details, json_line,
    progress_through_obstacles, read_ini, scenario_errors, start_sample, step_sample,
)


def position(client, course):
    p = client.simGetVehiclePose(VEHICLE).position
    return position_agl([p.x_val, p.y_val, -p.z_val], course)


def move(client, velocity_up, duration):
    vx, vy, vz = velocity_up
    if not client.moveByVelocityAsync(
        vx, vy, -vz, duration, drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(True, 0), vehicle_name=VEHICLE,
    ).get():
        raise RuntimeError("AirSim did not complete the physical velocity command")


def initialize_probe(client, course, point, yaw=0):
    client.reset()
    client.enableApiControl(True, VEHICLE)
    if not client.armDisarm(True, VEHICLE):
        raise RuntimeError("Failed to arm during commissioning")
    pose = airsim.Pose(airsim.Vector3r(*to_ned(point, course)), to_quaternion(0, 0, yaw))
    client.simSetVehiclePose(pose, True, VEHICLE)
    move(client, (0, 0, 0), 1)
    if math.dist(position(client, course), point) > .25 or collision_details(client)["has_collided"]:
        raise RuntimeError(f"Invalid commissioning start: expected {point}, observed {position(client, course)}")


def reference_flight(env, family, course, output):
    """Use the same forward, climb, yaw-rate commands and the same step size as the actor."""
    waypoints = np.array(course["families"][family]["reference_xyz_agl_m"])
    goal = waypoints[-1].tolist()
    spec = course["evaluation"]
    env.set_evaluation_scenarios([{
        "scenario_id": family + "_reference",
        "start_position": to_ned(waypoints[0].tolist(), course),
        "goal_position": to_ned(goal, course),
        "yaw_rad": 0.0,
    }])
    depth = DepthAudit(spec)
    dynamics = env.dynamic_model
    with depth.attach(env.client), (output / "reference_trajectory.jsonl").open("w", encoding="utf-8") as trace:
        env.reset(seed=spec["seed"])
        samples = [start_sample(env, course, depth.latest(0))]
        assert_reset_depth_consistent(samples[0], course)
        json_line(trace, samples[0])
        while True:
            x, y, z = samples[-1]["position_xyz_agl_m"]
            lookahead = min(x + 3.0, goal[0])
            y_reference = np.interp(lookahead, waypoints[:, 0], waypoints[:, 1])
            z_reference = np.interp(lookahead, waypoints[:, 0], waypoints[:, 2])
            target_yaw = math.atan2(y_reference - y, max(0.2, lookahead - x))
            yaw_error = math.remainder(target_yaw - dynamics.get_attitude()[2], math.tau)
            climb = 0.9 * (z_reference - z)
            if family == "side":
                # Keep the side witness altitude fixed despite roll-induced thrust changes.
                climb = 2.2 * (z_reference - z) - 0.25 * samples[-1]["velocity_xyz_up_mps"][2]
            action = np.clip(
                [min(2.5, 0.9 + max(0.0, goal[0] - x) / 3), climb, 2.2 * yaw_error],
                env.action_space.low, env.action_space.high,
            )
            before = depth.previous_time_ns
            _, _, terminated, truncated, info = env.step(action)
            sample = step_sample(info, course, depth.latest(before))
            samples.append(sample)
            json_line(trace, sample)
            if terminated or truncated:
                break

    passed, order = progress_through_obstacles(samples, family, course)
    expected = [obstacle["name"] for obstacle in obstacles(family, course)]
    points = np.array([row["position_xyz_agl_m"] for row in samples])
    z_range = [float(points[:, 2].min()), float(points[:, 2].max())]
    y = course["families"][family]["lane_y_m"]
    half_width = (course["lane_width_m"] - course["wall_thickness_m"]) / 2
    clearances = {
        "floor": z_range[0] - course["floor_agl_m"],
        "roof": course["roof_agl_m"] - z_range[1],
        "nearest_side_wall": float(half_width - max(abs(points[:, 1] - y))),
    }
    clearance_to_boxes = [
        attempt["center_clearance_to_obstacle_m"]
        for obstacle in passed for attempt in obstacle["passages"] if attempt["correct_side"]
    ]
    horizontal_error, altitude_error = scenario_errors(samples[-1]["position_xyz_agl_m"], goal)
    result = {
        "passed": bool(info["is_success"] and order == expected and (family != "side" or np.ptp(z_range) < .2)),
        "goal_reached": info["is_success"],
        "collision": info["is_crash"],
        "out_of_bounds": info["is_not_in_workspace"],
        "timeout": truncated,
        "steps": len(samples) - 1,
        "depth_frames_checked": len(samples),
        "min_finite_positive_depth_fraction": min(row["depth"]["positive_finite_fraction"] for row in samples),
        "absolute_goal_altitude_error_m": altitude_error,
        "goal_horizontal_error_m": horizontal_error,
        "height_range_agl_m": z_range,
        "min_center_clearance_to_shell_m": clearances,
        "min_center_clearance_to_obstacles_m": min(clearance_to_boxes, default=None),
        "obstacle_order": order,
        "obstacle_records": passed,
    }
    write_json(output / "reference.json", result)
    return result


def ray_box(origin, direction, axes):
    low_t, high_t = 0.0, math.inf
    for p, v, (low, high) in zip(origin, direction, axes, strict=True):
        if abs(v) < 1e-8:
            if not low <= p <= high:
                return math.inf
        else:
            first, last = sorted(((low - p) / v, (high - p) / v))
            low_t, high_t = max(low_t, first), min(high_t, last)
    return low_t if low_t <= high_t else math.inf


def depth_probe(client, course, output, name, point, expected, yaw=0.0, pitch=0.0):
    initialize_probe(client, course, point, yaw)
    # Inspect the front, floor, and roof along defined camera rays while the vehicle holds position.
    client.simSetCameraPose(
        "0", airsim.Pose(airsim.Vector3r(.25, 0, 0), to_quaternion(pitch, 0, 0)), VEHICLE,
    )
    raw = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True)], VEHICLE)[0]
    audit = DepthAudit(course["evaluation"])
    measured = audit.consume(raw)
    p = raw.camera_position
    origin = position_agl([p.x_val, p.y_val, -p.z_val], course)
    _, camera_pitch, camera_yaw = quaternion_to_euler(raw.camera_orientation)
    direction = (
        math.cos(camera_pitch) * math.cos(camera_yaw),
        math.cos(camera_pitch) * math.sin(camera_yaw),
        math.sin(camera_pitch),
    )
    expected_m, ray_hit = min(
        (ray_box(origin, direction, [x, y, z]), box)
        for box, x, y, z, _, _ in structural_boxes(course)
    )
    if measured["center_m"] is None or not math.isfinite(expected_m):
        raise RuntimeError(f"{name} did not produce a valid center ray: {measured}, {expected_m}")
    difference = abs(measured["center_m"] - expected_m)
    np.savez_compressed(output / f"depth_{name}.npz", depth_m=audit.frame)
    return {
        "name": name,
        "passed": ray_hit == expected and difference < max(.12, .035 * expected_m),
        "expected_surface": expected,
        "ray_surface": ray_hit,
        "ray_distance_m": expected_m,
        "error_m": difference,
        "camera_xyz_agl_m": origin,
        "camera_forward_xyz_up": list(direction),
        "depth": measured,
    }


def blocking_flight(client, course, object_names, name, point, velocity, expected):
    initialize_probe(client, course, point)
    samples = []
    hit = None
    for _ in range(36):
        move(client, velocity, .14)
        samples.append(position(client, course))
        reading = collision_details(client)
        if reading["has_collided"]:
            hit = reading
            break
    expected_names = {expected, f"Ch4_3D_{expected}", object_names[expected]}
    return {
        "name": name,
        "passed": hit is not None and hit["object_name"] in expected_names,
        "expected_contact": expected,
        "contact": hit,
        "initial_xyz_agl_m": point,
        "command_xyz_up_mps": velocity,
        "samples_xyz_agl_m": samples,
    }


def braking_probe(client, course, point, command, axis, duration):
    initialize_probe(client, course, point)
    move(client, command, duration)
    start = position(client, course)
    state = client.getMultirotorState(vehicle_name=VEHICLE)
    velocity = state.kinematics_estimated.linear_velocity
    speed = abs(list(velocity)[axis])
    trace = []
    settled_steps = 0
    for _ in range(35):
        move(client, (0, 0, 0), .10)
        p = position(client, course)
        v = client.getMultirotorState(vehicle_name=VEHICLE).kinematics_estimated.linear_velocity
        current = abs(list(v)[axis])
        trace.append({"displacement_m": p[axis] - start[axis], "speed_mps": current})
        settled_steps = settled_steps + 1 if current < .15 else 0
        if settled_steps >= 3:
            break
    return {
        "start_speed_mps": speed,
        "stop_distance_m": max(abs(row["displacement_m"]) for row in trace),
        "settled": settled_steps >= 3,
        "collision": collision_details(client)["has_collided"],
        "trace": trace,
    }


def commission_family(family, course, object_names, port, output):
    output.mkdir()
    cfg = read_ini(TRAIN_CONFIG)
    cfg["airsim"].update(
        ip=HOST, rpc_port=str(port), depth_only="True", cache_step_state="True", timeout_value="15",
    )
    env = AirsimGymEnv()
    env.set_config(cfg)
    client = env.client
    try:
        reference = reference_flight(env, family, course, output)
        lane_y = course["families"][family]["lane_y_m"]
        depths, collisions = [], []
        for obstacle in obstacles(family, course):
            front = obstacle["x_m"][0]
            name = obstacle["name"]
            lo, hi = obstacle["z_agl_m"]
            blocked_z = (lo + hi) / 2
            depths.append(depth_probe(client, course, output, name, [front - 5, lane_y, blocked_z], name))
            edge_y = lane_y if family == "side" else lane_y + 5.0
            collisions.append(blocking_flight(
                client, course, object_names, name, [front - 3, edge_y, blocked_z], [2, 0, 0], name,
            ))
        braking = {}
        if family == "over":
            depths += [
                depth_probe(client, course, output, "floor", [8, 0, 2.8], "Floor", pitch=-math.pi / 2),
                depth_probe(client, course, output, "roof", [8, 0, 2.8], "Roof", pitch=math.pi / 2),
                depth_probe(client, course, output, "divider", [12, 3, 2.8], "LaneWall_1", yaw=math.pi / 2),
            ]
            collisions += [
                blocking_flight(client, course, object_names, "floor", [8, 0, 1.05], [0, 0, -.65], "Floor"),
                blocking_flight(client, course, object_names, "roof", [8, 0, 5.35], [0, 0, .75], "Roof"),
                blocking_flight(client, course, object_names, "divider", [12, 3.4, 2.8], [0, 1.7, 0], "LaneWall_1"),
            ]
        elif family == "side":
            collisions += [
                blocking_flight(
                    client, course, object_names, "high_side_block", [21, lane_y, 5.6], [2, 0, 0],
                    "Side_FullHeightBlock",
                ),
                blocking_flight(
                    client, course, object_names, "low_side_block", [21, lane_y, .68], [2, 0, 0],
                    "Side_FullHeightBlock",
                ),
            ]
            braking = {
                "forward_4_to_0": braking_probe(client, course, [5, lane_y + 3.9, 2.6], [4, 0, 0], 0, 3.4),
                "climb_0_8_to_0": braking_probe(client, course, [8, lane_y, 2.7], [0, 0, .8], 2, 1.7),
            }
        elif family == "over_under":
            collisions.append(blocking_flight(
                client, course, object_names, "combo_all_high", [40, lane_y, 4.3], [2, 0, 0],
                "Combo02_RoofBlock",
            ))
        result = {
            "passed": (reference["passed"] and all(row["passed"] for row in depths + collisions)
                       and all(row["settled"] and not row["collision"] for row in braking.values())),
            "reference": reference,
            "depth_probes": depths,
            "blocking_flights": collisions,
            "braking": braking,
        }
        write_json(output / "commissioning.json", result)
        print(
            f"{family}: reference {'passed' if reference['passed'] else 'failed'}, "
            f"{sum(row['passed'] for row in depths)}/{len(depths)} depth probes, "
            f"{sum(row['passed'] for row in collisions)}/{len(collisions)} contact probes",
            flush=True,
        )
        return result
    finally:
        env.client.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--map-manifest", type=Path, default=REPORT / "map_manifest.json")
    parser.add_argument("--ue-exe", type=Path)
    parser.add_argument("--ue-project", type=Path, help="Use standalone UE Editor with this source project")
    parser.add_argument("--first-port", type=int, default=41966)
    args = parser.parse_args()
    course, manifest = read_json(COURSE_FILE), read_json(args.map_manifest)
    if (course["map"] != TRAINENV_3D_MAP or sha256(COURSE_FILE) != manifest["geometry_sha256"]
            or sha256(Path(manifest["map_file"])) != manifest["map_sha256"]):
        raise RuntimeError("The course geometry and the saved editor level have different identities")
    names = {row["name"]: row["ue_object_name"] for row in manifest["solids"]}
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    families = list(course["families"])
    ports = list(range(args.first_port, args.first_port + len(families)))
    executable = args.ue_exe or (DEFAULT_EDITOR if args.ue_project else DEFAULT_3D_EXE)
    with launch_trainenv(
        executable=executable.resolve(), template=SETTINGS_TEMPLATE, ports=ports, start_position=SPAWN,
        clock_speed=course["evaluation"]["clock_speed"], run_dir=run, timeout=180, map_path=TRAINENV_3D_MAP,
        editor_project=args.ue_project.resolve() if args.ue_project else None,
    ), ProcessPoolExecutor(max_workers=len(families)) as pool:
        workers = {
            name: pool.submit(commission_family, name, course, names, port, run / name)
            for name, port in zip(families, ports, strict=True)
        }
        results = {name: worker.result() for name, worker in workers.items()}
    success = all(row["passed"] for row in results.values())
    write_json(run / "summary.json", {"passed": success, "map_sha256": manifest["map_sha256"], "families": results})
    if not success:
        raise RuntimeError("At least one physical course commissioning check failed")


if __name__ == "__main__":
    main()
