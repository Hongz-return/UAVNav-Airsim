"""Verify the 40 original obstacles in the running UE 5.4 TrainEnv replica."""

import argparse
import math
from pathlib import Path
import time

import cosysairsim as airsim
import cv2
import numpy as np

from gym_env.envs.dynamics.multirotor_airsim import to_quaternion
from tools.trainenv_instances import DEFAULT_EDITOR, launch_trainenv
from tools.ue_benchmark.ch4_stage3_v2_course import SPAWN_NED, TransientBoxes, tunnel_roof
from tools.ue_benchmark.ch4_trainenv_3d_course import ROOT, read_json, sha256, write_json
from tools.ue_benchmark.ch4_trainenv_dynamic_layout import REPORT


VEHICLE = "SimpleFlight"
TEMPLATE = ROOT / "configs/ch4_stage3_trainenv_airsim.json"


def object_pose(client, name):
    p = client.simGetObjectPose(name).position
    xyz = np.array([p.x_val, p.y_val, -p.z_val], dtype=float)
    if not np.isfinite(xyz).all():
        raise RuntimeError(f"No runtime pose for {name}")
    return xyz


def pose_now(client, point, yaw=0.0, pitch=0.0):
    client.simSetVehiclePose(
        airsim.Pose(airsim.Vector3r(*point), to_quaternion(pitch, 0, yaw)),
        ignore_collision=True, vehicle_name=VEHICLE,
    )


def capture_view(client, output, name, point, yaw, pitch):
    pose_now(client, point, yaw=yaw, pitch=pitch)
    requests = [
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False),
    ]
    for _ in range(12):
        scene, raw = client.simGetImages(requests, vehicle_name=VEHICLE)
        depth = np.asarray(raw.image_data_float, dtype=np.float32)
        usable = np.isfinite(depth) & (depth > .1) & (depth < 40)
        if depth.size and usable.mean() > .05 and np.median(depth[usable]) > 1.0:
            break
        time.sleep(.1)
    else:
        raise RuntimeError(f"No valid scene depth for {name}")
    np.save(output / f"{name}_depth.npy", depth.reshape(raw.height, raw.width), allow_pickle=False)
    raw_pixels = np.frombuffer(scene.image_data_uint8, dtype=np.uint8)
    if scene.width and scene.height and raw_pixels.size == scene.width * scene.height * 3:
        image = raw_pixels.reshape(scene.height, scene.width, 3)
        if np.ptp(image) > 8:
            cv2.imwrite(str(output / f"{name}.png"), image)
    return {"name": name, "near_depth_fraction": float(usable.mean()),
            "usable_depth_median_m": float(np.median(depth[usable]))}


def measure_motion(client, map_data, output):
    survey = read_json(Path(map_data["survey"]))
    dynamic = {item["label"]: item for item in map_data["motions"]}
    static = {item["name"]: item["object_name"] for item in survey["solids"] if item["name"] not in dynamic}
    baseline = {label: object_pose(client, name) for label, name in static.items()}
    trace = []
    start = time.monotonic()
    while time.monotonic() - start < 18.0:
        trace.append({"t_s": time.monotonic() - start, "positions": {
            label: object_pose(client, row["actor"]).tolist() for label, row in dynamic.items()
        }})
        time.sleep(.35)
    ranges = []
    for label, mover in dynamic.items():
        track = np.array([row["positions"][label] for row in trace])
        excursion = np.asarray(mover["excursion_cm"])
        travel_axis = int(np.argmax(np.abs(excursion)))
        span = np.ptp(track, axis=0)
        ranges.append({"label": label, "actor": mover["actor"], "travel_axis": "xyz"[travel_axis],
                       "peak_to_peak_m": float(span[travel_axis]), "axis_span_m": span.tolist()})
    write_json(output / "obstacle_tracks.json", trace)
    write_json(output / "motion_ranges.json", ranges)
    if len(ranges) != 40 or any(not 1.0 <= row["peak_to_peak_m"] <= 3.55 for row in ranges):
        raise RuntimeError("The full forty-obstacle motion audit did not pass")
    if any(sum(d > .03 for d in row["axis_span_m"]) != 1 for row in ranges):
        raise RuntimeError("An obstacle left its prescribed motion axis")
    max_static_m = max(math.dist(old, object_pose(client, static[label])) for label, old in baseline.items())
    if max_static_m > .015:
        raise RuntimeError("A stationary component moved")
    return {"moving": len(ranges), "stationary_components": len(static),
            "min_peak_to_peak_m": min(row["peak_to_peak_m"] for row in ranges),
            "max_peak_to_peak_m": max(row["peak_to_peak_m"] for row in ranges),
            "max_static_displacement_m": max_static_m, "samples": len(trace)}


def verify_collision(client, map_data):
    mover = next(x for x in map_data["motions"] if x["label"] == "GreenPillar_038")
    x, y, _ = ((lo + hi) / 2 for lo, hi in mover["swept_local_bounds_m"])
    client.enableApiControl(True, vehicle_name=VEHICLE)
    client.armDisarm(True, vehicle_name=VEHICLE)
    deadline = time.monotonic() + 12
    while abs(object_pose(client, mover["actor"])[1] - y) < 1.45:
        if time.monotonic() >= deadline:
            raise RuntimeError("The moving pillar never cleared the probe placement")
        time.sleep(.055)
    last_collision = client.simGetCollisionInfo(vehicle_name=VEHICLE).time_stamp
    pose_now(client, (x, y, 1.0))
    for _ in range(70):
        client.moveByVelocityZAsync(0, 0, 1.0, .13, vehicle_name=VEHICLE).join()
        hit = client.simGetCollisionInfo(vehicle_name=VEHICLE)
        if hit.has_collided and hit.time_stamp > last_collision:
            if hit.object_name != mover["actor"]:
                raise RuntimeError(f"Unexpected contact {hit.object_name} while probing {mover['actor']}")
            return {"label": mover["label"], "actor": hit.object_name,
                    "aircraft_command": "hold horizontal position", "collision_timestamp": hit.time_stamp}
    raise RuntimeError(f"No physical contact observed when {mover['label']} swept across the aircraft")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=42610)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    map_data = read_json(REPORT / "map_manifest.json")
    target = Path(map_data["map_file"])
    if sha256(target) != map_data["map_sha256"] or sha256(Path(map_data["source_map_file"])) != map_data["source_map_sha256"]:
        raise RuntimeError("An audited map changed")
    project = target.parents[3] / "TrainEnvAirSim.uproject"
    with launch_trainenv(executable=DEFAULT_EDITOR, template=TEMPLATE, ports=[args.port], start_position=SPAWN_NED,
                         clock_speed=1.0, run_dir=output, timeout=150, map_path=map_data["map"], editor_project=project):
        client = airsim.MultirotorClient(ip="127.0.0.1", port=args.port, timeout_value=25)
        try:
            boxes = TransientBoxes(client)
            boxes.replace([tunnel_roof(2.05)], verify=True)
            motion = measure_motion(client, map_data, output)
            contact = verify_collision(client, map_data)
            boxes.clear()
            views = [capture_view(client, output, "route_view", (2, -15, 1), 0, 0),
                     capture_view(client, output, "overview", (-4, -38, -18), .45, -.38)]
        finally:
            client.client.close()
    write_json(output / "verification.json", {"passed": True, "map_sha256": map_data["map_sha256"],
                                               "motion": motion, "contact": contact, "views": views})
    print(motion, contact, flush=True)


if __name__ == "__main__":
    main()
