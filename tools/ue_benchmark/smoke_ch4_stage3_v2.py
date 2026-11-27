"""Commission the dynamic Stage 3 geometry in the packaged TrainEnv."""

import argparse
from dataclasses import asdict
import math
from pathlib import Path
import time

import cosysairsim as airsim
import matplotlib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from gym_env.envs.dynamics.multirotor_airsim import to_quaternion
from tools.trainenv_instances import DEFAULT_EXE, HOST, launch_trainenv
from tools.ue_benchmark.ch4_stage3_v2_course import (
    CLEAN_STRIPS, FLOOR_Z, ROOF_HEIGHT, SPAWN_NED, Box, TransientBoxes, sample_corridor, tunnel_roof,
)
from tools.ue_benchmark.ch4_trainenv_3d_course import read_json, write_json
from tools.ue_benchmark.verify_ch4_benchmark_maps import viewport_still
from tools.ue_benchmark.verify_ch4_trainenv_3d import VEHICLE, collision_details, move


REPORT = Path("reports/ch4_stage3_v2_20261125")
GEOMETRY_SURVEY = Path("reports/ch4_stage3_3d_20261125/original_trainenv_geometry.json")
TEMPLATE = Path("configs/ch4_stage3_trainenv_airsim.json")


def verify_strips(survey):
    solids = survey["solids"]
    if len(solids) != 173:
        raise RuntimeError("Expected all 173 AABBs from the original TrainEnv survey")
    for x0, x1, y in CLEAN_STRIPS:
        # The survey is in PlayerStart z-up, while the stages fly in vehicle z-up.
        interior = ((x0, x1), (y - 6, y + 6), (FLOOR_Z - SPAWN_NED[2], 3.42 - SPAWN_NED[2]))
        for solid in solids:
            if all(min(a[1], b[1]) - max(a[0], b[0]) > .0001
                   for a, b in zip(interior, solid["axes_local_z_up_m"], strict=True)):
                raise RuntimeError(f"Original mesh intrudes into corridor strip: {solid['name']}")
    return len(solids)


def native_boxes(survey, center_x, center_y):
    """Account for original pillars and boundary walls occluding the roof."""
    for solid in survey["solids"]:
        x, y, z = solid["axes_local_z_up_m"]
        if x[1] < center_x - 30 or x[0] > center_x + 30 or y[1] < center_y - 30 or y[0] > center_y + 30:
            continue
        yield Box(solid["name"], tuple(x), tuple(y), tuple(value + SPAWN_NED[2] for value in z))


def reset_probe(client, xyz, yaw=0.0):
    t0 = time.perf_counter()
    x, y, z = xyz
    client.reset()
    client.enableApiControl(True, VEHICLE)
    if not client.armDisarm(True, VEHICLE):
        raise RuntimeError("Could not arm smoke-test drone")
    pose = airsim.Pose(airsim.Vector3r(x, y, -z), to_quaternion(0, 0, yaw))
    client.simSetVehiclePose(pose, True, VEHICLE)
    move(client, (0, 0, 0), 1.0)
    actual = position(client)
    if math.dist(actual, xyz) > .25 or collision_details(client)["has_collided"]:
        raise RuntimeError(f"Unsafe reset requested={xyz} actual={actual}")
    return time.perf_counter() - t0


def position(client):
    x, y, zned = client.simGetVehiclePose(VEHICLE).position
    return x, y, -zned


def first_intersections(origin, rays, boxes):
    closest = np.full(rays.shape[:-1], np.inf)
    labels = np.full(closest.shape, -1, dtype=np.int16)
    for index, box in enumerate(boxes):
        lower, upper = np.asarray(box.bounds).T
        parallel = np.abs(rays) < 1e-9
        divisor = np.where(parallel, 1.0, rays)
        near = (lower - origin) / divisor
        far = (upper - origin) / divisor
        entry = np.max(np.where(parallel, -np.inf, np.minimum(near, far)), axis=-1)
        exit_ = np.min(np.where(parallel, np.inf, np.maximum(near, far)), axis=-1)
        outside_parallel = np.any(parallel & ((origin < lower) | (origin > upper)), axis=-1)
        hit = ~outside_parallel & (entry > 0) & (exit_ >= entry) & (entry < closest)
        closest[hit] = entry[hit]
        labels[hit] = index
    return closest, labels


def depth_probe(client, output, label, xyz, surface, boxes, *, yaw=0.0, pitch=0.0, gate=False):
    reset_wall = reset_probe(client, xyz, yaw)
    client.simSetCameraPose(
        "0", airsim.Pose(airsim.Vector3r(.46, 0, 0), to_quaternion(pitch, 0, 0)), VEHICLE,
    )
    camera = client.simGetCameraInfo("0", VEHICLE)
    raw = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)],
                              vehicle_name=VEHICLE)[0]
    scene = viewport_still(client, output / f"rgb_{label}.png", 320, 180)[..., ::-1]
    depth = np.asarray(raw.image_data_float, dtype=np.float32).reshape(raw.height, raw.width)
    if depth.shape != (180, 320):
        raise RuntimeError(f"Unexpected DepthPlanar shape {depth.shape}")
    focal = raw.width / (2 * math.tan(math.radians(camera.fov) / 2))
    py, px = np.mgrid[:raw.height, :raw.width]
    local = np.stack((np.ones_like(px), (px + .5 - raw.width / 2) / focal,
                      (py + .5 - raw.height / 2) / focal), axis=-1)
    q = raw.camera_orientation
    rays = Rotation.from_quat((q.x_val, q.y_val, q.z_val, q.w_val)).apply(local.reshape(-1, 3))
    rays = rays.reshape(local.shape) * [1, 1, -1]
    origin = np.asarray(list(raw.camera_position)) * [1, 1, -1]
    expected, labels = first_intersections(origin, rays, boxes)
    surface_index = next(index for index, box in enumerate(boxes) if box.key == surface)
    mask = binary_erosion((labels == surface_index) & (expected < 25), iterations=3)
    errors = np.abs(depth[mask] - expected[mask])
    valid = np.isfinite(depth) & (depth > 0)
    matches = errors <= np.maximum(.12, .035 * expected[mask])
    row, col = raw.height // 2, raw.width // 2
    checks = {
        "label": label, "surface": surface, "reset_wall_s": reset_wall,
        "camera_xyz_up_m": origin.tolist(), "shape_px": list(depth.shape),
        "rgb_range": [int(scene.min()), int(scene.max())],
        "finite_positive_fraction": float(valid.mean()), "surface_pixels": int(mask.sum()),
        "surface_match_fraction": float(matches.mean()) if matches.size else 0.0,
        "center_expected_m": float(expected[row, col]) if np.isfinite(expected[row, col]) else None,
        "center_observed_m": float(depth[row, col]),
        "p95_absolute_error_m": float(np.percentile(errors, 95)) if errors.size else None,
    }
    checks["passed"] = bool(valid.all() and mask.sum() >= 150 and checks["surface_match_fraction"] >= .99
                            and labels[row, col] == surface_index
                            and abs(depth[row, col] - expected[row, col]) <= max(.12, .035 * expected[row, col]))
    if gate:
        # Rows occupied by the gate span the whole projected corridor. The
        # remaining rays through the opening must see beyond its front face.
        gate_depth = expected[row, col]
        aperture = binary_erosion((labels != surface_index) & (expected > gate_depth + .75), iterations=3)
        achieved = aperture & (depth > gate_depth + .65)
        checks["opening_pixels"] = int(aperture.sum())
        checks["opening_clear_fraction"] = float(achieved.sum() / max(1, aperture.sum()))
        center_row = mask[row]
        checks["center_row_gate_pixels"] = int(center_row.sum())
        checks["passed"] = bool(checks["passed"] and checks["opening_pixels"] >= 120
                                and checks["opening_clear_fraction"] >= .97
                                and center_row[:raw.width // 3].any() and center_row[-raw.width // 3:].any())
    print(f"[smoke] {label}: {checks['center_observed_m']:.3f} m, "
          f"surface agreement {checks['surface_match_fraction']:.1%}, pass={checks['passed']}", flush=True)
    return checks, scene, depth


def collision_probe(client, label, xyz, velocity, box, object_name, axis, inside):
    reset_wall = reset_probe(client, xyz)
    hit = None
    for _ in range(40):
        move(client, velocity, .15)
        reading = collision_details(client)
        if reading["has_collided"]:
            hit = reading
            break
    terminal = position(client)
    limit = box.bounds[axis][0 if velocity[axis] > 0 else 1]
    impact = list(hit["impact_ned_m"]) if hit else None
    if impact:
        impact[2] *= -1
    sign = math.copysign(1, velocity[axis])
    passed = bool(hit and hit["object_name"] == object_name and abs(impact[axis] - limit) < .10
                  and sign * (terminal[axis] - limit) < .07)
    result = {
        "label": label, "passed": passed, "expected_actor": object_name, "contact": hit,
        "face_xyz_axis": axis, "face_coordinate_m": limit, "terminal_xyz_up_m": terminal,
        "started_inside_corridor": inside, "reset_wall_s": reset_wall,
    }
    print(f"[smoke] {label}: contact={hit and hit['object_name']}, pass={passed}", flush=True)
    return result


def render_evidence(output, captures):
    fig, axes = plt.subplots(len(captures), 2, figsize=(10, 2.26 * len(captures)), squeeze=False)
    for row, (audit, rgb, depth) in enumerate(captures):
        left, right = axes[row]
        left.imshow(rgb)
        left.set_title(audit["label"].replace("_", " ") + " | RGB", fontsize=9)
        im = right.imshow(np.clip(depth, 0, 25), vmin=0, vmax=25, cmap="turbo")
        expected = audit["center_expected_m"]
        expected_text = f"{expected:.2f}" if expected is not None else "unbounded"
        right.set_title(f"DepthPlanar | center {audit['center_observed_m']:.2f} m / {expected_text} m expected",
                        fontsize=9)
        left.set_axis_off()
        right.set_axis_off()
    fig.subplots_adjust(left=.02, right=.89, top=.97, bottom=.021, wspace=.06, hspace=.30)
    bar = fig.add_axes((.91, .10, .017, .80))
    fig.colorbar(im, cax=bar, label="DepthPlanar (m, clipped at 25 m for display)")
    fig.savefig(output / "rgb_depth_comparison.png", dpi=142)
    plt.close(fig)


def commission(client, output, audit, survey):
    material = TransientBoxes(client)
    floor = Box("map_floor", (-9., 171.), (-60., 60.), (FLOOR_Z - .3, FLOOR_Z))
    captures = []
    resets = audit["spawn_reset_wall_s"]
    audit["collisions"] = collisions = []

    def capture(*args, **kwargs):
        result = depth_probe(client, output, *args, **kwargs)
        captures.append(result)
        audit["depth"].append(result[0])

    def install(boxes):
        t0 = time.perf_counter()
        material.replace(boxes, verify=True)
        resets.append(time.perf_counter() - t0)

    try:
        client.simRunConsoleCommand("r.SkyAtmosphere.EditorNotifications 0")
        client.simRunConsoleCommand("r.MotionBlurQuality 0")
        open_scene = viewport_still(client, output / "rgb_open_scene.png", 320, 180)
        audit["open_scene_rgb_max"] = int(open_scene.max())
        for height, x in zip(ROOF_HEIGHT, (51.0, 110.0), strict=True):
            roof = tunnel_roof(height)
            install([roof])
            capture(f"tunnel_roof_{height:.1f}m", (x, 53.35, -1.0), roof.key,
                    (roof, *native_boxes(survey, x, 53.35)), pitch=.52)
            collisions.append(collision_probe(client, f"roof_collision_{height:.1f}m", (x, 53.35, -1.0),
                                              (0, 0, .8), roof, material.names[roof.key], 2, False))
        for strip, width, family in [(0, 8., "over_under"), (1, 12., "under_over"), (2, 12., "over_under")]:
            course = sample_corridor(family, np.random.default_rng(12 + strip), strip=strip, width=width)
            audit["corridors"].append({"strip": strip, "width_m": width, "family": family,
                                        "boxes": [asdict(box) for box in course["boxes"]]})
            boxes = course["boxes"]
            install(boxes)
            start_x, sy, sz = course["start_position"]
            yaw = 0.0 if course["direction"] > 0 else math.pi
            audit["corridors"][-1]["spawn_wall_s"] = reset_probe(client, (start_x, sy, -sz), yaw)
            y = course["center_y"]
            if strip == 0:
                for wall in boxes[:2]:
                    side = 1 if wall.key == "wall_max_y" else -1
                    capture(wall.key, (start_x + course["direction"] * 2, y, -.8), wall.key,
                            (*boxes, floor), yaw=side * math.pi / 2)
                    collisions.append(collision_probe(client, wall.key, (start_x, y + side * (width / 2 - 1.6), -.8),
                                                      (0, 1.6 * side, 0), wall, material.names[wall.key], 1, True))
                roof = boxes[2]
                capture(roof.key, (start_x + course["direction"] * 2, y, .4), roof.key,
                        (*boxes, floor), pitch=.65)
                collisions.append(collision_probe(client, roof.key, (start_x, y, 1.2),
                                                  (0, 0, .8), roof, material.names[roof.key], 2, True))
            for gate in course["gates"]:
                for side in (-1, 1):
                    edge = gate.x[0 if side < 0 else 1]
                    z = gate.z[1] - .42 if gate.key.startswith("over") else gate.z[0] + .42
                    capture(f"strip_{strip}_{gate.key}_{'west' if side < 0 else 'east'}",
                            (edge + side * 2.5, y, z), gate.key, (*boxes, floor),
                            yaw=0.0 if side < 0 else math.pi, gate=True)
                for lateral in (-1, 1):
                    xyz = (gate.x[0] - 2.6, y + lateral * (width / 2 - .65), gate.center[2])
                    collisions.append(collision_probe(
                        client, f"strip_{strip}_{gate.key}_edge_{lateral:+d}", xyz, (2, 0, 0), gate,
                        material.names[gate.key], 0, True,
                    ))
    finally:
        material.clear()
        audit["transient_objects_remaining"] = client.simListSceneObjects("Ch4Stage3V2_.*")
        if captures:
            examples = [frame for frame in captures if frame[0]["label"].startswith((
                "tunnel_roof", "wall_", "corridor_roof", "strip_0_",
            ))]
            render_evidence(output, examples)
            np.savez_compressed(output / "depth_m.npz", **{a["label"]: depth for a, _, depth in captures})
    reset_times = [probe["reset_wall_s"] for probe in audit["depth"] + collisions]
    audit["reset_wall_s"] = {"median": float(np.median(reset_times)), "max": max(reset_times)}
    audit["passed"] = (all(probe["passed"] for probe in audit["depth"] + collisions)
                       and all(probe["rgb_range"][1] > 1 for probe in audit["depth"])
                       and max(resets) < 2.0 and max(reset_times) < 3.0 and not audit["transient_objects_remaining"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPORT / "smoke_corridor")
    parser.add_argument("--port", type=int, default=42071)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    survey = read_json(GEOMETRY_SURVEY)
    audit = {"passed": False, "depth": [], "corridors": [], "spawn_reset_wall_s": [],
             "original_solids_checked": verify_strips(survey), "vehicle_spawn_ned": SPAWN_NED,
             "camera_color_capture": "Unreal FPV viewport"}
    try:
        with launch_trainenv(executable=DEFAULT_EXE, template=TEMPLATE.resolve(), ports=[args.port],
                             start_position=SPAWN_NED, clock_speed=2.0, run_dir=output, timeout=120,
                             show_viewport=True):
            client = airsim.MultirotorClient(ip=HOST, port=args.port, timeout_value=15)
            try:
                commission(client, output, audit, survey)
            finally:
                client.client.close()
        if not audit["passed"]:
            raise RuntimeError("Packaged geometry commissioning did not meet the depth, contact and timing gate")
    except BaseException as error:
        audit["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "smoke_summary.json", audit)


if __name__ == "__main__":
    main()
