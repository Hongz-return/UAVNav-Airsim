"""Physical and paired-metric checks for the packaged TrainEnv cross traffic."""

from contextlib import contextmanager
import math

import cosysairsim as airsim
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

from .ch4_stage3_crossing import CROSSING_KEY, CROSSING_SIZE_M, HALF_TRAVEL_M, SPEED_MPS
from .ch4_stage3_v2_course import Box, FLOOR_Z, SPAWN_NED
from .ch4_trainenv_3d_course import ROOT, read_json
from .smoke_ch4_stage3_v2 import first_intersections


SURVEY = ROOT / "reports/ch4_stage3_3d_20261125/original_trainenv_geometry.json"
MAX_UPDATE_GAP_S = .12
MAX_UNBLOCKED_LAG_M = .04
NEAR_SURFACE_M = .55


def surveyed_boxes():
    data = read_json(SURVEY)
    if len(data["solids"]) != 173:
        raise RuntimeError("Expected the original TrainEnv survey of 173 solid meshes")
    return tuple(
        Box(solid["name"], tuple(x), tuple(y), tuple(alt + SPAWN_NED[2] for alt in z))
        for solid in data["solids"] for x, y, z in [solid["axes_local_z_up_m"]]
    )


def distance_to_box(point, box):
    return math.sqrt(sum(max(lo - p, p - hi, 0.0)**2 for p, (lo, hi) in zip(
        point, box.bounds, strict=True,
    )))


def assert_surveyed_section(path, start, boxes):
    """Check the swept volume and two lateral bypasses against the solid survey."""
    half_x, half_y, half_z = (n / 2 for n in CROSSING_SIZE_M)
    section = (
        (path.x - half_x - NEAR_SURFACE_M, path.x + half_x + NEAR_SURFACE_M),
        (path.y - 5.2, path.y + 5.2),
        (path.z - half_z - .2, path.z + half_z + .2),
    )
    conflicts = [b.key for b in boxes if all(
        min(a[1], c[1]) > max(a[0], c[0]) for a, c in zip(section, b.bounds, strict=True)
    )]
    if conflicts:
        raise RuntimeError(f"Original solids intersect the intended cross traffic or its bypasses: {conflicts}")
    start_up = (*start[:2], -start[2])
    gap = distance_to_box(start_up, path.box_at(0.0, "stationary"))
    if gap < 2.0:
        raise RuntimeError("Cross traffic is too close to the reset pose")
    return {"spawn_center_clearance_m": gap,
            "clear_lateral_band_outside_sweep_each_side_m": 5.2 - HALF_TRAVEL_M - half_y}


@contextmanager
def capture_depth(client):
    """Capture the actual policy input frame, including its optical pose, once."""
    original = client.simGetImages
    captured = []

    def wrapper(requests, *args, **kwargs):
        images = original(requests, *args, **kwargs)
        if (len(requests) == len(images) == 1 and requests[0].pixels_as_float
                and requests[0].image_type == airsim.ImageType.DepthPlanar and images[0].width > 0):
            captured.append(images[0])
        return images

    client.simGetImages = wrapper
    try:
        yield captured
    finally:
        client.simGetImages = original


def depth_geometry(raw, obstacle, static, fov_deg, output):
    """Ray trace the known cube from the pose in the observed DepthPlanar frame."""
    height, width = int(raw.height), int(raw.width)
    if (height, width) != (180, 320):
        raise RuntimeError(f"Unexpected raw DepthPlanar dimensions: {(height, width)}")
    depth = np.asarray(raw.image_data_float, dtype=np.float32).reshape(height, width)
    origin = np.asarray(list(raw.camera_position)) * [1, 1, -1]
    q = raw.camera_orientation
    orientation = Rotation.from_quat([q.x_val, q.y_val, q.z_val, q.w_val])
    focal = width / (2 * math.tan(math.radians(fov_deg) / 2))
    py, px = np.mgrid[:height, :width]
    local = np.stack((np.ones_like(px), (px + .5 - width / 2) / focal,
                      (py + .5 - height / 2) / focal), axis=-1)
    rays = orientation.apply(local.reshape(-1, 3)).reshape(local.shape) * [1, 1, -1]
    floor = Box("map_floor", (-9., 171.), (-60., 60.), (FLOOR_Z - .3, FLOOR_Z))
    blockers = [b for b in static if distance_to_box(origin, b) <= 28]
    expected, labels = first_intersections(origin, rays, (obstacle, floor, *blockers))
    core = binary_erosion((labels == 0) & (expected < 25), iterations=2)
    ground = binary_erosion((labels == 1) & (expected < 9), iterations=2)
    errors = np.abs(depth[core] - expected[core])
    ground_error = np.abs(depth[ground] - expected[ground])
    pixels = int(core.sum())
    matched = float(np.mean(errors <= np.maximum(.12, .035 * expected[core]))) if pixels else 0.0
    valid = float(np.mean(np.isfinite(depth) & (depth > 0)))
    center = (height // 2, width // 2)
    visit = np.where(core)
    stats = {
        "shape_px": [height, width], "camera_xyz_up_m": origin.tolist(), "time_stamp_ns": int(raw.time_stamp),
        "object_xyz_up_m": obstacle.center, "positive_finite_fraction": valid,
        "range_at_center_m": float(depth[center]), "cube_core_pixels": pixels, "cube_match_fraction": matched,
        "cube_expected_median_m": float(np.median(expected[core])) if pixels else None,
        "cube_observed_median_m": float(np.median(depth[core])) if pixels else None,
        "cube_median_col": float(np.median(visit[1])) if pixels else None,
        "cube_error_p95_m": float(np.percentile(errors, 95)) if pixels else None,
        "ground_core_pixels": int(ground.sum()),
        "ground_match_fraction": float(np.mean(ground_error < .25)) if ground.any() else 0.0,
        "max_m": float(np.nanmax(depth)),
        "passed": bool(valid >= .99 and pixels >= 35 and matched >= .98 and np.nanmax(depth) > 1.2),
    }
    stats["passed"] = stats["passed"] and stats["ground_core_pixels"] >= 300 and stats["ground_match_fraction"] >= .95
    np.savez_compressed(output, depth_m=depth, cube_core=core, expected_m=expected.astype(np.float32))
    return stats


def current_obstacle(center):
    return Box(CROSSING_KEY, *((p - d / 2, p + d / 2) for p, d in zip(center, CROSSING_SIZE_M, strict=True)))


def plane_crossing(samples, x, direction):
    """Interpolate the first forward crossing between two measured poses."""
    for a, b in zip(samples, samples[1:]):
        pa, pb = a["drone_xyz_up_m"], b["drone_xyz_up_m"]
        if direction * (pa[0] - x) < 0 <= direction * (pb[0] - x):
            frac = (x - pa[0]) / (pb[0] - pa[0])
            time_s = a["t_s"] + frac * (b["t_s"] - a["t_s"])
            y, z = (pa[i] + frac * (pb[i] - pa[i]) for i in (1, 2))
            return {"t_s": time_s, "y_m": y, "z_m": z}
    return None


def episode_clearance(samples, scene, hit):
    path = scene.path
    toward = 1 if samples[0]["drone_xyz_up_m"][0] < path.x else -1
    plane = plane_crossing(samples, path.x, toward)
    closest = min(samples, key=lambda row: row["center_to_box_m"])
    updates = scene.updates
    times = [r["t_s"] for r in updates]
    y = [r["xyz_up_m"][1] for r in updates]
    progress = np.array(y) * path.direction
    crossing_indices = np.flatnonzero(progress >= path.direction * path.y)
    actor_crossing = None
    if len(crossing_indices):
        i = int(crossing_indices[0])
        if i > 0 and progress[i] > progress[i - 1]:
            fraction = (path.direction * path.y - progress[i - 1]) / (progress[i] - progress[i - 1])
            actor_crossing = times[i - 1] + fraction * (times[i] - times[i - 1])
    time_delta = None
    if plane is not None:
        actor_y = float(np.interp(plane["t_s"], times, y))
        plane["box_center_y_m"] = actor_y
        plane["lateral_center_offset_m"] = abs(plane["y_m"] - actor_y)
        time_delta = plane["t_s"] - actor_crossing if actor_crossing is not None else None
    own_hit = hit is not None and hit["has_collided"] and hit["object_name"] == scene.actor
    hit_time = (hit["timestamp_ns"] - scene.start_timestamp_ns) / 1e9 if hit else None
    if own_hit and scene.mode == "stationary":
        contact = "stationary_added_box"
    elif own_hit:
        # Collision is generated during integration using the previously
        # applied pose. Classify from the measured motion bracketing the hit.
        i = int(np.searchsorted(times, hit_time, side="left"))
        during_motion = (0 < i < len(times) and abs(y[i] - y[i - 1]) > .001
                         and path.start_time_s <= hit_time <= path.end_time_s)
        contact = "crossing_added_box" if during_motion else "parked_added_box"
    else:
        contact = "other_geometry" if hit else None
    return {
        "drone_at_plane": plane, "obstacle_center_crossing_s": actor_crossing,
        "drone_minus_obstacle_arrival_s": time_delta, "collision_t_s": hit_time, "contact": contact,
        "minimum_center_to_box_surface_m": float(closest["center_to_box_m"]),
        "closest_sample_t_s": closest["t_s"], "closest_sample_xyz_up_m": closest["drone_xyz_up_m"],
        "within_contact_band": own_hit or closest["center_to_box_m"] <= NEAR_SURFACE_M,
    }


def check_scene(scene, *, own_hit=False):
    end_error = math.dist(scene.center, scene.read_center())
    if (end_error > .008 or scene.max_moving_update_gap_s > MAX_UPDATE_GAP_S
            or (scene.max_position_lag_m > MAX_UNBLOCKED_LAG_M and not own_hit)):
        raise RuntimeError(f"Cross-traffic clock or mesh mismatch: end_error={end_error:.4f} m, "
                           f"update_gap={scene.max_moving_update_gap_s:.4f} s, lag={scene.max_position_lag_m:.4f} m")
    return {"max_update_gap_s": scene.max_moving_update_gap_s,
            "max_position_lag_m": scene.max_position_lag_m, "terminal_pose_error_m": end_error,
            "max_nominal_update_m": scene.max_moving_update_gap_s * SPEED_MPS}
