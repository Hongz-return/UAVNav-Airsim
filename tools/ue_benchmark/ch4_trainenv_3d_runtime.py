"""Shared AirSim telemetry for the TrainEnv_3D flight and commissioning runs."""

from contextlib import contextmanager
import json
import math

import cosysairsim as airsim
import numpy as np

from tools.experiment_config import (
    CH4_OBSERVATION_CONTRACT as CH4_OBSERVATION_CONTRACT,
    assert_observation_contract as assert_observation_contract,
    read_ini as read_ini,
)
from tools.ue_benchmark.ch4_trainenv_3d_course import ROOT, obstacles, position_agl, structural_boxes


REPORT = ROOT / "reports/ch4_trainenv_3d_20261125"
TRAIN_CONFIG = ROOT / "configs/ch4_train_trainenv_3d.ini"
SETTINGS_TEMPLATE = ROOT / "configs/ch4_trainenv_3d_airsim.json"
VEHICLE = "SimpleFlight"
SPAWN = [0.0, 0.0, 0.0]


def json_line(handle, document):
    handle.write(json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n")


class DepthAudit:
    """Measure the same float DepthPlanar response that the controller consumes."""

    def __init__(self, specification):
        self.shape = tuple(specification["depth_shape_px"])
        self.max_depth = specification["depth_max_encoding_m"]
        self.minimum_valid = specification["min_finite_positive_depth_fraction"]
        self.previous_time_ns = 0
        self.frame = None
        self.stats = None

    def consume(self, image):
        values = np.asarray(image.image_data_float, dtype=np.float32)
        shape = (int(image.height), int(image.width))
        if shape != self.shape or values.size != math.prod(self.shape):
            raise RuntimeError(f"Expected raw depth {self.shape}, got {shape} with {values.size} values")
        timestamp = int(image.time_stamp)
        valid_mask = np.isfinite(values) & (values > 0)
        valid = values[valid_mask]
        fraction = float(valid_mask.mean())
        if fraction < self.minimum_valid or timestamp <= self.previous_time_ns:
            raise RuntimeError(
                f"Bad raw depth: finite-positive fraction {fraction}, "
                f"timestamp {timestamp}, previous {self.previous_time_ns}"
            )
        self.previous_time_ns = timestamp
        self.frame = values.reshape(shape)
        h, w = shape
        center = self.frame[h // 2 - 1:h // 2 + 2, w // 2 - 1:w // 2 + 2]
        center = center[np.isfinite(center) & (center > 0)]
        self.stats = {
            "timestamp_ns": timestamp,
            "shape_px": list(shape),
            "positive_finite_fraction": fraction,
            "encoding_range_fraction": float(np.mean(valid_mask & (values <= self.max_depth))),
            "min_m": float(valid.min()),
            "max_m": float(valid.max()),
            "center_m": float(np.median(center)) if center.size else None,
            "camera_position_ned_m": list(image.camera_position),
        }
        return self.stats

    @contextmanager
    def attach(self, client):
        original = client.simGetImages

        def audited(requests, *args, **kwargs):
            images = original(requests, *args, **kwargs)
            if (len(requests) == 1 and requests[0].image_type == airsim.ImageType.DepthPlanar
                    and requests[0].pixels_as_float and images and images[0].width > 0):
                self.consume(images[0])
            return images

        client.simGetImages = audited
        try:
            yield
        finally:
            client.simGetImages = original

    def latest(self, previous_step_ns):
        if self.stats is None or self.previous_time_ns <= previous_step_ns:
            raise RuntimeError("Step did not consume a new raw DepthPlanar frame")
        return self.stats


def scenario_errors(position, goal):
    return math.hypot(position[0] - goal[0], position[1] - goal[1]), abs(position[2] - goal[2])


def assert_reset_depth_consistent(first_sample, course):
    """Reject uncompiled [0, 1] luminance fallback or occluded DepthPlanar frames at reset."""
    depth = first_sample["depth"]
    cam_ned = depth["camera_position_ned_m"]
    cam_agl = position_agl([cam_ned[0], cam_ned[1], -cam_ned[2]], course)
    nearest_surface_m = min(
        math.sqrt(sum(max(lo - a, 0.0, a - hi) ** 2 for a, (lo, hi) in zip(cam_agl, (ax, ay, az))))
        for _, ax, ay, az, *_ in structural_boxes(course)
    )
    center_m = depth.get("center_m")
    max_m = depth.get("max_m")
    if (
        center_m is None
        or max_m is None
        or max_m <= 1.2
        or center_m + 0.10 < nearest_surface_m
    ):
        raise RuntimeError(
            f"Reset DepthPlanar inconsistent with scene geometry: "
            f"center_m={center_m}, max_m={max_m}, nearest_surface_m={nearest_surface_m:.3f}"
        )


def collision_details(client):
    hit = client.simGetCollisionInfo(vehicle_name=VEHICLE)
    return {
        "has_collided": bool(hit.has_collided),
        "object_name": str(hit.object_name),
        "timestamp_ns": int(hit.time_stamp),
        "impact_ned_m": list(hit.impact_point),
        "normal_ned": list(hit.normal),
    }


def start_sample(env, course, depth):
    dynamics = env.dynamic_model
    position = position_agl(dynamics.get_position(), course)
    attitude = dynamics.get_attitude()
    dynamics.get_velocity()
    return {
        "step": 0,
        "position_xyz_agl_m": position,
        "attitude_rad": list(attitude),
        "velocity_xyz_up_mps": list(dynamics.last_velocity_world),
        "state_timestamp_ns": dynamics.last_state_timestamp,
        "depth": depth,
    }


def step_sample(info, course, depth):
    return {
        "step": info["step_num"],
        "position_xyz_agl_m": position_agl(info["position"], course),
        "velocity_xyz_up_mps": info["velocity_world"],
        "action": [float(value) for value in info["actual_action"]],
        "state_timestamp_ns": info["state_timestamp"],
        "depth": depth,
    }


def interpolate_plane(left, right, x):
    a, b = left["position_xyz_agl_m"], right["position_xyz_agl_m"]
    ratio = (x - a[0]) / (b[0] - a[0])
    return {
        "step": left["step"] + ratio * (right["step"] - left["step"]),
        "position_xyz_agl_m": [x, *(a[i] + ratio * (b[i] - a[i]) for i in (1, 2))],
    }


def progress_through_obstacles(samples, family, course):
    records = []
    sequence = []
    for obstacle in obstacles(family, course):
        x_in, x_out = obstacle["x_m"]
        x_mid = (x_in + x_out) / 2
        passage = None
        attempts, centers = [], []
        for a, b in zip(samples, samples[1:]):
            x0, x1 = a["position_xyz_agl_m"][0], b["position_xyz_agl_m"][0]
            if x1 < x_in:
                passage = None
            if x0 < x_mid <= x1 or x1 < x_mid <= x0:
                centers.append({**interpolate_plane(a, b, x_mid), "forward": x1 > x0})
            if x0 < x_in <= x1:
                passage = [interpolate_plane(a, b, x_in)]
            if passage is None:
                continue
            if x0 < x_out <= x1:
                passage.append(interpolate_plane(a, b, x_out))
                ys = [row["position_xyz_agl_m"][1] for row in passage]
                zs = [row["position_xyz_agl_m"][2] for row in passage]
                maneuver = obstacle["maneuver"]
                if maneuver == "over":
                    clearance = min(zs) - obstacle["z_agl_m"][1]
                elif maneuver == "under":
                    clearance = obstacle["z_agl_m"][0] - max(zs)
                else:
                    clearance = max(obstacle["y_m"][0] - max(ys), min(ys) - obstacle["y_m"][1])
                attempts.append({
                    "entry": passage[0],
                    "exit": passage[-1],
                    "height_range_agl_m": [min(zs), max(zs)],
                    "center_clearance_to_obstacle_m": clearance,
                    "correct_side": clearance > 0,
                })
                passage = None
            else:
                passage.append(b)
        valid = next((attempt for attempt in attempts if attempt["correct_side"]), None)
        if valid:
            sequence.append((valid["exit"]["step"], obstacle["name"]))
        records.append({
            "obstacle": obstacle["name"],
            "maneuver": obstacle["maneuver"],
            "center_plane_crossings": centers,
            "passages": attempts,
            "passed": valid is not None,
        })
    order = [name for _, name in sorted(sequence)]
    return records, order
