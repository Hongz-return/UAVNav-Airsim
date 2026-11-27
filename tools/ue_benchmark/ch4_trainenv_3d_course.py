"""Shared geometry and fixed scenario generation for the UE5.4 TrainEnv course."""

import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[2]
COURSE_FILE = ROOT / "configs/ch4_trainenv_3d_course.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def to_ned(point_agl, course):
    """AirSim pose is relative to the initial pawn at the zero-offset PlayerStart."""
    x, y, z_agl = point_agl
    return [x, y, round(course["player_start_ue_cm"][2] / 100 - z_agl, 5)]


def position_agl(point_z_up, course):
    x, y, z_up = point_z_up
    return [x, y, z_up + course["player_start_ue_cm"][2] / 100]


def obstacles(family, course):
    lane = course["families"][family]
    half_width = course["lane_width_m"] / 2
    for obstacle in lane["obstacles"]:
        y_offsets = obstacle.get("y_offset_m", [-half_width, half_width])
        yield {
            **obstacle,
            "y_m": [lane["lane_y_m"] + offset for offset in y_offsets],
        }


def structural_boxes(course):
    """Closed shell, full-height lane dividers, and exactly the five course obstacles."""
    x_lo, x_hi = course["interior_x_m"]
    floor, roof = course["floor_agl_m"], course["roof_agl_m"]
    wall = course["wall_thickness_m"]
    half_width = course["lane_width_m"] / 2
    centers = [lane["lane_y_m"] for lane in course["families"].values()]
    y_boundaries = [centers[0] - half_width, *(center + half_width for center in centers)]
    x = [x_lo - wall, x_hi + wall]
    y = [y_boundaries[0] - wall / 2, y_boundaries[-1] + wall / 2]
    yield "Floor", x, y, [floor - wall, floor], "Floor", "Structure"
    yield "Roof", x, y, [roof, roof + wall], "Boundary", "Structure"
    yield "StartEndWall", [x_lo - wall, x_lo], y, [floor, roof], "Boundary", "Structure"
    yield "GoalEndWall", [x_hi, x_hi + wall], y, [floor, roof], "Boundary", "Structure"
    for index, edge in enumerate(y_boundaries):
        yield (
            f"LaneWall_{index}", [x_lo, x_hi], [edge - wall / 2, edge + wall / 2],
            [floor, roof], "Boundary", "Structure",
        )
    materials = {"over": "BlueBlock", "under": "BrownPanel", "side": "GreenPillar"}
    for family in course["families"]:
        for box in obstacles(family, course):
            yield (
                box["name"], box["x_m"], box["y_m"], box["z_agl_m"],
                materials[box["maneuver"]], "Obstacles",
            )


def stratified(bounds, count, rng):
    low, high = bounds
    values = [low + (high - low) * (index + rng.random()) / count for index in range(count)]
    rng.shuffle(values)
    return [round(value, 4) for value in values]


def evaluation_scenarios(course):
    """One independently shuffled sample per equal-width bin of each variable."""
    spec = course["evaluation"]
    rng = random.Random(spec["seed"])
    count = spec["episodes_per_family"]
    scenarios = []
    for name, lane in course["families"].items():
        lane_y = lane["lane_y_m"]
        sample_start_z = stratified(lane["start_agl_m"], count, rng)
        sample_goal_z = (
            sample_start_z if lane.get("same_endpoint_altitude")
            else stratified(lane["goal_agl_m"], count, rng)
        )
        for index, (sx, sy, sz, gx, gy, gz, jitter) in enumerate(zip(
            stratified(spec["start_x_m"], count, rng),
            stratified(spec["start_y_offset_m"], count, rng),
            sample_start_z,
            stratified(spec["goal_x_m"], count, rng),
            stratified(spec["goal_y_offset_m"], count, rng),
            sample_goal_z,
            stratified(spec["yaw_offset_deg"], count, rng),
            strict=True,
        ), start=1):
            start = [sx, round(lane_y + sy, 4), sz]
            goal = [gx, round(lane_y + gy, 4), gz]
            yaw = math.atan2(gy - sy, gx - sx) + math.radians(jitter)
            scenarios.append({
                "scenario_id": f"{name}_{index:02d}",
                "family": name,
                "start_xyz_agl_m": start,
                "goal_xyz_agl_m": goal,
                "start_position": to_ned(start, course),
                "goal_position": to_ned(goal, course),
                "yaw_offset_deg": jitter,
                "yaw_rad": yaw,
                "yaw_deg": math.degrees(yaw),
            })
    return scenarios


def build_protocol(course):
    return {
        "schema_version": 1,
        "geometry_sha256": sha256(COURSE_FILE),
        "map": course["map"],
        "settings": course["evaluation"],
        "inference": "Original checkpoint encoding and history, deterministic actions, zero gradient updates.",
        "success": (
            "Horizontal error < 2.0 m and absolute altitude error < 0.5 m in the same step, "
            "with no collision or workspace violation. All prescribed obstacles must have been "
            "traversed forward in order with positive center clearance on the required side."
        ),
        "outcomes": ["success", "collision", "out_of_bounds", "timeout", "route_not_completed"],
        "depth": (
            "Audit the raw 320x180 DepthPlanar response consumed at reset and every policy step. "
            "Require increasing positive timestamps and >=99.9% finite positive depths. "
            "A sensor or RPC failure invalidates the run and is recorded as such."
        ),
        "scenarios": evaluation_scenarios(course),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lock the TrainEnv 3D evaluation scenarios before flight.")
    parser.add_argument("output", type=Path)
    output = parser.parse_args().output
    payload = build_protocol(read_json(COURSE_FILE))
    with output.open("x", encoding="utf-8") as target:
        json.dump(payload, target, indent=2, ensure_ascii=False, allow_nan=False)
        target.write("\n")
    print(f"{len(payload['scenarios'])} scenarios, SHA-256 {sha256(output)}")
