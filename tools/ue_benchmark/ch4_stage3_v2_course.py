"""Runtime-only geometry for the roofed TrainEnv curriculum."""

from dataclasses import dataclass
from math import isclose

import cosysairsim as airsim
import numpy as np


SPAWN_NED = (0.0, 0.0, 1.0)
# AirSim object poses use the PlayerStart origin. Vehicle and camera poses use
# the initial vehicle origin, 1 m below it in this package.
FLOOR_Z = -3.6 + SPAWN_NED[2]  # Vehicle-local z up.
TUNNEL_Z = (-2.2, 0.2)
ROOF_HEIGHT = (1.4, 2.6)  # Clearance above the Stage 2 nominal spawn at z=-1.
CORRIDOR_FAMILIES = {
    "over": ("over",), "under": ("under",),
    "over_under": ("over", "under"), "under_over": ("under", "over"),
}
TIGHT_CORRIDOR_FAMILIES = {
    **CORRIDOR_FAMILIES,
    "side": ("side",),
}
# Interior boxes at the 12 m maximum width are clear of the 173 original
# AABBs. The outer surfaces may meet adjoining static geometry.
CLEAN_STRIPS = ((27.0, 85.8, 53.35), (88.5, 146.5, 53.35), (37.0, 84.5, -53.35))


@dataclass(frozen=True)
class Box:
    key: str
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    @property
    def bounds(self):
        return self.x, self.y, self.z

    @property
    def size(self):
        return tuple(hi - lo for lo, hi in self.bounds)

    @property
    def center(self):
        return tuple((lo + hi) / 2 for lo, hi in self.bounds)


def tunnel_roof(height):
    if not ROOF_HEIGHT[0] <= height <= ROOF_HEIGHT[1]:
        raise ValueError(f"Tunnel roof clearance {height} m is outside {ROOF_HEIGHT}")
    z = -1.0 + height
    return Box("tunnel_roof", (-8.8, 170.8), (-59.5, 59.5), (z, z + .36))


def sample_corridor(family, rng, *, strip=None, width=None, recovery_demo=False, tight_goal=False):
    """Place walls and a roof around a randomized course on a surveyed strip."""
    modes = TIGHT_CORRIDOR_FAMILIES[family] if tight_goal else CORRIDOR_FAMILIES[family]
    if strip is None:
        if tight_goal and len(modes) == 2:
            strip = int(rng.integers(2))  # Use the two 58 m strips so 2-gate courses have full terminal approach.
        else:
            strip = int(rng.integers(len(CLEAN_STRIPS)))
    xmin, xmax, y = CLEAN_STRIPS[strip]
    if width is None:
        if tight_goal and family == "side":
            width = float(rng.uniform(10.5, 12.0))
        else:
            width = float(rng.uniform(8, 12))
    if not 8 <= width <= 12:
        raise ValueError("Corridor clear width must be between 8 and 12 meters")
    ymin, ymax = y - width / 2, y + width / 2
    x_bounds, y_bounds = (xmin, xmax), (ymin, ymax)
    floor = FLOOR_Z - .18
    roof = float(rng.uniform(3.20, 3.78) if tight_goal else rng.uniform(2.65, 3.42))
    boxes = [
        Box("wall_min_y", x_bounds, (ymin - .32, ymin), (floor, roof + .15)),
        Box("wall_max_y", x_bounds, (ymax, ymax + .32), (floor, roof + .15)),
        Box("corridor_roof", x_bounds, (ymin - .19, ymax + .19), (roof, roof + .36)),
    ]
    direction = int(rng.choice((-1, 1)))
    start_x = (xmin if direction > 0 else xmax) + direction * float(rng.uniform(2.6, 3.7))
    if tight_goal:
        # Leave 7.0-8.5 m between goal_x and a real back wall (matching TrainEnv_3D's GoalEndWall).
        far_edge = xmax if direction > 0 else xmin
        goal_x = far_edge - direction * float(rng.uniform(7.2, 8.6))
        end_wall_x = (far_edge - .36, far_edge) if direction > 0 else (far_edge, far_edge + .36)
        boxes.append(Box("goal_end_wall", end_wall_x, (ymin - .20, ymax + .20), (floor, roof + .15)))
    else:
        goal_x = (xmax if direction > 0 else xmin) - direction * float(rng.uniform(2.6, 3.7))
    length = abs(goal_x - start_x)
    if family == "side":
        side_h = float(rng.uniform(1.55, 4.35))
        start_z = FLOOR_Z + side_h
        goal_z = start_z
    else:
        start_height = (1.45, 2.20) if modes[0] == "over" else (3.70, 4.55) if tight_goal else (3.5, 4.4)
        goal_height = (1.50, 2.30) if modes[-1] == "over" or family == "over_under" else (3.70, 4.55)
        start_z = FLOOR_Z + float(rng.uniform(*start_height))
        goal_z = FLOOR_Z + float(rng.uniform(*(goal_height if tight_goal else start_height)))

    # The existing TrainEnv floor forms the fourth face. Gate meshes overlap
    # it/the roof and both side walls so there are no traversable seams.
    gate_fractions = [(0.40, 0.52)] if len(modes) == 1 else [(0.26, 0.34), (0.62, 0.70)]
    profile = [(0.0, start_z)]
    gates = []
    side_sign = int(rng.choice((-1, 1)))
    for index, (mode, fraction) in enumerate(zip(modes, gate_fractions, strict=True)):
        distance = length * float(rng.uniform(*fraction))
        half_length = float(rng.uniform(1.4, 2.1) if mode == "side" else rng.uniform(1.05, 1.65))
        center = start_x + direction * distance
        if mode == "over":
            crest = FLOOR_Z + float(rng.uniform(2.90, 3.60) if tight_goal else rng.uniform(2.85, 3.8))
            z_span, target_z = (floor, crest), crest + .95
            y_span = (ymin - .20, ymax + .20)
        elif mode == "under":
            underside = FLOOR_Z + float(rng.uniform(2.40, 3.05) if tight_goal else rng.uniform(1.95, 2.70))
            z_span, target_z = (underside, roof + .18), underside - .85
            y_span = (ymin - .20, ymax + .20)
        else:
            pillar_half_w = float(rng.uniform(1.6, 2.2))
            pillar_offset = float(rng.uniform(-0.35, 0.35))
            y_span = (y + pillar_offset - pillar_half_w, y + pillar_offset + pillar_half_w)
            z_span, target_z = (floor, roof + .18), start_z
        gate = Box(f"{mode}_{index}", (center - half_length, center + half_length), y_span, z_span)
        gates.append(gate)
        approach = distance - half_length - 2.5
        departure = distance + half_length + 1.8
        profile.extend(((approach, target_z), (departure, target_z)))
    profile.append((max(profile[-1][0] + 2.0, length - 4.0), goal_z))
    profile.append((length, goal_z))
    if recovery_demo:
        start_y = y + int(rng.choice((-1, 1))) * float(rng.uniform(1.65, 2.45))
        goal_y = y + float(rng.uniform(-1.15, 1.15))
    elif tight_goal:
        start_y = y + float(rng.uniform(-.90, .90))
        goal_y = y + float(rng.uniform(-1.15, 1.15))
    else:
        start_y = y + float(rng.uniform(-.30, .30))
        goal_y = y + float(rng.uniform(-.35, .35))
    return {
        "family": family, "strip": strip, "direction": direction, "length": length,
        "x_bounds": x_bounds, "y_bounds": y_bounds, "z_bounds": (FLOOR_Z + .25, roof - .25),
        "center_y": y, "roof_z": roof, "side_sign": side_sign, "tight_goal": tight_goal,
        "start_position": (start_x, start_y, -start_z), "goal_position": (goal_x, goal_y, -goal_z),
        "recovery_demo": recovery_demo,
        "boxes": tuple(boxes + gates), "gates": tuple(gates), "profile": tuple(profile),
    }


class TransientBoxes:
    """Own the static collision meshes spawned into one AirSim world."""

    def __init__(self, client):
        self.client = client
        self.names = {}
        self.generation = 0
        if "Cube" not in client.simListAssets():
            raise RuntimeError("Packaged TrainEnv does not contain the Cube asset")

    def replace(self, boxes, *, verify=False):
        self.clear()
        self.generation += 1
        try:
            for box in boxes:
                x, y, z = box.center
                global_ned = (x + SPAWN_NED[0], y + SPAWN_NED[1], -z + SPAWN_NED[2])
                pose = airsim.Pose(airsim.Vector3r(*global_ned), airsim.Quaternionr())
                name = self.client.simSpawnObject(
                    f"Ch4Stage3V2_{self.generation}_{box.key}", "Cube", pose,
                    airsim.Vector3r(*box.size),
                )
                if not name:
                    raise RuntimeError(f"AirSim did not spawn {box.key}")
                self.names[box.key] = name
                if verify:
                    position = self.client.simGetObjectPose(name).position
                    scale = self.client.simGetObjectScale(name)
                    if not all(isclose(a, b, abs_tol=.005) for a, b in zip(
                        (*position, *scale), (*global_ned, *box.size), strict=True,
                    )):
                        raise RuntimeError(f"{name} pose or scale disagrees with the requested mesh")
        except BaseException:
            self.clear()
            raise

    def clear(self):
        for key, name in list(self.names.items()):
            if not self.client.simDestroyObject(name):
                raise RuntimeError(f"Could not destroy transient AirSim object {name}")
            del self.names[key]


def profile_altitude(scenario, progress):
    distances, heights = zip(*scenario["profile"], strict=True)
    return float(np.interp(progress, distances, heights))
