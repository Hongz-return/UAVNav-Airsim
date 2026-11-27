"""Stage 3 route curriculum on the original packaged multi-obstacle TrainEnv."""

from itertools import combinations
import json
import math
from pathlib import Path

import cosysairsim as airsim
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from tools.trainenv_instances import TRAINENV_MAP
from .ch4_trainenv_3d_course import ROOT, position_agl, read_json


AUGMENTED_FAMILIES = {
    "over": ("over",), "under": ("under",),
    "over_under": ("over", "under"), "under_over": ("under", "over"),
}


def load_multiobstacle_course(path):
    course = read_json(path)
    if course["schema_version"] != 1 or course["map"] != TRAINENV_MAP:
        raise ValueError(f"Stage 3 requires the original multi-obstacle TrainEnv: {path}")
    return course


def available_routes(course):
    """Use takeoff and landing points surveyed in the unchanged source map."""
    routes = {1: [], 2: []}
    for track in course["tracks"]:
        starts = track["endpoint_x_m"]
        for count in routes:
            for centers in combinations(track["anchor_x_m"], count):
                if count == 2 and not 22 <= centers[1] - centers[0] <= 46:
                    continue
                before = [x for x in starts if 15 <= centers[0] - x <= 27]
                after = [x for x in starts if 15 <= x - centers[-1] <= 27]
                if before and after:
                    routes[count].append((track["y_m"], centers, before, after))
    if not all(routes.values()):
        raise ValueError("Expected usable single and sequential obstacle routes")
    return routes


def augmented_route(course, routes, family, rng, *, strict_maneuvers=False):
    modes = AUGMENTED_FAMILIES[family]
    y, centers, before, after = routes[len(modes)][int(rng.integers(len(routes[len(modes)])))]
    x0, x1 = float(rng.choice(before)), float(rng.choice(after))
    if rng.integers(2):
        x0, x1 = x1, x0
        centers = centers[::-1]
    x0 += rng.uniform(-0.5, 0.5)
    x1 += rng.uniform(-0.5, 0.5)
    direction = 1 if x0 < x1 else -1
    z0 = float(rng.uniform(-1.3, -0.7) if modes[0] == "over" else rng.uniform(1.80, 2.65))
    z1 = float(rng.uniform(-1.35, -0.45))
    if strict_maneuvers and modes[-1] == "under":
        # Returning high after the beam teaches descent despite a high goal.
        z1 = float(rng.uniform(1.55, 2.25))
    start = [x0, y + float(rng.uniform(-0.5, 0.5)), -z0]
    goal = [x1, y + float(rng.uniform(-0.5, 0.5)), -z1]
    profile = [(0.0, z0)]
    boxes = []
    for center, mode in zip(centers, modes, strict=True):
        x = center + float(rng.uniform(-0.5, 0.5))
        size_x = float(rng.uniform(2.2, 4.5))
        width = float(rng.uniform(7.0, 10.0) if strict_maneuvers else rng.uniform(4.6, 8.2))
        offset = float(rng.uniform(-0.45, 0.45))
        if mode == "over":
            z_lo = course["floor_local_z_up_m"]
            z_hi = float(rng.uniform(0.03, 0.65) if strict_maneuvers else rng.uniform(0.22, 1.18))
            target_z = z_hi + 1.15
        else:
            z_lo = float(rng.uniform(-0.42, 0.42) if strict_maneuvers else rng.uniform(0.42, 1.32))
            z_hi = 3.94
            target_z = z_lo - 1.18
        distance = direction * (x - x0)
        extension = size_x / 2 + 2.2
        profile.extend(((distance - extension, target_z), (distance + extension, target_z)))
        boxes.append({"mode": mode, "xyz_z_up_m": [x, y + offset, (z_lo + z_hi) / 2],
                      "size_m": [size_x, width, z_hi - z_lo]})
    distance = abs(x1 - x0)
    profile.extend(((distance - 8.0, z1), (distance, z1)))
    return {"family": family, "start_position": start, "goal_position": goal, "shape": profile,
            "boxes": boxes, "direction": direction, "reference_y": y}


class Stage3TrainingCourse:
    def __init__(self, cfg):
        config = cfg["stage3"]
        path = Path(config["course_file"])
        self.course = load_multiobstacle_course(ROOT / "configs" / path)
        self.routes = available_routes(self.course)
        self.original_weight = config.getfloat("original_altitude_weight")
        self.altitude_weight = config.getfloat("altitude_tracking_weight")
        self.side_weight = config.getfloat("side_tracking_weight")
        self.strict_maneuvers = config.getboolean("strict_maneuvers", fallback=False)
        self.guided_fraction = config.getfloat("guided_fraction", fallback=0.0)
        self.scenario = None
        self.objects = []
        self.episode_number = 0
        self.cube_verified = False

    @property
    def disable_virtual_ceiling(self):
        return bool(self.scenario and self.scenario["boxes"])

    @property
    def route_complete(self):
        return not self.strict_maneuvers or self.passed_gates == len(self.scenario["boxes"])

    def select_route(self, env):
        self._clear_objects(env.client)
        self.episode_number += 1
        rng = env.np_random
        weights = self.course["family_weights"]
        families = list(weights)
        family = str(rng.choice(families, p=np.array(list(weights.values())) / sum(weights.values())))
        if family in AUGMENTED_FAMILIES:
            scenario = augmented_route(self.course, self.routes, family, rng, strict_maneuvers=self.strict_maneuvers)
            self.profile_x, self.profile_z = zip(*scenario["shape"], strict=True)
            env.task.work_space_z = self.course["curriculum_vertical_workspace_m"]
            self._place_objects(env.client, scenario["boxes"])
            yaw_jitter = math.radians(10)
        else:
            route_index = int(rng.integers(len(env.training_routes)))
            start, goal = (list(point) for point in env.training_routes[route_index])
            if env.random_route_reverse and rng.integers(2):
                start, goal = goal, start
            if family == "side":
                start[2] += float(rng.uniform(-0.45, 0.45))
                goal[2] += float(rng.uniform(-0.45, 0.45))
            scenario = {"family": family, "route_index": route_index, "start_position": start,
                        "goal_position": goal, "boxes": []}
            env.task.work_space_z = env.work_space_z
            yaw_jitter = env.start_random_angle
        self.scenario = scenario
        x, y, z_ned = scenario["start_position"]
        self.last_position = (x, y, -z_ned)
        self.passed_gates = 0
        self.valid_crossing = False
        self.prior_course_distance = 0.0
        self.guidance_steps = 0
        self.used_guidance = False
        self.guided_episode = self.guided_fraction > 0 and self.disable_virtual_ceiling and rng.random() < self.guided_fraction
        env.dynamic_model.set_start(scenario["start_position"], random_angle=yaw_jitter, fixed_yaw=None)
        env.dynamic_model._set_goal_pose_single(scenario["goal_position"])

    def _clear_objects(self, client):
        while self.objects:
            name = self.objects[-1]
            if not client.simDestroyObject(name):
                raise RuntimeError(f"Could not remove temporary training obstacle {name}")
            self.objects.pop()

    def _place_objects(self, client, boxes):
        if not self.cube_verified:
            if "Cube" not in client.simListAssets():
                raise RuntimeError("The TrainEnv package is missing the Cube asset for temporary obstacles")
            self.cube_verified = True
        try:
            for index, box in enumerate(boxes):
                x, y, z = box["xyz_z_up_m"]
                pose = airsim.Pose(airsim.Vector3r(x, y, -z), airsim.Quaternionr())
                name = client.simSpawnObject(
                    f"Stage3TrainingBox_{self.episode_number}_{index}", "Cube", pose,
                    airsim.Vector3r(*box["size_m"]),
                )
                self.objects.append(name)
        except BaseException:
            self._clear_objects(client)
            raise

    def tracking_reward(self, relative_position):
        self.last_position = relative_position
        x, y, z = relative_position
        scenario = self.scenario
        if not scenario["boxes"]:
            deviation = min(abs(z + scenario["goal_position"][2]) / 2.0, 1.0)
            return -self.original_weight * deviation ** 2
        distance = (x - scenario["start_position"][0]) * scenario["direction"]
        z_ref = np.interp(distance, self.profile_x, self.profile_z)
        altitude_error = min(abs(z - z_ref) / 1.6, 1.0)
        side_error = min(abs(y - scenario["reference_y"]) / 3.0, 1.0)
        bonus = self._gate_reward(distance, y, z) if self.strict_maneuvers else 0.0
        return bonus - self.altitude_weight * altitude_error ** 2 - self.side_weight * side_error ** 2

    def _gate_reward(self, distance, y, z):
        if self.route_complete:
            return 0.0
        box = self.scenario["boxes"][self.passed_gates]
        x, yc, zc = box["xyz_z_up_m"]
        sx, sy, sz = box["size_m"]
        direction = self.scenario["direction"]
        center = direction * (x - self.scenario["start_position"][0])
        entry, exit_ = center - sx / 2, center + sx / 2
        if self.prior_course_distance < entry <= distance:
            self.valid_crossing = True
        elif distance < entry:
            self.valid_crossing = False
        if self.valid_crossing:
            vertical_clearance = z - (zc + sz / 2) if box["mode"] == "over" else (zc - sz / 2) - z
            self.valid_crossing = abs(y - yc) < sy / 2 - 0.25 and vertical_clearance > 0.40
        self.prior_course_distance = distance
        if distance >= exit_ and self.valid_crossing:
            self.passed_gates += 1
            self.valid_crossing = False
            return 2.0
        return 0.0

    def training_action(self, action):
        """Provide vertical demonstrations for a subset of physical gate approaches."""
        self.used_guidance = False
        if not self.guided_episode or self.route_complete:
            return action
        box = self.scenario["boxes"][self.passed_gates]
        x, y, z = self.last_position
        gx, gy, _ = box["xyz_z_up_m"]
        direction = self.scenario["direction"]
        forward_distance = direction * (gx - x)
        if not (-box["size_m"][0] / 2 - 0.5 <= forward_distance <= 15.0
                and abs(y - gy) <= box["size_m"][1] / 2 + 0.5):
            return action
        progress = direction * (x - self.scenario["start_position"][0])
        reference = np.interp(progress + 2.5, self.profile_x, self.profile_z)
        guided = np.array(action, copy=True)
        guided[0] = min(guided[0], 2.4)
        guided[1] = np.clip(1.15 * (reference - z), -0.8, 0.8)
        self.guidance_steps += 1
        self.used_guidance = True
        return guided

    def terminal_metrics(self, info, truncated):
        point = position_agl(info["position"], self.course)
        gx, gy, gz_ned = self.scenario["goal_position"]
        goal = position_agl((gx, gy, -gz_ned), self.course)
        return {
            "family": self.scenario["family"], "route_index": self.scenario.get("route_index"),
            "temporary_obstacles": self.scenario["boxes"],
            "gates_passed": self.passed_gates, "guidance_steps": self.guidance_steps,
            "route_not_completed": bool(info.get("stage3_route_not_completed", False)),
            "success": bool(info["is_success"]), "collision": bool(info["is_crash"]),
            "out_of_bounds": bool(info["is_not_in_workspace"] and not info["is_crash"]),
            "timeout": bool(truncated), "episode_steps": info["step_num"],
            "terminal_xyz_agl_m": point, "goal_xyz_agl_m": goal,
            "terminal_altitude_error_m": abs(point[2] - goal[2]),
        }


class Stage3CourseMetrics(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if not done:
                continue
            metrics = info["stage3_episode"]
            prefix = f"stage3/{metrics['family']}"
            for name in ("success", "collision", "out_of_bounds", "timeout", "terminal_altitude_error_m"):
                self.logger.record_mean(f"{prefix}/{name}", float(metrics[name]))
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps({
                    "checkpoint_step": self.num_timesteps, "training_updates": self.model._n_updates,
                    "alpha": self.model.ent_coef, "return": info["episode"]["r"], **metrics,
                }, allow_nan=False) + "\n")
        return True
