"""The two kinds of Stage 3 training episodes use separate reward envelopes."""

from dataclasses import asdict
import math

import numpy as np

from .ch4_stage3_crossing import CrossingScene, crossing_mode, sample_roofed_crossing
from .ch4_stage3_v2_course import (
    CORRIDOR_FAMILIES, FLOOR_Z, ROOF_HEIGHT, TIGHT_CORRIDOR_FAMILIES, TUNNEL_Z,
    TransientBoxes, profile_altitude, sample_corridor, tunnel_roof,
)


ROOFED_FAMILIES = ("original_roofed", "side_roofed")
EPISODE_CYCLE = (
    "original_roofed", "over", "side_roofed", "under",
    "original_roofed", "over_under", "side_roofed", "under_over",
)
EPISODE_CYCLE_TIGHT = (
    "original_roofed", "over", "original_roofed", "under", "side_roofed",
    "side", "original_roofed", "over_under", "side_roofed", "under_over",
)


class Stage3V2Course:
    disable_virtual_ceiling = True
    used_guidance = False

    def __init__(self, cfg):
        section = cfg["stage3_v2"]
        self.tight_goal = section.getboolean("tight_goal", fallback=False)
        self.tracking_weight = section.getfloat("altitude_tracking_weight")
        self.centerline_weight = section.getfloat("centerline_weight")
        self.gate_bonus = section.getfloat("gate_bonus")
        self.corridor_accept_radius = section.getfloat("corridor_accept_radius", fallback=2.0)
        self.corridor_altitude_tolerance = section.getfloat("corridor_altitude_tolerance", fallback=0.5)
        self.roofed_altitude_weight = cfg.getfloat("reward", "altitude_deviation_weight")
        self.tunnel_accept_radius = cfg.getfloat("environment", "accept_radius")
        self.tunnel_goal_tolerance = cfg.getfloat("environment", "accept_altitude_tolerance", fallback=None)
        self.tunnel_max_steps = cfg.getint("environment", "max_episode_steps")
        self.worker_parity = cfg.getint("airsim", "rpc_port") % 2
        self.next_family = None
        self.forced_family = None  # Only the offline demonstration collector uses this.
        self.recovery_demo = False
        self.geometry = None
        self.scenario = None
        self.crossing_mode = crossing_mode(cfg)
        self.crossing_arm = self.crossing_mode
        self.crossing_case = None
        self.crossing = None
        self.record_crossing_updates = False

    @property
    def route_complete(self):
        return self.passed_gates == len(self.scenario["gates"])

    def select_route(self, env):
        rng = env.np_random
        cycle = EPISODE_CYCLE_TIGHT if self.tight_goal else EPISODE_CYCLE
        corridor_families = TIGHT_CORRIDOR_FAMILIES if self.tight_goal else CORRIDOR_FAMILIES
        case = self.crossing_case
        if case is not None:
            family = "original_roofed"
        elif self.forced_family:
            family = self.forced_family
        else:
            if self.next_family is None:
                self.next_family = 2 * int(rng.integers(len(cycle) // 2)) + self.worker_parity
            family = cycle[self.next_family]
            self.next_family = (self.next_family + 1) % len(cycle)
        if self.geometry is None:
            self.geometry = TransientBoxes(env.client)
        corridor = family in corridor_families
        task = env.task
        if corridor:
            scenario = sample_corridor(family, rng, recovery_demo=self.recovery_demo, tight_goal=self.tight_goal)
            task.work_space_x = scenario["x_bounds"]
            task.work_space_y = scenario["y_bounds"]
            task.work_space_z = scenario["z_bounds"]
            task.accept_radius = self.corridor_accept_radius
            task.accept_altitude_tolerance = self.corridor_altitude_tolerance
            task.altitude_deviation_weight = 0.0
            env.max_episode_steps = 500
            yaw = math.radians(10)
        else:
            if case is None:
                scenario = self._roofed_route(env, family)
            else:
                scenario = {
                    "family": family, "start_position": case.start_position, "goal_position": case.goal_position,
                    "roof_clearance_above_spawn_m": case.roof_clearance_m,
                    "boxes": (tunnel_roof(case.roof_clearance_m),), "gates": (),
                }
            task.work_space_x = env.work_space_x
            task.work_space_y = env.work_space_y
            task.work_space_z = TUNNEL_Z
            task.accept_radius = self.tunnel_accept_radius
            task.accept_altitude_tolerance = self.tunnel_goal_tolerance
            task.altitude_deviation_weight = self.roofed_altitude_weight
            env.max_episode_steps = self.tunnel_max_steps
            yaw = env.start_random_angle
        self.crossing = None
        if not corridor and self.crossing_mode != "off":
            path, phase = (case.path, case.phase) if case is not None else sample_roofed_crossing(scenario, rng)
            box = path.box_at(0.0, self.crossing_arm)
            scenario["boxes"] += (box,)
            self.geometry.replace(scenario["boxes"], verify=True)
            self.crossing = CrossingScene(
                env.client, self.geometry.names[box.key], path, self.crossing_arm, phase,
                record_updates=self.record_crossing_updates,
            )
        else:
            self.geometry.replace(scenario["boxes"])
        self.scenario = scenario
        self.passed_gates = 0
        self.valid_crossing = False
        x, y, z = scenario["start_position"]
        self.last_position = (x, y, -z)
        self.min_z = self.max_z = -z
        if case is None:
            env.dynamic_model.set_start(scenario["start_position"], random_angle=yaw)
        else:
            env.dynamic_model.set_start(scenario["start_position"], random_angle=0.0, fixed_yaw=case.yaw_rad)
        env.dynamic_model._set_goal_pose_single(scenario["goal_position"])

    @staticmethod
    def _roofed_route(env, family):
        rng = env.np_random
        indices = range(len(env.training_routes))
        if family == "side_roofed":
            indices = [i for i, (a, b) in enumerate(env.training_routes) if a[1] != b[1]]
        index = int(rng.choice(indices))
        start, goal = (list(point) for point in env.training_routes[index])
        if env.random_route_reverse and rng.integers(2):
            start, goal = goal, start
        if family == "side_roofed":
            start[1] += float(rng.uniform(-.35, .35))
            goal[1] += float(rng.uniform(-.35, .35))
        roof_height = float(rng.uniform(*ROOF_HEIGHT))
        return {
            "family": family, "route_index": index, "start_position": start, "goal_position": goal,
            "roof_clearance_above_spawn_m": roof_height, "boxes": (tunnel_roof(roof_height),), "gates": (),
        }

    @staticmethod
    def training_action(action):
        return action

    def tracking_reward(self, position):
        previous = self.last_position
        self.last_position = position
        x, y, z = position
        self.min_z, self.max_z = min(self.min_z, z), max(self.max_z, z)
        scenario = self.scenario
        if scenario["family"] in ROOFED_FAMILIES:
            return 0.0  # The original altitude penalty already applies in BaseTask.
        direction = scenario["direction"]
        start_x = scenario["start_position"][0]
        progress = direction * (x - start_x)
        altitude_error = min(abs(z - profile_altitude(scenario, progress)) / 1.6, 1.0)
        half_width = (scenario["y_bounds"][1] - scenario["y_bounds"][0]) / 2
        if self.tight_goal:
            remaining = scenario["length"] - progress
            gx, gy, gz_ned = scenario["goal_position"]
            if self.route_complete and remaining < 15.0:
                target_y = gy
                lateral_error = min(abs(y - target_y) / max(half_width * 0.75, 1.0), 1.0)
            elif scenario["family"] == "side":
                lateral_error = 0.0  # Do not pull the drone into the central side pillar.
            else:
                lateral_error = min(abs(y - scenario["center_y"]) / half_width, 1.0)
            bonus = self._cross_gate(previous, position) if not self.route_complete else 0.0
            terminal_shaping = 0.0
            if self.route_complete:
                dxy = math.hypot(x - gx, y - gy)
                if dxy < 9.0:
                    z_err = min(abs(z + gz_ned) / 1.0, 1.0)
                    xy_err = min(dxy / 9.0, 1.0)
                    terminal_shaping = -0.22 * z_err**2 - 0.14 * xy_err**2
            return bonus + terminal_shaping - self.tracking_weight * altitude_error**2 - self.centerline_weight * lateral_error**2
        lateral_error = min(abs(y - scenario["center_y"]) / half_width, 1.0)
        bonus = self._cross_gate(previous, position) if not self.route_complete else 0.0
        return bonus - self.tracking_weight * altitude_error**2 - self.centerline_weight * lateral_error**2

    def _cross_gate(self, previous, current):
        scenario = self.scenario
        gate = scenario["gates"][self.passed_gates]
        direction = scenario["direction"]
        enter, leave = sorted(direction * x for x in gate.x)
        before, after = direction * previous[0], direction * current[0]
        if after < enter:
            self.valid_crossing = False
        if before < enter <= after:
            self.valid_crossing = True
        if not self.valid_crossing:
            return 0.0
        if after != before:
            weights = np.clip([(enter - before) / (after - before), (leave - before) / (after - before)], 0, 1)
        else:
            weights = [0, 1]
        ymin, ymax = scenario["y_bounds"]
        zmin, zmax = scenario["z_bounds"]
        for gap in weights:
            y, z = (previous[i] + gap * (current[i] - previous[i]) for i in (1, 2))
            if gate.key.startswith("side"):
                side_clear = (ymin + .25 <= y <= gate.y[0] - .30) or (gate.y[1] + .30 <= y <= ymax - .25)
                if not side_clear or not zmin <= z <= zmax:
                    self.valid_crossing = False
                    return 0.0
            else:
                clearance = z - gate.z[1] if gate.key.startswith("over") else gate.z[0] - z
                if not ymin + .25 <= y <= ymax - .25 or clearance < .30:
                    self.valid_crossing = False
                    return 0.0
        if after >= leave:
            self.passed_gates += 1
            self.valid_crossing = False
            return self.gate_bonus
        return 0.0

    def terminal_metrics(self, info, truncated):
        scenario = self.scenario
        x, y, z = info["position"]
        gx, gy, gz = scenario["goal_position"]
        metrics = {
            "family": scenario["family"], "route_index": scenario.get("route_index"),
            "strip": scenario.get("strip"), "gates_passed": self.passed_gates,
            "gates_total": len(scenario["gates"]), "route_complete": self.route_complete,
            "temporary_obstacles": [asdict(box) for box in scenario["boxes"]],
            "success": bool(info["is_success"]), "collision": bool(info["is_crash"]),
            "out_of_bounds": bool(info["is_not_in_workspace"] and not info["is_crash"]),
            "timeout": bool(truncated), "episode_steps": info["step_num"],
            "min_z_up_m": min(self.min_z, z), "max_z_up_m": max(self.max_z, z),
            "terminal_xyz_agl_m": [x, y, z - FLOOR_Z],
            "goal_xyz_agl_m": [gx, gy, -gz - FLOOR_Z],
            "terminal_altitude_error_m": abs(z + gz),
        }
        if self.crossing is not None:
            metrics["crossing"] = self.crossing.info()
        return metrics

    def close(self):
        if self.geometry is not None:
            self.geometry.clear()


def geometry_demo_action(scenario, position, yaw):
    """Follow a continuous, speed-limited path through the actual gate openings."""
    x, y, z = position
    gx, gy, gz_ned = scenario["goal_position"]
    goal_z = -gz_ned
    direction = scenario["direction"]
    progress = direction * (x - scenario["start_position"][0])
    near = profile_altitude(scenario, progress + .8)
    far = profile_altitude(scenario, progress + 3.8)
    slope = abs(far - near) / 3
    forward = min(2.55, .65 / max(.05, slope))
    target_z = profile_altitude(scenario, progress + 2.35)
    remaining = scenario["length"] - progress
    dxy = math.hypot(gx - x, gy - y)

    if scenario.get("family") == "side":
        gate = scenario["gates"][0]
        gate_x = sum(gate.x) / 2
        gate_dist = direction * (gate_x - x)
        sign = scenario.get("side_sign", 1)
        if -3.5 <= gate_dist <= 13.0:
            edge = gate.y[1] + 1.65 if sign > 0 else gate.y[0] - 1.65
            ymin, ymax = scenario["y_bounds"]
            target_y = float(np.clip(edge, ymin + 1.1, ymax - 1.1))
        else:
            target_y = gy
        target_z = goal_z
    else:
        target_y = scenario["center_y"] if remaining > 12.0 else gy

    if scenario["recovery_demo"]:
        exit_x = max(gx_edge * direction for gx_edge in scenario["gates"][-1].x)
        exit_progress = exit_x - direction * scenario["start_position"][0]
        target_y = np.interp(progress + 1.5, (exit_progress + .2, scenario["length"] - 5.0),
                             (scenario["start_position"][1], gy))
        if remaining < 9:
            forward = min(forward, 1.9)

    if scenario.get("tight_goal") and (remaining < 11.0 or dxy < 11.0):
        # Precision terminal guidance directly onto (gx, gy, goal_z) inside the tight goal cylinder.
        desired_yaw = math.atan2(gy - y, gx - x)
        forward = min(forward, max(0.85, 0.55 + 0.22 * dxy))
        target_z = goal_z
        vz_gain = 1.85
    else:
        desired_yaw = math.atan2(target_y - y, 4 * direction)
        vz_gain = 1.55
    yaw_error = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
    return np.array([forward, np.clip(vz_gain * (target_z - z), -.8, .8),
                     np.clip(2.2 * yaw_error, -math.pi / 3, math.pi / 3)], dtype=np.float32)


def terminal_landing_action(goal_position, position, yaw):
    """Guide the last few meters of roofed demos cleanly into the tight 2.0m x 0.5m goal."""
    x, y, z = position
    gx, gy, gz_ned = goal_position
    goal_z = -gz_ned
    dxy = math.hypot(gx - x, gy - y)
    desired_yaw = math.atan2(gy - y, gx - x)
    yaw_error = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
    forward = max(0.8, min(2.0, 0.5 + 0.25 * dxy))
    return np.array([forward, np.clip(1.8 * (goal_z - z), -.8, .8),
                     np.clip(2.2 * yaw_error, -math.pi / 3, math.pi / 3)], dtype=np.float32)
