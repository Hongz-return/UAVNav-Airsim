"""One swept cross-traffic obstacle for the original roofed TrainEnv routes."""

from dataclasses import dataclass
import math

import cosysairsim as airsim
import gymnasium as gym
import numpy as np

from .ch4_stage3_v2_course import Box, SPAWN_NED


CROSSING_MODES = ("off", "stationary", "moving")
CROSSING_KEY = "cross_traffic"
PHASE_OFFSETS_S = {"early": -2.2, "near": 0.0, "late": 2.2}
CROSSING_SIZE_M = (1.10, 1.40, 1.40)
HALF_TRAVEL_M = 3.0
SPEED_MPS = 1.35
REFERENCE_SPEED_MPS = 2.65
# Clear cross-sections surveyed on TrainEnv, in order of the three spawn lanes.
SPAWN_LANES = (-15.0, 0.0, 15.0)
APPROACH_DISTANCES = {0.0: (10.5, 12.5, 17.5), 162.0: (10.5, 11.0, 17.0)}


def crossing_mode(cfg):
    mode = cfg.get("stage3_v2", "crossing_mode", fallback="off")
    if mode not in CROSSING_MODES:
        raise ValueError(f"stage3_v2.crossing_mode must be one of {CROSSING_MODES}")
    return mode


def enable_crossing_clock(cfg):
    """Apply the clock contract only to the opt-in configuration."""
    if crossing_mode(cfg) == "off":
        return
    if cfg.has_option("scene", "runtime_script"):
        raise ValueError("TrainEnv cross traffic and the UE sequence controller share a clock callback")
    cfg["airsim"]["synchronous_steps"] = "True"


def ready_reset_observation(env, obs):
    """Reacquire a bad reset frame while physics and the obstacle are paused."""
    scale = 255 / env.perception.max_depth_meters
    for attempt in range(5):
        lower = obs["image"][-10:, :, 0]
        # The surveyed level reset sees the floor in these downward rays.
        # A saturated frame or a frame filled by the body cannot pass.
        if np.mean((lower > 1.2 * scale) & (lower < 10.0 * scale)) >= .5:
            return obs, attempt
        if attempt < 4:
            obs = env._get_obs()
    raise RuntimeError("AirSim reset DepthPlanar still lacks near ground geometry after five captures")


@dataclass(frozen=True)
class CrossingPath:
    """A single run through a lane followed by a stop, in vehicle-local z up."""

    x: float
    y: float
    z: float
    direction: int
    center_time_s: float

    def __post_init__(self):
        if self.direction not in (-1, 1):
            raise ValueError("Cross-traffic lateral direction must be -1 or 1")
        if not all(math.isfinite(value) for value in (self.x, self.y, self.z, self.center_time_s)):
            raise ValueError("Cross-traffic pose and times must be finite")
        if self.start_time_s < 0:
            raise ValueError("Cross traffic must start at or after the episode clock starts")

    @property
    def start_time_s(self):
        return self.center_time_s - HALF_TRAVEL_M / SPEED_MPS

    @property
    def end_time_s(self):
        return self.center_time_s + HALF_TRAVEL_M / SPEED_MPS

    def center_at(self, time_s, mode):
        travel = 0.0 if mode == "stationary" else min(
            2 * HALF_TRAVEL_M, max(0.0, (time_s - self.start_time_s) * SPEED_MPS),
        )
        return (self.x, self.y + self.direction * (travel - HALF_TRAVEL_M), self.z)

    def box_at(self, time_s, mode):
        bounds = tuple((v - s / 2, v + s / 2) for v, s in zip(
            self.center_at(time_s, mode), CROSSING_SIZE_M, strict=True,
        ))
        return Box(CROSSING_KEY, *bounds)


@dataclass(frozen=True)
class CrossingCase:
    """A roofed route with an identical reset state in both comparison arms."""

    start_position: tuple[float, float, float]
    goal_position: tuple[float, float, float]
    yaw_rad: float
    roof_clearance_m: float
    path: CrossingPath
    phase: str


def sample_roofed_crossing(scenario, rng):
    sx, sy, _ = scenario["start_position"]
    gx, gy, _ = scenario["goal_position"]
    if sx not in APPROACH_DISTANCES or {sx, gx} != set(APPROACH_DISTANCES):
        raise ValueError("Cross traffic needs the surveyed roofed TrainEnv route endpoints x=0 and x=162")
    lane_index = min(range(len(SPAWN_LANES)), key=lambda index: abs(sy - SPAWN_LANES[index]))
    if abs(sy - SPAWN_LANES[lane_index]) > .36 or min(abs(gy - lane) for lane in SPAWN_LANES) > .36:
        raise ValueError("Cross traffic needs the surveyed TrainEnv spawn and goal lanes at y=-15, 0 or 15")
    distance = APPROACH_DISTANCES[sx][lane_index]
    phase = str(rng.choice(tuple(PHASE_OFFSETS_S)))
    return CrossingPath(
        sx + math.copysign(distance, gx - sx),
        sy + (gy - sy) * distance / abs(gx - sx),
        -1.0, int(rng.choice((-1, 1))),
        distance / REFERENCE_SPEED_MPS + .8 + PHASE_OFFSETS_S[phase],
    ), phase


class CrossingScene:
    """Advance the real mesh at paused physics boundaries using AirSim's clock."""

    def __init__(self, client, actor, path, mode, phase, *, record_updates):
        self.client, self.actor = client, actor
        self.path, self.mode, self.phase = path, mode, phase
        self.center = path.center_at(0.0, mode)
        self.requested_center = self.center
        self.time_s = 0.0
        self.max_moving_update_gap_s = 0.0
        self.max_position_lag_m = 0.0
        self.updates = [] if record_updates else None

    def start(self, dynamics):
        self.start_timestamp_ns = int(self.client.getMultirotorState(vehicle_name=dynamics.vehicle_name).timestamp)
        self.center = self.read_center()
        self._record()
        # The stationary arm uses the same subdivided physics windows.
        dynamics.scene_step_callback = self.advance

    def read_center(self):
        x, y, z = self.client.simGetObjectPose(self.actor).position
        center = (x - SPAWN_NED[0], y - SPAWN_NED[1], SPAWN_NED[2] - z)
        if not all(math.isfinite(value) for value in center):
            raise RuntimeError(f"Lost cross-traffic actor {self.actor}")
        return center

    def advance(self, timestamp_ns):
        elapsed = (int(timestamp_ns) - self.start_timestamp_ns) / 1e9
        if elapsed < self.time_s:
            raise RuntimeError("Cross-traffic simulation time moved backwards")
        if elapsed == self.time_s:
            return
        if self.time_s < self.path.end_time_s and elapsed > self.path.start_time_s:
            self.max_moving_update_gap_s = max(self.max_moving_update_gap_s, elapsed - self.time_s)
        self.time_s = elapsed
        requested = self.path.center_at(elapsed, self.mode)
        if requested != self.requested_center:
            x, y, z = requested
            pose = airsim.Pose(airsim.Vector3r(x + SPAWN_NED[0], y + SPAWN_NED[1], SPAWN_NED[2] - z))
            # Sweep the same mesh used by DepthPlanar. Keep the real pose if
            # Unreal clips the sweep on contact.
            self.client.simSetObjectPose(self.actor, pose, teleport=False)
            self.center = self.read_center()
            self.requested_center = requested
            self.max_position_lag_m = max(self.max_position_lag_m, math.dist(self.center, requested))
        self._record()

    def _record(self):
        if self.updates is not None:
            self.updates.append({
                "t_s": self.time_s, "xyz_up_m": self.center, "requested_xyz_up_m": self.requested_center,
            })

    def info(self):
        return {"actor": self.actor, "mode": self.mode, "phase": self.phase,
                "t_s": self.time_s, "xyz_up_m": self.center}


class CrossingCourseEnv(gym.Wrapper):
    """Keep traffic fixed through reset and start its timing at the first observation."""

    def reset(self, *, seed=None, options=None):
        base = self.env.unwrapped
        if base._evaluation_scenarios is not None:
            raise ValueError("Use evaluate_trainenv_crossing for preset cross-traffic routes")
        course = base.stage3_course
        dynamics = base.dynamic_model
        dynamics.scene_step_callback = None
        options = {} if options is None else dict(options)
        course.crossing_case = options.pop("crossing_case", None)
        course.crossing_arm = options.pop("crossing_arm", course.crossing_mode)
        course.record_crossing_updates = options.pop("record_crossing_updates", False)
        if course.crossing_arm not in CROSSING_MODES[1:]:
            raise ValueError(f"Unknown cross-traffic arm: {course.crossing_arm}")
        try:
            obs, info = self.env.reset(seed=seed, options=options)
        finally:
            course.crossing_case = None
        if course.crossing is not None:
            obs, retries = ready_reset_observation(base, obs)
            course.crossing.start(dynamics)
            info["stage3_crossing"] = course.crossing.info()
            info["stage3_crossing"]["reset_depth_recaptures"] = retries
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped
        scene = base.stage3_course.crossing
        if scene is not None:
            scene.advance(info["state_timestamp"])
            if scene.time_s >= scene.path.end_time_s:
                # Both arms use fine windows until the planned motion stops.
                base.dynamic_model.scene_step_callback = None
            info["stage3_crossing"] = scene.info()
            if info["is_crash"]:
                hit = base.task.last_collision_info
                info["stage3_crossing"]["collision"] = {
                    "has_collided": bool(hit.has_collided), "object_name": str(hit.object_name),
                    "timestamp_ns": int(hit.time_stamp), "impact_ned_m": list(hit.impact_point),
                    "normal_ned": list(hit.normal),
                }
        return obs, reward, terminated, truncated, info

    def close(self):
        base = self.env.unwrapped
        base.dynamic_model.scene_step_callback = None
        try:
            base.stage3_course.close()
        finally:
            try:
                self.env.close()
            finally:
                base.client.client.close()
