"""Multirotor dynamics using Cosys-AirSim."""

from contextlib import contextmanager

import cosysairsim as airsim
import numpy as np
import math
import json
import time
from gymnasium import spaces


def to_quaternion(pitch, roll, yaw):
    """Convert Euler angles to Quaternionr."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)

    w = cy * cr * cp + sy * sr * sp
    x = cy * sr * cp - sy * cr * sp
    y = cy * cr * sp + sy * sr * cp
    z = sy * cr * cp - cy * sr * sp
    return airsim.Quaternionr(x, y, z, w)


def quaternion_to_euler(q):
    """Convert Quaternionr to Euler angles (roll, pitch, yaw)."""
    w = getattr(q, "w_val", getattr(q, "w", 0.0))
    x = getattr(q, "x_val", getattr(q, "x", 0.0))
    y = getattr(q, "y_val", getattr(q, "y", 0.0))
    z = getattr(q, "z_val", getattr(q, "z", 0.0))

    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))

    t2 = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(t2)

    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll, pitch, yaw]


@contextmanager
def step_state_reads(dynamics):
    """Read a fresh pose and physics state once after the action has completed."""
    if not dynamics.synchronous_steps and getattr(dynamics, "cache_step_state", False) is not True:
        yield
        return
    pose = dynamics.client.simGetVehiclePose(dynamics.vehicle_name)
    state = dynamics.client.getMultirotorState(vehicle_name=dynamics.vehicle_name)
    dynamics._paused_readings = (pose, state)
    try:
        yield
    finally:
        dynamics._paused_readings = None


paused_state_reads = step_state_reads


class MultirotorDynamicsAirsim:
    """Multirotor dynamics controller for Cosys-AirSim."""

    def __init__(self, cfg, vehicle_name="", client=None):
        # Config
        self.navigation_3d = cfg.getboolean('options', 'navigation_3d')
        self.using_velocity_state = cfg.getboolean('options', 'using_velocity_state')
        self.distance_norm_scale = cfg.getfloat('environment', 'distance_norm_scale', fallback=0.0)
        self.max_vertical_difference = cfg.getfloat('environment', 'max_vertical_difference', fallback=5.0)
        if not math.isfinite(self.max_vertical_difference) or self.max_vertical_difference <= 0:
            raise ValueError("environment.max_vertical_difference must be finite and positive")
        self.align_start_yaw_to_goal = cfg.getboolean('environment', 'align_start_yaw_to_goal', fallback=False)
        self.dt = cfg.getfloat('multirotor', 'dt')
        self.vehicle_name = vehicle_name or cfg.get('airsim', 'vehicle_name', fallback='')
        self.airsim_ip = cfg.get('airsim', 'ip', fallback='')
        self.airsim_port = cfg.getint('airsim', 'rpc_port', fallback=41451)
        self.airsim_timeout = cfg.getfloat('airsim', 'timeout_value', fallback=3600.0)
        self.synchronous_steps = cfg.getboolean('airsim', 'synchronous_steps', fallback=False)
        self.cache_step_state = cfg.getboolean('airsim', 'cache_step_state', fallback=False)

        # AirSim client (shared if provided, else create new)
        if client is not None:
            self.client = client
            self._owns_client = False
        else:
            self.client = airsim.MultirotorClient(
                ip=self.airsim_ip,
                port=self.airsim_port,
                timeout_value=self.airsim_timeout,
            )
            self.client.confirmConnection()
            self._owns_client = True

        if self.synchronous_steps:
            settings = json.loads(self.client.getSettingsString())
            if settings.get('ClockType') != 'SteppableClock':
                raise ValueError('synchronous_steps requires AirSim ClockType=SteppableClock')

        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

        # Position config
        self.start_position = [0, 0, 0]          # NED (z down) for AirSim pose
        self.start_position_enu = [0, 0, 0]      # ENU (z up) for math
        self.start_random_angle = 0
        self.start_fixed_yaw = None
        self.goal_position = [0, 0, 0]           # ENU (z up) for math
        self.goal_distance = 0

        # Action space config
        self.v_xy_max = cfg.getfloat('multirotor', 'v_xy_max')
        self.v_xy_min = cfg.getfloat('multirotor', 'v_xy_min')
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')
        self.yaw_rate_max_deg = cfg.getfloat('multirotor', 'yaw_rate_max_deg')
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)

        # State feature length and action space
        if self.navigation_3d:
            self.state_feature_length = 6 if self.using_velocity_state else 3
            self.action_space = spaces.Box(
                low=np.array([self.v_xy_min, -self.v_z_max, -self.yaw_rate_max_rad], dtype=np.float32),
                high=np.array([self.v_xy_max, self.v_z_max, self.yaw_rate_max_rad], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.state_feature_length = 4 if self.using_velocity_state else 2
            self.action_space = spaces.Box(
                low=np.array([self.v_xy_min, -self.yaw_rate_max_rad], dtype=np.float32),
                high=np.array([self.v_xy_max, self.yaw_rate_max_rad], dtype=np.float32),
                dtype=np.float32,
            )

        self.state_norm = np.zeros(self.state_feature_length)

    def reset(self, rng=None):
        """Reset drone to start position and take off."""
        if self.synchronous_steps:
            self.client.simPause(True)
        try:
            self._reset_vehicle(rng)
        finally:
            if self.synchronous_steps:
                self.client.simPause(True)

    def _reset_vehicle(self, rng):
        # Only reset all if we own the client (single vehicle mode)
        if self._owns_client:
            self.client.reset()

        self.client.enableApiControl(True, self.vehicle_name)
        if not self.client.armDisarm(True, self.vehicle_name):
            raise RuntimeError("Failed to arm drone during episode reset")

        # Set pose for this vehicle
        random_source = rng if rng is not None else np.random
        if self.start_fixed_yaw is not None:
            yaw = self.start_fixed_yaw
        else:
            base_yaw = (
                math.atan2(
                    self.goal_position[1] - self.start_position_enu[1],
                    self.goal_position[0] - self.start_position_enu[0],
                )
                if self.align_start_yaw_to_goal
                else 0.0
            )
            yaw = base_yaw + self.start_random_angle * (2.0 * random_source.random() - 1.0)
        if self.synchronous_steps:
            state = airsim.KinematicsState()
            state.position = airsim.Vector3r(*self.start_position)
            state.orientation = to_quaternion(0, 0, yaw)
            state.linear_velocity = state.angular_velocity = airsim.Vector3r()
            state.linear_acceleration = state.angular_acceleration = airsim.Vector3r()
            self.client.simSetKinematics(state, True, self.vehicle_name)
            self._advance_synchronously(0, 0, 0, airsim.YawMode(True, 0), duration=1.0)
            self._validate_reset_state(yaw)
            return

        pose = self.client.simGetVehiclePose(self.vehicle_name)
        pose.position.x_val = self.start_position[0]
        pose.position.y_val = self.start_position[1]
        pose.position.z_val = self.start_position[2]
        pose.orientation = to_quaternion(0, 0, yaw)
        self.client.simSetVehiclePose(pose, True, self.vehicle_name)
        # Spawn is already airborne; takeoff rejects a falling vehicle. Settle before observing.
        settled = self.client.moveByVelocityAsync(
            0, 0, 0, 1.0, vehicle_name=self.vehicle_name,
        ).get()
        if not settled:
            raise RuntimeError("Failed to settle drone during episode reset")

    def _validate_reset_state(self, requested_yaw):
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        kinematics = state.kinematics_estimated
        position = list(kinematics.position)
        velocity = list(kinematics.linear_velocity)
        render_position = list(self.client.simGetVehiclePose(self.vehicle_name).position)
        collision = self.client.simGetCollisionInfo(self.vehicle_name)
        roll, pitch, yaw = quaternion_to_euler(kinematics.orientation)
        yaw_error = (yaw - requested_yaw + math.pi) % (2 * math.pi) - math.pi
        error = math.dist(position, self.start_position)
        speed = float(np.linalg.norm(velocity))
        self.last_reset_state = {
            'requested_position_ned_m': list(self.start_position),
            'position_ned_m': position, 'velocity_ned_mps': velocity,
            'position_error_m': error, 'speed_mps': speed,
            'render_position_ned_m': render_position,
            'render_physics_difference_m': math.dist(position, render_position),
            'attitude_rad': [roll, pitch, yaw], 'yaw_error_rad': yaw_error,
            'collision': bool(state.collision.has_collided or collision.has_collided),
        }
        if (not all(math.isfinite(v) for v in position + velocity + render_position + [roll, pitch, yaw])
                or error > 0.2 or speed > 0.15 or max(abs(roll), abs(pitch), abs(yaw_error)) > 0.05
                or math.dist(position, render_position) > 0.05 or self.last_reset_state['collision']):
            raise RuntimeError('Invalid airborne reset state: ' + json.dumps(self.last_reset_state))

    def set_action(self, action):
        """Execute velocity command."""
        v_forward = float(action[0])
        # Reduce yaw rate multiplier from 2 to 1 to prevent excessive spinning
        yaw_rate = float(action[-1])  # Removed * 2
        v_z = float(action[1]) if self.navigation_3d and len(action) >= 3 else 0.0

        # Convert to world frame
        yaw = self.get_attitude()[2]
        vx = v_forward * math.cos(yaw)
        vy = v_forward * math.sin(yaw)
        vz = -v_z  # NED convention

        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate))
        if self.synchronous_steps:
            self._advance_synchronously(vx, vy, vz, yaw_mode)
        else:
            completed = self.client.moveByVelocityAsync(
                vx, vy, vz, self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=yaw_mode,
                vehicle_name=self.vehicle_name,
            ).get()
            if not completed:
                raise RuntimeError("Failed to execute drone velocity command")

    def _advance_synchronously(self, vx, vy, vz, yaw_mode, duration=None):
        # Keep the command alive across each physics window; paused inference must not trigger SimpleFlight's 60ms timeout.
        before = self.client.getMultirotorState(vehicle_name=self.vehicle_name).timestamp
        control_duration = self.dt if duration is None else duration
        future = self.client.moveByVelocityAsync(
            vx, vy, vz, control_duration + 1.0,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=yaw_mode, vehicle_name=self.vehicle_name,
        )
        advanced = 0.0
        minimum_window = 0.002
        scene_step = getattr(self, 'scene_step_callback', None)
        last_progress = time.monotonic()
        try:
            while advanced < control_duration - 1e-6:
                if time.monotonic() - last_progress > 5.0:
                    raise TimeoutError('Synchronous AirSim physics did not advance')
                previous_advance = advanced
                window = control_duration - advanced
                if scene_step is not None:
                    window = min(window, 0.05)
                self.client.simContinueForTime(max(minimum_window, window))
                timestamp = self.client.getMultirotorState(vehicle_name=self.vehicle_name).timestamp
                advanced = (timestamp - before) / 1e9
                if scene_step is not None:
                    scene_step(timestamp)
                if advanced > previous_advance:
                    last_progress = time.monotonic()
                    minimum_window = 0.002
                else:
                    # A short trailing wall-time window can end before one
                    # SteppableClock physics tick is scheduled.
                    minimum_window = 0.010
        finally:
            self.client.simPause(True)
            self.client.cancelLastTask(self.vehicle_name)
            # False is expected after cancellation; get() still propagates RPC errors.
            future.get()
        self.last_control_sim_seconds = advanced

    def set_start(self, position, random_angle, fixed_yaw=None):
        """Set start position and random angle range."""
        self.start_position = list(position)
        # AirSim uses NED; convert to ENU for distance math (z up)
        self.start_position_enu = [self.start_position[0], self.start_position[1], -self.start_position[2]]
        self.start_random_angle = random_angle
        self.start_fixed_yaw = fixed_yaw

    def _set_goal_pose_single(self, goal_position):
        """Set fixed goal position."""
        goal_ned = [float(p) for p in goal_position]
        # Convert goal z to ENU (z up) for all geometry calculations
        self.goal_position = [goal_ned[0], goal_ned[1], -goal_ned[2]]

        dx = self.goal_position[0] - self.start_position_enu[0]
        dy = self.goal_position[1] - self.start_position_enu[1]
        dz = self.goal_position[2] - self.start_position_enu[2]
        if self.navigation_3d:
            self.goal_distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        else:
            self.goal_distance = math.sqrt(dx * dx + dy * dy)

    def get_position(self):
        """Get current position [x, y, z] in ENU frame."""
        p = self._vehicle_pose().position
        return [p.x_val, p.y_val, -p.z_val]

    def get_velocity(self):
        """Get current body-frame velocity [v_forward, v_up, yaw_rate]."""
        readings = getattr(self, "_paused_readings", None)
        state = readings[1] if readings is not None else self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        lv = state.kinematics_estimated.linear_velocity
        av = state.kinematics_estimated.angular_velocity
        self.last_state_timestamp = int(state.timestamp)
        self.last_velocity_world = [float(lv.x_val), float(lv.y_val), -float(lv.z_val)]

        # World/NED velocities
        vx, vy, vz = lv.x_val, lv.y_val, lv.z_val
        yaw = self.get_attitude()[2]

        # Rotate to body frame (forward/right/up), keep up positive
        v_forward = math.cos(yaw) * vx + math.sin(yaw) * vy
        v_up = -vz
        yaw_rate = av.z_val
        return [v_forward, v_up, yaw_rate]

    def _vehicle_pose(self):
        readings = getattr(self, "_paused_readings", None)
        return readings[0] if readings is not None else self.client.simGetVehiclePose(self.vehicle_name)

    def get_attitude(self):
        """Get current attitude [roll, pitch, yaw] in radians."""
        q = self._vehicle_pose().orientation
        return quaternion_to_euler(q)

    def get_distance_to_goal_2d(self):
        """Get 2D distance to goal (XY plane only)."""
        pos = self.get_position()
        dx = pos[0] - self.goal_position[0]
        dy = pos[1] - self.goal_position[1]
        return math.sqrt(dx * dx + dy * dy)

    def get_distance_to_goal_3d(self):
        """Get 3D distance to goal (includes Z)."""
        pos = self.get_position()
        dx = pos[0] - self.goal_position[0]
        dy = pos[1] - self.goal_position[1]
        dz = pos[2] - self.goal_position[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _get_state_feature(self):
        """Get normalized state features in [0, 1] range (float32 precision)."""
        pos = self.get_position()
        self.state_altitude_above_start = pos[2] - self.start_position_enu[2]
        vel = self.get_velocity()

        distance = self.get_distance_to_goal_3d() if self.navigation_3d else self.get_distance_to_goal_2d()
        relative_z = pos[2] - self.goal_position[2]
        relative_yaw = self._get_relative_yaw()

        # Normalize directly to [0, 1] for better precision
        if self.distance_norm_scale > 0:
            dist_norm = distance / self.distance_norm_scale
        else:
            dist_norm = distance / max(self.goal_distance, 1e-6)
        z_norm = relative_z / self.max_vertical_difference / 2 + 0.5
        yaw_norm = relative_yaw / math.pi / 2 + 0.5
        v_forward_norm = (vel[0] - self.v_xy_min) / (self.v_xy_max - self.v_xy_min)
        v_up_norm = vel[1] / self.v_z_max / 2 + 0.5
        vyaw_norm = vel[2] / self.yaw_rate_max_rad / 2 + 0.5

        # Clip to [0, 1] and keep as float32
        state_norm = np.clip(
            [dist_norm, z_norm, yaw_norm, v_forward_norm, v_up_norm, vyaw_norm],
            0.0, 1.0
        ).astype(np.float32)

        if self.navigation_3d:
            self.state_norm = state_norm[:3] if not self.using_velocity_state else state_norm
        else:
            self.state_norm = np.array([state_norm[0], state_norm[2], state_norm[3], state_norm[5]], dtype=np.float32)
            if not self.using_velocity_state:
                self.state_norm = self.state_norm[:2]

        return self.state_norm

    def _get_relative_yaw(self):
        """Get yaw angle to goal (-pi to pi)."""
        pos = self.get_position()
        dx = self.goal_position[0] - pos[0]
        dy = self.goal_position[1] - pos[1]
        target_yaw = math.atan2(dy, dx)

        current_yaw = self.get_attitude()[2]
        error = target_yaw - current_yaw

        if error > math.pi:
            error -= 2 * math.pi
        elif error < -math.pi:
            error += 2 * math.pi
        return error
