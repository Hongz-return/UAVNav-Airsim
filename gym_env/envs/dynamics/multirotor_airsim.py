"""Multirotor dynamics using Cosys-AirSim."""

import cosysairsim as airsim
import numpy as np
import math
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


class MultirotorDynamicsAirsim:
    """Multirotor dynamics controller for Cosys-AirSim."""

    def __init__(self, cfg, vehicle_name="", client=None):
        # Config
        self.navigation_3d = cfg.getboolean('options', 'navigation_3d')
        self.using_velocity_state = cfg.getboolean('options', 'using_velocity_state')
        self.dt = cfg.getfloat('multirotor', 'dt')
        self.vehicle_name = vehicle_name or cfg.get('airsim', 'vehicle_name', fallback='')
        self.airsim_ip = cfg.get('airsim', 'ip', fallback='')
        self.airsim_port = cfg.getint('airsim', 'rpc_port', fallback=41451)
        self.airsim_timeout = cfg.getfloat('airsim', 'timeout_value', fallback=3600.0)

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

        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)

        # Position config
        self.start_position = [0, 0, 0]          # NED (z down) for AirSim pose
        self.start_position_enu = [0, 0, 0]      # ENU (z up) for math
        self.start_random_angle = 0
        self.start_fixed_yaw = None
        self.goal_position = [0, 0, 0]           # ENU (z up) for math
        self.goal_distance = 0

        # Command state (for logging)
        self.v_xy_sp = 0
        self.v_z_sp = 0
        self.yaw_sp = 0

        # Action space config
        self.v_xy_max = cfg.getfloat('multirotor', 'v_xy_max')
        self.v_xy_min = cfg.getfloat('multirotor', 'v_xy_min')
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')
        self.yaw_rate_max_deg = cfg.getfloat('multirotor', 'yaw_rate_max_deg')
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.max_vertical_difference = 5

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

        # State arrays
        self.state_raw = np.zeros(6)
        self.state_norm = np.zeros(self.state_feature_length)

    def reset(self, rng=None):
        """Reset drone to start position and take off."""
        # Only reset all if we own the client (single vehicle mode)
        if self._owns_client:
            self.client.reset()
            time.sleep(0.1)

        # Set pose for this vehicle
        random_source = rng if rng is not None else np.random
        yaw = (
            self.start_fixed_yaw
            if self.start_fixed_yaw is not None
            else self.start_random_angle * random_source.random()
        )
        pose = self.client.simGetVehiclePose(self.vehicle_name)
        pose.position.x_val = self.start_position[0]
        pose.position.y_val = self.start_position[1]
        pose.position.z_val = self.start_position[2]
        pose.orientation = to_quaternion(0, 0, yaw)
        self.client.simSetVehiclePose(pose, True, self.vehicle_name)

        time.sleep(0.05)
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
        self.client.takeoffAsync(vehicle_name=self.vehicle_name).join()

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

        self.client.moveByVelocityAsync(
            vx, vy, vz, self.dt,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate)),
            vehicle_name=self.vehicle_name,
        ).join()

        # Store for state computation
        self.v_xy_sp = v_forward
        self.v_z_sp = v_z
        self.yaw_sp = yaw + yaw_rate * self.dt

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
        p = self.client.simGetVehiclePose(self.vehicle_name).position
        return [p.x_val, p.y_val, -p.z_val]

    def get_velocity(self):
        """Get current body-frame velocity [v_forward, v_up, yaw_rate]."""
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        lv = state.kinematics_estimated.linear_velocity
        av = state.kinematics_estimated.angular_velocity

        # World/NED velocities
        vx, vy, vz = lv.x_val, lv.y_val, lv.z_val
        yaw = self.get_attitude()[2]

        # Rotate to body frame (forward/right/up), keep up positive
        v_forward = math.cos(yaw) * vx + math.sin(yaw) * vy
        v_up = -vz
        yaw_rate = av.z_val
        return [v_forward, v_up, yaw_rate]

    def get_attitude(self):
        """Get current attitude [roll, pitch, yaw] in radians."""
        q = self.client.simGetVehiclePose(self.vehicle_name).orientation
        return quaternion_to_euler(q)

    def get_attitude_cmd(self):
        """Get commanded attitude."""
        return [0.0, 0.0, self.yaw_sp]

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
        vel = self.get_velocity()

        distance = self.get_distance_to_goal_3d() if self.navigation_3d else self.get_distance_to_goal_2d()
        relative_z = pos[2] - self.goal_position[2]
        relative_yaw = self._get_relative_yaw()

        # Normalize directly to [0, 1] for better precision
        dist_norm = distance / max(self.goal_distance, 1e-6)
        z_norm = relative_z / self.max_vertical_difference / 2 + 0.5
        yaw_norm = relative_yaw / math.pi / 2 + 0.5
        v_forward_norm = (vel[0] - self.v_xy_min) / (self.v_xy_max - self.v_xy_min)
        v_up_norm = vel[1] / self.v_z_max / 2 + 0.5
        vyaw_norm = vel[2] / self.yaw_rate_max_rad / 2 + 0.5

        self.state_raw = np.array([
            distance, relative_z, math.degrees(relative_yaw),
            vel[0], vel[1], math.degrees(vel[2])
        ])

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
