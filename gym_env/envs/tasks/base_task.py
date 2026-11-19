"""Base class for navigation tasks (reward + termination)."""

from abc import ABC
from configparser import ConfigParser
import math

import numpy as np


class BaseTask(ABC):
    """
    Base class for navigation tasks.

    Provides a shared reward function (progress + obstacle penalty + smoothness
    + time penalty + terminal rewards) that can be used by all subclasses.
    """

    def __init__(self, cfg: ConfigParser, env):
        """
        Initialize task with config and environment reference.

        Args:
            cfg: ConfigParser with task parameters
            env: Reference to AirsimGymEnv for accessing state
        """
        self.env = env
        self.reward_cfg = {}
        self.reward_type = "reward_final"
        # Cached termination flags (set by is_done(), used by get_info())
        self._cached_success = False
        self._cached_crash = False
        self._cached_outside = False
        # For action smoothness penalty (shared across all tasks)
        self._prev_action = None
        self._action_delta_scale = None
        self.last_reward_components = {}
        self._load_config(cfg)

    def _load_config(self, cfg: ConfigParser):
        """Load task parameters from config file."""
        # Common parameters
        self.accept_radius = cfg.getfloat("environment", "accept_radius")
        self.crash_distance = cfg.getfloat("environment", "crash_distance")

        # Workspace boundaries
        import ast

        self.work_space_x = ast.literal_eval(cfg.get("environment", "work_space_x"))
        self.work_space_y = ast.literal_eval(cfg.get("environment", "work_space_y"))
        self.work_space_z = ast.literal_eval(cfg.get("environment", "work_space_z"))

        # Load reward config
        self._load_reward_config(cfg)

    def _load_reward_config(self, cfg: ConfigParser):
        """Load reward parameters. Override in subclass for task-specific params."""
        self.reward_type = cfg.get("options", "reward_type", fallback="reward_final")
        self.reward_cfg = {
            "reach": cfg.getfloat("reward", "reward_reach"),
            "crash": cfg.getfloat("reward", "reward_crash"),
            "outside": cfg.getfloat("reward", "reward_outside"),
            "timeout": cfg.getfloat("reward", "reward_timeout", fallback=0.0),
            "obs_safe_margin": cfg.getfloat("reward", "obstacle_safe_margin"),
            "obs_collision_margin": cfg.getfloat("reward", "obstacle_collision_margin", fallback=0.3),
        }

        self.d_safe = self.reward_cfg["obs_safe_margin"]
        self.d_collision = self.reward_cfg["obs_collision_margin"]

        if self.reward_type == "reward_v2":
            self.progress_scale = cfg.getfloat("reward", "progress_scale", fallback=20.0)
            self.obstacle_penalty_weight = cfg.getfloat("reward", "obstacle_penalty_weight", fallback=2.0)
            self.time_penalty = cfg.getfloat("reward", "time_penalty", fallback=0.015)
            self.smoothness_weight = cfg.getfloat("reward", "action_smoothness_weight", fallback=0.2)
            self.dense_reward_clip = cfg.getfloat("reward", "dense_reward_clip", fallback=2.5)
        else:
            self.progress_scale = cfg.getfloat("reward", "progress_scale", fallback=2.5)
            self.obstacle_penalty_weight = cfg.getfloat("reward", "obstacle_penalty_weight", fallback=1.0)
            self.time_penalty = cfg.getfloat("reward", "time_penalty", fallback=0.1)
            self.smoothness_weight = cfg.getfloat("reward", "action_smoothness_weight", fallback=0.05)
            self.dense_reward_clip = cfg.getfloat("reward", "dense_reward_clip", fallback=2.0)

        action_range = np.asarray(self.env.action_space.high - self.env.action_space.low, dtype=np.float32)
        self._action_delta_scale = np.maximum(action_range, 1e-6)

        self.r_goal = self.reward_cfg["reach"]
        self.r_fail_crash = self.reward_cfg["crash"]
        self.r_fail_outside = self.reward_cfg["outside"]
        self.r_fail_timeout = self.reward_cfg["timeout"]

    def compute_reward(self, done: bool, action: np.ndarray) -> float:
        """
        Default reward: progress, obstacle penalty, time penalty, smoothness.

        Exact dense shaping depends on ``reward_type`` from ``[options]``.
        Terminal rewards are read from ``[reward]``.
        """
        reward = 0.0
        components = dict(progress=0.0, obstacle=0.0, time=0.0, smoothness=0.0,
                          clipping=0.0, terminal=0.0, timeout=0.0)

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            distance_delta = self.env.previous_distance_from_des_point - distance_now
            self.env.previous_distance_from_des_point = distance_now
            r_progress = self._compute_progress_reward(distance_delta)

            obstacle_penalty = self._compute_obstacle_penalty(self.env.min_distance_to_obstacles)
            smooth_penalty = self._compute_smoothness_penalty(action)

            components.update(
                progress=float(r_progress),
                obstacle=float(-self.obstacle_penalty_weight * obstacle_penalty),
                time=float(-self.time_penalty),
                smoothness=float(-self.smoothness_weight * smooth_penalty),
            )
            dense_reward = sum(components.values())
            reward = float(np.clip(dense_reward, -self.dense_reward_clip, self.dense_reward_clip))
            components['clipping'] = reward - dense_reward

        else:
            if self._cached_success:
                reward = self.r_goal
            elif self._cached_crash:
                reward = self.r_fail_crash
            elif self._cached_outside:
                reward = self.r_fail_outside
            components['terminal'] = float(reward)

        self.last_reward_components = components
        return reward

    def _compute_progress_reward(self, distance_delta: float) -> float:
        if self.reward_type == "reward_v2":
            distance_ref = max(self.env.dynamic_model.goal_distance, self.accept_radius, 1e-6)
            return self.progress_scale * (distance_delta / distance_ref)
        return self.progress_scale * distance_delta

    def _compute_obstacle_penalty(self, obstacle_distance: float) -> float:
        if obstacle_distance >= self.d_safe:
            return 0.0
        if obstacle_distance <= self.d_collision:
            return 1.0

        ratio = (self.d_safe - obstacle_distance) / max(self.d_safe - self.d_collision, 1e-6)
        if self.reward_type == "reward_v2":
            return ratio * ratio
        return ratio

    def _compute_smoothness_penalty(self, action: np.ndarray) -> float:
        action = np.asarray(action, dtype=np.float32)
        if self._prev_action is None:
            smooth_penalty = 0.0
        elif self.reward_type == "reward_v2":
            normalized_delta = (action - self._prev_action) / self._action_delta_scale
            smooth_penalty = float(np.linalg.norm(normalized_delta) / math.sqrt(action.size))
        else:
            smooth_penalty = float(np.linalg.norm(action - self._prev_action))
        self._prev_action = action.copy()
        return smooth_penalty

    def compute_timeout_reward(self, action: np.ndarray) -> float:
        """Penalty applied when an episode ends by time limit truncation."""
        del action
        return self.r_fail_timeout

    def is_done(self) -> bool:
        """
        Check if episode should terminate (not truncate).

        Timeout (max_episode_steps) is handled by airsim_env.py as truncation,
        not termination. This allows the critic to bootstrap on timeout.
        """
        self._cached_crash = self.is_crashed()
        self._cached_outside = self.is_not_inside_workspace()
        # Success must be mutually exclusive with failure states to avoid
        # contradictory reward/metric labels on the same transition.
        self._cached_success = (
            not self._cached_crash and not self._cached_outside and self.is_in_desired_pose()
        )
        return self._cached_outside or self._cached_success or self._cached_crash

    def is_not_inside_workspace(self) -> bool:
        """Check if drone is outside workspace boundaries."""
        pos = self.env.dynamic_model.get_position()
        return (
            pos[0] < self.work_space_x[0]
            or pos[0] > self.work_space_x[1]
            or pos[1] < self.work_space_y[0]
            or pos[1] > self.work_space_y[1]
            or pos[2] < self.work_space_z[0]
            or pos[2] > self.work_space_z[1]
        )

    def is_in_desired_pose(self) -> bool:
        """Check if drone reached goal."""
        return self.get_distance_to_goal_3d() < self.accept_radius

    def is_crashed(self) -> bool:
        """Check if drone crashed (collision detection or configured near-collision)."""
        collision_info = self.env.client.simGetCollisionInfo()
        if collision_info.has_collided:
            return True
        if self.crash_distance > 0:
            return self.env.min_distance_to_obstacles <= self.crash_distance
        return False

    def get_distance_to_goal_3d(self) -> float:
        """Calculate 3D distance to goal in ENU frame."""
        pos = self.env.dynamic_model.get_position()
        goal = self.env.dynamic_model.goal_position
        return math.sqrt(
            (pos[0] - goal[0]) ** 2
            + (pos[1] - goal[1]) ** 2
            + (pos[2] - goal[2]) ** 2
        )

    def get_info(self) -> dict:
        """Return info dict for step(). Uses cached flags from is_done()."""
        return {
            "is_success": self._cached_success,
            "is_crash": self._cached_crash,
            "is_not_in_workspace": self._cached_outside,
            "step_num": self.env.step_num,
        }

    def reset(self):
        """Called on episode reset. Override for task-specific reset logic."""
        self._prev_action = None
