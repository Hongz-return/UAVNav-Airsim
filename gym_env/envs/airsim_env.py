"""
AirSim Gym environment for UAV navigation.
"""

import ast
from contextlib import suppress
import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .dynamics.multirotor_airsim import MultirotorDynamicsAirsim, step_state_reads
from .perception import get_perception
from .tasks import get_task


_LOGGER = logging.getLogger(__name__)


class AirsimGymEnv(gym.Env):
    """Gymnasium environment for AirSim UAV navigation."""

    def __init__(self) -> None:
        super().__init__()
        np.set_printoptions(formatter={"float": "{: 4.2f}".format}, suppress=True)
        print("init airsim-gym-env")

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 1, 1), dtype=np.uint8)

        self.task = None
        self.perception = None
        self.training_routes = []
        self.stage3_course = None
        self.navrl_forest = False
        self._navrl_goal_path = None
        self.print_airsim_log = False
        # Evaluation-only switch: keep episodes alive after collision events
        # and let the evaluator aggregate collision statistics separately.
        self.continue_after_collision = False
        self._evaluation_scenarios = None
        self._evaluation_scenario_index = 0
        self.current_scenario = None
        self.record_evaluation_timing = False

    def set_config(self, cfg):
        """Configure environment from .ini config file."""
        self.env_name = cfg.get("options", "env_name")
        self.dynamic_name = cfg.get("options", "dynamic_name")
        self.perception_type = cfg.get("options", "perception")
        self.navrl_forest = cfg.has_section("navrl_forest")
        self.print_airsim_log = cfg.getboolean("options", "print_airsim_log", fallback=False)
        reward_type = cfg.get("options", "reward_type", fallback="reward_final")
        use_pid_residual = cfg.getboolean("options", "use_pid_residual", fallback=False)

        print(f"Environment: {self.env_name}, Dynamics: {self.dynamic_name}, Perception: {self.perception_type}")
        print(f"  Reward: {reward_type}")

        if use_pid_residual:
            raise ValueError(
                "Legacy PID control configs are no longer supported. "
                "Use the current NoPID configs and retrain/evaluate with compatible checkpoints."
            )

        if self.dynamic_name != "Multirotor":
            raise ValueError(f"Only 'Multirotor' dynamics supported, got: {self.dynamic_name}")

        self.dynamic_model = MultirotorDynamicsAirsim(cfg)
        self.client = self.dynamic_model.client
        self.state_feature_length = self.dynamic_model.state_feature_length
        self.action_space = self.dynamic_model.action_space

        self._navrl_goal_path = None
        goal_dir = cfg.get("navrl_forest", "goal_dir", fallback=None)
        if self.navrl_forest and goal_dir is not None:
            self._navrl_goal_stage = cfg.getint("navrl_forest", "stage")
            if not goal_dir.strip() or self._navrl_goal_stage < 1:
                raise ValueError("NavRL live goals require a non-empty goal_dir and a positive stage")
            self._navrl_goal_path = Path(goal_dir) / f"live_goal_{self.dynamic_model.airsim_port}.json"
            self._navrl_goal_write_warned = False

        self._load_environment_config(cfg)

        perception_cls = get_perception(self.perception_type)
        self.perception = perception_cls(cfg, self.client)
        if self.perception_type in ["vector", "depth_state", "depth_motion"]:
            self.perception.set_state_feature_length(self.state_feature_length)
        self.observation_space = self.perception.get_observation_space()
        obs_desc = getattr(self.observation_space, "shape", self.observation_space)
        print(f"  Perception: {self.perception_type} -> obs_space: {obs_desc}")

        task_cls = get_task(self.env_name)
        self.task = task_cls(cfg, self)
        print(f"  Task: {self.env_name} -> {task_cls.__name__}")

        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self._episode_reward_components = {}
        self.previous_distance_from_des_point = 0
        self.min_distance_to_obstacles = float("inf")

        self.max_episode_steps = cfg.getint("environment", "max_episode_steps")
        self.action_repeat = cfg.getint("multirotor", "action_repeat", fallback=1)

        print(f"  Action repeat: {self.action_repeat} (each action lasts {self.action_repeat * self.dynamic_model.dt:.2f}s)")

    def _load_environment_config(self, cfg):
        """Load start/goal positions and workspace from config."""
        parse = ast.literal_eval

        start_position = parse(cfg.get("environment", "start_position"))
        goal_position = parse(cfg.get("environment", "goal_position"))
        self.start_random_angle = cfg.getfloat("environment", "start_random_angle")
        self.random_route_reverse = cfg.getboolean("environment", "random_route_reverse", fallback=False)
        routes = cfg.get("environment", "training_routes", fallback=None)
        self.training_routes = self._parse_training_routes(routes) if routes and routes.strip() else []

        self.dynamic_model.set_start(start_position, random_angle=self.start_random_angle)
        self.dynamic_model._set_goal_pose_single(goal_position)

        self.work_space_x = parse(cfg.get("environment", "work_space_x"))
        self.work_space_y = parse(cfg.get("environment", "work_space_y"))
        self.work_space_z = parse(cfg.get("environment", "work_space_z"))

        print(f"  Start: {start_position}, Goal: {goal_position}")
        print(f"  Workspace X: {self.work_space_x}, Y: {self.work_space_y}, Z: {self.work_space_z}")

    @staticmethod
    def _parse_training_routes(routes_config):
        try:
            routes = ast.literal_eval(routes_config)
        except (ValueError, SyntaxError) as error:
            raise ValueError("environment.training_routes must be a list of routes") from error
        if not isinstance(routes, list) or not routes:
            raise ValueError("environment.training_routes must be a non-empty list")

        parsed = []
        for route in routes:
            if isinstance(route, dict):
                route = (route.get("start"), route.get("goal"))
            if not isinstance(route, (list, tuple)) or len(route) != 2 or not all(
                isinstance(point, (list, tuple)) and len(point) == 3
                and all(isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool) for coordinate in point)
                for point in route
            ):
                raise ValueError("Each training route needs a start and a goal with three numeric coordinates each")
            start, goal = route
            parsed.append((start, goal))
        return parsed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._apply_next_evaluation_scenario()
        route_idx = None
        route_reversed = False
        live_goal_ned = None
        if self.stage3_course is not None and self._evaluation_scenarios is None:
            self.stage3_course.select_route(self)
        elif self.navrl_forest and self._evaluation_scenarios is None:
            from tools.navrl_forest_runtime import sample_training_route

            start, goal, yaw = sample_training_route(self.np_random)
            self.dynamic_model.set_start(start, random_angle=0.0, fixed_yaw=yaw)
            self.dynamic_model._set_goal_pose_single(goal)
            live_goal_ned = goal
        elif self._evaluation_scenarios is None and self.training_routes:
            route_idx = int(self.np_random.integers(len(self.training_routes)))
            start, goal = self.training_routes[route_idx]
            if self.random_route_reverse:
                route_reversed = bool(self.np_random.integers(2))
                if route_reversed:
                    start, goal = goal, start
            self.dynamic_model.set_start(start, random_angle=self.start_random_angle, fixed_yaw=None)
            self.dynamic_model._set_goal_pose_single(goal)
        if self._navrl_goal_path is not None and live_goal_ned is not None:
            try:
                self._navrl_goal_path.unlink(missing_ok=True)
            except OSError as error:
                self._warn_navrl_goal_write(error)
        self.dynamic_model.reset(rng=self.np_random)
        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self._episode_reward_components = {}
        self.dynamic_model.goal_distance = self.dynamic_model.get_distance_to_goal_3d()
        self.previous_distance_from_des_point = self.dynamic_model.goal_distance

        if self.task:
            self.task.reset()

        disable_augmentation = self.navrl_forest or self._evaluation_scenarios is not None or (
            self.stage3_course is not None and self.stage3_course.disable_virtual_ceiling
        )
        self.perception.reset(rng=self.np_random, disable_augmentation=disable_augmentation)
        obs = self._get_obs()
        if self.record_evaluation_timing:
            self.dynamic_model.get_velocity()
            self.episode_start_state_timestamp = self.dynamic_model.last_state_timestamp
            self.episode_start_wall_time = time.perf_counter()
        if self._navrl_goal_path is not None and live_goal_ned is not None:
            self._publish_navrl_goal(live_goal_ned)
        return obs, {
            "scenario_id": (
                self.current_scenario.get("scenario_id")
                if self.current_scenario is not None
                else None
            ),
            "route_index": route_idx,
            "route_reversed": route_reversed,
        }

    def _publish_navrl_goal(self, goal_ned):
        x, y, z = goal_ned
        payload = json.dumps({
            "stage": self._navrl_goal_stage,
            "rpc_port": self.dynamic_model.airsim_port,
            "episode": self.episode_num,
            "goal_position": [float(x), float(y), float(z)],
            "published_at": time.time(),
        }, allow_nan=False)
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._navrl_goal_path.parent,
                prefix=f".{self._navrl_goal_path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload + "\n")
            temporary.replace(self._navrl_goal_path)
        except OSError as error:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
            self._warn_navrl_goal_write(error)

    def _warn_navrl_goal_write(self, error):
        if not self._navrl_goal_write_warned:
            _LOGGER.warning("Could not update NavRL live goal %s: %s", self._navrl_goal_path, error)
            self._navrl_goal_write_warned = True

    def set_evaluation_scenarios(self, scenarios):
        """Install an ordered, reproducible evaluation scenario sequence."""
        if not scenarios:
            raise ValueError("Evaluation scenario list must not be empty")
        self._evaluation_scenarios = list(scenarios)
        self._evaluation_scenario_index = 0

    def _apply_next_evaluation_scenario(self):
        if self._evaluation_scenarios is None:
            self.current_scenario = None
            return
        if self._evaluation_scenario_index >= len(self._evaluation_scenarios):
            raise RuntimeError("Evaluation scenario manifest exhausted")

        scenario = self._evaluation_scenarios[self._evaluation_scenario_index]
        self._evaluation_scenario_index += 1
        start_position = scenario["start_position"]
        goal_position = scenario["goal_position"]
        yaw_rad = float(scenario["yaw_rad"])
        self.dynamic_model.set_start(start_position, random_angle=0.0, fixed_yaw=yaw_rad)
        self.dynamic_model._set_goal_pose_single(goal_position)
        self.current_scenario = scenario

    def _execute_action(self, action):
        """Execute the policy action directly."""
        if self.stage3_course is not None and self._evaluation_scenarios is None:
            action = self.stage3_course.training_action(action)
        self.dynamic_model.set_action(action)
        return action

    def step(self, action):
        """Execute action, optionally with action_repeat for smoother control."""
        self.step_num += 1
        reward_components = {}
        reward = 0.0
        repeats = max(1, self.action_repeat)
        for substep in range(repeats):
            actual_action = self._execute_action(action)
            with step_state_reads(self.dynamic_model):
                obs = self._get_obs()
                done = self.task.is_done()
                step_reward = self.task.compute_reward(done, actual_action)
                for name, value in self.task.last_reward_components.items():
                    reward_components[name] = reward_components.get(name, 0.0) + value
                reward += step_reward
                self.cumulated_episode_reward += step_reward
                self.total_step += 1
                info = self.task.get_info()
                if done or substep + 1 == repeats:
                    info["actual_action"] = actual_action
                    if (self.stage3_course is not None and self._evaluation_scenarios is None
                            and self.stage3_course.used_guidance):
                        info["stage3_guided_action"] = actual_action
                    info["position"] = list(self.dynamic_model.get_position())
                    info["velocity"] = list(self.dynamic_model.get_velocity())
                    info["velocity_world"] = list(self.dynamic_model.last_velocity_world)
                    info["state_timestamp"] = self.dynamic_model.last_state_timestamp
                    info["min_distance_to_obstacles"] = float(self.min_distance_to_obstacles)
                    break

        terminated = done
        if self.continue_after_collision and info.get("is_crash", False):
            terminated = False
            info["collision_event"] = True
            info["collision_ignored_termination"] = True
        else:
            info["collision_event"] = bool(info.get("is_crash", False))
        truncated = (not terminated) and (self.step_num >= self.max_episode_steps)
        if terminated or truncated:
            info["terminal_position"] = list(info["position"])
        if truncated:
            timeout_reward = float(self.task.compute_timeout_reward(actual_action))
            if timeout_reward != 0.0:
                reward += timeout_reward
                self.cumulated_episode_reward += timeout_reward
                info["timeout_reward"] = timeout_reward
                reward_components['timeout'] = reward_components.get('timeout', 0.0) + timeout_reward
            info["TimeLimit.truncated"] = True

        info['reward_components'] = reward_components
        for name, value in reward_components.items():
            self._episode_reward_components[name] = self._episode_reward_components.get(name, 0.0) + value
        if terminated or truncated:
            info['episode_reward_components'] = dict(self._episode_reward_components)
            if self.stage3_course is not None and self._evaluation_scenarios is None:
                info["stage3_episode"] = self.stage3_course.terminal_metrics(info, truncated)

        if self.print_airsim_log and self.step_num % 10 == 0:
            self.client.simPrintLogMessage("Train: ", f"EP:{self.episode_num} Step:{self.step_num}")
            self.client.simPrintLogMessage("Reward: ", f"{reward:.3f} total:{self.cumulated_episode_reward:.3f}")

        if self.record_evaluation_timing:
            info['wall_monotonic_time'] = time.perf_counter()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Get observation from perception module."""
        state_feature = self.dynamic_model._get_state_feature()
        dz_above_start = (
            self.dynamic_model.state_altitude_above_start if self._evaluation_scenarios is None else None
        )
        obs = self.perception.get_observation(state_feature, dz_above_start=dz_above_start)
        self.min_distance_to_obstacles = self.perception.min_distance_to_obstacles

        if isinstance(self.observation_space, spaces.Dict):
            obs["image"] = np.clip(
                obs["image"],
                self.observation_space["image"].low,
                self.observation_space["image"].high,
            ).astype(self.observation_space["image"].dtype, copy=False)

            obs["state"] = np.clip(
                obs["state"],
                self.observation_space["state"].low,
                self.observation_space["state"].high,
            ).astype(self.observation_space["state"].dtype, copy=False)

            return obs

        dtype = self.observation_space.dtype
        if np.issubdtype(dtype, np.integer):
            obs = np.clip(obs, 0, 255)
        else:
            obs = np.clip(obs, 0.0, 1.0)
        return obs.astype(dtype, copy=False)
