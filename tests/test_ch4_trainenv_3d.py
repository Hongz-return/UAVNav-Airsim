"""Regression checks for the independent 3D course and frozen evaluation semantics."""

from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cosysairsim as airsim
from gymnasium import spaces
import numpy as np

from gym_env.envs.airsim_env import AirsimGymEnv
from gym_env.envs.tasks.base_task import BaseTask
from scripts.evaluation.evaluate_trainenv_3d import completed_episodes
from scripts.training.train_trainenv import main as train_main
from tools.trainenv_instances import (
    DEFAULT_3D_EXE, DEFAULT_EDITOR, DEFAULT_EXE, DEPTH_SIZE, TRAINENV_3D_MAP, TRAINENV_MAP,
    _probe_depth_ready, trainenv_command,
)
from tools.ue_benchmark.ch4_trainenv_3d_course import COURSE_FILE, build_protocol, obstacles, read_json, to_ned
from tools.ue_benchmark.ch4_trainenv_3d_runtime import (
    SETTINGS_TEMPLATE, TRAIN_CONFIG, DepthAudit, assert_reset_depth_consistent,
    progress_through_obstacles, read_ini,
)


class TrainEnv3DTests(unittest.TestCase):
    def setUp(self):
        self.course = read_json(COURSE_FILE)

    def test_explicit_horizontal_and_vertical_success_leave_original_sphere_intact(self):
        model = SimpleNamespace(
            goal_position=[10.0, 0.0, 1.0], get_position=Mock(), vehicle_name="",
        )
        env = SimpleNamespace(
            dynamic_model=model, action_space=spaces.Box(-1, 1, shape=(3,)),
            client=Mock(), min_distance_to_obstacles=25.0, step_num=1,
        )
        env.client.simGetCollisionInfo.return_value.has_collided = False
        old_cfg = read_ini(TRAIN_CONFIG.with_name("ch4_train_trainenv.ini"))
        old_cfg["environment"]["accept_radius"] = "2"
        old_cfg["environment"]["work_space_z"] = "[-0.7, 4.7]"
        old_task, new_task = BaseTask(old_cfg, env), BaseTask(read_ini(TRAIN_CONFIG), env)
        for position, old, new in (
            ([10, 0, 1.7], True, False),
            ([11.99, 0, 1.4], False, True),
            ([12, 0, 1.0], False, False),
            ([10, 0, 0.5], True, False),
        ):
            with self.subTest(position=position):
                model.get_position.return_value = position
                self.assertEqual(old_task.is_in_desired_pose(), old)
                self.assertEqual(new_task.is_in_desired_pose(), new)
        model.get_position.return_value = model.goal_position
        env.client.simGetCollisionInfo.return_value.has_collided = True
        self.assertTrue(new_task.is_done())
        self.assertFalse(new_task.get_info()["is_success"])

    def test_fixed_40_and_balanced_training_starts(self):
        protocol = build_protocol(self.course)
        expected = dict.fromkeys(self.course["families"], 10)
        scenarios = protocol["scenarios"]
        self.assertEqual(Counter(row["family"] for row in scenarios), expected)
        self.assertEqual(len({tuple(row["start_position"]) for row in scenarios}), 40)
        self.assertEqual(len({tuple(row["goal_position"]) for row in scenarios}), 40)
        self.assertEqual(protocol, build_protocol(self.course))
        for row in scenarios:
            self.assertEqual(row["start_position"], to_ned(row["start_xyz_agl_m"], self.course))
            self.assertEqual(row["goal_position"], to_ned(row["goal_xyz_agl_m"], self.course))
            self.assertLessEqual(abs(row["yaw_offset_deg"]), 10)
            if row["family"] == "side":
                self.assertEqual(row["start_position"][2], row["goal_position"][2])
        cfg = read_ini(TRAIN_CONFIG)
        routes = AirsimGymEnv._parse_training_routes(cfg["environment"]["training_routes"])
        route_counts = Counter()
        for start, goal in routes:
            lane = min(self.course["families"].values(), key=lambda s: abs(start[1] - s["lane_y_m"]))
            route_counts[lane["lane_y_m"]] += 1
            self.assertLess(abs(goal[1] - lane["lane_y_m"]), 1.2)
            self.assertLess(start[0], goal[0])
        self.assertEqual(route_counts, dict.fromkeys([0, 12, 24, 36], 5))
        self.assertFalse(cfg.getboolean("environment", "random_route_reverse"))
        self.assertEqual(cfg.getfloat("reward", "altitude_deviation_weight"), 0)
        self.assertGreater(cfg.getfloat("reward", "altitude_boundary_weight"), 0)
        self.assertEqual(cfg.getfloat("perception", "virtual_ceiling_prob"), 0)

    def test_combo_checks_order_and_rejects_flying_high_through_second_gate(self):
        family = "over_under"
        points = self.course["families"][family]["reference_xyz_agl_m"]
        trajectory = [{"step": i, "position_xyz_agl_m": p} for i, p in enumerate(points)]
        expected = [box["name"] for box in obstacles(family, self.course)]
        records, order = progress_through_obstacles(trajectory, family, self.course)
        self.assertEqual(order, expected)
        self.assertTrue(all(record["passed"] for record in records))
        high = [{"step": i, "position_xyz_agl_m": [p[0], p[1], 4.3]} for i, p in enumerate(points)]
        records, order = progress_through_obstacles(high, family, self.course)
        self.assertEqual(order, expected[:1])
        self.assertFalse(records[1]["passed"])
        self.assertEqual(len(records[1]["center_plane_crossings"]), 1)

    def test_raw_depth_audit_rejects_stale_and_invalid_frames(self):
        audit = DepthAudit(self.course["evaluation"])
        frame = SimpleNamespace(
            height=180, width=320, time_stamp=55,
            image_data_float=np.linspace(1, 30, 180 * 320), camera_position=airsim.Vector3r(),
        )
        self.assertEqual(audit.consume(frame)["positive_finite_fraction"], 1.0)
        self.assertEqual(audit.latest(0)["timestamp_ns"], 55)
        with self.assertRaisesRegex(RuntimeError, "new raw"):
            audit.latest(55)
        with self.assertRaisesRegex(RuntimeError, "Bad raw depth"):
            audit.consume(frame)
        frame.time_stamp = 56
        frame.image_data_float[:100] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Bad raw depth"):
            audit.consume(frame)

    def test_training_entry_uses_the_selected_cooked_map_and_the_correct_ned_origin(self):
        for config, executable, map_path, spawn, extra in (
            (TRAIN_CONFIG.with_name("ch4_train_trainenv.ini"), DEFAULT_EXE, TRAINENV_MAP, [0, 0, 1], []),
            (TRAIN_CONFIG, DEFAULT_EDITOR, TRAINENV_3D_MAP, [0, 0, 0], []),
            (TRAIN_CONFIG, DEFAULT_3D_EXE, TRAINENV_3D_MAP, [0, 0, 0], ["--ue-exe", str(DEFAULT_3D_EXE)]),
        ):
            with self.subTest(config=config), TemporaryDirectory() as temp:
                with (
                    patch("scripts.training.train_trainenv.launch_trainenv", return_value=nullcontext()) as launch,
                    patch("scripts.training.train_trainenv.run_training") as train,
                ):
                    train_main(["-c", str(config), "--instances", "1", "--run-dir", str(Path(temp) / "run"), *extra])
                train.assert_called_once()
                selection = launch.call_args.kwargs
                self.assertEqual(selection["executable"], executable.resolve())
                self.assertEqual(selection["map_path"], map_path)
                self.assertEqual(selection["start_position"], spawn)
                self.assertEqual(trainenv_command(executable, "settings", "log", map_path)[1], map_path)
                if config == TRAIN_CONFIG:
                    self.assertEqual(selection["template"], SETTINGS_TEMPLATE)
                if executable == DEFAULT_EDITOR:
                    project = Path(read_ini(TRAIN_CONFIG)["trainenv"]["editor_project"])
                    self.assertEqual(selection["editor_project"], project.resolve())
                    self.assertEqual(trainenv_command(executable, "settings", "log", map_path, project)[2], map_path)
                else:
                    self.assertIsNone(selection["editor_project"])
        self.assertEqual(trainenv_command(DEFAULT_EXE, "settings", "log")[1], TRAINENV_MAP)

    def test_resume_retains_only_complete_episodes_from_the_locked_prefix(self):
        scenarios = build_protocol(self.course)["scenarios"][:10]
        with TemporaryDirectory() as temp:
            family = Path(temp) / "over"
            family.mkdir()
            trajectory = family / "over_01.jsonl"
            trajectory.write_text("{}\n{}\n", encoding="utf-8")
            recorded = {
                "scenario_id": "over_01",
                "steps": 1,
                "depth_frames_checked": 2,
                "trajectory": "over/over_01.jsonl",
            }
            summary = family / "episodes.jsonl"
            summary.write_text(json.dumps(recorded) + "\n", encoding="utf-8")
            self.assertEqual(completed_episodes(family, scenarios), [recorded])
            trajectory.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Incomplete trace"):
                completed_episodes(family, scenarios)
            recorded["scenario_id"] = "over_02"
            summary.write_text(json.dumps(recorded) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "locked scenario prefix"):
                completed_episodes(family, scenarios)

    def test_depth_ready_probe_rejects_uncompiled_luminance_fallback(self):
        client = Mock()
        w, h = DEPTH_SIZE
        fallback = SimpleNamespace(
            width=w, height=h, image_data_float=np.full(w * h, 0.85, dtype=np.float32),
        )
        physical = SimpleNamespace(
            width=w, height=h, image_data_float=np.linspace(3.1, 51.0, w * h, dtype=np.float32),
        )
        client.simGetImages.side_effect = [[fallback], [physical]]
        self.assertIsNone(_probe_depth_ready(client, "SimpleFlight"))
        self.assertAlmostEqual(_probe_depth_ready(client, "SimpleFlight"), 51.0, places=3)

    def test_reset_depth_consistency_rejects_luminance_fallback(self):
        bad = {
            "depth": {
                "camera_position_ned_m": [4.403, 12.369, -3.513],
                "center_m": 0.276,
                "max_m": 0.662,
            }
        }
        with self.assertRaisesRegex(RuntimeError, "inconsistent with scene geometry"):
            assert_reset_depth_consistent(bad, self.course)
        good = {
            "depth": {
                "camera_position_ned_m": [4.403, 12.369, -3.513],
                "center_m": 19.594,
                "max_m": 51.219,
            }
        }
        assert_reset_depth_consistent(good, self.course)


if __name__ == "__main__":
    unittest.main()
