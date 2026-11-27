import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from scripts.evaluation.evaluate_trainenv_3d import freeze_candidate
from scripts.training import train_single
from tools.trainenv_instances import TRAINENV_MAP
from tools.ue_benchmark.ch4_stage3_training import (
    AUGMENTED_FAMILIES, Stage3TrainingCourse, augmented_route, available_routes, load_multiobstacle_course,
)
from tools.ue_benchmark.ch4_trainenv_3d_course import ROOT, read_json, sha256
from tools.ue_benchmark.ch4_trainenv_3d_runtime import assert_observation_contract, read_ini


CONFIG = ROOT / "configs/ch4_stage3_trainenv_3d.ini"


class Stage3Tests(unittest.TestCase):
    def test_stage3_uses_original_map_and_ordered_multiobstacle_routes(self):
        cfg = read_ini(CONFIG)
        course = load_multiobstacle_course(CONFIG.parent / cfg["stage3"]["course_file"])
        routes = available_routes(course)
        self.assertEqual(course["map"], TRAINENV_MAP)
        self.assertEqual(cfg["trainenv"]["map_path"], TRAINENV_MAP)
        self.assertTrue(all(len(choices) > 10 for choices in routes.values()))
        rng = np.random.default_rng(11)
        for family, modes in AUGMENTED_FAMILIES.items():
            for _ in range(32):
                route = augmented_route(course, routes, family, rng)
                self.assertEqual(tuple(box["mode"] for box in route["boxes"]), modes)
                distances = [p[0] for p in route["shape"]]
                self.assertTrue(all(a < b for a, b in zip(distances, distances[1:])))
                xs = [box["xyz_z_up_m"][0] for box in route["boxes"]]
                self.assertEqual(xs, sorted(xs, reverse=route["direction"] < 0))

    def test_isolated_resume_leaves_the_source_checkpoint_and_directory_untouched(self):
        cfg = read_ini(CONFIG)
        assert_observation_contract(cfg)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source_dir = root / "source" / "models"
            source_dir.mkdir(parents=True)
            source = source_dir / "checkpoint_495000_steps.zip"
            source.write_bytes(b"frozen source")
            # Newer files in the source directory do not block an isolated resume.
            (source_dir / "checkpoint_500000_steps.zip").write_bytes(b"other run output")
            output = root / "new_run"
            args = train_single.parse_args(["--config", str(CONFIG), "--resume", str(source), "--timesteps", "20"])
            model = Mock(num_timesteps=495000, device="cuda", n_envs=1)
            env = Mock()
            with (patch.object(train_single, "build_runtime_env", return_value=env),
                  patch.object(train_single, "load_model", return_value=model) as load):
                train_single.run_training(args, cfg, run_dir=output)
            self.assertEqual(load.call_args.args[1], str(source))
            self.assertIs(load.call_args.kwargs["load_replay"], False)
            self.assertIs(model.learn.call_args.kwargs["reset_num_timesteps"], False)
            self.assertEqual(model.learn.call_args.args[0], 20)
            self.assertEqual(read_json(output / "run.json")["resume_checkpoint_sha256"], sha256(source))
            self.assertEqual(source.read_bytes(), b"frozen source")
            self.assertFalse((source.parent.parent / "run.json").exists())
            env.close.assert_called_once_with()

    def test_strict_stage3_counts_only_ordered_crossings_with_vertical_clearance(self):
        cfg = read_ini(CONFIG)
        cfg.set("stage3", "strict_maneuvers", "True")
        curriculum = Stage3TrainingCourse(cfg)
        curriculum.scenario = {
            "start_position": [30, 0, 0], "direction": -1,
            "boxes": [
                {"mode": "over", "xyz_z_up_m": [20, 0, -1], "size_m": [4, 6, 2]},
                {"mode": "under", "xyz_z_up_m": [10, 0, 2], "size_m": [4, 6, 2]},
            ],
        }
        curriculum.passed_gates = 0
        curriculum.valid_crossing = False
        curriculum.prior_course_distance = 0
        self.assertEqual(curriculum._gate_reward(9, 0, 1), 0)
        self.assertEqual(curriculum._gate_reward(12, 0, 1), 2)
        self.assertEqual(curriculum._gate_reward(19, 3.5, 0), 0)
        self.assertEqual(curriculum._gate_reward(22, 0, 0), 0)
        self.assertFalse(curriculum.route_complete)
        curriculum._gate_reward(17, 0, 0)
        curriculum._gate_reward(19, 0, 0)
        self.assertEqual(curriculum._gate_reward(22, 0, 0), 2)
        self.assertTrue(curriculum.route_complete)
        self.assertEqual(curriculum._gate_reward(24, 0, 0), 0)

    def test_candidate_resume_refuses_changed_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config").mkdir()
            (root / "models").mkdir()
            (root / "evaluation").mkdir()
            shutil.copyfile(CONFIG, root / "config/config.ini")
            source = root / "models/checkpoint_498000_steps.zip"
            source.write_bytes(b"tested candidate")
            copied, _, identity = freeze_candidate(source, 498000, root / "evaluation", False)
            self.assertEqual(identity["checkpoint_sha256"], sha256(copied))
            source.write_bytes(b"changed candidate")
            with self.assertRaisesRegex(RuntimeError, "Candidate identity differs"):
                freeze_candidate(source, 498000, root / "evaluation", True)
