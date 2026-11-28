"""Regression coverage for the checkpoint evaluated by the forest gate."""

import json
from configparser import ConfigParser
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
from unittest.mock import Mock, call, patch

from scripts.training.train_navrl_curriculum import ForestPromotion, launch_stage, load_protocol, run_alternating_stage
from tools.navrl_forest_runtime import fixed_forest_routes


class _Model:
    num_timesteps = 100

    def save(self, stem):
        Path(stem + ".zip").write_text(str(self.num_timesteps), encoding="utf-8")


def _scenarios(stage, round_id):
    return {"scenarios": [{"scenario_id": f"{stage}:{round_id}:{case}"} for case in range(50)]}


def _gate(stage, rounds):
    passed = len(rounds) >= 2 and all(
        row["collision_free_successes"] >= 41 for row in rounds[-2:]
    )
    return {"stage": stage, "decision": "advance" if passed else "stay"}


class ForestProtocolTests(unittest.TestCase):
    def test_stage_uses_requested_clock_speed(self):
        args = SimpleNamespace(ue_exe=Path("editor.exe"), ue_project=Path("Forest.uproject"),
                               startup_timeout=120, clock_speed=2.0)
        stage = SimpleNamespace(map_path="/Game/Training/Maps/v3")
        with patch("scripts.training.train_navrl_curriculum.launch_trainenv") as launch:
            launch_stage(args, stage, [41811], Path("run"), Path("settings.json"))
        self.assertEqual(launch.call_args.kwargs["clock_speed"], 2.0)
        self.assertEqual(launch.call_args.kwargs["clock_type"], "SteppableClock")
        self.assertEqual(launch.call_args.kwargs["editor_project"], Path("Forest.uproject"))

    def test_stage_omits_editor_project_for_packaged_executable(self):
        args = SimpleNamespace(ue_exe=Path("Ch4Tunnel.exe"), ue_project=Path("Forest.uproject"),
                               startup_timeout=120, clock_speed=2.0)
        stage = SimpleNamespace(map_path="/Game/Training/Maps/v3")
        with patch("scripts.training.train_navrl_curriculum.launch_trainenv") as launch:
            launch_stage(args, stage, [41811], Path("run"), Path("settings.json"))
        self.assertIsNone(launch.call_args.kwargs["editor_project"])
        self.assertIn("-NoVSync", launch.call_args.kwargs["unreal_args"])

    def test_default_loads_full_field_and_explicit_versions_resolve_older_maps(self):
        with TemporaryDirectory() as tmp:
            project = Path(tmp) / "Forest.uproject"
            content = project.parent / "Content" / "Training" / "Maps"
            content.mkdir(parents=True)
            stages = [SimpleNamespace(map_path=f"/Game/Training/Maps/v{version}") for version in (1, 2, 3)]
            for version in (1, 2, 3):
                (content / f"v{version}.umap").touch()
            rules = {
                "STAGES": (stages[0],), "V2_STAGES": (stages[1],), "V3_STAGES": (stages[2],),
                "evaluation_scenarios": lambda stage, round_id, scene_version=1: (stage, round_id, scene_version),
            }
            with patch("scripts.training.train_navrl_curriculum.runpy.run_path", side_effect=lambda _: rules.copy()):
                for version, selected in enumerate(
                    (load_protocol(project, 1), load_protocol(project, 2), load_protocol(project)), 1,
                ):
                    self.assertEqual(selected["STAGES"], (stages[version - 1],))
                    self.assertEqual(selected["evaluation_scenarios"](1, 7), (1, 7, version))


class ForestPromotionTests(unittest.TestCase):
    def test_parallel_eval_releases_training_worlds_and_resumes_full_state(self):
        active = []
        trained = []
        assessed = []

        @contextmanager
        def launch(*args):
            active.append("training")
            try:
                yield None
            finally:
                active.pop()

        def train(args, cfg, run_dir, simulator_job):
            self.assertEqual(active, ["training"])
            self.assertTrue(args.restore_replay)
            state = torch.load(args.resume, weights_only=False)
            self.assertIn("replay_buffer", state)
            state["num_timesteps"] += args.timesteps
            trained.append(state["num_timesteps"])
            (run_dir / "models").mkdir(parents=True)
            torch.save(state, run_dir / "models/model_sb3.zip")

        def evaluate(checkpoint, directory, cases):
            self.assertFalse(active)
            weights = torch.load(checkpoint, weights_only=False)
            self.assertNotIn("replay_buffer", weights)
            assessed.append(weights["num_timesteps"])
            return [dict(scenario_id=c["scenario_id"], collision_free_success=False) for c in cases]

        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "original.zip"
            torch.save({"replay_buffer": {}, "num_timesteps": 100000}, source)
            gate = self.callback(folder, evaluate)
            gate.first_eval_step = 100000
            cfg = ConfigParser()
            cfg.add_section("environment")
            with patch("scripts.training.train_navrl_curriculum.launch_stage", launch), \
                 patch("scripts.training.train_navrl_curriculum.run_training", train):
                spent, full = run_alternating_stage(None, gate.stage, folder, None, cfg,
                    [1, 2], gate, SimpleNamespace(restore_replay=True), source, 20)
            self.assertEqual(trained, [100010, 100020])
            self.assertEqual(assessed, [100000, 100010, 100020])
            self.assertEqual(spent, 20)
            self.assertTrue(full.is_file())

    def test_resume_evaluates_exact_checkpoint_before_next_training_step(self):
        with TemporaryDirectory() as tmp:
            def evaluate(checkpoint, directory, cases):
                return [{"scenario_id": c["scenario_id"], "collision_free_success": False} for c in cases]

            callback = self.callback(Path(tmp), evaluate)
            callback.model.num_timesteps = 100000
            callback.first_eval_step = 100000
            callback._on_training_start()
            self.assertEqual([row["num_timesteps"] for row in callback.rounds], [100000])
            self.assertEqual(callback.next_evaluation, callback.interval)
            callback.model.num_timesteps += callback.interval
            callback._on_step()
            self.assertEqual(len(callback.rounds), 2)

    def test_switch_keeps_model_and_requires_two_new_random_passes(self):
        def score(checkpoint, directory, cases):
            return [{"scenario_id": row["scenario_id"], "collision_free_success": i < 40,
                     "crossed_core": True, "core_path_fraction": 1.0} for i, row in enumerate(cases)]

        with TemporaryDirectory() as tmp:
            callback = self.callback(Path(tmp), score)
            callback.rules["EPISODES_PER_ROUND"] = 50
            callback.rules["decide"] = lambda stage, rows: {
                "decision": "advance" if len(rows) >= 2 and
                all(r["collision_free_successes"] >= 40 for r in rows[-2:]) else "stay"}
            callback.fixed_routes = fixed_forest_routes((22, (-12, 0, 12), .75))
            workers = Mock()
            callback.model.get_env = lambda: workers
            for step in (110, 120, 130, 140):
                callback.model.num_timesteps = step
                self.assertEqual(callback._on_step(), step != 140)
            workers.env_method.assert_called_once_with("enable_random_forest_routes")
            rounds = sorted((Path(tmp) / "evaluations").iterdir())
            results = [json.loads((p / "result.json").read_text()) for p in rounds]
            self.assertEqual([r["decision"] for r in results], ["stay", "switch_to_random", "stay", "advance"])
            self.assertEqual([r["route_phase"] for r in results], ["fixed", "fixed", "random", "random"])
            cases = json.loads((rounds[0] / "scenario_manifest.json").read_text())["scenarios"]
            routes = [(tuple(c["start_position"]), tuple(c["goal_position"])) for c in cases]
            self.assertEqual(len(set(routes)), 18)
            self.assertLessEqual(max(map(routes.count, routes)) - min(map(routes.count, routes)), 1)

    def callback(self, folder, evaluate):
        callback = ForestPromotion(
            SimpleNamespace(number=1), {"evaluation_scenarios": _scenarios, "decide": _gate},
            folder, 30, 10, evaluate,
        )
        callback.model = _Model()
        callback._on_training_start()
        return callback

    def test_handoff_is_the_checkpoint_that_passed_on_the_second_round(self):
        scores = iter([40, 41, 41])

        def evaluate(checkpoint, directory, cases):
            self.assertEqual(checkpoint.parent, directory)
            cutoff = next(scores)
            return [{"scenario_id": row["scenario_id"], "collision_free_success": i < cutoff}
                    for i, row in enumerate(cases)]

        with TemporaryDirectory() as tmp:
            callback = self.callback(Path(tmp), evaluate)
            callback.model.num_timesteps = 109
            self.assertTrue(callback._on_step())
            self.assertFalse(callback.rounds)
            for step in (110, 120):
                callback.model.num_timesteps = step
                self.assertTrue(callback._on_step())
                self.assertIsNone(callback.promotion_checkpoint)
            callback.model.num_timesteps = 130
            self.assertFalse(callback._on_step())
            checkpoint = callback.promotion_checkpoint
            self.assertEqual(checkpoint.read_text(encoding="utf-8"), "130")
            result = json.loads((checkpoint.parent / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["collision_free_successes"], 41)
            self.assertEqual(len(result["outcomes"]), 50)

    def test_an_incomplete_round_cannot_advance(self):
        with TemporaryDirectory() as tmp:
            callback = self.callback(Path(tmp), lambda *_: [])
            callback.model.num_timesteps = 110
            with self.assertRaisesRegex(RuntimeError, "prescribed scenario"):
                callback._on_step()
            self.assertIsNone(callback.promotion_checkpoint)
            self.assertFalse(callback.rounds)
            self.assertFalse(list(Path(tmp).rglob("result.json")))

    def test_continuous_training_worlds_pause_for_the_independent_evaluator(self):
        with TemporaryDirectory() as tmp:
            callback = self.callback(Path(tmp), Mock(side_effect=RuntimeError("evaluation failed")))
            callback.continuous_training = True
            workers = Mock()
            callback.model.get_env = lambda: workers
            callback.model.num_timesteps = 110
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                callback._on_step()
            self.assertEqual(workers.env_method.call_args_list, [
                call("pause_for_evaluation", True), call("pause_for_evaluation", False),
            ])

    def test_v2_records_core_crossings_separately_from_arrivals(self):
        def v2_scenarios(stage, round_id):
            cases = _scenarios(stage, round_id)
            for case in cases["scenarios"]:
                case["crossing_axis"] = 0
            return cases

        def evaluate(_checkpoint, _directory, cases):
            return [
                {
                    "scenario_id": case["scenario_id"],
                    "collision_free_success": i < 41,
                    "crossed_core": i < 30,
                    "core_path_fraction": 0.5 if i < 30 else 0.0,
                }
                for i, case in enumerate(cases)
            ]

        with TemporaryDirectory() as tmp:
            callback = self.callback(Path(tmp), evaluate)
            callback.rules["evaluation_scenarios"] = v2_scenarios
            callback.model.num_timesteps = 110
            self.assertTrue(callback._on_step())
            result_path = Path(tmp) / "evaluations" / "round_001" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["collision_free_successes"], 41)
            self.assertEqual(result["core_crossing_episodes"], 30)
            self.assertAlmostEqual(result["core_crossing_rate"], 0.6)
            self.assertAlmostEqual(result["mean_core_path_fraction"], 0.3)
