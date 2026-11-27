"""Regression coverage for the checkpoint evaluated by the forest gate."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from scripts.training.train_navrl_curriculum import ForestPromotion


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


class ForestPromotionTests(unittest.TestCase):
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
