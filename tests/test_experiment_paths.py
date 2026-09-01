"""Path conventions that keep training runs and evaluation results aligned."""

import unittest
from pathlib import Path

from tools.experiment_paths import (
    EVALUATION_RESULTS_ROOT,
    TRAIN_RUNS_ROOT,
    evaluation_parent_for_run,
    normalize_training_relative,
    training_run_relative,
)


class NormalizeTrainingRelativeTest(unittest.TestCase):
    def test_strips_obsolete_outer_batch_layer(self):
        relative = Path("2026_07_28_07_47_57_517511/tunnel_mixed/ma/seed_0/run_a")
        self.assertEqual(
            normalize_training_relative(relative),
            Path("tunnel_mixed/ma/seed_0/run_a"),
        )

    def test_keeps_canonical_layout_untouched(self):
        relative = Path("tunnel/sac/seed_2/run_b")
        self.assertEqual(normalize_training_relative(relative), relative)

    def test_keeps_batch_id_when_it_is_the_only_component(self):
        relative = Path("2026_07_28_07_47_57_517511")
        self.assertEqual(normalize_training_relative(relative), relative)


class TrainingRunRelativeTest(unittest.TestCase):
    def test_resolves_run_below_the_canonical_root(self):
        run_dir = TRAIN_RUNS_ROOT / "tunnel_obs" / "lstm" / "seed_1" / "run_c"
        self.assertEqual(
            training_run_relative(run_dir),
            Path("tunnel_obs/lstm/seed_1/run_c"),
        )

    def test_rejects_runs_outside_the_canonical_root(self):
        with self.assertRaises(ValueError):
            training_run_relative(Path("/tmp/some_other_place/run_d"))


class EvaluationParentTest(unittest.TestCase):
    def test_mirrors_training_identity_below_the_results_root(self):
        run_dir = TRAIN_RUNS_ROOT / "tunnel" / "coordatt" / "seed_0" / "run_e"
        self.assertEqual(
            evaluation_parent_for_run(run_dir),
            EVALUATION_RESULTS_ROOT / "tunnel" / "coordatt" / "seed_0" / "run_e",
        )

    def test_honours_a_custom_output_root(self):
        run_dir = TRAIN_RUNS_ROOT / "tunnel" / "coordatt" / "seed_0" / "run_e"
        custom = Path("/tmp/custom_results")
        self.assertEqual(
            evaluation_parent_for_run(run_dir, custom),
            custom / "tunnel" / "coordatt" / "seed_0" / "run_e",
        )


if __name__ == "__main__":
    unittest.main()
