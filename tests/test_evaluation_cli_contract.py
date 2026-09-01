"""Every launcher must speak the CLI that evaluate_single actually accepts.

The launchers spawn ``scripts.evaluation.evaluate_single`` in a subprocess, so
a renamed script or an option that was never declared only shows up as a failed
job hours into a batch. ``evaluate_single`` itself pulls in torch, SB3 and
cosysairsim, so its option list is read from the source with ``ast`` instead of
by importing it.
"""

import argparse
import ast
import unittest
from pathlib import Path

from scripts.evaluation.evaluate_completed_runs import (
    EvaluationJob,
    build_evaluation_command,
)
from scripts.evaluation.record_parallel import build_eval_command
from scripts.training.train_parallel import ROOT

EVALUATOR_MODULE = "scripts.evaluation.evaluate_single"
EVALUATOR_SOURCE = ROOT / "scripts" / "evaluation" / "evaluate_single.py"


def declared_options(source: Path) -> set[str]:
    """Collect every option string passed to parser.add_argument in a script."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    options: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        for argument in node.args:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                options.add(argument.value)
    return options


def emitted_options(command: list[str]) -> set[str]:
    return {part for part in command if part.startswith("-") and part != "-"}


class EvaluatorOptionsTest(unittest.TestCase):
    def setUp(self):
        self.accepted = declared_options(EVALUATOR_SOURCE)

    def test_source_scan_finds_the_evaluator_options(self):
        self.assertIn("--best", self.accepted)
        self.assertIn("--scenario-seed", self.accepted)

    def test_batch_evaluator_emits_only_accepted_options(self):
        command = build_evaluation_command(
            EvaluationJob(
                scene="tunnel",
                method="sac",
                seed=0,
                run_dir=Path("Logs/TrainData/Runs/tunnel/sac/seed_0/run_a"),
                target_steps=80_000,
                port=41451,
            ),
            episodes=100,
            airsim_ip="127.0.0.1",
            scenario_seed=2026,
            yaw_range_deg=10.0,
        )
        self.assertEqual(command[1:3], ["-m", EVALUATOR_MODULE])
        self.assertLessEqual(emitted_options(command), self.accepted)

    def test_demo_recorder_emits_only_accepted_options(self):
        args = argparse.Namespace(
            episodes=8, airsim_ip="127.0.0.1", scenario_seed=2026
        )
        for job in (
            {"run_dir": Path("run"), "model": "best"},
            {"run_dir": Path("run"), "model": "checkpoint", "step": 40_000},
            {"run_dir": Path("run"), "model": "final"},
        ):
            with self.subTest(model=job["model"]):
                command = build_eval_command(job, 41451, args)
                self.assertEqual(command[1:3], ["-m", EVALUATOR_MODULE])
                self.assertLessEqual(emitted_options(command), self.accepted)


if __name__ == "__main__":
    unittest.main()
