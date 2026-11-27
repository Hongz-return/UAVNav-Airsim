"""Job planning for parallel training, and the scene budgets it defaults to."""

import argparse
from configparser import ConfigParser
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.training.train_parallel import (
    METHOD_CONFIGS,
    ROOT,
    SCENES,
    build_batches,
    build_jobs,
    default_ports,
    read_start_position,
    write_runtime_config,
)

# Budgets are the experiment protocol, so RETRAINING.md is the source of truth
# and the launcher defaults have to agree with it.
RETRAINING_BUDGET_ROW = re.compile(
    r"^\|\s*`(?P<scene>\w+)`\s*\|[^|]*\|\s*(?P<steps>[\d,]+)\s*steps\s*\|$"
)


def documented_budgets() -> dict[str, int]:
    text = (ROOT / "RETRAINING.md").read_text(encoding="utf-8")
    budgets = {}
    for line in text.splitlines():
        match = RETRAINING_BUDGET_ROW.match(line.strip())
        if match:
            budgets[match["scene"]] = int(match["steps"].replace(",", ""))
    return budgets


def make_args(**overrides) -> argparse.Namespace:
    defaults = {
        "scene": "tunnel",
        "methods": ["sac", "coordatt", "lstm"],
        "seeds": [0, 1, 2],
        "ports": None,
        "max_parallel": 3,
        "log_root": Path("Logs/TrainData/Runs"),
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class SceneBudgetTest(unittest.TestCase):
    def test_defaults_match_the_documented_experiment_matrix(self):
        documented = documented_budgets()
        self.assertEqual(set(documented), set(SCENES))
        for scene, steps in documented.items():
            with self.subTest(scene=scene):
                self.assertEqual(SCENES[scene]["timesteps"], steps)

    def test_every_scene_declares_a_task_name_and_map(self):
        for scene, spec in SCENES.items():
            with self.subTest(scene=scene):
                self.assertIn(spec["env_name"], {"Tunnel", "Obs"})
                self.assertTrue(spec["map_path"].startswith("/Game/"))

    def test_env3_defaults_to_the_packaged_dynamic_map(self):
        self.assertEqual(SCENES["tunnel_activate"]["map_path"],
                         "/Game/Tunnel_Col_Vol1/Maps/Tunnel_Activate")

    def test_railway_scene_uses_calibrated_goal_for_all_methods(self):
        scene = SCENES['rail_single_dynamic']
        self.assertTrue(scene['requires_existing_airsim'])
        with TemporaryDirectory() as tmp:
            for method in ('ma', 'ma_motion', 'ma_aux', 'ma_motion_aux'):
                target = Path(tmp) / f'{method}.ini'
                write_runtime_config(ROOT / METHOD_CONFIGS[method], target,
                                     env_name=scene['env_name'], airsim_ip='127.0.0.1',
                                     port=41571, goal_position=scene['goal_position'])
                cfg = ConfigParser()
                cfg.read(target, encoding='utf-8')
                self.assertEqual(cfg.get('environment', 'start_position'), '[0, 0, -1.5]')
                self.assertEqual(cfg.get('environment', 'goal_position'), repr(scene['goal_position']))

    def test_every_method_config_exists(self):
        for method, config in METHOD_CONFIGS.items():
            with self.subTest(method=method):
                self.assertTrue((ROOT / config).is_file())


class JobPlanningTest(unittest.TestCase):
    def test_ch4_ablation_plan_and_config_differences(self):
        methods = ["ma", "ma_motion", "ma_aux", "ma_motion_aux"]
        jobs = build_jobs(make_args(scene="tunnel_activate", methods=methods), run_id="CH4")
        batches = build_batches(jobs, 3)
        self.assertEqual(len(jobs), 12)
        self.assertEqual([[job.method for job in batch] for batch in batches], [[method] * 3 for method in methods])
        baseline = ConfigParser()
        baseline.read(ROOT / METHOD_CONFIGS["ma"], encoding="utf-8")
        for method in methods:
            with self.subTest(method=method):
                cfg = ConfigParser()
                cfg.read(ROOT / METHOD_CONFIGS[method], encoding="utf-8")
                for section in baseline.sections():
                    for key, value in baseline.items(section):
                        if (section, key) not in {("options", "project_name"), ("options", "perception")}:
                            self.assertEqual(cfg.get(section, key), value, (section, key))
                self.assertEqual(cfg.get("options", "perception"), "depth_motion" if "motion" in method else "depth_state")
                self.assertEqual(cfg.getfloat("aux", "aux_weight", fallback=0), 0.1 if "aux" in method else 0)

    def test_jobs_are_method_major(self):
        jobs = build_jobs(make_args(), run_id="RUN")
        self.assertEqual(
            [(job.method, job.seed) for job in jobs],
            [
                ("sac", 0), ("sac", 1), ("sac", 2),
                ("coordatt", 0), ("coordatt", 1), ("coordatt", 2),
                ("lstm", 0), ("lstm", 1), ("lstm", 2),
            ],
        )

    def test_each_wave_of_three_seeds_is_one_method(self):
        jobs = build_jobs(make_args(), run_id="RUN")
        batches = build_batches(jobs, 3)
        self.assertEqual(len(batches), 3)
        for batch in batches:
            self.assertEqual(len({job.method for job in batch}), 1)
            self.assertEqual(len({job.port for job in batch}), 3)

    def test_single_seed_packs_methods_into_one_wave(self):
        jobs = build_jobs(make_args(seeds=[0]), run_id="RUN")
        batches = build_batches(jobs, 3)
        self.assertEqual(len(batches), 1)
        self.assertEqual(
            [job.method for job in batches[0]], ["sac", "coordatt", "lstm"]
        )

    def test_default_ports_are_ten_apart(self):
        self.assertEqual(default_ports(3), [41451, 41461, 41471])

    def test_rejects_duplicate_methods(self):
        with self.assertRaises(ValueError):
            build_jobs(make_args(methods=["sac", "sac"]), run_id="RUN")

    def test_rejects_duplicate_seeds(self):
        with self.assertRaises(ValueError):
            build_jobs(make_args(seeds=[0, 0]), run_id="RUN")

    def test_rejects_duplicate_ports(self):
        with self.assertRaises(ValueError):
            build_jobs(make_args(ports=[41451, 41451, 41461]), run_id="RUN")

    def test_rejects_too_few_ports(self):
        with self.assertRaises(ValueError):
            build_jobs(make_args(ports=[41451]), run_id="RUN")

    def test_build_batches_rejects_zero_parallelism(self):
        jobs = build_jobs(make_args(), run_id="RUN")
        with self.assertRaises(ValueError):
            build_batches(jobs, 0)


class StartPositionTest(unittest.TestCase):
    def test_all_methods_share_one_start_position(self):
        positions = {
            tuple(read_start_position(ROOT / config))
            for config in METHOD_CONFIGS.values()
        }
        self.assertEqual(len(positions), 1)

    def test_takeoff_height_matches_the_fixed_protocol(self):
        position = read_start_position(ROOT / METHOD_CONFIGS["sac"])
        self.assertEqual(position[2], -1.5)


if __name__ == "__main__":
    unittest.main()
