"""Verify that NavRL V2 distinguishes a core traversal from a boundary route."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from configparser import ConfigParser
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
import json

from tools.navrl_forest_evaluation import _evaluate_scenario, _path_metrics, evaluate_checkpoint


class CoreTraversalTests(unittest.TestCase):
    def test_four_workers_own_their_histories_and_report_every_case_once(self):
        ports = [41815, 41816, 41817, 41818]
        cfg = ConfigParser()
        cfg.read_dict({"airsim": {"rpc_port": ports[0]}, "navrl_forest": {}, "options": {"algo": "SAC"}})
        cases = [{"scenario_id": str(i)} for i in range(50)]
        barrier = Barrier(4, timeout=10)
        first = set()
        persisted = []
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint.zip"
            checkpoint.touch()
            (root / "scenario_manifest.json").touch()

            def simulate(model, env, transpose, case, *, seed, on_reset):
                port = model.port
                self.assertEqual(env.port, port)
                on_reset(case["scenario_id"])
                if port not in first:
                    first.add(port)
                    barrier.wait()
                marker = json.loads((root / f"active_scenario_{port}.json").read_text())
                self.assertEqual(marker["scenario_id"], case["scenario_id"])
                return dict(case, collision_free_success=True)

            def make_env(config):
                env = Mock()
                gym = Mock(port=config.getint("airsim", "rpc_port"))
                env.venv.envs = [gym]
                return env

            mod = "tools.navrl_forest_evaluation."
            with patch(mod + "build_runtime_env", side_effect=make_env), patch(mod + "use_mamba_backend", return_value=True), \
                 patch(mod + "load_model", side_effect=lambda *a, cfg, **k: Mock(port=cfg.getint("airsim","rpc_port"))), \
                 patch(mod + "_evaluate_scenario", side_effect=simulate):
                rows = evaluate_checkpoint(checkpoint, cfg, cases, seed=0, ports=ports, on_outcome=persisted.append)
            self.assertEqual([r["scenario_id"] for r in rows], [c["scenario_id"] for c in cases])
            self.assertCountEqual(persisted, rows)
            self.assertEqual(set(ports), {row["evaluation_port"] for row in rows})

    def test_segment_is_clipped_to_the_core_even_with_both_endpoints_outside(self):
        outcome = _path_metrics([(2, 0), (46, 0)], 0, [46, 0, 0])
        self.assertEqual(outcome["horizontal_path_m"], 44)
        self.assertAlmostEqual(outcome["core_path_fraction"], 40 / 44)
        self.assertTrue(outcome["crossed_core"])

    def test_perimeter_route_and_boundary_contact_do_not_cross_the_core(self):
        detour = _path_metrics([(2, 0), (2, 22), (46, 22), (46, 0)], 0, [46, 0, 0])
        self.assertEqual(detour["horizontal_path_m"], 88)
        self.assertEqual(detour["core_path_fraction"], 0)
        self.assertFalse(detour["crossed_core"])
        self.assertFalse(_path_metrics([(2, 20), (46, 20)], 0, [46, 20, 0])["crossed_core"])

    def test_exact_centerline_samples_count_only_when_the_path_keeps_crossing(self):
        starts_on_positive_y = [(24, 22), (24, 0)]
        goal = [24, -22, 0]
        self.assertTrue(_path_metrics(starts_on_positive_y + [(24, -22)], 1, goal)["crossed_core"])
        self.assertFalse(_path_metrics(starts_on_positive_y + [(24, 22)], 1, goal)["crossed_core"])

    def test_evaluator_uses_actual_settled_start_and_preserves_legacy_output(self):
        scenario = {"scenario_id": "case", "start_position": [2.1, 0, 0], "goal_position": [46, 0, 0]}
        env = Mock()
        env.unwrapped = SimpleNamespace(
            max_episode_steps=1,
            dynamic_model=SimpleNamespace(last_reset_state={"position_ned_m": [2, 0, 0]}),
        )
        env.reset.return_value = (None, {"scenario_id": "case"})
        env.step.return_value = (None, 0, True, False, {
            "is_crash": False, "is_success": True, "is_not_in_workspace": False, "position": [46, 0, 0],
        })
        model = Mock()
        model.predict.return_value = (None, None)
        identity = lambda obs: obs

        legacy = _evaluate_scenario(model, env, identity, scenario)
        self.assertNotIn("crossed_core", legacy)
        publish = Mock()
        v2 = _evaluate_scenario(model, env, identity, {**scenario, "crossing_axis": 0}, on_reset=publish)
        publish.assert_called_once_with("case")
        self.assertEqual(v2["horizontal_path_m"], 44)
        self.assertTrue(v2["crossed_core"])
        self.assertEqual(v2["collision_free_success"], legacy["collision_free_success"])
        publish.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "expected"):
            _evaluate_scenario(model, env, identity, {**scenario, "scenario_id": "wrong"}, on_reset=publish)
        publish.assert_not_called()
