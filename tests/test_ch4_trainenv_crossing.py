"""Regress the opt-in clock, two paired poses and existing Stage 3 family cycle."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cosysairsim as airsim
import numpy as np

from gym_env.envs.dynamics.multirotor_airsim import MultirotorDynamicsAirsim
from scripts.training import train_trainenv
from tools.ue_benchmark.ch4_stage3_crossing import (
    CROSSING_KEY, HALF_TRAVEL_M, SPEED_MPS, CrossingCase, CrossingPath, CrossingScene, crossing_mode,
    ready_reset_observation,
)
from tools.ue_benchmark.ch4_stage3_v2_training import Stage3V2Course
from tools.ue_benchmark.ch4_trainenv_3d_runtime import read_ini


CFG = Path(__file__).resolve().parents[1] / "configs/ch4_train_pure_masac_3d_scratch.ini"


class CrossTrafficTests(unittest.TestCase):
    def test_short_synchronous_window_retries_with_time_for_a_physics_tick(self):
        dynamics = MultirotorDynamicsAirsim.__new__(MultirotorDynamicsAirsim)
        dynamics.client = Mock()
        dynamics.client.getMultirotorState.side_effect = [
            SimpleNamespace(timestamp=ts) for ts in (0, 99_000_000, 99_000_000, 108_000_000)
        ]
        dynamics.dt = .1
        dynamics.vehicle_name = "SimpleFlight"
        dynamics._advance_synchronously(0., 0., 0., airsim.YawMode(True, 0.))
        windows = [call.args[0] for call in dynamics.client.simContinueForTime.call_args_list]
        self.assertEqual(windows, [.1, .002, .010])
        self.assertAlmostEqual(dynamics.last_control_sim_seconds, .108)

    def test_reset_reacquires_saturated_and_body_occluded_views(self):
        sky = {"image": np.full((60, 90, 1), 255, dtype=np.uint8)}
        body = {"image": np.zeros((60, 90, 1), dtype=np.uint8)}
        floor = {"image": np.full((60, 90, 1), 34, dtype=np.uint8)}
        env = SimpleNamespace(perception=SimpleNamespace(max_depth_meters=25), _get_obs=Mock(side_effect=[body, floor]))
        result, attempts = ready_reset_observation(env, sky)
        self.assertIs(result, floor)
        self.assertEqual(attempts, 2)
        env._get_obs = Mock(return_value=sky)
        with self.assertRaisesRegex(RuntimeError, "near ground"):
            ready_reset_observation(env, sky)
        self.assertEqual(env._get_obs.call_count, 4)

    def test_one_continuous_traverse_with_a_matched_stationary_start(self):
        for direction in (-1, 1):
            path = CrossingPath(12., 1., -1., direction, 5.)
            self.assertEqual(path.box_at(0., "stationary"), path.box_at(0., "moving"))
            start = 1. - direction * HALF_TRAVEL_M
            end = 1. + direction * HALF_TRAVEL_M
            self.assertEqual(path.center_at(path.start_time_s - .1, "moving")[1], start)
            self.assertAlmostEqual(path.center_at(path.start_time_s + .2, "moving")[1],
                                   start + direction * .2 * SPEED_MPS)
            self.assertAlmostEqual(path.center_at(5., "moving")[1], 1.)
            self.assertAlmostEqual(path.center_at(path.end_time_s, "moving")[1], end)
            self.assertEqual(path.center_at(100., "moving")[1], end)
            self.assertEqual(path.center_at(100., "stationary")[1], start)

    def test_sweeps_depend_on_simulation_timestamp_and_log_observed_pose(self):
        path = CrossingPath(12., 1., -1., 1, HALF_TRAVEL_M / SPEED_MPS)
        client = Mock()
        first = path.center_at(0., "moving")
        initial_pose = airsim.Pose(airsim.Vector3r(first[0], first[1], 1. - first[2]))
        client.simGetObjectPose.return_value = initial_pose
        client.getMultirotorState.return_value.timestamp = 4_000_000_000

        def swept(actor, pose, *, teleport):
            self.assertEqual(actor, "mesh")
            self.assertFalse(teleport)
            client.simGetObjectPose.return_value = pose

        client.simSetObjectPose.side_effect = swept
        scene = CrossingScene(client, "mesh", path, "moving", "near", record_updates=True)
        dynamics = SimpleNamespace(vehicle_name="SimpleFlight")
        scene.start(dynamics)
        dynamics.scene_step_callback(4_000_000_000)
        client.simSetObjectPose.assert_not_called()
        dynamics.scene_step_callback(4_050_000_000)
        self.assertAlmostEqual(scene.info()["t_s"], .05)
        self.assertAlmostEqual(scene.center[1], first[1] + .05 * SPEED_MPS)
        self.assertAlmostEqual(scene.updates[-1]["xyz_up_m"][1], scene.center[1])
        self.assertAlmostEqual(scene.max_moving_update_gap_s, .05)
        with self.assertRaisesRegex(RuntimeError, "backwards"):
            dynamics.scene_step_callback(4_020_000_000)

    def test_default_disabled_and_opt_in_does_not_replace_three_dimensional_gate(self):
        cfg = read_ini(CFG)
        cfg["airsim"]["rpc_port"] = "42072"
        self.assertEqual(crossing_mode(cfg), "off")
        env = SimpleNamespace(
            np_random=np.random.default_rng(31), task=SimpleNamespace(), client=Mock(),
            work_space_x=(-7., 169.), work_space_y=(-58., 58.),
            training_routes=[([0., 0., 1.], [162., 0., 1.])],
            random_route_reverse=False, start_random_angle=.45, dynamic_model=Mock(),
        )
        disabled = Stage3V2Course(cfg)
        disabled.geometry = Mock()
        disabled.forced_family = "original_roofed"
        disabled.select_route(env)
        self.assertEqual([b.key for b in disabled.scenario["boxes"]], ["tunnel_roof"])
        self.assertIsNone(disabled.crossing)

        cfg["stage3_v2"]["crossing_mode"] = "moving"
        active = Stage3V2Course(cfg)
        active.geometry = Mock()
        active.geometry.names = {CROSSING_KEY: "mesh"}
        case = CrossingCase((0., 0., 1.), (162., 0., 1.), .1, 2., CrossingPath(12., 0., -1., 1, 5.), "near")
        active.crossing_case = case
        active.select_route(env)
        self.assertEqual([b.key for b in active.scenario["boxes"]], ["tunnel_roof", CROSSING_KEY])
        self.assertEqual(active.scenario["start_position"], case.start_position)
        env.dynamic_model.set_start.assert_called_with(case.start_position, random_angle=0., fixed_yaw=case.yaw_rad)
        active.crossing_case = None
        active.forced_family = "over"
        active.select_route(env)
        self.assertIsNone(active.crossing)
        self.assertEqual([b.key for b in active.scenario["gates"]], ["over_0"])
        self.assertEqual(env.max_episode_steps, 500)

    def test_training_entry_applies_matching_clock_only_when_selected(self):
        for mode in (None, "stationary", "moving"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory, patch.object(
                train_trainenv, "launch_trainenv",
            ) as launch, patch.object(train_trainenv, "run_training") as train:
                argv = ["-c", str(CFG), "--instances", "1", "--timesteps", "1",
                        "--run-dir", str(Path(directory) / "owned")]
                if mode is not None:
                    argv.extend(("--crossing-mode", mode))
                train_trainenv.main(argv)
                configured = train.call_args.args[1]
                self.assertEqual(crossing_mode(configured), mode or "off")
                self.assertEqual(configured.getboolean("airsim", "synchronous_steps", fallback=False), bool(mode))
                self.assertEqual(launch.call_args.kwargs["clock_type"], "SteppableClock" if mode else None)
                self.assertEqual(launch.call_args.kwargs["clock_speed"], 1.0 if mode else 2.0)


if __name__ == "__main__":
    unittest.main()
