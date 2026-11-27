"""Reset must report controller failures before an episode starts."""
from configparser import ConfigParser
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cosysairsim as airsim

from gym_env.envs.dynamics.multirotor_airsim import MultirotorDynamicsAirsim


class MultirotorResetTest(unittest.TestCase):
    def make_dynamics(self):
        dynamics = MultirotorDynamicsAirsim.__new__(MultirotorDynamicsAirsim)
        dynamics.client = Mock()
        dynamics.client.armDisarm.return_value = True
        dynamics.client.simGetVehiclePose.return_value = airsim.Pose()
        dynamics.client.simGetCollisionInfo.return_value = SimpleNamespace(has_collided=False)
        dynamics._owns_client = True
        dynamics.synchronous_steps = False
        dynamics._forest_reset = False
        dynamics.vehicle_name = 'SimpleFlight'
        dynamics.start_fixed_yaw = 0.0
        dynamics.start_random_angle = 0.0
        dynamics.start_position = [0, 0, -1.5]
        return dynamics

    def test_controller_rpc_errors_are_not_silenced_by_join(self):
        dynamics = self.make_dynamics()
        dynamics.client.moveByVelocityAsync.return_value.get.side_effect = RuntimeError('controller failed')
        with self.assertRaisesRegex(RuntimeError, 'controller failed'):
            dynamics.reset()

    def test_unsuccessful_arm_or_settle_rejects_reset(self):
        for arm_success in (False, True):
            with self.subTest(arm_success=arm_success):
                dynamics = self.make_dynamics()
                dynamics.client.armDisarm.return_value = arm_success
                dynamics.client.moveByVelocityAsync.return_value.get.return_value = False
                with self.assertRaisesRegex(RuntimeError, 'Failed to (arm|settle) drone'):
                    dynamics.reset()
                if not arm_success:
                    dynamics.client.moveByVelocityAsync.assert_not_called()

    def test_action_rpc_error_or_unsuccessful_result_is_reported(self):
        dynamics = self.make_dynamics()
        dynamics.navigation_3d = True
        dynamics.dt = 0.1
        dynamics.get_attitude = Mock(return_value=[0, 0, 0])
        future = dynamics.client.moveByVelocityAsync.return_value
        future.get.side_effect = RuntimeError('controller failed')
        with self.assertRaisesRegex(RuntimeError, 'controller failed'):
            dynamics.set_action([1, 0, 0])
        future.get.side_effect = None
        future.get.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'Failed to execute drone velocity command'):
            dynamics.set_action([1, 0, 0])

    def test_synchronous_control_counts_simulation_time_and_leaves_physics_paused(self):
        dynamics = self.make_dynamics()
        dynamics.dt = 0.1
        dynamics.client.getMultirotorState.side_effect = [
            SimpleNamespace(timestamp=t) for t in [0, 60_000_000, 102_000_000]
        ]
        dynamics.client.moveByVelocityAsync.return_value.get.return_value = False
        dynamics._advance_synchronously(1, 0, 0, airsim.YawMode())
        self.assertAlmostEqual(dynamics.last_control_sim_seconds, .102)
        self.assertEqual(dynamics.client.simContinueForTime.call_count, 2)
        dynamics.client.simPause.assert_called_with(True)
        dynamics.client.cancelLastTask.assert_called_once_with('SimpleFlight')

    def test_navrl_clock_speed_scales_wall_windows_only(self):
        for forest, speed, expected in ((True, 2, (.05, .02)), (True, 1, (.1, .04)),
                                        (True, None, (.1, .04)), (False, 2, (.1, .04))):
            with self.subTest(forest=forest, speed=speed):
                cfg = ConfigParser()
                cfg.read_dict({
                    'options': {'navigation_3d': True, 'using_velocity_state': False},
                    'multirotor': {'dt': .1, 'v_xy_max': 2, 'v_xy_min': 0, 'v_z_max': 1, 'yaw_rate_max_deg': 90},
                    'airsim': {'synchronous_steps': True},
                })
                if forest:
                    cfg.add_section('navrl_forest')
                client = Mock()
                settings = {'ClockType': 'SteppableClock'}
                if speed is not None:
                    settings['ClockSpeed'] = speed
                client.getSettingsString.return_value = json.dumps(settings)
                client.getMultirotorState.side_effect = [
                    SimpleNamespace(timestamp=t) for t in (0, 60_000_000, 102_000_000)
                ]
                dynamics = MultirotorDynamicsAirsim(cfg, client=client)
                dynamics._advance_synchronously(1, 0, 0, airsim.YawMode())
                client.getSettingsString.assert_called_once_with()
                self.assertAlmostEqual(dynamics.last_control_sim_seconds, .102)
                self.assertEqual(client.simContinueForTime.call_count, len(expected))
                for call, window in zip(client.simContinueForTime.call_args_list, expected):
                    self.assertAlmostEqual(call.args[0], window)

    def test_synchronous_rpc_error_still_propagates_after_physics_is_paused(self):
        dynamics = self.make_dynamics()
        dynamics.dt = 0.1
        dynamics.client.getMultirotorState.side_effect = [
            SimpleNamespace(timestamp=t) for t in [0, 102_000_000]
        ]
        dynamics.client.moveByVelocityAsync.return_value.get.side_effect = RuntimeError('controller failed')
        with self.assertRaisesRegex(RuntimeError, 'controller failed'):
            dynamics._advance_synchronously(1, 0, 0, airsim.YawMode())
        dynamics.client.simPause.assert_called_with(True)

    def test_synchronous_reset_keeps_setup_paused_and_clears_motion(self):
        dynamics = self.make_dynamics()
        dynamics.synchronous_steps = True
        dynamics._advance_synchronously = Mock()
        dynamics._validate_reset_state = Mock()
        dynamics.reset()
        self.assertNotIn(False, [call.args[0] for call in dynamics.client.simPause.call_args_list])
        state = dynamics.client.simSetKinematics.call_args.args[0]
        self.assertEqual(list(state.position), [0, 0, -1.5])
        self.assertEqual(list(state.linear_velocity), [0, 0, 0])
        self.assertEqual(list(state.angular_velocity), [0, 0, 0])
        self.assertTrue(dynamics._advance_synchronously.call_args.args[3].is_rate)

    def test_continuous_forest_settles_the_pawn_then_runs_freely(self):
        dynamics = self.make_dynamics()
        dynamics._forest_reset = True
        dynamics._advance_synchronously = Mock()
        dynamics._validate_reset_state = Mock()
        dynamics.reset()
        dynamics.client.simSetObjectPose.assert_called_once()
        dynamics.client.simSetKinematics.assert_called_once()
        dynamics._validate_reset_state.assert_called_once_with(0.0)
        dynamics.client.simPause.assert_called_with(False)

        dynamics._validate_reset_state.side_effect = RuntimeError('invalid spawn')
        with self.assertRaisesRegex(RuntimeError, 'invalid spawn'):
            dynamics.reset()
        dynamics.client.simPause.assert_called_with(True)

    def test_invalid_reset_is_rejected_before_an_episode_can_start(self):
        dynamics = self.make_dynamics()
        dynamics.client.getMultirotorState.return_value = SimpleNamespace(
            kinematics_estimated=airsim.KinematicsState(), collision=SimpleNamespace(has_collided=True))
        with self.assertRaisesRegex(RuntimeError, 'Invalid airborne reset state'):
            dynamics._validate_reset_state(0)
