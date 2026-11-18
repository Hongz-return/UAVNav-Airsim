"""Scene timing and static labels must survive the shared train/eval wrapper."""
import tempfile
import sys
import unittest
from configparser import ConfigParser
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_env.wrappers.sequence_scene import SequenceSceneWrapper
from gym_env.wrappers.ue_sequence_clock import LocalSequenceClock
from tools.runtime_support import CheckpointMonitor


class SceneEnv(gym.Env):
    def __init__(self):
        self.action_space=spaces.Box(-1,1,(3,),dtype=np.float32)
        self.observation_space=spaces.Box(0,1,(1,),dtype=np.float32)
        self.dynamic_model=SimpleNamespace(synchronous_steps=True,vehicle_name='SimpleFlight',dt=.1,last_control_sim_seconds=.102,last_reset_state={})
        self.client=Mock();self.client.getMultirotorState.return_value=SimpleNamespace(timestamp=100)
        self._evaluation_scenarios=None;self._evaluation_scenario_index=0;self.current_scenario=None
        self.step_num=0;self.cumulated_episode_reward=0.

    def reset(self,*,seed=None,options=None):
        if self._evaluation_scenarios:
            self.current_scenario=self._evaluation_scenarios[self._evaluation_scenario_index]
            self._evaluation_scenario_index+=1
        return np.zeros(1,np.float32),{}

    def step(self,action):
        self.step_num=1
        return np.zeros(1,np.float32),0.,True,False,{
            'position':[45.,2.25,-1.5],'is_success':False,'is_crash':False,'is_not_in_workspace':False}


class SequenceSceneTest(unittest.TestCase):
    def make_scene(self,root):
        script=root/'runtime.py';script.touch()
        cfg=ConfigParser();cfg.read_dict({'airsim':{'rpc_port':'41691'},'scene':{
            'runtime_script':str(script),'play_rate':'.4','training_phases_s':'[0,75]',
            'training_yaw_degrees':'3','record_dir':str(root/'records')}})
        env=SequenceSceneWrapper(SceneEnv(),cfg)
        env.command=Mock(side_effect=lambda *a,**kw:{'poses':{'cart':[0.,0.,0.]}})
        return env

    def test_moving_scene_uses_physics_elapsed_time_instead_of_autoplay(self):
        with tempfile.TemporaryDirectory() as directory:
            env=self.make_scene(Path(directory))
            env.unwrapped._evaluation_scenarios=[{'condition':'moving_reverse','phase_s':75}]
            env.reset(seed=7)
            env.unwrapped.dynamic_model.scene_step_callback(200_000_100)
            env.command.assert_called_with('advance',phase_s=75.,elapsed_s=.2,play_rate=.4)
            self.assertNotIn('play',[call.args[0] for call in env.command.call_args_list])

    def test_static_label_rejects_obstacle_movement(self):
        with tempfile.TemporaryDirectory() as directory:
            env=self.make_scene(Path(directory))
            env.unwrapped._evaluation_scenarios=[{'condition':'static','phase_s':0}]
            env.reset(seed=7)
            self.assertIsNone(env.unwrapped.dynamic_model.scene_step_callback)
            env.command.return_value={'poses':{'cart':[.1,0.,0.]}}
            env.command.side_effect=None
            with self.assertRaisesRegex(RuntimeError,'Static obstacle drift'):
                env.step(np.zeros(3,np.float32))

    def test_bound_clock_only_advances_on_new_physics_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.make_scene(Path(directory))
            prepared = {'map': 'tunnel', 'sequence': 'reverse', 'poses': {'cart': [0., 0., 0.]}}
            env.command = Mock(return_value=prepared)
            env._fast_sync_enabled = True
            clock = env._sequence_clock = Mock()
            env.unwrapped._evaluation_scenarios = [{'condition': 'moving_reverse', 'phase_s': 75}]
            env.reset(seed=7)
            clock.bind.assert_called_once_with(prepared, 75., .4)
            start = env.scene_start_timestamp
            for offset in (0, 36_000_000, 36_000_000, 75_000_000):
                env._sync_scene(start + offset)
            self.assertEqual(clock.advance.call_args_list, [call(.036), call(.075)])
            self.assertAlmostEqual(env.maximum_obstacle_update_interval_s, .039)
            clock.advance.side_effect = RuntimeError("sequence evaluation failed")
            with self.assertRaisesRegex(RuntimeError, "sequence evaluation failed"):
                env._sync_scene(start + 110_000_000)
            self.assertEqual(env.scene_elapsed_s, .075)
            env.close()
            clock.close.assert_called_once()

    @unittest.skipUnless(sys.platform == 'win32', 'Windows named shared memory')
    def test_console_delivery_requires_game_thread_acknowledgement(self):
        client = Mock()
        client.simRunConsoleCommand.return_value = True
        clock = LocalSequenceClock(client)
        try:
            with self.assertRaisesRegex(RuntimeError, 'was not acknowledged'):
                clock.bind({'map': 'world', 'sequence': 'sequence', 'poses': {}}, 0., .4)
        finally:
            clock.close()

    def test_training_scenario_draws_are_reproducible_for_both_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            first=self.make_scene(Path(directory));second=self.make_scene(Path(directory))
            for seed in [4,None,None,None]:
                first.reset(seed=seed);second.reset(seed=seed)
                self.assertEqual(first.scene_case,second.scene_case)
                self.assertEqual(first.unwrapped.dynamic_model.start_fixed_yaw,second.unwrapped.dynamic_model.start_fixed_yaw)

    def test_checkpoint_reset_reproduces_condition_phase_and_yaw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            live, fresh = self.make_scene(root), self.make_scene(root)
            monitor = CheckpointMonitor(live)
            monitor.reset(seed=7)
            monitor.step(np.zeros(3, np.float32))
            monitor.reset()
            restored = CheckpointMonitor(fresh)
            restored.pending_reset_origin = monitor.reset_origin
            restored.reset(seed=91)
            self.assertEqual(fresh.scene_case, live.scene_case)
            self.assertEqual(fresh.unwrapped.dynamic_model.start_fixed_yaw,
                             live.unwrapped.dynamic_model.start_fixed_yaw)
            self.assertEqual(fresh.unwrapped.np_random.random(), live.unwrapped.np_random.random())
