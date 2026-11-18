"""Reset native UE obstacle sequences with the same protocol in train and eval."""
import ast
import json
import sys
import time
import uuid
from pathlib import Path

import gymnasium as gym
import numpy as np

from .ue_sequence_clock import LocalSequenceClock


_TUNNEL_MAP = '/Game/RailwayTunnelPrototype/Maps/03_Experiments/L_Tunnel_MotionInteraction'


class SequenceSceneWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        if not env.unwrapped.dynamic_model.synchronous_steps:
            raise ValueError('Paired sequence experiments require synchronous_steps')
        self.runtime_script=Path(cfg.get('scene','runtime_script')).resolve()
        if not self.runtime_script.is_file(): raise FileNotFoundError(self.runtime_script)
        self.client=env.unwrapped.client
        self.port=cfg.getint('airsim','rpc_port')
        local_host = cfg.get('airsim', 'ip', fallback='').strip().lower() in {'', '127.0.0.1', 'localhost', '::1'}
        tunnel_clock = cfg.get('scene', 'map_asset', fallback='') == _TUNNEL_MAP
        self._fast_sync_enabled = (sys.platform == 'win32' and local_host and
                                   cfg.getboolean('scene', 'fast_scene_sync', fallback=tunnel_clock))
        self._sequence_clock = None
        self.fast_sync_active = False
        self.play_rate=cfg.getfloat('scene','play_rate')
        self.phases=ast.literal_eval(cfg.get('scene','training_phases_s'))
        self.yaw_degrees=cfg.getfloat('scene','training_yaw_degrees',fallback=0)
        self.episode=0
        self.record_dir=Path(cfg.get('scene','record_dir')).resolve()
        self.record_dir.mkdir(parents=True,exist_ok=True)

    def command(self, mode, **values):
        token=uuid.uuid4().hex
        root=self.runtime_script.parent
        request=root/f'scene_{self.port}_request.json'
        temporary=request.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(token=token,mode=mode,**values)),encoding='utf-8')
        temporary.replace(request)
        if not self.client.simRunConsoleCommand('py "'+str(self.runtime_script)+'"'):
            raise RuntimeError('UE rejected the sequence-control script')
        deadline=time.monotonic()+20
        response=root/f'scene_{self.port}_response.json'
        while time.monotonic()<deadline:
            if response.exists():
                result=json.loads(response.read_text('utf-8'))
                if result.get('token')==token:
                    if 'error' in result: raise RuntimeError(result['error'])
                    if result['workers']!=3 or result['carts']!=2:
                        raise RuntimeError('Obstacle count changed during the experiment')
                    return result
            time.sleep(.01)
        raise TimeoutError('No response from the reserved sequence controller')

    def reset(self, *, seed=None, options=None):
        base=self.env.unwrapped
        if seed is not None: gym.Env.reset(base,seed=seed)
        rng=base.np_random
        base.dynamic_model.scene_step_callback=None
        self.fast_sync_active = False
        scenarios=base._evaluation_scenarios
        if scenarios is not None:
            scenario=scenarios[base._evaluation_scenario_index]
            condition=scenario['condition'];phase=float(scenario['phase_s'])
        else:
            condition=str(rng.choice(['moving_forward','moving_reverse']))
            phase=float(rng.choice(self.phases))
            base.dynamic_model.start_fixed_yaw=float(np.deg2rad(rng.uniform(-self.yaw_degrees,self.yaw_degrees)))
        if condition not in ['static','moving_forward','moving_reverse']: raise ValueError(condition)
        direction='reverse' if condition=='moving_reverse' else 'forward'
        self.scene_start=self.command('prepare',direction=direction,phase_s=phase,play_rate=self.play_rate)
        obs,info=self.env.reset(seed=None,options=options)
        self.scene_case={'condition':condition,'phase_s':phase,'play_rate':self.play_rate}
        self.initial_drone_state=dict(base.dynamic_model.last_reset_state)
        self.episode+=1;self.minimum_height_m=float('inf');self.maximum_step_error_s=0.
        self.scene_start_timestamp=self.client.getMultirotorState(vehicle_name=base.dynamic_model.vehicle_name).timestamp
        self.scene_elapsed_s=0.
        self.maximum_obstacle_update_interval_s=0.
        if condition != 'static':
            if self._fast_sync_enabled and 'map' in self.scene_start and 'sequence' in self.scene_start:
                if self._sequence_clock is None:
                    self._sequence_clock = LocalSequenceClock(self.client)
                self._sequence_clock.bind(self.scene_start, phase, self.play_rate)
                self.fast_sync_active = True
            base.dynamic_model.scene_step_callback = self._sync_scene
        info['scene_case']=dict(self.scene_case)
        return obs,info

    def _sync_scene(self, timestamp):
        elapsed=(timestamp-self.scene_start_timestamp)/1e9
        if self.fast_sync_active:
            if elapsed == self.scene_elapsed_s:
                return
            self._sequence_clock.advance(elapsed)
        else:
            self.command('advance', phase_s=self.scene_case['phase_s'], elapsed_s=elapsed, play_rate=self.play_rate)
        self.maximum_obstacle_update_interval_s=max(self.maximum_obstacle_update_interval_s,elapsed-self.scene_elapsed_s)
        self.scene_elapsed_s=elapsed

    def close(self):
        try:
            if self._sequence_clock is not None:
                self._sequence_clock.close()
        finally:
            self.env.close()

    def step(self, action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        base=self.env.unwrapped
        self.minimum_height_m=min(self.minimum_height_m,3.8+info['position'][2])
        self.maximum_step_error_s=max(self.maximum_step_error_s,abs(base.dynamic_model.last_control_sim_seconds-base.dynamic_model.dt))
        if terminated or truncated:
            end=self.command('pause')
            displacement={name:float(np.linalg.norm(np.array(p)-self.scene_start['poses'][name])) for name,p in end['poses'].items()}
            if self.scene_case['condition']=='static' and max(displacement.values())>.05:
                raise RuntimeError('Static obstacle drift exceeded 0.5 mm')
            record={'episode':self.episode,**self.scene_case,'scenario_id':info.get('scenario_id') or (base.current_scenario or {}).get('scenario_id'),
                'steps':base.step_num,'reward':float(base.cumulated_episode_reward),'success':bool(info['is_success']),
                'initial_drone_state':self.initial_drone_state,
                'crash':bool(info['is_crash']),'outside':bool(info['is_not_in_workspace']),'truncated':bool(truncated),
                'terminal_position_z_up_m':info['position'],'minimum_height_m':self.minimum_height_m,
                'obstacle_simulation_elapsed_s':self.scene_elapsed_s,
                'maximum_obstacle_update_interval_s':self.maximum_obstacle_update_interval_s,
                'maximum_step_duration_error_s':self.maximum_step_error_s,'start':self.scene_start,'end':end,'actor_displacement_cm':displacement}
            with (self.record_dir/'episodes.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            info['scene_episode']=record
        return obs,reward,terminated,truncated,info
