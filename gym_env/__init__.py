from gymnasium.envs.registration import register

register(
    id='airsim-env-v0',
    entry_point='gym_env.envs:AirsimGymEnv'
)