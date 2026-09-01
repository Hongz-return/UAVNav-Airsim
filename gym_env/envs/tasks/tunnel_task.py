"""Tunnel navigation task - fly through a tunnel to reach goal.

Uses the default reward function from BaseTask (progress − obstacle − time − smoothness).
Override ``_load_reward_config()`` only if Tunnel-specific parameters differ from defaults.
"""

from .base_task import BaseTask


class TunnelTask(BaseTask):
    """
    Tunnel navigation task.

    Reward parameters are loaded from the ``[reward]`` config section via
    ``BaseTask._load_reward_config()``.  No additional task-specific reward
    logic is required; the default implementation covers progress, obstacle
    proximity, action smoothness, and terminal rewards.
    """
    pass
