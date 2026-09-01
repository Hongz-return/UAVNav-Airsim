"""Obstacle environment navigation task - fly through obstacles to reach goal.

Uses the default reward function from BaseTask (progress − obstacle − time − smoothness).
Same reward structure as TunnelTask; only environment layout differs.
"""

from .base_task import BaseTask


class ObsTask(BaseTask):
    """
    Obstacle environment navigation task.

    Reward parameters are loaded from the ``[reward]`` config section via
    ``BaseTask._load_reward_config()``.  No additional task-specific reward
    logic is required.
    """
    pass
