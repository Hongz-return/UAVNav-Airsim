"""Task modules for different navigation scenarios."""

from .base_task import BaseTask
from .tunnel_task import TunnelTask
from .obs_task import ObsTask

# Registry for easy lookup by name
TASK_REGISTRY = {
    'Tunnel': TunnelTask,
    'Forest': TunnelTask,
    'Trees': TunnelTask,
    'Obs': ObsTask,
    'Tunnel_Obs': ObsTask,
}


def get_task(task_name: str) -> type:
    """Get task class by name."""
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_name]
