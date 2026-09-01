"""Perception modules for different sensor inputs."""

from .base_perception import BasePerception
from .depth_state_perception import DepthStatePerception
from .vector_perception import VectorPerception

# Registry for easy lookup by name
PERCEPTION_REGISTRY = {
    'depth_state': DepthStatePerception,
    'vector': VectorPerception,
}


def get_perception(perception_type: str) -> type:
    """Get perception class by name."""
    if perception_type not in PERCEPTION_REGISTRY:
        raise ValueError(f"Unknown perception: {perception_type}. Available: {list(PERCEPTION_REGISTRY.keys())}")
    return PERCEPTION_REGISTRY[perception_type]
