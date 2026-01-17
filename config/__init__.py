"""Configuration module for fire evacuation simulation."""

from .settings import (
    MATERIALS_PROPS,
    GridConfig,
    IgnitionConfig,
    MaterialConfig,
    PathConfig,
    SimulationConfig,
    get_material_index,
    load_simulation_config,
)

__all__ = [
    "MATERIALS_PROPS",
    "GridConfig",
    "IgnitionConfig",
    "MaterialConfig",
    "PathConfig",
    "SimulationConfig",
    "get_material_index",
    "load_simulation_config",
]
