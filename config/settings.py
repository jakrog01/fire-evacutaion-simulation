from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

MATERIALS_PROPS = np.array(
    [
        [9999.0, 0.0, 0.0], 
        [9999.0, 0.0, 0.0],  
        [300.0, 5.0, 0.5], 
        [250.0, 15.0, 2.0], 
        [180.0, 80.0, 5.0], 
    ],
    dtype=np.float32,
)


@dataclass
class MaterialConfig:
    air_flashpoint: float = 9999.0
    wall_flashpoint: float = 9999.0
    wood_flashpoint: float = 300.0
    cloth_flashpoint: float = 250.0
    foam_flashpoint: float = 180.0

    air_burn_rate: float = 0.0
    wall_burn_rate: float = 0.0
    wood_burn_rate: float = 5.0
    cloth_burn_rate: float = 15.0
    foam_burn_rate: float = 80.0

    air_smoke_rate: float = 0.0
    wall_smoke_rate: float = 0.0
    wood_smoke_rate: float = 0.5
    cloth_smoke_rate: float = 2.0
    foam_smoke_rate: float = 5.0

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [
                [self.air_flashpoint, self.air_burn_rate, self.air_smoke_rate],
                [self.wall_flashpoint, self.wall_burn_rate, self.wall_smoke_rate],
                [self.wood_flashpoint, self.wood_burn_rate, self.wood_smoke_rate],
                [self.cloth_flashpoint, self.cloth_burn_rate, self.cloth_smoke_rate],
                [self.foam_flashpoint, self.foam_burn_rate, self.foam_smoke_rate],
            ],
            dtype=np.float32,
        )


@dataclass
class IgnitionConfig:
    initial_temperature: float = 1000.0
    ignition_radius: int = 5
    primary_material: Literal["foam", "wood", "cloth"] = "foam"
    secondary_material: Literal["foam", "wood", "cloth", "none"] = "wood"


@dataclass
class GridConfig:
    """Grid generation settings."""

    grid_size: float = 0.1
    scale_factor: float = 18.59

    thick_wall: int = 2
    thick_os_cut: int = 6
    margin_os: int = 2
    thick_door: int = 4


@dataclass
class PathConfig:
    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("output"))
    config_dir: Path = field(default_factory=lambda: Path("config"))

    geometry_file: str = "stationnc.json"
    maps_subdir: str = "maps"

    def get_geometry_path(self) -> Path:
        return self.input_dir / self.geometry_file

    def get_maps_output_dir(self) -> Path:
        return self.output_dir / self.maps_subdir

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.get_maps_output_dir().mkdir(parents=True, exist_ok=True)


@dataclass
class SimulationConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    materials: MaterialConfig = field(default_factory=MaterialConfig)
    ignition: IgnitionConfig = field(default_factory=IgnitionConfig)

    save_figures: bool = True
    figure_dpi: int = 150
    figure_format: str = "png"


def get_material_index(material_name: str) -> int:
    mapping = {
        "air": 0,
        "wall": 1,
        "brick": 1,
        "wood": 2,
        "cloth": 3,
        "foam": 4,
    }
    return mapping.get(material_name.lower(), 0)


def load_simulation_config() -> SimulationConfig:
    try:
        import tyro

        return tyro.cli(SimulationConfig)
    except ImportError:
        return SimulationConfig()
