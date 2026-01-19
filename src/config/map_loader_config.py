from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from .settings import SimulationConfig


@dataclass
class GridConfig:
    thick_wall: int = 2
    thick_os_cut: int = 6
    margin_os: int = 2
    thick_door: int = 4


@dataclass
class MapLoaderConfig:
    geometry: dict = field(default_factory=dict)
    grid: GridConfig = field(default_factory=GridConfig)
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)

    grid_size: float = 1.0
    scale_factor: float = 1.0

    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    @property
    def thick_wall(self) -> int:
        return self.grid.thick_wall

    @property
    def thick_os_cut(self) -> int:
        return self.grid.thick_os_cut

    @property
    def margin_os(self) -> int:
        return self.grid.margin_os

    @property
    def thick_door(self) -> int:
        return self.grid.thick_door

    @property
    def ignition_temperature(self) -> float:
        return self.sim_config.ignition.initial_temperature
       

    @property
    def ignition_radius(self) -> int:
        return self.sim_config.ignition.ignition_radius

    @property
    def ignition_x(self) -> int:
        return self.sim_config.ignition.x
    
    @property
    def ignition_y(self) -> int:
        return self.sim_config.ignition.y

    @property
    def ignition_on(self) -> List[str]:
        if self.sim_config is not None and self.sim_config.ignition.on is not None:
            return self.sim_config.ignition.on
        return ["floor"]

    def get_maps_output_dir(self) -> Path:
        return self.output_dir / "maps"

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.get_maps_output_dir().mkdir(parents=True, exist_ok=True)
