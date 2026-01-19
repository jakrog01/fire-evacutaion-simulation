from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class IgnitionConfig:
    initial_temperature: float = 350.0
    ignition_radius: int = 4
    on: List[str] = field(default_factory=lambda: ["floor", "wall", "ceil"])
    x: int = 0
    y: int = 0


@dataclass
class PathConfig:
    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("sim_output"))
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
    ignition: IgnitionConfig = field(default_factory=IgnitionConfig)

    save_figures: bool = True
    figure_dpi: int = 150
    figure_format: str = "png"


def load_simulation_config(args: list[str] | None = None) -> SimulationConfig:
    try:
        import tyro

        if args is None:
            return tyro.cli(SimulationConfig)
        return tyro.cli(SimulationConfig, args=args)
    except ImportError:
        return SimulationConfig()
