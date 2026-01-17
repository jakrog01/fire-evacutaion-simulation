import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from config.settings import (
    MATERIALS_PROPS,
    SimulationConfig,
)


@dataclass
class SimConfig:
    grid_size: float = 0.1
    scale_factor: float = 18.59
    geometry: dict = field(default_factory=dict)

    thick_wall: int = 2
    thick_os_cut: int = 6
    margin_os: int = 2
    thick_door: int = 4

    input_dir: Path = field(default_factory=lambda: Path("input"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    materials: np.ndarray = field(default_factory=lambda: MATERIALS_PROPS.copy())

    ignition_temperature: float = 1000.0
    ignition_radius: int = 5
    ignition_point: tuple[float, float] | None = None

    def get_maps_output_dir(self) -> Path:
        return self.output_dir / "maps"

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.get_maps_output_dir().mkdir(parents=True, exist_ok=True)


def load_config(
    path: str | Path,
    sim_config: SimulationConfig | None = None,
) -> tuple[dict, SimConfig]:
    if sim_config is not None:
        input_dir = sim_config.paths.input_dir
        output_dir = sim_config.paths.output_dir
    else:
        input_dir = Path("input")
        output_dir = Path("output")

    path = Path(path)
    if not path.is_absolute():
        if not path.parent.parts:
            path = input_dir / path
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config from {path}: {e}")
        data = {}

    ignition_point = None
    geometry = data.get("geometry", {})
    if "ignition_point" in geometry:
        pos = geometry["ignition_point"].get("pos")
        if pos:
            ignition_point = (pos[0], pos[1])

    materials = MATERIALS_PROPS.copy()
    ignition_temp = 1000.0
    ignition_radius = 5

    if sim_config is not None:
        materials = sim_config.materials.to_numpy()
        ignition_temp = sim_config.ignition.initial_temperature
        ignition_radius = sim_config.ignition.ignition_radius

    config = SimConfig(
        grid_size=data.get("grid_size", 0.1),
        scale_factor=data.get("scale_factor", 18.59),
        geometry=geometry,
        thick_wall=sim_config.grid.thick_wall if sim_config else 2,
        thick_os_cut=sim_config.grid.thick_os_cut if sim_config else 6,
        margin_os=sim_config.grid.margin_os if sim_config else 2,
        thick_door=sim_config.grid.thick_door if sim_config else 4,
        input_dir=input_dir,
        output_dir=output_dir,
        materials=materials,
        ignition_temperature=ignition_temp,
        ignition_radius=ignition_radius,
        ignition_point=ignition_point,
    )

    return data, config
