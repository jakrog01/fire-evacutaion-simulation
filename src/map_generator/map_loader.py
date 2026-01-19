import json
from pathlib import Path

from ..config import GridConfig, MapLoaderConfig
from ..config.settings import load_simulation_config, SimulationConfig


def load_config(
    path: str | Path,
    ml_config: MapLoaderConfig | None = None,
) -> tuple[dict, MapLoaderConfig]:
    if ml_config is not None and getattr(ml_config, "sim_config", None) is not None:
        sim_cfg = ml_config.sim_config
    else:
        try:
            sim_cfg = load_simulation_config([])
        except Exception:
            sim_cfg = SimulationConfig()

    if sim_cfg is not None:
        input_dir = sim_cfg.paths.input_dir
        output_dir = sim_cfg.paths.output_dir
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

    geometry = data.get("geometry", {})
    
    if sim_cfg is not None:
        sim_cfg.ignition.initial_temperature = sim_cfg.ignition.initial_temperature
        sim_cfg.ignition.ignition_radius = sim_cfg.ignition.ignition_radius
        grid = ml_config.grid if (ml_config is not None) else GridConfig()

    config = MapLoaderConfig(
        geometry=geometry,
        grid=grid,
        sim_config=sim_cfg,
        grid_size=data.get("grid_size", 1.0),
        scale_factor=data.get("scale_factor", 1.0),
        input_dir=input_dir,
        output_dir=output_dir,
    )

    if "ignition_point" in geometry and sim_cfg is not None:
        pos = geometry["ignition_point"].get("pos")
        if pos and len(pos) >= 2:
            gx = int(pos[0] / config.scale_factor / config.grid_size)
            gy = int(pos[1] / config.scale_factor / config.grid_size)
            sim_cfg.ignition.x = gx
            sim_cfg.ignition.y = gy

        on = geometry["ignition_point"].get("on")
        if on:
            sim_cfg.ignition.on = on

    return data, config
