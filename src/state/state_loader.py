from pathlib import Path

import numpy as np

from ..config import SimulationConstants, map_fuel_capacity_grid
from .simulation_state import SimulationState


def initialize_state(
    nav: np.ndarray,
    fuel_power_floor: np.ndarray,
    fuel_power_ceil: np.ndarray | None = None,
    ambient_temp: float = SimulationConstants.AMBIENT_TEMPERATURE,
    wall_code: int = SimulationConstants.WALL_NAV_CODE,
    void_code: int = SimulationConstants.VOID_NAV_CODE,
) -> SimulationState:
    H, W = nav.shape

    temperature_floor = np.full((H, W), ambient_temp, dtype=np.float32)
    temperature_ceil = np.full((H, W), ambient_temp, dtype=np.float32)

    heat_delta_floor = np.zeros((H, W), dtype=np.float32)
    heat_delta_ceil = np.zeros((H, W), dtype=np.float32)

    burn_energy_floor = np.zeros((H,W), dtype=np.float32)
    burn_energy_ceil = np.zeros((H,W), dtype=np.float32)

    fp_floor = fuel_power_floor.astype(np.float32)
    if fuel_power_ceil is not None:
        fp_ceil = fuel_power_ceil.astype(np.float32)
    else:
        fp_ceil = np.zeros((H, W), dtype=np.float32)

    rem_floor = map_fuel_capacity_grid(fp_floor)
    rem_ceil = map_fuel_capacity_grid(fp_ceil)

    return SimulationState(
        temperature_floor=temperature_floor,
        temperature_ceil=temperature_ceil,
        fuel_power_floor=fp_floor,
        fuel_power_ceil=fp_ceil,
        fuel_remaining_floor=rem_floor,
        fuel_remaining_ceil=rem_ceil,
        heat_delta_floor=heat_delta_floor,
        heat_delta_ceil=heat_delta_ceil,
        burn_energy_floor=burn_energy_floor,
        burn_energy_ceil=burn_energy_ceil,
        nav=nav,
        height=H,
        width=W,
    )


def load_state_from_grids(
    nav: np.ndarray,
    f_block: np.ndarray,
    f_ceil: np.ndarray | None = None,
    fire_nonblocking: np.ndarray | None = None,
    wall_code: int = SimulationConstants.WALL_NAV_CODE,
    void_code: int = SimulationConstants.VOID_NAV_CODE,
) -> SimulationState:
    H, W = nav.shape

    temperature_floor = np.full(
        (H, W), SimulationConstants.AMBIENT_TEMPERATURE, dtype=np.float32
    )
    temperature_ceil = np.full(
        (H, W), SimulationConstants.AMBIENT_TEMPERATURE, dtype=np.float32
    )

    fp_floor = f_block.astype(np.float32)
    if f_ceil is not None:
        fp_ceil = f_ceil.astype(np.float32)
    else:
        fp_ceil = np.zeros((H, W), dtype=np.float32)

    different_values = np.sum((fp_floor != fp_ceil) & (fp_floor > 0) & (fp_ceil > 0))
    print(
        f"Data isolation check: {different_values} cells have different floor/ceil values"
    )

    # Log counts for fuel factors 10 and 11 on floor and ceil
    count_fp10 = int(np.sum(fp_floor == 10.0))
    count_fp11 = int(np.sum(fp_floor == 11.0))
    print(f"Fuel factor counts (floor): fp10={count_fp10}, fp11={count_fp11}")

    count_fp10_ceil = int(np.sum(fp_ceil == 10.0))
    count_fp11_ceil = int(np.sum(fp_ceil == 11.0))
    print(f"Fuel factor counts (ceil):  fp10={count_fp10_ceil}, fp11={count_fp11_ceil}")

    if fire_nonblocking is not None:
        fire = fire_nonblocking.astype(np.float32)
        mask = fire > np.float32(SimulationConstants.AMBIENT_TEMPERATURE)

        temperature_floor[mask] = fire[mask]

        ceil_mask = mask & (fp_ceil > 0.0)
        temperature_ceil[ceil_mask] = fire[ceil_mask]

    floor_capacity = map_fuel_capacity_grid(fp_floor)
    ceil_capacity = map_fuel_capacity_grid(fp_ceil)

    return SimulationState(
        temperature_floor=temperature_floor,
        temperature_ceil=temperature_ceil,
        fuel_power_floor=fp_floor,
        fuel_power_ceil=fp_ceil,
        fuel_remaining_floor=floor_capacity,
        fuel_remaining_ceil=ceil_capacity,
        heat_delta_floor=np.zeros((H, W), dtype=np.float32),
        heat_delta_ceil=np.zeros((H, W), dtype=np.float32),
        burn_energy_floor=np.zeros((H,W), dtype=np.float32),
        burn_energy_ceil=np.zeros((H,W), dtype=np.float32),
        nav=nav,
        height=H,
        width=W,
    )


def load_state_from_files(
    matrices_dir: Path | str,
    nav_file: str = "nav.npy",
    f_block_file: str = "f_block.npy",
    f_ceil_file: str = "f_ceil.npy",
    fire_file: str = "fire_nonblocking.npy",
    wall_code: int = SimulationConstants.WALL_NAV_CODE,
    void_code: int = SimulationConstants.VOID_NAV_CODE,
) -> SimulationState:
    matrices_dir = Path(matrices_dir)

    nav = np.load(matrices_dir / nav_file)
    f_block = np.load(matrices_dir / f_block_file)

    f_ceil_path = matrices_dir / f_ceil_file
    f_ceil = np.load(f_ceil_path) if f_ceil_path.exists() else None

    fire_path = matrices_dir / fire_file
    fire_nonblocking = np.load(fire_path) if fire_path.exists() else None

    return load_state_from_grids(
        nav,
        f_block,
        f_ceil,
        fire_nonblocking,
        wall_code,
        void_code,
    )


def save_state(state: SimulationState, output_dir: Path | str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = {}

    temp_floor_path = output_dir / "temperature_floor.npy"
    np.save(temp_floor_path, state.temperature_floor)
    saved["temperature_floor"] = temp_floor_path

    temp_ceil_path = output_dir / "temperature_ceil.npy"
    np.save(temp_ceil_path, state.temperature_ceil)
    saved["temperature_ceil"] = temp_ceil_path

    fuel_power_floor_path = output_dir / "fuel_power_floor.npy"
    np.save(fuel_power_floor_path, state.fuel_power_floor)
    saved["fuel_power_floor"] = fuel_power_floor_path

    fuel_power_ceil_path = output_dir / "fuel_power_ceil.npy"
    np.save(fuel_power_ceil_path, state.fuel_power_ceil)
    saved["fuel_power_ceil"] = fuel_power_ceil_path

    return saved
