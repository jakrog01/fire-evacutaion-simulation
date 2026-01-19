from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationState:
    temperature_floor: np.ndarray
    temperature_ceil: np.ndarray
    fuel_power_floor: np.ndarray
    fuel_power_ceil: np.ndarray
    fuel_remaining_floor: np.ndarray
    fuel_remaining_ceil: np.ndarray
    heat_delta_floor: np.ndarray
    heat_delta_ceil: np.ndarray
    burn_energy_floor: np.ndarray
    burn_energy_ceil: np.ndarray
    nav: np.ndarray
    height: int
    width: int

    def copy(self) -> "SimulationState":
        return SimulationState(
            temperature_floor=self.temperature_floor.copy(),
            temperature_ceil=self.temperature_ceil.copy(),
            fuel_power_floor=self.fuel_power_floor.copy(),
            fuel_power_ceil=self.fuel_power_ceil.copy(),
            fuel_remaining_floor=self.fuel_remaining_floor.copy(),
            fuel_remaining_ceil=self.fuel_remaining_ceil.copy(),
            heat_delta_floor=self.heat_delta_floor.copy(),
            heat_delta_ceil=self.heat_delta_ceil.copy(),
            burn_energy_floor=self.burn_energy_floor.copy(),
            burn_energy_ceil=self.burn_energy_ceil.copy(),
            nav=self.nav.copy(),
            height=self.height,
            width=self.width,
        )

    def get_max_temperature(self) -> float:
        return max(
            float(self.temperature_floor.max()),
            float(self.temperature_ceil.max()),
        )

    def get_burning_cell_count(self, threshold: float = 100.0) -> int:
        floor_burning = np.sum(self.temperature_floor > threshold)
        ceil_burning = np.sum(self.temperature_ceil > threshold)
        return int(floor_burning + ceil_burning)

    def is_wall(self, y: int, x: int, wall_code: int = 1, void_code: int = 255) -> bool:
        if not (0 <= y < self.height and 0 <= x < self.width):
            return True
        nav_val = self.nav[y, x]
        return nav_val == wall_code or nav_val == void_code
