import numpy as np

from src.config.fire_constants import FUEL_POWER_MAX

from ..config import SimulationConstants
from .geometry import GeometryProcessor
from .map_loader import MapLoaderConfig


class FireMatrixGenerator:

    def __init__(
        self,
        config: MapLoaderConfig,
        geometry: GeometryProcessor,
        nav: np.ndarray,
        f_block: np.ndarray,
        f_ceil: np.ndarray,
    ):
        self.config = config
        self.geo = geometry
        self.nav = nav
        self.f_block = f_block
        self.f_ceil = f_ceil
        self.H, self.W = nav.shape

    def generate_all(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        fire_blocking = self._generate_fire_matrix(blocking=True)
        fire_nonblocking = self._generate_fire_matrix(blocking=False)

        return fire_blocking, fire_nonblocking

    def _generate_fire_matrix(self, blocking: bool = True) -> np.ndarray:
        temperature = np.full((self.H, self.W), 20.0, dtype=np.float32)
        return apply_ignition(
            temperature,
            self.config,
            self.nav,
            self.f_block,
            self.f_ceil,
            void_code=SimulationConstants.VOID_NAV_CODE,
            wall_code=SimulationConstants.WALL_NAV_CODE,
        )


def apply_ignition(
    temperature: np.ndarray,
    config: MapLoaderConfig,
    nav: np.ndarray,
    f_block: np.ndarray,
    f_ceil: np.ndarray,
    void_code: int = SimulationConstants.VOID_NAV_CODE,
    wall_code: int = SimulationConstants.WALL_NAV_CODE,
) -> np.ndarray:
    H, W = temperature.shape
    result = temperature.copy()

    fuel_value = float(FUEL_POWER_MAX)
    base_temp = float(config.ignition_temperature)

    for dy in range(-config.ignition_radius, config.ignition_radius + 1):
        for dx in range(-config.ignition_radius, config.ignition_radius + 1):
            y = config.ignition_y + dy
            x = config.ignition_x + dx

            if not (0 <= y < H and 0 <= x < W):
                continue

            dist_sq = dy * dy + dx * dx
            if dist_sq > config.ignition_radius * config.ignition_radius:
                continue

            allowed_layers = config.ignition_on
            is_wall = nav[y, x] == wall_code
            is_interior = (nav[y, x] != wall_code) and (nav[y, x] != void_code)

            if "wall" in allowed_layers and is_wall:
                if fuel_value > float(f_block[y, x]):
                    f_block[y, x] = np.float32(fuel_value)
                if fuel_value > float(f_ceil[y, x]):
                    f_ceil[y, x] = np.float32(fuel_value)

            if ("floor" in allowed_layers) and is_interior:
                if fuel_value > float(f_block[y, x]):
                    f_block[y, x] = np.float32(fuel_value)

            if ("ceil" in allowed_layers) and is_interior:
                if fuel_value > float(f_ceil[y, x]):
                    f_ceil[y, x] = np.float32(fuel_value)

            has_floor_fuel = float(f_block[y, x]) > 0.0
            has_ceil_fuel = float(f_ceil[y, x]) > 0.0
            if not (has_floor_fuel or has_ceil_fuel):
                continue

            dist = dist_sq**0.5
            falloff = max(0.0, 1.0 - (dist / (config.ignition_radius + 1.0)))
            temp_val = base_temp * falloff

            if temp_val > float(result[y, x]):
                result[y, x] = np.float32(temp_val)

    return result
