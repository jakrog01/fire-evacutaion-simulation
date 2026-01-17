import numpy as np

from .config import SimConfig
from .geometry import GeometryProcessor


class MaterialCode:
    AIR = 0
    WALL = 1
    WOOD = 2
    CLOTH = 3
    FOAM = 4


class FireMatrixGenerator:

    def __init__(
        self,
        config: SimConfig,
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

        self.materials = config.materials

    def generate_all(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        material_blocking = self._generate_material_map(blocking=True)
        material_nonblocking = self._generate_material_map(blocking=False)

        smoke_upper = self._generate_smoke_layer(material_blocking, layer="upper")
        smoke_lower = self._generate_smoke_layer(material_blocking, layer="lower")

        fire_blocking = self._generate_fire_matrix(material_blocking, blocking=True)
        fire_nonblocking = self._generate_fire_matrix(
            material_nonblocking, blocking=False
        )

        fire_blocking = self._apply_ignition_point(fire_blocking, material_blocking)

        return smoke_upper, smoke_lower, fire_blocking, fire_nonblocking

    def _generate_material_map(self, blocking: bool = True) -> np.ndarray:
        material_map = np.zeros((self.H, self.W), dtype=np.uint8)
        fuel = self.f_block if blocking else self.f_ceil

        material_map[:] = MaterialCode.AIR

        material_map[self.nav == 1] = MaterialCode.WALL
        material_map[self.nav == 255] = MaterialCode.WALL 

        walkable_mask = (
            (self.nav == 0)
            | (self.nav == 3)
            | (self.nav == 4)
            | (self.nav == 5)
            | (self.nav == 6)
        )

        material_map[walkable_mask & (fuel >= 7)] = MaterialCode.FOAM
        material_map[walkable_mask & (fuel >= 5) & (fuel < 7)] = MaterialCode.CLOTH
        material_map[walkable_mask & (fuel >= 2) & (fuel < 5)] = MaterialCode.WOOD

        obstacle_mask = (self.nav == 7) | (self.nav == 8)
        material_map[obstacle_mask & (fuel >= 7)] = MaterialCode.FOAM
        material_map[obstacle_mask & (fuel >= 5) & (fuel < 7)] = MaterialCode.CLOTH
        material_map[obstacle_mask & (fuel >= 2) & (fuel < 5)] = MaterialCode.WOOD
        material_map[obstacle_mask & (fuel < 2)] = MaterialCode.WALL

        return material_map

    def _generate_smoke_layer(
        self, material_map: np.ndarray, layer: str = "upper"
    ) -> np.ndarray:
        smoke = np.zeros((self.H, self.W), dtype=np.float32)

        return smoke

    def _generate_fire_matrix(
        self, material_map: np.ndarray, blocking: bool = True
    ) -> np.ndarray:
        temperature = np.full((self.H, self.W), 20.0, dtype=np.float32)
        return temperature

    def _apply_ignition_point(
        self,
        fire_matrix: np.ndarray,
        material_map: np.ndarray,
    ) -> np.ndarray:
        if self.config.ignition_point is None:
            return fire_matrix
        ign_x, ign_y = self.config.ignition_point
        gx, gy = self.geo.to_grid((ign_x, ign_y))

        if not (0 <= gx < self.W and 0 <= gy < self.H):
            print(
                f"Warning: Ignition point ({gx}, {gy}) outside grid bounds "
                f"({self.W}, {self.H})"
            )
            return fire_matrix

        radius = self.config.ignition_radius
        init_temp = self.config.ignition_temperature 

        y_coords, x_coords = np.ogrid[: self.H, : self.W]
        dist_sq = (x_coords - gx) ** 2 + (y_coords - gy) ** 2
        ignition_mask = dist_sq <= (radius**2)

        ignition_mask = ignition_mask & (material_map != MaterialCode.WALL)

        inner_radius = radius // 2
        inner_mask = ignition_mask & (dist_sq <= (inner_radius**2))
        outer_mask = ignition_mask & ~inner_mask

        material_map[inner_mask] = MaterialCode.FOAM
        material_map[outer_mask] = MaterialCode.WOOD

        fire_matrix[ignition_mask] = init_temp

        return fire_matrix

    def get_material_properties(
        self, material_map: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flashpoints = self.materials[material_map, 0]
        burn_rates = self.materials[material_map, 1]
        smoke_rates = self.materials[material_map, 2]

        return flashpoints, burn_rates, smoke_rates
