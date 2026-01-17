from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SimConfig, load_config
from .fire_matrices import FireMatrixGenerator
from .geometry import GeometryProcessor
from .maps import MapGenerator
from .potentials import PotentialFieldGenerator
from .visualization import MapVisualizer


@dataclass
class FireSimulationMaps:

    smoke_upper: np.ndarray
    smoke_lower: np.ndarray
    fire_blocking: np.ndarray
    fire_nonblocking: np.ndarray


class GridGenerator:
    def __init__(self, json_path: str, sim_config=None):
        self._data, self._config = load_config(json_path, sim_config)
        self._geometry = GeometryProcessor(self._config)
        self._map_generator = MapGenerator(self._config, self._geometry)

        self.nav: np.ndarray | None = None
        self.f_block: np.ndarray | None = None
        self.f_ceil: np.ndarray | None = None
        self.pot_patron: np.ndarray | None = None
        self.pot_staff: np.ndarray | None = None

        self.smoke_upper: np.ndarray | None = None
        self.smoke_lower: np.ndarray | None = None
        self.fire_blocking: np.ndarray | None = None
        self.fire_nonblocking: np.ndarray | None = None

    @property
    def config(self) -> SimConfig:
        return self._config

    @property
    def H(self) -> int:
        return self._map_generator.H

    @property
    def W(self) -> int:
        return self._map_generator.W

    def generate_maps(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.nav, self.f_block, self.f_ceil = self._map_generator.generate()
        return self.nav, self.f_block, self.f_ceil

    def generate_potentials(self) -> tuple[np.ndarray, np.ndarray]:
        if self.nav is None:
            raise RuntimeError("Maps must be generated before potentials")

        potential_gen = PotentialFieldGenerator(
            self._config,
            self._geometry,
            self.nav,
            self._map_generator.b_centroid,
        )

        self.pot_patron = potential_gen.generate("patron")
        self.pot_staff = potential_gen.generate("staff")

        return self.pot_patron, self.pot_staff

    def generate_fire_matrices(self) -> FireSimulationMaps:
        if self.nav is None or self.f_block is None or self.f_ceil is None:
            raise RuntimeError("Maps must be generated before fire matrices")

        fire_gen = FireMatrixGenerator(
            self._config,
            self._geometry,
            self.nav,
            self.f_block,
            self.f_ceil,
        )

        (
            self.smoke_upper,
            self.smoke_lower,
            self.fire_blocking,
            self.fire_nonblocking,
        ) = fire_gen.generate_all()

        return FireSimulationMaps(
            smoke_upper=self.smoke_upper,
            smoke_lower=self.smoke_lower,
            fire_blocking=self.fire_blocking,
            fire_nonblocking=self.fire_nonblocking,
        )

    def visualize_final(self) -> None:
        if any(
            x is None
            for x in [
                self.nav,
                self.f_block,
                self.f_ceil,
                self.pot_patron,
                self.pot_staff,
            ]
        ):
            raise RuntimeError(
                "Maps and potentials must be generated before visualization"
            )

        visualizer = MapVisualizer(
            self.nav,
            self.f_block,
            self.f_ceil,
            self.pot_patron,
            self.pot_staff,
        )
        visualizer.show()

    def save_visualizations(
        self, output_dir: Path | str | None = None
    ) -> dict[str, Path]:
        if any(
            x is None
            for x in [
                self.nav,
                self.f_block,
                self.f_ceil,
                self.pot_patron,
                self.pot_staff,
            ]
        ):
            raise RuntimeError(
                "Maps and potentials must be generated before saving visualizations"
            )

        if output_dir is None:
            output_dir = self._config.get_maps_output_dir()
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        visualizer = MapVisualizer(
            self.nav,
            self.f_block,
            self.f_ceil,
            self.pot_patron,
            self.pot_staff,
            fire_blocking=self.fire_blocking,
            output_dir=output_dir,
        )

        saved_paths = visualizer.save_all()
        print(f"Saved {len(saved_paths)} visualization(s) to {output_dir}")

        return saved_paths

    def save_matrices(self, output_dir: Path | str | None = None) -> dict[str, Path]:
        if output_dir is None:
            output_dir = self._config.output_dir / "matrices"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        matrices_to_save = {
            "nav": self.nav,
            "f_block": self.f_block,
            "f_ceil": self.f_ceil,
            "pot_patron": self.pot_patron,
            "pot_staff": self.pot_staff,
            "smoke_upper": self.smoke_upper,
            "smoke_lower": self.smoke_lower,
            "fire_blocking": self.fire_blocking,
            "fire_nonblocking": self.fire_nonblocking,
        }

        for name, matrix in matrices_to_save.items():
            if matrix is not None:
                filepath = output_dir / f"{name}.npy"
                np.save(filepath, matrix)
                saved_paths[name] = filepath

        print(f"Saved {len(saved_paths)} matrix file(s) to {output_dir}")
        return saved_paths
