from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


class MapVisualizer:
    NAV_CODES = {
        0: ("Walkable", "#FFFFFF"),
        1: ("Wall/Window", "#1a1a1a"),
        3: ("Door", "#4CAF50"),
        4: ("Staff Door", "#FF9800"),
        5: ("Exit", "#2196F3"),
        6: ("Staff Exit", "#9C27B0"),
        7: ("Obstacle (Blocking)", "#795548"),
        8: ("Obstacle (Non-blocking)", "#BCAAA4"),
        255: ("Void", "#263238"),
    }

    def __init__(
        self,
        nav: np.ndarray,
        f_block: np.ndarray,
        f_ceil: np.ndarray,
        pot_patron: np.ndarray,
        pot_staff: np.ndarray,
        fire_blocking: np.ndarray | None = None,
        output_dir: Path | None = None,
        dpi: int = 150,
        figure_format: str = "png",
    ):
        self.nav = nav
        self.f_block = f_block
        self.f_ceil = f_ceil
        self.pot_patron = pot_patron
        self.pot_staff = pot_staff
        self.fire_blocking = fire_blocking
        self.output_dir = output_dir
        self.dpi = dpi
        self.figure_format = figure_format

    def show(self) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        ax = axes.flatten()

        self._plot_nav(ax[0])
        self._plot_blocking(ax[1])
        self._plot_ceiling(ax[2])
        self._plot_patron_potential(ax[3])
        self._plot_staff_potential(ax[4])

        ax[5].axis("off")
        plt.tight_layout()
        plt.show()

    def save_all(self) -> dict[str, Path]:
        if self.output_dir is None:
            raise ValueError("Output directory must be set to save figures")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        saved_paths["nav_architecture"] = self._save_single_plot(
            self._plot_nav, "nav_architecture", figsize=(12, 10)
        )
        saved_paths["fuel_blocking"] = self._save_single_plot(
            self._plot_blocking, "fuel_blocking", figsize=(10, 8)
        )
        saved_paths["fuel_ceiling"] = self._save_single_plot(
            self._plot_ceiling, "fuel_ceiling", figsize=(10, 8)
        )
        saved_paths["potential_patron"] = self._save_single_plot(
            self._plot_patron_potential, "potential_patron", figsize=(10, 8)
        )
        saved_paths["potential_staff"] = self._save_single_plot(
            self._plot_staff_potential, "potential_staff", figsize=(10, 8)
        )

        if self.fire_blocking is not None:
            saved_paths["fire_on_architecture"] = self._save_fire_overlay()

        saved_paths["overview"] = self._save_overview()

        return saved_paths

    def _save_single_plot(
        self,
        plot_func: callable,
        name: str,
        figsize: tuple[int, int] = (10, 8),
    ) -> Path:
        """Save a single plot to file."""
        fig, ax = plt.subplots(figsize=figsize)
        plot_func(ax)
        plt.tight_layout()

        filepath = self.output_dir / f"{name}.{self.figure_format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return filepath

    def _save_overview(self) -> Path:
        fig, axes = plt.subplots(2, 3, figsize=(24, 14))
        ax = axes.flatten()

        self._plot_nav(ax[0])
        self._plot_blocking(ax[1])
        self._plot_ceiling(ax[2])
        self._plot_patron_potential(ax[3])
        self._plot_staff_potential(ax[4])

        ax[5].axis("off")
        plt.tight_layout()

        filepath = self.output_dir / f"overview.{self.figure_format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return filepath

    def _plot_nav(self, ax: plt.Axes) -> None:
        rgb_image = np.zeros((*self.nav.shape, 3), dtype=np.uint8)

        for code, (_, hex_color) in self.NAV_CODES.items():
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            mask = self.nav == code
            rgb_image[mask] = [r, g, b]

        ax.imshow(rgb_image, interpolation="nearest")
        ax.set_title("Architecture (Nav)")

        legend_patches = [
            mpatches.Patch(color=color, label=label)
            for code, (label, color) in self.NAV_CODES.items()
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            framealpha=0.9,
        )

    def _plot_blocking(self, ax: plt.Axes) -> None:
        im = ax.imshow(self.f_block, cmap="magma", vmin=0, vmax=10, origin="upper")
        ax.set_title("Blocking Fuel")
        plt.colorbar(im, ax=ax)

    def _plot_ceiling(self, ax: plt.Axes) -> None:
        im = ax.imshow(self.f_ceil, cmap="magma", vmin=0, vmax=10, origin="upper")
        ax.set_title("Ceiling Fuel")
        plt.colorbar(im, ax=ax)

    def _plot_patron_potential(self, ax: plt.Axes) -> None:
        pot_disp = self.pot_patron.copy()
        pot_disp[pot_disp == np.inf] = np.nan

        cmap_pot = plt.colormaps["jet"].copy()
        cmap_pot.set_bad(color="black")

        im = ax.imshow(pot_disp, cmap=cmap_pot, origin="upper")
        ax.set_title("Patron Potential")
        plt.colorbar(im, ax=ax)

    def _plot_staff_potential(self, ax: plt.Axes) -> None:
        pot_disp = self.pot_staff.copy()
        pot_disp[pot_disp == np.inf] = np.nan

        cmap_pot = plt.colormaps["jet"].copy()
        cmap_pot.set_bad(color="black")

        im = ax.imshow(pot_disp, cmap=cmap_pot, origin="upper")
        ax.set_title("Staff Potential")
        plt.colorbar(im, ax=ax)

    def _plot_fire_on_architecture(self, ax: plt.Axes) -> None:
        rgb_image = np.zeros((*self.nav.shape, 3), dtype=np.uint8)

        for code, (_, hex_color) in self.NAV_CODES.items():
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            mask = self.nav == code
            rgb_image[mask] = [r, g, b]

        ax.imshow(rgb_image, interpolation="nearest")

        if self.fire_blocking is not None:
            fire_display = self.fire_blocking.copy()
            fire_display[fire_display <= 20] = np.nan

            cmap_fire = plt.colormaps["hot"].copy()
            cmap_fire.set_bad(alpha=0)

            im = ax.imshow(
                fire_display,
                cmap=cmap_fire,
                alpha=0.8,
                interpolation="nearest",
                vmin=50,
                vmax=1200,
            )
            plt.colorbar(im, ax=ax, label="Temperature (°C)")

        ax.set_title("Fire Ignition on Architecture")

        legend_patches = [
            mpatches.Patch(color=color, label=label)
            for code, (label, color) in self.NAV_CODES.items()
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.15, 1),
            fontsize=7,
            framealpha=0.9,
        )

    def _save_fire_overlay(self) -> Path:
        fig, ax = plt.subplots(figsize=(14, 10))
        self._plot_fire_on_architecture(ax)
        plt.tight_layout()

        filepath = self.output_dir / f"fire_on_architecture.{self.figure_format}"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

        return filepath
