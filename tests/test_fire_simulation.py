"""
Fire Simulation Tests - Using Actual Station Map.

Tests fire propagation physics on the real station layout from output/matrices/.
Generates GIF visualization showing floor and ceiling temperature over time.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.constants import (
    AMBIENT_TEMPERATURE,
    DT,
    MAX_TEMPERATURE,
    VOID_NAV_CODE,
)
from src.physics.fire_kernel import update_fire_kernel
from src.state import SimulationState, load_state_from_files

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
MATRICES_DIR = PROJECT_ROOT / "sim_output" / "matrices"
TEST_OUTPUT_DIR = PROJECT_ROOT / "test_output"


# =============================================================================
# LOAD STATE FROM ACTUAL MAP
# =============================================================================


def load_station_state() -> SimulationState:
    """Load simulation state from the actual station map matrices."""
    if not MATRICES_DIR.exists():
        raise FileNotFoundError(
            f"Matrices directory not found: {MATRICES_DIR}\n"
            "Run main.py first to generate the maps."
        )

    state = load_state_from_files(MATRICES_DIR)
    print(f"Loaded station map: {state.height} x {state.width}")
    print(f"Initial max temperature: {state.temperature_floor.max():.1f}°C")
    return state


def run_simulation_step(state: SimulationState, dt: float = DT) -> None:
    update_fire_kernel(
        state.temperature_floor,
        state.temperature_ceil,
        state.fuel_power_floor,
        state.fuel_power_ceil,
        state.fuel_remaining_floor,
        state.fuel_remaining_ceil,
        state.heat_delta_floor,
        state.heat_delta_ceil,
        state.burn_energy_floor,
        state.burn_energy_ceil,
        state.nav,
        DT,
    )


def run_simulation(state: SimulationState, num_steps: int, dt: float = DT) -> None:
    """Run simulation for specified number of steps."""
    for _ in range(num_steps):
        run_simulation_step(state, dt)


# =============================================================================
# PHYSICS TESTS
# =============================================================================


class TestFirePhysics:
    """Tests for fire propagation physics using actual station map."""

    def setup_method(self):
        """Load station state for each test (pytest compatible)."""
        self.state = load_station_state()

    def setup(self):
        """Load station state (for manual running)."""
        self.state = load_station_state()

    def test_fire_spreads_from_ignition(self):
        """Test that fire spreads from initial ignition point."""
        initial_hot_cells = np.sum(self.state.temperature_floor > 100)

        # Run 500 steps (0.5 seconds)
        run_simulation(self.state, 500)

        final_hot_cells = np.sum(self.state.temperature_floor > 100)

        assert (
            final_hot_cells >= initial_hot_cells
        ), f"Fire should spread: {initial_hot_cells} → {final_hot_cells} hot cells"
        print(f"Hot cells: {initial_hot_cells} → {final_hot_cells}")

    def test_temperature_increases(self):
        """Test that temperature increases during simulation."""
        initial_max = self.state.temperature_floor.max()

        run_simulation(self.state, 500)

        final_max = self.state.temperature_floor.max()

        # Temperature should increase or stay at max
        assert (
            final_max >= initial_max
        ), f"Temperature should not decrease: {initial_max:.1f}°C → {final_max:.1f}°C"
        print(f"Max temperature: {initial_max:.1f}°C → {final_max:.1f}°C")

    def test_fuel_consumption(self):
        """Test that fuel is consumed as fire burns."""
        pytest.skip("Fuel consumability removed from model")

    def test_vertical_heat_transfer(self):
        """Test that heat transfers from floor to ceiling."""
        # Find a hot spot on floor
        hot_y, hot_x = np.unravel_index(
            np.argmax(self.state.temperature_floor), self.state.temperature_floor.shape
        )

        initial_ceil_temp = self.state.temperature_ceil[hot_y, hot_x]

        run_simulation(self.state, 1000)

        final_ceil_temp = self.state.temperature_ceil[hot_y, hot_x]

        assert (
            final_ceil_temp >= initial_ceil_temp
        ), f"Ceiling should heat up: {initial_ceil_temp:.1f}°C → {final_ceil_temp:.1f}°C"
        print(
            f"Ceiling temp at hot spot: {initial_ceil_temp:.1f}°C → {final_ceil_temp:.1f}°C"
        )

    def test_walls_block_fire(self):
        """Test that walls/voids remain at ambient temperature."""
        run_simulation(self.state, 1000)

        void_mask = self.state.nav == VOID_NAV_CODE
        if np.any(void_mask):
            void_temps = self.state.temperature_floor[void_mask]
            assert np.all(
                void_temps <= AMBIENT_TEMPERATURE + 5
            ), "Void areas should stay near ambient"

    def test_temperature_capped(self):
        """Test that temperature doesn't exceed MAX_TEMPERATURE."""
        run_simulation(self.state, 2000)

        assert (
            self.state.temperature_floor.max() <= MAX_TEMPERATURE
        ), f"Floor temp exceeds max: {self.state.temperature_floor.max():.1f}°C"
        assert (
            self.state.temperature_ceil.max() <= MAX_TEMPERATURE
        ), f"Ceiling temp exceeds max: {self.state.temperature_ceil.max():.1f}°C"


# =============================================================================
# TEMPERATURE COLORMAP
# =============================================================================


def create_fire_colormap():
    """Create colormap: dark blue (cold) → red → orange → yellow → white (hot)."""
    colors = [
        (0.00, "#1a1a2e"),  # 20°C - Dark blue (ambient)
        (0.05, "#16213e"),  # ~95°C - Deep blue
        (0.10, "#0f3460"),  # ~170°C - Blue
        (0.20, "#e94560"),  # ~315°C - Red (ignition zone)
        (0.35, "#ff6b35"),  # ~540°C - Orange
        (0.55, "#ffc93c"),  # ~835°C - Yellow
        (0.80, "#ffffff"),  # ~1200°C - White
        (1.00, "#ffffff"),  # 1500°C - White (max)
    ]

    return LinearSegmentedColormap.from_list("fire", colors)


# =============================================================================
# GIF GENERATION
# =============================================================================


def generate_fire_gif(
    output_path: Path,
    simulation_seconds: float = 100.0,
    frame_interval: int = 100,
    frame_duration_ms: int = 200,
) -> dict:
    """
    Generate GIF showing fire propagation on actual station map.

    Args:
        output_path: Where to save the GIF
        simulation_seconds: Total simulation time
        frame_interval: Capture frame every N steps
        frame_duration_ms: Duration per frame in GIF (ms)

    Returns:
        Statistics dictionary
    """
    print(f"\n{'='*60}")
    print("FIRE SIMULATION - STATION MAP")
    print(f"{'='*60}")

    # Load state
    state = load_station_state()

    # Calculate steps from simulation time and dt
    total_steps = int(simulation_seconds / DT)
    num_frames = total_steps // frame_interval

    print("\nConfiguration:")
    print(f"  Simulation time: {simulation_seconds}s")
    print(f"  Time step (dt): {DT}s ({1/DT:.0f} Hz)")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Frames: {num_frames} (every {frame_interval} steps)")
    print(f"  Frame duration: {frame_duration_ms}ms")

    # Collect frames and track per-burning-cell metrics
    frames_floor = []
    frames_ceil = []
    times = []
    stats = {
        "times": [],
        "max_temp_floor": [],
        "max_temp_ceil": [],
        "hot_cells_floor": [],
        "hot_cells_ceil": [],
        # per-step series for fuel & burning-cell metrics
        "fuel_energy_cumulative": [],
        "burning_cells": [],
        "mean_fuel_per_burning_cell": [],
        "mean_delta_temp_floor": [],
        "mean_delta_temp_ceil": [],
    }

    # Parameters / accumulators
    BURNING_TEMP = 100.0  # threshold to consider a cell "burning"
    total_fuel_energy = 0.0  # cumulative integrated fuel power (arb. units)
    burning_cell_seconds = 0.0  # sum over time of burning cells * dt
    ever_burned_mask = np.zeros_like(state.temperature_floor, dtype=bool)
    total_delta_temp_floor = 0.0
    total_delta_temp_ceil = 0.0

    print("\nRunning simulation...")

    # Run simulation step-by-step to accumulate metrics
    # Track per-layer accumulators (floor / ceil) and combined
    total_fuel_energy_floor = 0.0
    total_fuel_energy_ceil = 0.0
    burning_cell_seconds_floor = 0.0
    burning_cell_seconds_ceil = 0.0

    for step in range(total_steps + 1):
        # Snapshot before the step (used to detect burning cells at step start)
        before_floor = state.temperature_floor.copy()
        before_ceil = state.temperature_ceil.copy()

        # Per-layer burning masks
        floor_burning_mask = before_floor > BURNING_TEMP
        ceil_burning_mask = before_ceil > BURNING_TEMP
        burning_mask = floor_burning_mask | ceil_burning_mask

        burning_count_total = int(np.sum(burning_mask))
        burning_count_floor = int(np.sum(floor_burning_mask))
        burning_count_ceil = int(np.sum(ceil_burning_mask))

        # Log counts of fuel factors 10 and 11 every step
        count_fp10_floor = int(np.sum(state.fuel_power_floor == 10.0))
        count_fp11_floor = int(np.sum(state.fuel_power_floor == 11.0))
        count_fp10_ceil = int(np.sum(state.fuel_power_ceil == 10.0))
        count_fp11_ceil = int(np.sum(state.fuel_power_ceil == 11.0))

        # Integrate fuel_power over burning cells for each layer
        fuel_power_floor = (
            float(state.fuel_power_floor[floor_burning_mask].sum())
            if burning_count_floor > 0
            else 0.0
        )
        fuel_power_ceil = (
            float(state.fuel_power_ceil[ceil_burning_mask].sum())
            if burning_count_ceil > 0
            else 0.0
        )

        fuel_step_floor = fuel_power_floor * DT
        fuel_step_ceil = fuel_power_ceil * DT
        fuel_step_total = fuel_step_floor + fuel_step_ceil

        total_fuel_energy_floor += float(fuel_step_floor)
        total_fuel_energy_ceil += float(fuel_step_ceil)
        total_fuel_energy = total_fuel_energy_floor + total_fuel_energy_ceil

        burning_cell_seconds_floor += burning_count_floor * DT
        burning_cell_seconds_ceil += burning_count_ceil * DT
        burning_cell_seconds += burning_count_total * DT

        ever_burned_mask |= burning_mask
        # Keep per-layer ever-burned masks for layer-specific counts
        if "ever_burned_mask_floor" not in locals():
            ever_burned_mask_floor = np.zeros_like(state.temperature_floor, dtype=bool)
            ever_burned_mask_ceil = np.zeros_like(state.temperature_floor, dtype=bool)
        ever_burned_mask_floor |= floor_burning_mask
        ever_burned_mask_ceil |= ceil_burning_mask

        # Advance the simulation for this step (unless we're at final snapshot)
        if step < total_steps:
            run_simulation_step(state)

            # Measure positive temperature increases on cells that were burning at step start (per-layer)
            if burning_count_floor > 0:
                after_floor = state.temperature_floor
                delta_floor = (
                    after_floor[floor_burning_mask] - before_floor[floor_burning_mask]
                )
                positive_delta_floor = np.clip(delta_floor, a_min=0.0, a_max=None)
                total_delta_temp_floor += float(positive_delta_floor.sum())
            else:
                positive_delta_floor = np.array([])

            if burning_count_ceil > 0:
                after_ceil = state.temperature_ceil
                delta_ceil = (
                    after_ceil[ceil_burning_mask] - before_ceil[ceil_burning_mask]
                )
                positive_delta_ceil = np.clip(delta_ceil, a_min=0.0, a_max=None)
                total_delta_temp_ceil += float(positive_delta_ceil.sum())
            else:
                positive_delta_ceil = np.array([])

        # Record frames and per-frame stats at configured intervals
        if step % frame_interval == 0:
            t = step * DT

            frames_floor.append(state.temperature_floor.copy())
            frames_ceil.append(state.temperature_ceil.copy())
            times.append(t)

            stats["times"].append(t)
            stats["max_temp_floor"].append(float(state.temperature_floor.max()))
            stats["max_temp_ceil"].append(float(state.temperature_ceil.max()))
            stats["hot_cells_floor"].append(
                int(np.sum(state.temperature_floor > BURNING_TEMP))
            )
            stats["hot_cells_ceil"].append(
                int(np.sum(state.temperature_ceil > BURNING_TEMP))
            )

            # per-layer + total counts
            stats.setdefault("burning_cells_floor", []).append(burning_count_floor)
            stats.setdefault("burning_cells_ceil", []).append(burning_count_ceil)
            stats.setdefault("burning_cells_total", []).append(burning_count_total)

            # cumulative fuel per-layer & combined
            stats.setdefault("fuel_energy_cumulative_floor", []).append(
                float(total_fuel_energy_floor)
            )
            stats.setdefault("fuel_energy_cumulative_ceil", []).append(
                float(total_fuel_energy_ceil)
            )
            stats.setdefault("fuel_energy_cumulative_total", []).append(
                float(total_fuel_energy)
            )

            # Per-burning-cell means for this step (0 when no burning cells) per-layer + combined
            if burning_count_floor > 0:
                stats.setdefault("mean_fuel_per_burning_cell_floor", []).append(
                    float(fuel_step_floor / burning_count_floor)
                )
                stats.setdefault("mean_delta_temp_floor", []).append(
                    float(
                        positive_delta_floor.mean()
                        if positive_delta_floor.size
                        else 0.0
                    )
                )
            else:
                stats.setdefault("mean_fuel_per_burning_cell_floor", []).append(0.0)
                stats.setdefault("mean_delta_temp_floor", []).append(0.0)

            if burning_count_ceil > 0:
                stats.setdefault("mean_fuel_per_burning_cell_ceil", []).append(
                    float(fuel_step_ceil / burning_count_ceil)
                )
                stats.setdefault("mean_delta_temp_ceil", []).append(
                    float(
                        positive_delta_ceil.mean() if positive_delta_ceil.size else 0.0
                    )
                )
            else:
                stats.setdefault("mean_fuel_per_burning_cell_ceil", []).append(0.0)
                stats.setdefault("mean_delta_temp_ceil", []).append(0.0)

            # combined mean fuel per burning cell (total burning cells)
            if burning_count_total > 0:
                stats.setdefault("mean_fuel_per_burning_cell_total", []).append(
                    float(fuel_step_total / burning_count_total)
                )
            else:
                stats.setdefault("mean_fuel_per_burning_cell_total", []).append(0.0)

            frame_num = step // frame_interval
            if frame_num % 10 == 0:
                mean_fuel_floor = stats.get("mean_fuel_per_burning_cell_floor", [0.0])[
                    -1
                ]
                mean_fuel_ceil = stats.get("mean_fuel_per_burning_cell_ceil", [0.0])[-1]
                mean_fuel_total = stats.get("mean_fuel_per_burning_cell_total", [0.0])[
                    -1
                ]
                mean_delta_floor = stats.get("mean_delta_temp_floor", [0.0])[-1]
                mean_delta_ceil = stats.get("mean_delta_temp_ceil", [0.0])[-1]

                print(
                    f"  t={t:.2f}s | Floor: {stats['max_temp_floor'][-1]:.0f}°C | "
                    f"Ceil: {stats['max_temp_ceil'][-1]:.0f}°C | "
                    f"Hot floor/ceil: {stats['hot_cells_floor'][-1]}/{stats['hot_cells_ceil'][-1]} | "
                    f"Fuel cum (f/c/t): {stats['fuel_energy_cumulative_floor'][-1]:.2f}/{stats['fuel_energy_cumulative_ceil'][-1]:.2f}/{stats['fuel_energy_cumulative_total'][-1]:.2f} | "
                    f"Mean fuel (f/c/t): {mean_fuel_floor:.3f}/{mean_fuel_ceil:.3f}/{mean_fuel_total:.3f} | "
                    f"Mean ΔT (f/c): {mean_delta_floor:.2f}/{mean_delta_ceil:.2f} :"
                    f" Fuel factors - floor: fp10={count_fp10_floor}, fp11={count_fp11_floor} | ceil: fp10={count_fp10_ceil}, fp11={count_fp11_ceil}"
                )

    # After simulation, compute summary metrics for burning cells
    ever_burned_cells = int(np.sum(ever_burned_mask))
    avg_fuel_per_burned_cell_total = (
        total_fuel_energy / ever_burned_cells if ever_burned_cells > 0 else 0.0
    )
    avg_fuel_per_burned_cell_per_sec = (
        total_fuel_energy / burning_cell_seconds if burning_cell_seconds > 0 else 0.0
    )

    total_delta_temp = total_delta_temp_floor + total_delta_temp_ceil
    avg_delta_temp_per_burned_cell_total = (
        total_delta_temp / ever_burned_cells if ever_burned_cells > 0 else 0.0
    )
    avg_delta_temp_per_burned_cell_per_sec = (
        total_delta_temp / burning_cell_seconds if burning_cell_seconds > 0 else 0.0
    )

    # Compute time-averaged (per-frame) means excluding frames with zero burning cells
    def _time_avg(series, counts):
        vals = [v for v, c in zip(series, counts) if c > 0]
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    timeavg_mean_fuel_total = _time_avg(
        stats.get("mean_fuel_per_burning_cell_total", []),
        stats.get("burning_cells_total", []),
    )
    timeavg_mean_fuel_floor = _time_avg(
        stats.get("mean_fuel_per_burning_cell_floor", []),
        stats.get("burning_cells_floor", []),
    )
    timeavg_mean_fuel_ceil = _time_avg(
        stats.get("mean_fuel_per_burning_cell_ceil", []),
        stats.get("burning_cells_ceil", []),
    )

    timeavg_mean_delta_floor = _time_avg(
        stats.get("mean_delta_temp_floor", []), stats.get("burning_cells_floor", [])
    )
    timeavg_mean_delta_ceil = _time_avg(
        stats.get("mean_delta_temp_ceil", []), stats.get("burning_cells_ceil", [])
    )

    # Per-layer ever-burned counts (safe if not created)
    ever_burned_cells_floor = (
        int(np.sum(ever_burned_mask_floor))
        if "ever_burned_mask_floor" in locals()
        else 0
    )
    ever_burned_cells_ceil = (
        int(np.sum(ever_burned_mask_ceil)) if "ever_burned_mask_ceil" in locals() else 0
    )

    avg_fuel_per_burned_cell_floor_total = (
        total_fuel_energy_floor / ever_burned_cells_floor
        if ever_burned_cells_floor > 0
        else 0.0
    )
    avg_fuel_per_burned_cell_floor_per_sec = (
        total_fuel_energy_floor / burning_cell_seconds_floor
        if burning_cell_seconds_floor > 0
        else 0.0
    )

    avg_fuel_per_burned_cell_ceil_total = (
        total_fuel_energy_ceil / ever_burned_cells_ceil
        if ever_burned_cells_ceil > 0
        else 0.0
    )
    avg_fuel_per_burned_cell_ceil_per_sec = (
        total_fuel_energy_ceil / burning_cell_seconds_ceil
        if burning_cell_seconds_ceil > 0
        else 0.0
    )

    stats.update(
        {
            "total_fuel_energy": float(total_fuel_energy),
            "total_fuel_energy_floor": float(total_fuel_energy_floor),
            "total_fuel_energy_ceil": float(total_fuel_energy_ceil),
            "burning_cell_seconds": float(burning_cell_seconds),
            "burning_cell_seconds_floor": float(burning_cell_seconds_floor),
            "burning_cell_seconds_ceil": float(burning_cell_seconds_ceil),
            "ever_burned_cells": ever_burned_cells,
            "ever_burned_cells_floor": ever_burned_cells_floor,
            "ever_burned_cells_ceil": ever_burned_cells_ceil,
            "avg_fuel_per_burned_cell_total": float(avg_fuel_per_burned_cell_total),
            "avg_fuel_per_burned_cell_per_sec": float(avg_fuel_per_burned_cell_per_sec),
            "avg_fuel_per_burned_cell_floor_total": float(
                avg_fuel_per_burned_cell_floor_total
            ),
            "avg_fuel_per_burned_cell_floor_per_sec": float(
                avg_fuel_per_burned_cell_floor_per_sec
            ),
            "avg_fuel_per_burned_cell_ceil_total": float(
                avg_fuel_per_burned_cell_ceil_total
            ),
            "avg_fuel_per_burned_cell_ceil_per_sec": float(
                avg_fuel_per_burned_cell_ceil_per_sec
            ),
            "total_delta_temp_floor": float(total_delta_temp_floor),
            "total_delta_temp_ceil": float(total_delta_temp_ceil),
            "avg_delta_temp_per_burned_cell_total": float(
                avg_delta_temp_per_burned_cell_total
            ),
            "avg_delta_temp_per_burned_cell_per_sec": float(
                avg_delta_temp_per_burned_cell_per_sec
            ),
        }
    )

    # Attach time-averaged per-frame means to stats (exclude frames with zero burning cells)
    stats["timeavg_mean_fuel_per_burning_cell_total"] = float(timeavg_mean_fuel_total)
    stats["timeavg_mean_fuel_per_burning_cell_floor"] = float(timeavg_mean_fuel_floor)
    stats["timeavg_mean_fuel_per_burning_cell_ceil"] = float(timeavg_mean_fuel_ceil)
    stats["timeavg_mean_delta_temp_floor"] = float(timeavg_mean_delta_floor)
    stats["timeavg_mean_delta_temp_ceil"] = float(timeavg_mean_delta_ceil)

    # Create animation
    print("\nGenerating GIF...")

    fire_cmap = create_fire_colormap()
    vmin, vmax = AMBIENT_TEMPERATURE, MAX_TEMPERATURE

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor("#0a0a0a")

    im_floor = axes[0].imshow(frames_floor[0], cmap=fire_cmap, vmin=vmin, vmax=vmax)
    im_ceil = axes[1].imshow(frames_ceil[0], cmap=fire_cmap, vmin=vmin, vmax=vmax)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#0a0a0a")

    axes[0].set_title(
        "FLOOR LEVEL", fontsize=16, color="white", fontweight="bold", pad=10
    )
    axes[1].set_title(
        "CEILING LEVEL", fontsize=16, color="white", fontweight="bold", pad=10
    )

    # Add time text on each panel (more visible)
    time_text_floor = axes[0].text(
        0.02,
        0.98,
        "t = 0.00s",
        transform=axes[0].transAxes,
        fontsize=14,
        color="yellow",
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
    )
    time_text_ceil = axes[1].text(
        0.02,
        0.98,
        "t = 0.00s",
        transform=axes[1].transAxes,
        fontsize=14,
        color="yellow",
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
    )

    # Max temp text on each panel
    temp_text_floor = axes[0].text(
        0.98,
        0.02,
        f"Max: {frames_floor[0].max():.0f}°C",
        transform=axes[0].transAxes,
        fontsize=12,
        color="white",
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
    )
    temp_text_ceil = axes[1].text(
        0.98,
        0.02,
        f"Max: {frames_ceil[0].max():.0f}°C",
        transform=axes[1].transAxes,
        fontsize=12,
        color="white",
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="black", alpha=0.7),
    )

    # Colorbars with temperature levels
    for ax, im in [(axes[0], im_floor), (axes[1], im_ceil)]:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([20, 150, 300, 500, 800, 1000, 1500])
        cbar.set_ticklabels(
            ["20°C", "150°C", "300°C", "500°C", "800°C", "1000°C", "1500°C"]
        )
        cbar.ax.yaxis.set_tick_params(color="white", labelsize=9)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    # Title with time info
    title = fig.suptitle(
        f"SIMULATION TIME: 0.00s\n"
        f"Floor Max: {frames_floor[0].max():.0f}°C | Ceiling Max: {frames_ceil[0].max():.0f}°C",
        fontsize=14,
        color="white",
        y=0.98,
        fontweight="bold",
    )

    plt.tight_layout(rect=(0, 0, 1, 0.93))

    def update(frame_idx):
        im_floor.set_array(frames_floor[frame_idx])
        im_ceil.set_array(frames_ceil[frame_idx])

        t = times[frame_idx]
        max_f = stats["max_temp_floor"][frame_idx]
        max_c = stats["max_temp_ceil"][frame_idx]

        # Update time text on panels
        time_text_floor.set_text(f"t = {t:.2f}s")
        time_text_ceil.set_text(f"t = {t:.2f}s")

        # Update max temp text
        temp_text_floor.set_text(f"Max: {max_f:.0f}°C")
        temp_text_ceil.set_text(f"Max: {max_c:.0f}°C")

        title.set_text(
            f"SIMULATION TIME: {t:.2f}s\n"
            f"Floor Max: {max_f:.0f}°C | Ceiling Max: {max_c:.0f}°C"
        )

        return [
            im_floor,
            im_ceil,
            title,
            time_text_floor,
            time_text_ceil,
            temp_text_floor,
            temp_text_ceil,
        ]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames_floor), interval=frame_duration_ms, blit=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(1000.0 / frame_duration_ms)))
    anim.save(str(output_path), writer="pillow", fps=fps)
    plt.close(fig)

    print(f"\n✓ GIF saved: {output_path}")
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Final time: {stats['times'][-1]:.2f}s")
    print(f"Final floor max temp: {stats['max_temp_floor'][-1]:.0f}°C")
    print(f"Final ceiling max temp: {stats['max_temp_ceil'][-1]:.0f}°C")
    print(
        f"Final hot cells (>100°C) floor/ceil: {stats['hot_cells_floor'][-1]}/{stats['hot_cells_ceil'][-1]}"
    )

    # Fuel / burning-cell summary (floor / ceil / total)
    print(
        f"Total fuel energy (cum.) floor/ceil/total: {stats['total_fuel_energy_floor']:.2f}/{stats['total_fuel_energy_ceil']:.2f}/{stats['total_fuel_energy']:.2f}"
    )
    print(
        f"Ever burned cells floor/ceil/unique: {stats['ever_burned_cells_floor']}/{stats['ever_burned_cells_ceil']}/{stats['ever_burned_cells']}"
    )

    print(
        f"Avg fuel per burned cell (floor total / per sec): {stats['avg_fuel_per_burned_cell_floor_total']:.6f} / {stats['avg_fuel_per_burned_cell_floor_per_sec']:.8f} (arb. units)"
    )
    print(
        f"Avg fuel per burned cell (ceil total / per sec):  {stats['avg_fuel_per_burned_cell_ceil_total']:.6f} / {stats['avg_fuel_per_burned_cell_ceil_per_sec']:.8f} (arb. units)"
    )
    print(
        f"Avg fuel per burned cell (combined total / per sec): {stats['avg_fuel_per_burned_cell_total']:.6f} / {stats['avg_fuel_per_burned_cell_per_sec']:.8f} (arb. units)"
    )

    print(
        f"Total ΔT contributed (floor/ceil): {stats['total_delta_temp_floor']:.6f}°C / {stats['total_delta_temp_ceil']:.6f}°C"
    )
    print(
        f"Avg ΔT per burned cell (combined total / per sec): {stats['avg_delta_temp_per_burned_cell_total']:.6f}°C / {stats['avg_delta_temp_per_burned_cell_per_sec']:.8f}°C/s"
    )

    # Time-averaged (per-frame) means (exclude frames with zero burning cells)
    print(
        f"Time-avg mean fuel per burning cell (floor/ceil/total): {stats['timeavg_mean_fuel_per_burning_cell_floor']:.6f} / {stats['timeavg_mean_fuel_per_burning_cell_ceil']:.6f} / {stats['timeavg_mean_fuel_per_burning_cell_total']:.6f} (arb. units)"
    )
    print(
        f"Time-avg mean ΔT per burning cell (floor/ceil): {stats['timeavg_mean_delta_temp_floor']:.6f} / {stats['timeavg_mean_delta_temp_ceil']:.6f} °C"
    )

    return stats


def generate_stats_plot(stats: dict, output_path: Path):
    """Generate plot showing simulation statistics over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")

    times = stats["times"]

    # Temperature
    ax = axes[0, 0]
    ax.plot(times, stats["max_temp_floor"], "r-", lw=2, label="Floor")
    ax.plot(times, stats["max_temp_ceil"], "orange", lw=2, label="Ceiling")
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Max Temperature (°C)", color="white")
    ax.set_title("Maximum Temperature", color="white", fontweight="bold")
    ax.legend(facecolor="#2a2a4e", labelcolor="white")
    ax.set_facecolor("#2a2a4e")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.3)

    # Hot cells
    ax = axes[0, 1]
    ax.plot(times, stats["hot_cells_floor"], "r-", lw=2)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title("Hot Cells (>100°C)", color="white", fontweight="bold")
    ax.set_facecolor("#2a2a4e")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.3)

    # Fuel - show cumulative fuel energy per layer and total
    ax = axes[1, 0]
    ax.plot(
        times,
        stats.get("fuel_energy_cumulative_floor", []),
        "lime",
        lw=2,
        label="Floor cumulative",
    )
    ax.plot(
        times,
        stats.get("fuel_energy_cumulative_ceil", []),
        "cyan",
        lw=2,
        label="Ceil cumulative",
    )
    ax.plot(
        times,
        stats.get("fuel_energy_cumulative_total", []),
        "white",
        lw=2,
        label="Total cumulative",
    )
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Fuel (arb. units)", color="white")
    ax.set_title(
        "Cumulative Fuel Energy (burning cells)", color="white", fontweight="bold"
    )
    ax.set_facecolor("#2a2a4e")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.3)
    ax.legend(facecolor="#2a2a4e", labelcolor="white")

    # Mean generated temperature increase per burning cell
    ax = axes[1, 1]
    # ΔT (per burning cell)
    (l1,) = ax.plot(
        times,
        stats.get("mean_delta_temp_floor", []),
        "r-",
        lw=2,
        label="Floor ΔT (mean)",
    )
    (l2,) = ax.plot(
        times,
        stats.get("mean_delta_temp_ceil", []),
        "orange",
        lw=2,
        label="Ceil ΔT (mean)",
    )
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("ΔT per burning cell (°C)", color="white")
    ax.set_title(
        "Mean ΔT & Mean Fuel per Burning Cell", color="white", fontweight="bold"
    )
    ax.set_facecolor("#2a2a4e")
    ax.tick_params(colors="white")
    ax.grid(True, alpha=0.3)

    # Mean fuel per burning cell on secondary axis
    ax2 = ax.twinx()
    (l3,) = ax2.plot(
        times,
        stats.get("mean_fuel_per_burning_cell_floor", []),
        "lime",
        lw=1.5,
        linestyle="--",
        label="Mean fuel floor",
    )
    (l4,) = ax2.plot(
        times,
        stats.get("mean_fuel_per_burning_cell_ceil", []),
        "cyan",
        lw=1.5,
        linestyle="--",
        label="Mean fuel ceil",
    )
    (l5,) = ax2.plot(
        times,
        stats.get("mean_fuel_per_burning_cell_total", []),
        "white",
        lw=1.5,
        linestyle="-.",
        label="Mean fuel total",
    )
    ax2.set_ylabel("Mean fuel per burning cell (arb. units)", color="white")
    ax2.tick_params(colors="white")

    # Combine legends
    handles = [l1, l2, l3, l4, l5]
    labels = [h.get_label() for h in handles]
    ax.legend(
        handles, labels, facecolor="#2a2a4e", labelcolor="white", loc="upper left"
    )

    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)

    print(f"✓ Stats plot saved: {output_path}")


# =============================================================================
# PYTEST FIXTURES & TEST FUNCTIONS
# =============================================================================


@pytest.fixture(scope="module")
def output_dir():
    """Create test output directory."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return TEST_OUTPUT_DIR


def test_generate_fire_gif(output_dir):
    """Generate fire simulation GIF from actual station map."""
    gif_path = output_dir / "fire_simulation.gif"

    stats = generate_fire_gif(
        output_path=gif_path,
        simulation_seconds=50.0,
        frame_interval=100,
        frame_duration_ms=200,
    )

    stats_path = output_dir / "fire_stats.png"
    generate_stats_plot(stats, stats_path)

    assert gif_path.exists()
    assert stats_path.exists()


# =============================================================================
# MAIN
# =============================================================================


def test():
    print("\n" + "=" * 60)
    print("FIRE SIMULATION TEST SUITE")
    print("=" * 60)

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run physics tests
    print("\n--- Physics Tests ---\n")

    test_class = TestFirePhysics()
    test_class.setup()

    tests = [
        ("Fire spreads", test_class.test_fire_spreads_from_ignition),
        ("Temperature increases", test_class.test_temperature_increases),
        ("Vertical heat transfer", test_class.test_vertical_heat_transfer),
        ("Walls block fire", test_class.test_walls_block_fire),
        ("Temperature capped", test_class.test_temperature_capped),
    ]

    passed = 0
    for name, test_func in tests:
        try:
            test_class.setup()  # Fresh state for each test
            test_func()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")

    print(f"\n--- {passed}/{len(tests)} tests passed ---")

    # Generate GIF
    print("\n--- Generating Fire Simulation GIF ---")

    gif_path = TEST_OUTPUT_DIR / "fire_simulation.gif"
    stats = generate_fire_gif(
        output_path=gif_path,
        simulation_seconds=150.0,
        frame_interval=100,
        frame_duration_ms=200,
    )

    stats_path = TEST_OUTPUT_DIR / "fire_stats.png"
    generate_stats_plot(stats, stats_path)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {TEST_OUTPUT_DIR}")
