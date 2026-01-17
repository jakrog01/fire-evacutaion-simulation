from config import load_simulation_config
from fire_evac_sim import GridGenerator


def main():
    sim_config = load_simulation_config()
    grid = GridGenerator("stationnc.json", sim_config=sim_config)
    print("Generating architecture maps...")
    grid.generate_maps()
    print("Generating potential fields...")
    grid.generate_potentials()
    print("Generating fire simulation matrices...")
    fire_maps = grid.generate_fire_matrices()

    grid.config.ensure_output_dirs()

    print("Saving visualizations...")
    grid.save_visualizations()

    print("Saving matrices...")
    grid.save_matrices()

    print("\n=== Generation Complete ===")
    print(f"Grid dimensions: {grid.H} x {grid.W}")
    print(f"Fire blocking shape: {fire_maps.fire_blocking.shape}")
    print(f"Ignition temperature: {grid.config.ignition_temperature}°C")

    if grid.config.ignition_point:
        print(f"Ignition point: {grid.config.ignition_point}")

    max_temp = fire_maps.fire_blocking.max()
    if max_temp > 100:
        print(f"Max initial temperature: {max_temp}°C (fire ignited)")
    else:
        print("Warning: No ignition point detected in fire matrix")


if __name__ == "__main__":
    main()
