from src import GridGenerator

def main():
    grid = GridGenerator("stationnc.json")
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

    if grid.config.ignition_temperature:
        print(f"Ignition point: {grid.config.ignition_temperature}")


if __name__ == "__main__":
    main()
