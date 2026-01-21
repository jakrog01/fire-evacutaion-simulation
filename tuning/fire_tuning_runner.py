import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

import argparse
import json
from pathlib import Path

import numpy as np

from src.physics.fire_kernel import update_fire_kernel
from src.state.state_loader import load_state_from_files
from tuning.roi_runtime import burn_ratio_for_rois, load_rois


def run(
    matrices_dir, roi_json, out_dir, t_end, dt, sample_dt, burn_threshold, burn_tau
):
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_state_from_files(matrices_dir)

    rois = load_rois(roi_json)
    roi_names = np.array([r.name for r in rois])

    steps = int(np.ceil(t_end / dt))
    sample_every = max(1, int(np.round(sample_dt / dt)))
    n_samples = (steps // sample_every) + 1

    t_axis = np.empty(n_samples, dtype=np.float32)
    curves = np.empty((n_samples, len(rois)), dtype=np.float32)

    s = 0
    t_axis[s] = 0.0
    curves[s] = burn_ratio_for_rois(
        state.burn_energy_floor,
        state.burn_energy_ceil,
        rois,
        burn_threshold,
    )

    for i in range(1, steps + 1):
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
            dt,
            burn_tau=burn_tau,
        )

        if i % sample_every == 0:
            s += 1
            t_axis[s] = i * dt
            curves[s] = burn_ratio_for_rois(
                state.burn_energy_floor,
                state.burn_energy_ceil,
                rois,
                burn_threshold,
            )

    np.savez_compressed(
        out_dir / "roi_curves.npz", t=t_axis, curves=curves, roi_names=roi_names
    )

    metrics = {
        "dt": dt,
        "t_end": t_end,
        "burn_tau": burn_tau,
        "steps": steps,
        "sample_dt": sample_dt,
        "burn_threshold": burn_threshold,
        "matrices_dir": str(matrices_dir),
        "roi_json": str(roi_json),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", type=Path, default=Path("sim_output/matrices"))
    ap.add_argument(
        "--roi_json",
        type=Path,
        default=Path("scripts/output/roi_groups_important.json"),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t_end", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--sample_dt", type=float, default=0.05)
    ap.add_argument("--burn_threshold", type=float, default=25.0)
    ap.add_argument("--burn_tau", type=float, default=1.5)
    args = ap.parse_args()

    run(
        matrices_dir=args.matrices,
        roi_json=args.roi_json,
        out_dir=args.out,
        t_end=args.t_end,
        dt=args.dt,
        sample_dt=args.sample_dt,
        burn_threshold=args.burn_threshold,
        burn_tau=args.burn_tau,
    )


if __name__ == "__main__":
    main()
