import argparse
import json
from pathlib import Path

import numpy as np

from src.config.fire_constants import make_default_tables
from src.physics.fire_kernel import make_default_params, update_fire_kernel
from src.state.state_loader import load_state_from_files
from tuning.fire_signal import compute_fire_active_mask
from tuning.loss import compute_loss
from tuning.roi_runtime import fire_ratio_for_rois_separate, load_rois


def run(
    matrices_dir,
    roi_json,
    out_dir,
    t_end,
    dt,
    sample_dt,
    burn_threshold,
    burn_tau,
    milestones_json,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    state = load_state_from_files(matrices_dir)

    rois = load_rois(roi_json)
    base_names = [r.name for r in rois]
    N = len(rois)
    roi_names = np.array(
        [f"{n}__FLOOR" for n in base_names] + [f"{n}__CEIL" for n in base_names]
    )

    steps = int(np.ceil(t_end / dt))
    sample_every = max(1, int(np.round(sample_dt / dt)))
    n_samples = (steps // sample_every) + 1

    t_axis = np.empty(n_samples, dtype=np.float32)
    curves = np.empty((n_samples, 2 * N), dtype=np.float32)

    params = make_default_params()
    tables = make_default_tables()
    ign_table, heat_table, diff_table, reach_table, cap_table, after_burn_discrete = (
        tables
    )

    s = 0
    t_axis[s] = 0.0

    mask_f = compute_fire_active_mask(
        state.temperature_floor,
        state.fuel_power_floor,
        state.fuel_remaining_floor,
        state.nav,
        params,
        ign_table,
        heat_table,
        cap_table,
    )
    mask_c = compute_fire_active_mask(
        state.temperature_ceil,
        state.fuel_power_ceil,
        state.fuel_remaining_ceil,
        state.nav,
        params,
        ign_table,
        heat_table,
        cap_table,
    )
    rf, rc = fire_ratio_for_rois_separate(mask_f, mask_c, rois)
    curves[s, :N] = rf
    curves[s, N:] = rc

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
            params,
            tables,
            dt=dt,
            burn_tau=burn_tau,
        )

        if i % sample_every == 0:
            s += 1
            t_axis[s] = i * dt

            mask_f = compute_fire_active_mask(
                state.temperature_floor,
                state.fuel_power_floor,
                state.fuel_remaining_floor,
                state.nav,
                params,
                ign_table,
                heat_table,
                cap_table,
            )
            mask_c = compute_fire_active_mask(
                state.temperature_ceil,
                state.fuel_power_ceil,
                state.fuel_remaining_ceil,
                state.nav,
                params,
                ign_table,
                heat_table,
                cap_table,
            )
            rf, rc = fire_ratio_for_rois_separate(mask_f, mask_c, rois)
            curves[s, :N] = rf
            curves[s, N:] = rc

    np.savez_compressed(
        out_dir / "roi_curves.npz", t=t_axis, curves=curves, roi_names=roi_names
    )

    metrics = {
        "dt": dt,
        "t_end": t_end,
        "burn_tau": burn_tau,
        "steps": steps,
        "sample_dt": sample_dt,
        "burn_threshold_note": "unused for roi_signal=FIRE_ACTIVE_MASK",
        "matrices_dir": str(matrices_dir),
        "roi_json": str(roi_json),
        "milestones_json": str(milestones_json),
        "roi_signal": "FIRE_ACTIVE_MASK_SEPARATE",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    loss_out = compute_loss(out_dir / "roi_curves.npz", milestones_json)
    (out_dir / "loss.json").write_text(json.dumps(loss_out, indent=2))

    if loss_out.get("missing_rois"):
        raise SystemExit(f"Missing ROIs in curves: {loss_out['missing_rois']}")

    be_sum = state.burn_energy_floor + state.burn_energy_ceil
    print("burn_energy max:", float(be_sum.max()))
    print("burn_energy nonzero:", int(np.count_nonzero(be_sum > 0.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", type=Path, default=Path("sim_output/matrices"))
    ap.add_argument(
        "--roi_json",
        type=Path,
        default=Path("scripts/output/roi_groups_important.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("tuning/tuning_runs/run0"),
    )
    ap.add_argument("--t_end", type=float, default=60.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--sample_dt", type=float, default=0.05)
    ap.add_argument("--burn_threshold", type=float, default=25.0)
    ap.add_argument("--burn_tau", type=float, default=1.5)
    ap.add_argument(
        "--milestones_json", type=Path, default=Path("tuning/objectives.json")
    )
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
        milestones_json=args.milestones_json,
    )


if __name__ == "__main__":
    main()
