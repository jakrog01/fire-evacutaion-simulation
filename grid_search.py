#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from src.config.fire_constants import make_default_tables
from src.physics.fire_kernel import (
    P_CONVECTION_RATE,
    P_DIFFUSION_SCALE,
    P_ENERGY_TO_TEMPERATURE,
    P_EX_MAX,
    P_EX_MIN,
    P_HRR_SCALE,
    P_IGNITION_SHIFT,
    P_RADIATION_DOWN_RATE,
    P_REACH_SCALE,
    P_VERTICAL_TRANSFER_THRESHOLD,
    P_WALL_SYNC_RATE,
    make_default_params,
    update_fire_kernel,
)
from src.state.state_loader import load_state_from_files
from tuning.fire_signal import compute_fire_active_mask
from tuning.loss import compute_loss
from tuning.roi_runtime import fire_ratio_for_rois_separate, load_rois


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    files = sorted([p for p in path.rglob("*") if p.is_file()])
    h = hashlib.sha256()
    for p in files:
        rel = str(p.relative_to(path)).encode("utf-8")
        h.update(rel)
        h.update(b"\x00")
        h.update(sha256_file(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def get_git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.STDOUT
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def get_versions() -> dict:
    v = {"python": sys.version.replace("\n", " "), "platform": platform.platform()}
    try:
        import numpy as _np

        v["numpy"] = _np.__version__
    except Exception:
        v["numpy"] = ""
    try:
        import numba as _nb

        v["numba"] = _nb.__version__
    except Exception:
        v["numba"] = ""
    return v


def loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    a = math.log(lo)
    b = math.log(hi)
    return float(math.exp(rng.uniform(a, b)))


def sample_param_set(rng: np.random.Generator) -> dict:
    return {
        "energy_to_temperature": loguniform(rng, 0.08, 0.40),
        "vertical_transfer_threshold": float(rng.uniform(20.0, 80.0)),
        "wall_sync_rate": loguniform(rng, 80.0, 400.0),
        "convection_rate": loguniform(rng, 0.04, 0.16),
        "radiation_down_rate": loguniform(rng, 0.003, 0.020),
        "hrr_scale": loguniform(rng, 0.60, 2.50),
        "ignition_shift": float(rng.uniform(-25.0, 15.0)),
        "reach_scale": float(rng.uniform(0.80, 1.80)),
        "diffusion_scale": float(rng.uniform(0.70, 1.70)),
        "ex_min": float(rng.uniform(0.12, 0.35)),
        "ex_max": float(rng.uniform(1.20, 3.00)),
        "burn_tau": float(rng.uniform(0.80, 2.50)),
    }


def params_to_array(p: dict) -> np.ndarray:
    arr = make_default_params(ex_min=float(p["ex_min"]), ex_max=float(p["ex_max"]))
    arr[P_ENERGY_TO_TEMPERATURE] = float(p["energy_to_temperature"])
    arr[P_VERTICAL_TRANSFER_THRESHOLD] = float(p["vertical_transfer_threshold"])
    arr[P_WALL_SYNC_RATE] = float(p["wall_sync_rate"])
    arr[P_CONVECTION_RATE] = float(p["convection_rate"])
    arr[P_RADIATION_DOWN_RATE] = float(p["radiation_down_rate"])
    arr[P_HRR_SCALE] = float(p["hrr_scale"])
    arr[P_IGNITION_SHIFT] = float(p["ignition_shift"])
    arr[P_REACH_SCALE] = float(p["reach_scale"])
    arr[P_DIFFUSION_SCALE] = float(p["diffusion_scale"])
    arr[P_EX_MIN] = float(p["ex_min"])
    arr[P_EX_MAX] = float(p["ex_max"])
    return arr.astype(np.float32, copy=False)


def make_run_id(index: int, params: dict, meta: dict) -> str:
    payload = {"index": int(index), "params": params, "meta": meta}
    b = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(b)[:16]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def is_finite_number(x) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def validate_success(run_dir: Path) -> tuple[bool, str]:
    metrics_p = run_dir / "metrics.json"
    loss_p = run_dir / "loss.json"
    curves_p = run_dir / "roi_curves.npz"
    if not metrics_p.exists():
        return False, "missing_metrics"
    if not loss_p.exists():
        return False, "missing_loss"
    if not curves_p.exists():
        return False, "missing_roi_curves"
    try:
        loss_j = json.loads(loss_p.read_text())
    except Exception:
        return False, "loss_unreadable"
    if "loss" not in loss_j:
        return False, "loss_missing_key"
    if not is_finite_number(loss_j["loss"]):
        return False, "loss_nan_inf"
    missing = loss_j.get("missing_rois", [])
    if missing:
        return False, "missing_rois"
    return True, "ok"


def read_manifest_entry(manifest_path: Path, offsets_path: Path, index: int) -> dict:
    if offsets_path.exists():
        offsets = np.load(offsets_path, allow_pickle=False)
        if index < 0 or index >= int(offsets.shape[0]):
            raise ValueError("bad_index")
        off = int(offsets[index])
        with manifest_path.open("rb") as f:
            f.seek(off)
            line = f.readline()
        return json.loads(line.decode("utf-8"))
    with manifest_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise ValueError("index_not_found")


def simulate_and_score(
    matrices_dir: Path,
    roi_json: Path,
    objectives_json: Path,
    run_dir: Path,
    t_end: float,
    dt: float,
    sample_dt: float,
    params_dict: dict,
) -> dict:
    ensure_dir(run_dir)

    rois = load_rois(roi_json)
    base_names = [r.name for r in rois]
    n_rois = len(rois)
    roi_names = np.array(
        [f"{n}__FLOOR" for n in base_names] + [f"{n}__CEIL" for n in base_names]
    )

    state = load_state_from_files(matrices_dir)
    tables = make_default_tables()
    ign_table, heat_table, diff_table, reach_table, cap_table, after_burn_discrete = (
        tables
    )

    params_arr = params_to_array(params_dict)
    burn_tau = float(params_dict["burn_tau"])

    steps = int(np.ceil(t_end / dt))
    sample_every = max(1, int(np.round(sample_dt / dt)))
    n_samples = (steps // sample_every) + 1

    t_axis = np.empty(n_samples, dtype=np.float32)
    curves = np.empty((n_samples, 2 * n_rois), dtype=np.float32)

    s = 0
    t_axis[s] = 0.0

    mask_f = compute_fire_active_mask(
        state.temperature_floor,
        state.fuel_power_floor,
        state.fuel_remaining_floor,
        state.nav,
        params_arr,
        ign_table,
        heat_table,
        cap_table,
    )
    mask_c = compute_fire_active_mask(
        state.temperature_ceil,
        state.fuel_power_ceil,
        state.fuel_remaining_ceil,
        state.nav,
        params_arr,
        ign_table,
        heat_table,
        cap_table,
    )
    rf, rc = fire_ratio_for_rois_separate(mask_f, mask_c, rois)
    curves[s, :n_rois] = rf
    curves[s, n_rois:] = rc

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
            params_arr,
            tables,
            dt=dt,
            burn_tau=burn_tau,
        )

        if i % sample_every == 0:
            s += 1
            t_axis[s] = float(i * dt)

            mask_f = compute_fire_active_mask(
                state.temperature_floor,
                state.fuel_power_floor,
                state.fuel_remaining_floor,
                state.nav,
                params_arr,
                ign_table,
                heat_table,
                cap_table,
            )
            mask_c = compute_fire_active_mask(
                state.temperature_ceil,
                state.fuel_power_ceil,
                state.fuel_remaining_ceil,
                state.nav,
                params_arr,
                ign_table,
                heat_table,
                cap_table,
            )
            rf, rc = fire_ratio_for_rois_separate(mask_f, mask_c, rois)
            curves[s, :n_rois] = rf
            curves[s, n_rois:] = rc

    np.savez_compressed(
        run_dir / "roi_curves.npz", t=t_axis, curves=curves, roi_names=roi_names
    )

    loss_out = compute_loss(run_dir / "roi_curves.npz", objectives_json)
    write_json(run_dir / "loss.json", loss_out)
    return loss_out


def cmd_generate(args) -> int:
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)
    ensure_dir(out_root / "runs")

    matrices_dir = Path(args.matrices).resolve()
    roi_json = Path(args.roi_json).resolve()
    objectives_json = Path(args.objectives_json).resolve()

    if not matrices_dir.exists():
        raise SystemExit(f"missing_matrices_dir: {matrices_dir}")
    if not roi_json.exists():
        raise SystemExit(f"missing_roi_json: {roi_json}")
    if not objectives_json.exists():
        raise SystemExit(f"missing_objectives_json: {objectives_json}")

    repo_root = Path.cwd()
    meta = {
        "n_runs": int(args.n_runs),
        "seed": int(args.seed),
        "created_unix": int(time.time()),
        "matrices_dir": str(matrices_dir),
        "roi_json": str(roi_json),
        "objectives_json": str(objectives_json),
        "t_end": float(args.t_end),
        "dt": float(args.dt),
        "sample_dt": float(args.sample_dt),
        "param_space_version": "v1",
        "git_commit": get_git_commit(repo_root),
        "versions": get_versions(),
        "input_hashes": {
            "roi_json_sha256": sha256_file(roi_json),
            "objectives_json_sha256": sha256_file(objectives_json),
            "matrices_dir_sha256": sha256_dir(matrices_dir),
        },
        "param_space": {
            "energy_to_temperature": [0.08, 0.40, "loguniform"],
            "vertical_transfer_threshold": [20.0, 80.0, "uniform"],
            "wall_sync_rate": [80.0, 400.0, "loguniform"],
            "convection_rate": [0.04, 0.16, "loguniform"],
            "radiation_down_rate": [0.003, 0.020, "loguniform"],
            "hrr_scale": [0.60, 2.50, "loguniform"],
            "ignition_shift": [-25.0, 15.0, "uniform"],
            "reach_scale": [0.80, 1.80, "uniform"],
            "diffusion_scale": [0.70, 1.70, "uniform"],
            "ex_min": [0.12, 0.35, "uniform"],
            "ex_max": [1.20, 3.00, "uniform"],
            "burn_tau": [0.80, 2.50, "uniform"],
        },
    }

    write_json(out_root / "meta.json", meta)

    manifest_path = out_root / "manifest.jsonl"
    offsets_path = out_root / "manifest_offsets.npy"

    rng = np.random.default_rng(int(args.seed))

    offsets = np.empty(int(args.n_runs), dtype=np.int64)
    with manifest_path.open("wb") as mf:
        for i in range(int(args.n_runs)):
            params = sample_param_set(rng)
            run_id = make_run_id(i, params, meta)
            run_rel = f"runs/run_{i:05d}_{run_id}"
            entry = {
                "index": int(i),
                "run_id": run_id,
                "run_rel": run_rel,
                "params": params,
            }
            offsets[i] = int(mf.tell())
            mf.write((json.dumps(entry, sort_keys=True) + "\n").encode("utf-8"))

    np.save(offsets_path, offsets, allow_pickle=False)
    return 0


def cmd_run_one(args) -> int:
    out_root = Path(args.out_root).resolve()
    manifest = Path(args.manifest).resolve()
    offsets = Path(args.offsets).resolve()

    meta_path = out_root / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"missing_meta_json: {meta_path}")

    meta = json.loads(meta_path.read_text())
    entry = read_manifest_entry(manifest, offsets, int(args.index))

    matrices_dir = Path(meta["matrices_dir"])
    roi_json = Path(meta["roi_json"])
    objectives_json = Path(meta["objectives_json"])

    run_dir = out_root / entry["run_rel"]
    ensure_dir(run_dir)

    ok, reason = validate_success(run_dir)
    if ok:
        return 0

    for stale in [
        "metrics.json",
        "loss.json",
        "roi_curves.npz",
        "failed.json",
        "run.log",
    ]:
        p = run_dir / stale
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    retries = int(args.retries)
    t0 = time.time()

    last_exc = ""
    for attempt in range(1, retries + 2):
        try:
            loss_out = simulate_and_score(
                matrices_dir=matrices_dir,
                roi_json=roi_json,
                objectives_json=objectives_json,
                run_dir=run_dir,
                t_end=float(meta["t_end"]),
                dt=float(meta["dt"]),
                sample_dt=float(meta["sample_dt"]),
                params_dict=entry["params"],
            )

            ok2, reason2 = validate_success(run_dir)
            if not ok2:
                raise RuntimeError(f"post_validate_failed:{reason2}")

            metrics = {
                "index": int(entry["index"]),
                "run_id": str(entry["run_id"]),
                "run_dir": str(run_dir),
                "argv": sys.argv,
                "git_commit": meta.get("git_commit", ""),
                "versions": get_versions(),
                "input_hashes": meta.get("input_hashes", {}),
                "matrices_dir": meta["matrices_dir"],
                "roi_json": meta["roi_json"],
                "objectives_json": meta["objectives_json"],
                "t_end": float(meta["t_end"]),
                "dt": float(meta["dt"]),
                "sample_dt": float(meta["sample_dt"]),
                "params": entry["params"],
                "loss": float(loss_out.get("loss", float("nan"))),
                "missing_rois": loss_out.get("missing_rois", []),
                "elapsed_s": float(time.time() - t0),
                "attempt": int(attempt),
                "ok": True,
            }
            write_json(run_dir / "metrics.json", metrics)
            (run_dir / "run.log").write_text(
                json.dumps({"ok": True, "reason": "ok", "attempt": attempt}, indent=2)
            )
            return 0
        except Exception as e:
            last_exc = traceback.format_exc()
            if attempt <= retries:
                continue
            failed = {
                "ok": False,
                "index": int(entry["index"]),
                "run_id": str(entry["run_id"]),
                "reason": "exception",
                "elapsed_s": float(time.time() - t0),
                "attempt": int(attempt),
                "exception_type": type(e).__name__,
                "exception": str(e),
                "traceback": last_exc,
            }
            write_json(run_dir / "failed.json", failed)
            (run_dir / "run.log").write_text(
                json.dumps({"ok": False, "attempt": attempt}, indent=2)
            )
            return 2

    return 2


def cmd_collect(args) -> int:
    out_root = Path(args.out_root).resolve()
    runs_dir = out_root / "runs"
    if not runs_dir.exists():
        raise SystemExit(f"missing_runs_dir: {runs_dir}")

    rows = []
    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        ok, reason = validate_success(run_dir)
        loss_val = float("nan")
        if (run_dir / "loss.json").exists():
            try:
                loss_val = float(
                    json.loads((run_dir / "loss.json").read_text()).get(
                        "loss", float("nan")
                    )
                )
            except Exception:
                pass
        rows.append(
            {
                "run_dir": str(run_dir),
                "ok": bool(ok),
                "reason": str(reason),
                "loss": float(loss_val) if is_finite_number(loss_val) else float("nan"),
            }
        )

    out = {"n": len(rows), "rows": rows}
    write_json(out_root / "collect.json", out)
    return 0


def cmd_report(args) -> int:
    out_root = Path(args.out_root).resolve()
    collect_path = out_root / "collect.json"
    if not collect_path.exists():
        raise SystemExit("run_collect_first")

    c = json.loads(collect_path.read_text())
    rows = [r for r in c["rows"] if r.get("ok") and is_finite_number(r.get("loss"))]
    rows.sort(key=lambda x: float(x["loss"]))

    k = int(args.k)
    top = rows[:k]
    write_json(out_root / "topk.json", {"k": k, "top": top})
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("--out_root", type=str, required=True)
    g.add_argument("--matrices", type=str, default="sim_output/matrices")
    g.add_argument(
        "--roi_json", type=str, default="scripts/output/roi_groups_important.json"
    )
    g.add_argument("--objectives_json", type=str, default="tuning/objectives.json")
    g.add_argument("--n_runs", type=int, default=10000)
    g.add_argument("--seed", type=int, default=12345)
    g.add_argument("--t_end", type=float, default=30.0)
    g.add_argument("--dt", type=float, default=0.001)
    g.add_argument("--sample_dt", type=float, default=0.05)

    r = sub.add_parser("run")
    r.add_argument("--out_root", type=str, required=True)
    r.add_argument("--manifest", type=str, required=True)
    r.add_argument("--offsets", type=str, required=True)
    r.add_argument("--index", type=int, required=True)
    r.add_argument("--retries", type=int, default=1)

    c = sub.add_parser("collect")
    c.add_argument("--out_root", type=str, required=True)

    p = sub.add_parser("report")
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--k", type=int, default=5)

    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()

    if args.cmd == "generate":
        return cmd_generate(args)
    if args.cmd == "run":
        return cmd_run_one(args)
    if args.cmd == "collect":
        return cmd_collect(args)
    if args.cmd == "report":
        return cmd_report(args)
    raise SystemExit("bad_cmd")


if __name__ == "__main__":
    raise SystemExit(main())
