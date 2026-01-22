import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Expect:
    mode: str
    min_val: float
    max_val: float
    w: float
    early_w: float


def _interp_at(times, values, t):
    return float(np.interp(t, times, values))


def _window_score(times, roi_curve, t0, t1, expect: Expect):
    tm = 0.5 * (t0 + t1)

    v0 = _interp_at(times, roi_curve, t0)
    vm = _interp_at(times, roi_curve, tm)
    v1 = _interp_at(times, roi_curve, t1)

    if expect.mode == "OFF":
        v = max(v0, vm, v1)
        p0 = max(0.0, v0 - expect.max_val)
        p1 = max(0.0, v - expect.max_val)
    else:
        v = min(v0, vm, v1)
        p0 = max(0.0, expect.min_val - v0)
        p1 = max(0.0, expect.min_val - v)

    return expect.w * (expect.early_w * p0 * p0 + p1 * p1)


def load_milestones(path: Path):
    j = json.loads(path.read_text())
    start_threshold = float(j.get("start_threshold", 0.05))
    full_threshold = float(j.get("full_threshold", 0.7))

    if "roi_expectations" in j:
        items = j["roi_expectations"]
    elif "milestones" in j:
        items = j["milestones"]
    else:
        raise ValueError(
            "Objectives JSON must contain 'roi_expectations' or 'milestones'"
        )

    ms = []
    for m in items:
        t0 = float(m["t0"])
        t1 = float(m["t1"])
        ex = {}
        for roi_name, cfg in m["expect"].items():
            mode = str(cfg["mode"]).upper()
            w = float(cfg.get("w", 1.0))
            early_w = float(cfg.get("early_w", 2.0))

            if mode == "OFF":
                ex[roi_name] = Expect(
                    mode=mode,
                    min_val=0.0,
                    max_val=float(cfg["max"]),
                    w=w,
                    early_w=early_w,
                )
            elif mode == "ON":
                ex[roi_name] = Expect(
                    mode=mode,
                    min_val=float(cfg.get("min", start_threshold)),
                    max_val=1.0,
                    w=w,
                    early_w=early_w,
                )
            elif mode == "FULL":
                ex[roi_name] = Expect(
                    mode=mode,
                    min_val=float(cfg.get("min", full_threshold)),
                    max_val=1.0,
                    w=w,
                    early_w=early_w,
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

        ms.append((t0, t1, ex))

    return ms


def compute_loss(curves_npz: Path, milestones_json: Path) -> dict:
    d = np.load(curves_npz, allow_pickle=True)
    t = d["t"].astype(np.float32, copy=False)
    curves = d["curves"].astype(np.float32, copy=False)
    roi_names = [str(x) for x in d["roi_names"].tolist()]

    roi_to_idx = {name: i for i, name in enumerate(roi_names)}
    ms = load_milestones(milestones_json)

    loss = 0.0
    missing = []

    for t0, t1, ex in ms:
        for roi_name, expect in ex.items():
            idx = roi_to_idx.get(roi_name)
            if idx is None:
                missing.append(roi_name)
                continue
            loss += _window_score(t, curves[:, idx], t0, t1, expect)

    out = {"loss": float(loss), "missing_rois": sorted(set(missing))}
    return out
