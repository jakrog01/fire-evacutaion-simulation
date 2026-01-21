import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _decode_counts(counts, shape_hw, start_val, expected_ones):
    h, w = shape_hw
    total = h * w
    c = list(counts)
    s = int(sum(c))
    if s == total + 1:
        c[-1] -= 1
        s -= 1
    if s != total or c[-1] < 0:
        raise ValueError(f"Bad RLE sum: {s} vs {total}")
    flat = np.empty(total, dtype=np.uint8)
    idx = 0
    val = int(start_val)
    for run in c:
        if run:
            flat[idx : idx + run] = val
        idx += run
        val ^= 1
    m = flat.reshape((h, w)).astype(bool)
    if expected_ones is not None and int(m.sum()) != int(expected_ones):
        return None
    return m


def decode_mask_rle(mask_rle, shape_hw, area_cells):
    m0 = _decode_counts(mask_rle, shape_hw, 0, area_cells)
    if m0 is not None:
        return m0
    m1 = _decode_counts(mask_rle, shape_hw, 1, area_cells)
    if m1 is not None:
        return m1
    m0 = _decode_counts(mask_rle, shape_hw, 0, None)
    m1 = _decode_counts(mask_rle, shape_hw, 1, None)
    s0 = int(m0.sum())
    s1 = int(m1.sum())
    d0 = abs(s0 - int(area_cells))
    d1 = abs(s1 - int(area_cells))
    return m0 if d0 <= d1 else m1


@dataclass(frozen=True)
class ROI:
    name: str
    y0: int
    y1: int
    x0: int
    x1: int
    mask_bbox: np.ndarray
    area_cells: int


def load_rois(path: Path):
    j = json.loads(path.read_text())
    groups = j["groups"]
    rois = []
    for name, g in groups.items():
        bb = g["bbox"]
        y0, y1, x0, x1 = int(bb["y0"]), int(bb["y1"]), int(bb["x0"]), int(bb["x1"])
        shape_hw = tuple(g["mask_rle_shape_hw"])
        area_cells = int(g["area_cells"])
        mask = decode_mask_rle(g["mask_rle"], shape_hw, area_cells)
        rois.append(
            ROI(
                name=name,
                y0=y0,
                y1=y1,
                x0=x0,
                x1=x1,
                mask_bbox=mask,
                area_cells=area_cells,
            )
        )
    return rois


def burn_ratio_for_rois(burn_energy_floor, burn_energy_ceil, rois, threshold):
    out = np.empty(len(rois), dtype=np.float32)
    for i, r in enumerate(rois):
        bf = burn_energy_floor[r.y0 : r.y1, r.x0 : r.x1]
        bc = burn_energy_ceil[r.y0 : r.y1, r.x0 : r.x1]
        active = ((bf + bc) > threshold) & r.mask_bbox
        out[i] = float(active.sum()) / float(r.area_cells)
    return out
