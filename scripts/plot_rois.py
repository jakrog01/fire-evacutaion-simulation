import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

PAD_CELLS = 2
LINE_WIDTH_CELLS = 2


def dilate_mask(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return mask
    h, w = mask.shape
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy > r * r:
                continue
            ys0 = max(0, dy)
            ys1 = min(h, h + dy)
            xs0 = max(0, dx)
            xs1 = min(w, w + dx)
            out[ys0:ys1, xs0:xs1] |= mask[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
    return out


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return y0, y1, x0, x1


def mask_to_row_segments(mask: np.ndarray):
    segs = []
    h, w = mask.shape
    for y in range(h):
        row = mask[y]
        if not row.any():
            continue
        x = 0
        while x < w:
            while x < w and not row[x]:
                x += 1
            if x >= w:
                break
            x0 = x
            while x < w and row[x]:
                x += 1
            x1 = x
            segs.append([int(y), int(x0), int(x1)])
    return segs


def world_to_grid_xy(x: float, y: float, scale_factor: float, grid_size: float):
    gx = x / (scale_factor * grid_size)
    gy = y / (scale_factor * grid_size)
    return gx, gy


def parse_point_xy(p):
    if isinstance(p, dict):
        return float(p.get("x")), float(p.get("y"))
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return float(p[0]), float(p[1])
    raise TypeError(f"Unsupported point format: {type(p)}")


def build_zone_mask(
    zone: dict, shape_hw, scale_factor: float, grid_size: float
) -> np.ndarray:
    h, w = shape_hw
    img = Image.new("L", (w, h), 0)
    dr = ImageDraw.Draw(img)

    if "points" in zone and zone["points"]:
        pts = []
        for p in zone["points"]:
            gx, gy = world_to_grid_xy(
                parse_point_xy(p)[0], parse_point_xy(p)[1], scale_factor, grid_size
            )
            pts.append((int(round(gx)), int(round(gy))))
        if len(pts) >= 3:
            dr.polygon(pts, fill=1)

    elif "center" in zone and "radius" in zone:
        c = zone["center"]
        gx, gy = world_to_grid_xy(
            parse_point_xy(c)[0], parse_point_xy(c)[1], scale_factor, grid_size
        )
        r = float(zone["radius"]) / (scale_factor * grid_size)
        x0, y0 = gx - r, gy - r
        x1, y1 = gx + r, gy + r
        dr.ellipse((x0, y0, x1, y1), fill=1)

    elif "start" in zone and "end" in zone:
        s, e = zone["start"], zone["end"]
        x0, y0 = world_to_grid_xy(
            parse_point_xy(s)[0], parse_point_xy(s)[1], scale_factor, grid_size
        )
        x1, y1 = world_to_grid_xy(
            parse_point_xy(e)[0], parse_point_xy(e)[1], scale_factor, grid_size
        )
        dr.line((x0, y0, x1, y1), fill=1, width=int(LINE_WIDTH_CELLS))

    else:
        return np.zeros((h, w), dtype=bool)

    arr = np.array(img, dtype=np.uint8)
    return arr.astype(bool)


def build_zone_masks_from_json(
    geometry: dict, nav_shape, scale_factor: float, grid_size: float
):
    zone_masks = {}
    for z in geometry.get("fuel_zones", []):
        name = z.get("name", "NONAME")
        fp = float(z.get("fuel_power", z.get("fuel_power_factor", 0.0)))
        mask = build_zone_mask(z, nav_shape, scale_factor, grid_size)
        zone_masks[name] = {
            "mask": mask,
            "fuel_power": fp,
            "type": z.get("type", "fuel_zone"),
        }
    for z in geometry.get("wall_fuel_zones", []):
        name = z.get("name", "NONAME")
        fp = float(z.get("fuel_power", z.get("fuel_power_factor", 0.0)))
        mask = build_zone_mask(z, nav_shape, scale_factor, grid_size)
        zone_masks[name] = {
            "mask": mask,
            "fuel_power": fp,
            "type": z.get("type", "wall_fuel_zone"),
        }
    return zone_masks


def zm(zone_masks: dict, *names: str):
    base = next(iter(zone_masks.values()))["mask"]
    m = np.zeros_like(base, dtype=bool)
    ok = []
    for n in names:
        if n in zone_masks:
            m |= zone_masks[n]["mask"]
            ok.append(n)
    return m, ok


def build_milestone_rois(zone_masks: dict):
    alcove_walls, alcove_wall_members = zm(
        zone_masks, "StageWall1", "StageWall2", "StageWall3"
    )

    base = next(iter(zone_masks.values()))["mask"] if zone_masks else None
    if base is None:
        return {}

    stage_ceiling = (
        zone_masks["StageCeiling"]["mask"]
        if "StageCeiling" in zone_masks
        else np.zeros_like(base)
    )
    stage_floor = (
        zone_masks["StageFloor"]["mask"]
        if "StageFloor" in zone_masks
        else np.zeros_like(base)
    )
    stage_rest_walls, stage_rest_wall_members = zm(zone_masks, "StageWall4", "StageWall5")

    alcove_ceiling = np.zeros_like(base)
    bb = mask_bbox(alcove_walls)
    if bb is not None:
        y0, y1, x0, x1 = bb
        pad = int(PAD_CELLS)
        y0 = max(0, y0 - pad)
        y1 = min(base.shape[0], y1 + pad)
        x0 = max(0, x0 - pad)
        x1 = min(base.shape[1], x1 + pad)
        alcove_ceiling[y0:y1, x0:x1] = stage_ceiling[y0:y1, x0:x1]

    stage_ceiling_rest = stage_ceiling & (~alcove_ceiling)

    roi_alcove = alcove_walls | alcove_ceiling
    roi_stage = stage_floor | stage_rest_walls | stage_ceiling_rest
    roi_alcove_stage = roi_alcove | roi_stage

    roi_main_room, _ = zm(zone_masks, "MainRoom")
    roi_exit_passage, _ = zm(zone_masks, "ExitPassageFuel")

    roi_sunroom, _ = zm(
        zone_masks,
        "SunRoomFloor",
        "SunRoomCeiling",
        "SunRoomWall1",
        "SunRoomWall2",
        "SunRoomWall3",
        "SunRoomWall4",
        "SunRoomWall5",
    )

    roi_dance_rail, _ = zm(
        zone_masks,
        "DanceFloor",
        "DanceFloorCeiling",
        "railingfire",
        "DanceFloorWall1",
        "DanceFloorWall2",
    )

    roi_bar, _ = zm(zone_masks, "BarFloor", "BarCeiling", "Bar")
    roi_bar_fuels, _ = zm(zone_masks, "BarFuel1", "BarFuel2", "BarFuel3")

    rois = {
        "ROI_ALCOVE": {
            "mask": roi_alcove,
            "members": alcove_wall_members
            + (["StageCeiling(part)"] if alcove_ceiling.any() else []),
        },
        "ROI_STAGE": {
            "mask": roi_stage,
            "members": (["StageFloor"] if stage_floor.any() else [])
            + stage_rest_wall_members
            + (["StageCeiling(rest)"] if stage_ceiling_rest.any() else []),
        },
        "ROI_ALCOVE_STAGE": {
            "mask": roi_alcove_stage,
            "members": ["ROI_ALCOVE", "ROI_STAGE"],
        },
        "ROI_DANCE_AND_RAILING": {
            "mask": roi_dance_rail,
            "members": [
                "DanceFloor",
                "DanceFloorCeiling",
                "railingfire",
                "DanceFloorWall1",
                "DanceFloorWall2",
            ],
        },
        "ROI_MAIN_ROOM": {"mask": roi_main_room, "members": ["MainRoom"]},
        "ROI_EXIT_PASSAGE": {"mask": roi_exit_passage, "members": ["ExitPassageFuel"]},
        "ROI_SUNROOM": {
            "mask": roi_sunroom,
            "members": [
                "SunRoomFloor",
                "SunRoomCeiling",
                "SunRoomWall1",
                "SunRoomWall2",
                "SunRoomWall3",
                "SunRoomWall4",
                "SunRoomWall5",
            ],
        },
        "ROI_BAR": {"mask": roi_bar, "members": ["BarFloor", "BarCeiling", "Bar"]},
        "ROI_BAR_FUELS": {
            "mask": roi_bar_fuels,
            "members": ["BarFuel1", "BarFuel2", "BarFuel3"],
        },
    }

    for k in list(rois.keys()):
        rois[k]["mask"] = dilate_mask(rois[k]["mask"], PAD_CELLS)

    rois = {k: v for k, v in rois.items() if v["mask"].any()}
    return rois





def export_rois_json(rois: dict, out_json: Path, meta: dict):
    out = {"meta": meta, "rois": {}}
    for name, item in rois.items():
        m = item["mask"]
        bb = mask_bbox(m)
        if bb is None:
            continue
        out["rois"][name] = {
            "bbox": list(bb),
            "segments": mask_to_row_segments(m),
            "cell_count": int(m.sum()),
            "members": item.get("members", []),
        }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")


def plot_overlays(
    nav: np.ndarray,
    items: dict,
    out_path: Path,
    title: str,
    alpha_fill=0.18,
    linewidth=1.2,
    label_max=80,
):
    nav_img = nav.astype(float)
    vmin = float(np.min(nav_img))
    vmax = (
        float(np.max(nav_img))
        if float(np.max(nav_img)) != float(np.min(nav_img))
        else float(np.min(nav_img) + 1.0)
    )

    fig = plt.figure(figsize=(12, 8), dpi=160)
    ax = plt.gca()
    ax.imshow(nav_img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    cmap = plt.get_cmap("tab20")
    keys = list(items.keys())

    for i, k in enumerate(keys):
        m = items[k]["mask"]
        if not m.any():
            continue
        color = cmap(i % 20)
        overlay = np.zeros((m.shape[0], m.shape[1], 4), dtype=float)
        overlay[..., 0] = color[0]
        overlay[..., 1] = color[1]
        overlay[..., 2] = color[2]
        overlay[..., 3] = alpha_fill * m.astype(float)
        ax.imshow(overlay)

        ax.contour(m.astype(float), levels=[0.5], linewidths=linewidth, colors=[color])

        bb = mask_bbox(m)
        if bb is not None:
            y0, y1, x0, x1 = bb
            cy = (y0 + y1) // 2
            cx = (x0 + x1) // 2
            label = k
            if len(label) > label_max:
                label = label[: label_max - 3] + "..."
            ax.text(
                cx,
                cy,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    fc=(0.9, 0.9, 0.9, 0.85),
                    ec=(0.2, 0.2, 0.2, 0.7),
                    lw=0.8,
                ),
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    import argparse

    script_root = Path(__file__).resolve().parent
    repo_root = script_root.parent

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json", type=str, default=str(repo_root / "input" / "stationnc.json")
    )
    ap.add_argument(
        "--nav",
        type=str,
        default=str(repo_root / "sim_output" / "matrices" / "nav.npy"),
    )
    ap.add_argument("--out", type=str, default=str(repo_root / "scripts" / "output"))
    args = ap.parse_args()

    json_path = Path(args.json)
    nav_path = Path(args.nav)
    out_dir = Path(args.out)

    if not json_path.exists():
        alt = Path("/mnt/data/stationnc.json")
        if alt.exists():
            json_path = alt

    data = json.loads(json_path.read_text(encoding="utf-8"))
    grid_size = float(data.get("grid_size", 1.0))
    scale_factor = float(data.get("scale_factor", 1.0))
    geometry = data.get("geometry", {})

    nav = np.load(nav_path)

    zone_masks = build_zone_masks_from_json(
        geometry, nav.shape, scale_factor, grid_size
    )
    rois = build_milestone_rois(zone_masks)

    out_dir.mkdir(parents=True, exist_ok=True)

    fuel_items = {
        k: {"mask": v["mask"]} for k, v in zone_masks.items() if v["mask"].any()
    }
    plot_overlays(
        nav,
        fuel_items,
        out_dir / "roi_fuel_zones.png",
        title="Fuel zones (raw) over nav",
        alpha_fill=0.20,
        linewidth=1.0,
        label_max=60,
    )

    plot_overlays(
        nav,
        rois,
        out_dir / "roi_groups_important.png",
        title="ROI groups (important milestones) over nav",
        alpha_fill=0.18,
        linewidth=1.2,
        label_max=60,
    )

    meta = {
        "pad_cells": int(PAD_CELLS),
        "grid_size": grid_size,
        "scale_factor": scale_factor,
        "nav_shape": [int(nav.shape[0]), int(nav.shape[1])],
    }
    export_rois_json(rois, out_dir / "roi_milestones.json", meta=meta)

    print(f"[OK] wrote: {out_dir / 'roi_fuel_zones.png'}")
    print(f"[OK] wrote: {out_dir / 'roi_groups_important.png'}")
    print(f"[OK] wrote: {out_dir / 'roi_milestones.json'}")


if __name__ == "__main__":
    main()
