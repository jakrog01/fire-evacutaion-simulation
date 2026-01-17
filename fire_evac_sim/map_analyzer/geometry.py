import numpy as np

from .config import SimConfig


class GeometryProcessor:
    def __init__(self, config: SimConfig):
        self.config = config

    def to_grid(self, pt: list | tuple | None) -> tuple[int, int]:
        if not pt:
            return 0, 0
        return (
            int(pt[0] / self.config.scale_factor / self.config.grid_size),
            int(pt[1] / self.config.scale_factor / self.config.grid_size),
        )

    def get_poly(self, obj: dict) -> np.ndarray | None:
        if "points" not in obj:
            return None
        return np.array([self.to_grid(p) for p in obj["points"]], np.int32).reshape(
            (-1, 1, 2)
        )

    def get_dimensions(self) -> tuple[int, int]:
        mx, my = 0, 0
        all_objs = []
        geo = self.config.geometry

        for k in [
            "rooms",
            "fuel_zones",
            "obstacle_polygons",
            "doors",
            "windows",
            "objects",
        ]:
            all_objs.extend(geo.get(k, []))

        for item in all_objs:
            pts = []
            if "points" in item:
                pts.extend(item["points"])
            if "start" in item:
                pts.extend([item["start"], item["end"]])
            for p in pts:
                gx, gy = self.to_grid(p)
                mx, my = max(mx, gx), max(my, gy)

        return my + 40, mx + 40

    def shorten_line(
        self, p1: tuple, p2: tuple, margin_px: int
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        v = p2 - p1
        d = np.linalg.norm(v)

        if d <= 2 * margin_px:
            return tuple(p1.astype(int)), tuple(p2.astype(int))

        u = v / d
        return (
            tuple((p1 + u * margin_px).astype(int)),
            tuple((p2 - u * margin_px).astype(int)),
        )

    def calc_building_centroid(self) -> tuple[float, float]:
        xs, ys = [], []
        for room in self.config.geometry.get("rooms", []):
            pts = self.get_poly(room)
            if pts is not None:
                xs.append(np.mean(pts[:, 0, 0]))
                ys.append(np.mean(pts[:, 0, 1]))

        if not xs:
            return (0, 0)

        return np.mean(xs), np.mean(ys)
