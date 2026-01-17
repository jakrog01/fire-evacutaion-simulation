import cv2
import numpy as np

from .config import SimConfig
from .drawing import create_safe_zone, draw_razor_line
from .geometry import GeometryProcessor


class MapGenerator:
    def __init__(self, config: SimConfig, geometry: GeometryProcessor):
        self.config = config
        self.geo = geometry
        self.nav: np.ndarray | None = None
        self.f_block: np.ndarray | None = None
        self.f_ceil: np.ndarray | None = None
        self.H: int = 0
        self.W: int = 0
        self.b_centroid: tuple[float, float] = (0, 0)

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H, W = self.geo.get_dimensions()
        self.H, self.W = H, W

        nav = np.full((H, W), 255, dtype=np.uint8)
        f_block = np.zeros((H, W), dtype=np.float32)
        f_ceil = np.zeros((H, W), dtype=np.float32)

        self._draw_rooms(nav, f_block, f_ceil)
        self._draw_open_spaces(nav, f_block)
        self._draw_obstacles(nav, f_block, f_ceil)
        self._draw_fuel_zones(f_block, f_ceil)
        self._draw_doors_windows(nav, f_block)

        self.nav = nav
        self.f_block = f_block
        self.f_ceil = f_ceil
        self.b_centroid = self.geo.calc_building_centroid()

        return nav, f_block, f_ceil

    def _draw_rooms(
        self, nav: np.ndarray, f_block: np.ndarray, f_ceil: np.ndarray
    ) -> None:
        for room in self.config.geometry.get("rooms", []):
            pts = self.geo.get_poly(room)
            if pts is None:
                continue

            cv2.fillPoly(nav, [pts], 0)
            cv2.polylines(nav, [pts], True, 1, self.config.thick_wall)
            cv2.fillPoly(f_block, [pts], 4.0)
            cv2.polylines(f_block, [pts], True, 6.0, self.config.thick_wall)
            cv2.fillPoly(f_ceil, [pts], 3.0)

    def _get_candidates(self) -> list[dict]:
        geometry = self.config.geometry
        return (
            geometry.get("objects", [])
            + geometry.get("doors", [])
            + geometry.get("windows", [])
        )

    def _draw_open_spaces(self, nav: np.ndarray, f_block: np.ndarray) -> None:
        for item in self._get_candidates():
            name = item.get("name", "")
            is_open = item.get("is_open_space", False) or "OS" in name

            if not is_open:
                continue

            p1, p2 = self._extract_line_points(item)
            if not p1:
                continue

            sp1, sp2 = self.geo.shorten_line(p1, p2, margin_px=self.config.margin_os)
            draw_razor_line(nav, sp1, sp2, 0, self.config.thick_os_cut)
            draw_razor_line(f_block, sp1, sp2, 0.0, self.config.thick_os_cut)

    def _draw_obstacles(
        self, nav: np.ndarray, f_block: np.ndarray, f_ceil: np.ndarray
    ) -> None:
        geometry = self.config.geometry
        obs_list = geometry.get("obstacle_polygons", []) + geometry.get("obstacles", [])

        for obs in obs_list:
            pts = self.geo.get_poly(obs)
            if pts is None:
                continue

            code = 7 if obs.get("path_blocking", True) else 8
            cv2.fillPoly(nav, [pts], code)
            cv2.fillPoly(f_block, [pts], 6.0)
            cv2.fillPoly(f_ceil, [pts], 6.0)

    def _draw_fuel_zones(self, f_block: np.ndarray, f_ceil: np.ndarray) -> None:
        zones = self.config.geometry.get("fuel_zones", [])

        for zone in zones:
            pts = self.geo.get_poly(zone)
            if pts is None:
                continue

            power = zone.get("fuel_power", 6.0)
            z_type = zone.get("type", "fuel_zone")

            mask = np.zeros((self.H, self.W), dtype=np.uint8)

            if z_type == "wall_fuel_zone" and "start" in zone:
                s = self.geo.to_grid(zone["start"])
                e = self.geo.to_grid(zone["end"])
                cv2.line(mask, s, e, 1, self.config.thick_wall)
                f_block[mask == 1] = np.maximum(f_block[mask == 1], power)
            else:
                cv2.fillPoly(mask, [pts], 1)
                f_block[mask == 1] = np.maximum(f_block[mask == 1], power)
                if z_type != "wall_fuel_zone_strict":
                    f_ceil[mask == 1] = np.maximum(f_ceil[mask == 1], power)

    def _draw_doors_windows(self, nav: np.ndarray, f_block: np.ndarray) -> None:
        for item in self._get_candidates():
            name = item.get("name", "")
            is_open = item.get("is_open_space", False) or "OS" in name

            if is_open:
                continue

            itype = item.get("type", "").lower()
            is_door_or_window = (
                "door" in itype
                or "door" in name.lower()
                or "exit" in name.lower()
                or "window" in itype
            )

            if not is_door_or_window:
                continue

            p1, p2 = self._extract_line_points(item)
            if not p1:
                continue

            is_win = "window" in itype or "window" in name.lower()
            is_staff = item.get("staff_only", False)
            is_exit = item.get("is_exit", False)

            code = self._get_door_code(is_win, is_exit, is_staff)
            draw_razor_line(nav, p1, p2, code, self.config.thick_door)

            f_val = 2.0 if is_win else 3.0
            draw_razor_line(f_block, p1, p2, f_val, self.config.thick_door)

            if is_exit:
                create_safe_zone(nav, p1, p2)

    def _extract_line_points(
        self, item: dict
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        if "start" in item:
            return self.geo.to_grid(item["start"]), self.geo.to_grid(item["end"])
        elif "points" in item:
            return (
                self.geo.to_grid(item["points"][0]),
                self.geo.to_grid(item["points"][1]),
            )
        return None, None

    @staticmethod
    def _get_door_code(is_win: bool, is_exit: bool, is_staff: bool) -> int:
        if is_win:
            return 1
        if is_exit and is_staff:
            return 6
        if is_exit:
            return 5
        if is_staff:
            return 4
        return 3
