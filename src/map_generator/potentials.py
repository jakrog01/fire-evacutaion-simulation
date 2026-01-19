import heapq

import numpy as np

from .map_loader import MapLoaderConfig
from .geometry import GeometryProcessor


class PotentialFieldGenerator:
    FAM_WEIGHT = 10.0
    PUSH_DIST = 15.0
    COST_STRAIGHT = 1.0
    COST_DIAGONAL = 1.414

    MOVES = [
        (0, 1, COST_STRAIGHT),
        (0, -1, COST_STRAIGHT),
        (1, 0, COST_STRAIGHT),
        (-1, 0, COST_STRAIGHT),
        (1, 1, COST_DIAGONAL),
        (-1, -1, COST_DIAGONAL),
        (1, -1, COST_DIAGONAL),
        (-1, 1, COST_DIAGONAL),
    ]

    def __init__(
        self,
        config: MapLoaderConfig,
        geometry: GeometryProcessor,
        nav: np.ndarray,
        centroid: tuple[float, float],
    ):
        self.config = config
        self.geo = geometry
        self.nav = nav
        self.centroid = centroid
        self.H, self.W = nav.shape

    def generate(self, agent_type: str) -> np.ndarray:
        return self._run_dijkstra(agent_type)

    def _get_blocked_cells(self, agent_type: str) -> set[int]:
        base_blocked = {1, 7, 255}
        if agent_type == "patron":
            return base_blocked | {4, 6}
        return base_blocked

    def _get_weighted_seeds(self, agent_type: str) -> list[tuple[int, int, float]]:
        seeds = []
        candidates = self._get_exit_candidates()
        bx, by = self.centroid

        for item in candidates:
            if not item.get("is_exit", False):
                continue

            is_staff = item.get("staff_only", False)
            if agent_type == "patron" and is_staff:
                continue

            fam = item.get("exit_familiarity", 1.0)
            start_val = -(fam * self.FAM_WEIGHT)

            p1, p2 = self._extract_line_points(item)
            if p1 and p2:
                target_x, target_y = self._calculate_push_point(p1, p2, bx, by)
                seeds.append((int(target_x), int(target_y), start_val))

        return seeds

    def _get_exit_candidates(self) -> list[dict]:
        geometry = self.config.geometry
        return (
            geometry.get("objects", [])
            + geometry.get("doors", [])
            + geometry.get("windows", [])
        )

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

    def _calculate_push_point(
        self,
        p1: tuple[int, int],
        p2: tuple[int, int],
        bx: float,
        by: float,
    ) -> tuple[float, float]:
        dx = (p1[0] + p2[0]) / 2.0
        dy = (p1[1] + p2[1]) / 2.0
        vx = dx - bx
        vy = dy - by
        norm = np.sqrt(vx * vx + vy * vy)

        if norm > 0:
            return dx + (vx / norm) * self.PUSH_DIST, dy + (vy / norm) * self.PUSH_DIST

        return dx, dy

    def _run_dijkstra(self, agent_type: str) -> np.ndarray:
        blocked = self._get_blocked_cells(agent_type)
        field = np.full((self.H, self.W), np.inf)
        visited = np.zeros((self.H, self.W), dtype=bool)
        queue = []

        seeds = self._get_weighted_seeds(agent_type)
        for sx, sy, val in seeds:
            if 0 <= sx < self.W and 0 <= sy < self.H:
                if self.nav[sy, sx] not in blocked:
                    if val < field[sy, sx]:
                        field[sy, sx] = val
                        heapq.heappush(queue, (val, sx, sy))

        while queue:
            dist, cx, cy = heapq.heappop(queue)
            if visited[cy, cx]:
                continue
            visited[cy, cx] = True

            for dx, dy, cost in self.MOVES:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.W and 0 <= ny < self.H:
                    if self.nav[ny, nx] in blocked:
                        continue
                    new_dist = dist + cost
                    if new_dist < field[ny, nx]:
                        field[ny, nx] = new_dist
                        heapq.heappush(queue, (new_dist, nx, ny))

        return field
