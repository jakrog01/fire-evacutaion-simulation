import cv2
import numpy as np
from src.config.constants import SAFE_ZONE_RADIUS

def draw_razor_line(
    img: np.ndarray,
    p1: tuple,
    p2: tuple,
    color: float | int,
    thickness: int,
) -> None:
    p1_arr = np.array(p1, dtype=float)
    p2_arr = np.array(p2, dtype=float)
    v = p2_arr - p1_arr
    length = np.linalg.norm(v)

    if length == 0:
        return

    u = v / length
    perp = np.array([-u[1], u[0]])
    half_w = thickness / 2.0

    poly = np.array(
        [
            p1_arr + perp * half_w,
            p2_arr + perp * half_w,
            p2_arr - perp * half_w,
            p1_arr - perp * half_w,
        ],
        dtype=np.int32,
    )

    c_val = float(color) if img.dtype == np.float32 else int(color)
    cv2.fillPoly(img, [poly], color=c_val)


def create_safe_zone(nav: np.ndarray, p1: tuple, p2: tuple, radius: int = 20) -> None:
    cx = int((p1[0] + p2[0]) / 2)
    cy = int((p1[1] + p2[1]) / 2)

    if radius is None:
        radius = SAFE_ZONE_RADIUS

    mask = np.zeros_like(nav)
    cv2.circle(mask, (cx, cy), radius, 1, -1)

    nav[(mask == 1) & (nav == 255)] = 0
