import cv2
import numpy as np


def draw_razor_line(
    img: np.ndarray,
    p1: tuple,
    p2: tuple,
    color: float | int,
    thickness: int,
) -> None:
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    v = p2 - p1
    length = np.linalg.norm(v)

    if length == 0:
        return

    u = v / length
    perp = np.array([-u[1], u[0]])
    half_w = thickness / 2.0

    poly = np.array(
        [
            p1 + perp * half_w,
            p2 + perp * half_w,
            p2 - perp * half_w,
            p1 - perp * half_w,
        ],
        dtype=np.int32,
    )

    c_val = float(color) if img.dtype == np.float32 else int(color)
    cv2.fillPoly(img, [poly], color=c_val)


def create_safe_zone(nav: np.ndarray, p1: tuple, p2: tuple, radius: int = 20) -> None:
    cx = int((p1[0] + p2[0]) / 2)
    cy = int((p1[1] + p2[1]) / 2)

    mask = np.zeros_like(nav)
    cv2.circle(mask, (cx, cy), radius, 1, -1)

    nav[(mask == 1) & (nav == 255)] = 0
