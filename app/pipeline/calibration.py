"""
Birds-eye-view (BEV) homography. Converts image-plane points to a calibrated
metric top-down view so speeds and distances can be expressed in real units.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.config import settings


class BEVCalibration:
    def __init__(
        self,
        src_points: Optional[list] = None,
        real_width_m: Optional[float] = None,
        real_height_m: Optional[float] = None,
        px_per_m: Optional[float] = None,
    ):
        self.src_points = np.array(
            src_points if src_points is not None else settings.default_src_points,
            dtype=np.float32,
        )
        self.real_width_m = real_width_m if real_width_m is not None else settings.default_real_width_m
        self.real_height_m = real_height_m if real_height_m is not None else settings.default_real_height_m
        self.px_per_m = px_per_m if px_per_m is not None else settings.default_bev_px_per_m

        bev_w = self.real_width_m * self.px_per_m
        bev_h = self.real_height_m * self.px_per_m
        self.dst_points = np.array(
            [[0, 0], [bev_w, 0], [bev_w, bev_h], [0, bev_h]], dtype=np.float32
        )
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

    def img_to_bev_px(self, points_xy: np.ndarray) -> np.ndarray:
        if len(points_xy) == 0:
            return points_xy
        pts = points_xy.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def img_to_bev_m(self, points_xy: np.ndarray) -> np.ndarray:
        return self.img_to_bev_px(points_xy) / self.px_per_m

    def as_dict(self) -> dict:
        return {
            "src_points": self.src_points.tolist(),
            "real_width_m": self.real_width_m,
            "real_height_m": self.real_height_m,
            "px_per_m": self.px_per_m,
            "used_defaults": (
                np.array_equal(self.src_points, np.array(settings.default_src_points, dtype=np.float32))
                and self.real_width_m == settings.default_real_width_m
                and self.real_height_m == settings.default_real_height_m
            ),
        }
