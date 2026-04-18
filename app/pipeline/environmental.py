"""
Environmental disturbance detector: global flow magnitude + scene SSIM
drop + background-keypoint displacement (static objects moving in world).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from app.config import settings
from app.pipeline.calibration import BEVCalibration


class EnvironmentalMonitor:
    def __init__(self, calib: BEVCalibration) -> None:
        self.calib = calib
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_small: Optional[np.ndarray] = None
        self.bg_kp: Optional[np.ndarray] = None
        self.bg_kp_origin: Optional[np.ndarray] = None
        self.bg_refresh_interval = 90
        self.frames_since_seed = 0

    def _seed_keypoints(self, gray: np.ndarray) -> None:
        kp = cv2.goodFeaturesToTrack(
            gray, maxCorners=80, qualityLevel=0.01, minDistance=20, blockSize=7
        )
        self.bg_kp = kp
        self.bg_kp_origin = kp.copy() if kp is not None else None
        self.frames_since_seed = 0

    def step(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (320, 180))

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_small = small
            self._seed_keypoints(gray)
            return {"flow_mag": 0.0, "ssim_drop": 0.0, "bg_shift_m": 0.0, "anomaly": False}

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_small, small, None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        mag = float(np.linalg.norm(flow, axis=2).mean())

        ssim_val = ssim(self.prev_small, small)
        ssim_drop = float(1.0 - ssim_val)

        bg_shift_m = 0.0
        if self.bg_kp is not None and len(self.bg_kp) > 0:
            new_kp, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.bg_kp, None,
                winSize=(21, 21), maxLevel=3,
            )
            if new_kp is not None and status is not None:
                good = status.flatten() == 1
                if good.sum() > 10:
                    orig = self.bg_kp_origin[good].reshape(-1, 2)
                    now = new_kp[good].reshape(-1, 2)
                    orig_bev = self.calib.img_to_bev_m(orig)
                    now_bev = self.calib.img_to_bev_m(now)
                    dists = np.linalg.norm(now_bev - orig_bev, axis=1)
                    bg_shift_m = float(np.percentile(dists, 90))
                    self.bg_kp = new_kp

        self.frames_since_seed += 1
        if self.frames_since_seed >= self.bg_refresh_interval:
            self._seed_keypoints(gray)

        self.prev_gray = gray
        self.prev_small = small

        anomaly = (
            mag > settings.flow_magnitude_thresh
            or ssim_drop > settings.ssim_drop_thresh
            or bg_shift_m > settings.background_displacement_m
        )
        return {
            "flow_mag": mag,
            "ssim_drop": ssim_drop,
            "bg_shift_m": bg_shift_m,
            "anomaly": bool(anomaly),
        }
