"""
Per-track speed estimator using a constant-velocity Kalman filter in BEV
meter coordinates, plus a sliding-window counter of unique "fast" vehicles.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from app.config import settings


class TrackSpeedKF:
    """4-state (x, y, vx, vy) Kalman filter; input in BEV meters."""

    def __init__(self, fps: float) -> None:
        self.dt = 1.0 / fps
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array(
            [[1, 0, self.dt, 0],
             [0, 1, 0, self.dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=float,
        )
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.kf.P *= 10.0
        self.kf.R *= 0.5
        self.kf.Q *= 0.05
        self.initialized = False
        self.n_updates = 0
        self.smoothed_kmh = 0.0

    def update(self, xy_m: np.ndarray) -> Optional[float]:
        if not self.initialized:
            self.kf.x = np.array([xy_m[0], xy_m[1], 0.0, 0.0], dtype=float)
            self.initialized = True
            self.n_updates = 1
            return None
        self.kf.predict()
        self.kf.update(xy_m)
        self.n_updates += 1
        vx, vy = self.kf.x[2], self.kf.x[3]
        kmh = math.hypot(vx, vy) * 3.6
        a = settings.speed_ema_alpha
        self.smoothed_kmh = a * kmh + (1 - a) * self.smoothed_kmh
        return self.smoothed_kmh

    def ready(self) -> bool:
        return self.n_updates >= settings.min_track_frames


class SpeedEstimator:
    """Owns Kalman filters for each active track_id."""

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.kfs: Dict[int, TrackSpeedKF] = {}

    def step(self, track_id: int, xy_m: np.ndarray) -> Optional[float]:
        kf = self.kfs.get(track_id)
        if kf is None:
            kf = TrackSpeedKF(self.fps)
            self.kfs[track_id] = kf
        sp = kf.update(xy_m)
        if sp is None or not kf.ready():
            return None
        # Sanity gate on absurd speeds from ID swaps
        if not (1.0 < sp < 250.0):
            return None
        return sp

    def evict(self, keep_ids: set) -> None:
        """Drop KFs for tracks that no longer exist."""
        drop = [tid for tid in self.kfs if tid not in keep_ids]
        for tid in drop:
            del self.kfs[tid]


class FastVehicleWindow:
    """Unique fast-vehicle track_ids observed within the last N seconds."""

    def __init__(self, fps: float, window_sec: float) -> None:
        self.window_frames = int(fps * window_sec)
        self.events: deque = deque()

    def add(self, frame_idx: int, track_id: int) -> None:
        self.events.append((frame_idx, track_id))

    def unique_count(self, current_frame: int) -> int:
        cutoff = current_frame - self.window_frames
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()
        return len({tid for _, tid in self.events})
