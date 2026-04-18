"""
Crowd-anomaly detection.

Two-tier:
  1. Rules: local density (>= crowd_count neighbors within crowd_radius_m)
     + velocity dispersion + speed threshold.
  2. LSTM autoencoder: trained on warmup descriptors (assumed normal).
     Reconstruction error > mu + k*sigma => anomaly.

Final signal = AND of the two (reduces false positives substantially).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from app.config import settings


class CrowdRuleDetector:
    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.hist: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=int(fps)))

    def update(self, track_ids: np.ndarray, bev_m: np.ndarray) -> dict:
        for tid, xy in zip(track_ids, bev_m):
            self.hist[tid].append(xy)

        active = set(track_ids.tolist())
        for tid in list(self.hist.keys()):
            if tid not in active and len(self.hist[tid]) < 3:
                del self.hist[tid]

        if len(bev_m) < settings.crowd_count:
            return {"triggered": False, "n_fast": 0, "dispersion": 0.0, "mean_speed_kmh": 0.0}

        n_in_cluster = 0
        velocities: List[np.ndarray] = []
        for tid, xy in zip(track_ids, bev_m):
            h = self.hist[tid]
            if len(h) < 4:
                continue
            neigh = np.linalg.norm(bev_m - xy, axis=1)
            if (neigh < settings.crowd_radius_m).sum() >= settings.crowd_count:
                n_in_cluster += 1
                dv = (np.array(h[-1]) - np.array(h[0])) / (len(h) / self.fps)
                velocities.append(dv)

        if n_in_cluster < settings.crowd_count or len(velocities) < 4:
            return {"triggered": False, "n_fast": 0, "dispersion": 0.0, "mean_speed_kmh": 0.0}

        vels = np.array(velocities)
        speeds_kmh = np.linalg.norm(vels, axis=1) * 3.6
        fast_count = int((speeds_kmh > settings.crowd_speed_kmh).sum())

        norms = np.linalg.norm(vels, axis=1, keepdims=True) + 1e-6
        dirs = vels / norms
        dispersion = float(dirs.std(axis=0).sum())

        triggered = (
            fast_count >= settings.crowd_count // 2
            or (dispersion > settings.crowd_dispersion_thresh
                and speeds_kmh.mean() > settings.crowd_speed_kmh * 0.6)
        )
        return {
            "triggered": triggered,
            "n_fast": fast_count,
            "dispersion": dispersion,
            "mean_speed_kmh": float(speeds_kmh.mean()),
        }


class MotionLSTMAE(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 32) -> None:
        super().__init__()
        self.enc = nn.LSTM(in_dim, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.enc(x)
        rep = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        dec_out, _ = self.dec(rep)
        return self.out(dec_out)


def build_motion_descriptor(
    flow_mag: float,
    ssim_drop: float,
    bg_shift_m: float,
    n_persons: int,
    mean_speed_kmh: float,
    dispersion: float,
    density: float,
    fast_vehicle_count: int,
) -> np.ndarray:
    return np.array(
        [flow_mag, ssim_drop, bg_shift_m, n_persons,
         mean_speed_kmh, dispersion, density, fast_vehicle_count],
        dtype=np.float32,
    )


class CrowdLSTMConfirmer:
    """Rolling-buffer LSTM autoencoder over 8-dim motion descriptors."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.model: Optional[MotionLSTMAE] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.threshold: float = float("inf")
        self.buffer: Deque[np.ndarray] = deque(maxlen=settings.lstm_seq_len)

    def fit(self, descriptors: np.ndarray) -> None:
        if len(descriptors) < settings.lstm_seq_len + 5:
            # Not enough warmup data; disable LSTM confirmation
            self.model = None
            return

        self.mean = descriptors.mean(axis=0)
        self.std = descriptors.std(axis=0) + 1e-6
        norm = (descriptors - self.mean) / self.std
        seq = np.array([norm[i:i + settings.lstm_seq_len]
                        for i in range(len(norm) - settings.lstm_seq_len)])
        seq_t = torch.from_numpy(seq).float().to(self.device)

        self.model = MotionLSTMAE(in_dim=descriptors.shape[1], hidden=settings.lstm_hidden).to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()
        self.model.train()
        for _ in range(settings.lstm_epochs):
            opt.zero_grad()
            rec = self.model(seq_t)
            loss = loss_fn(rec, seq_t)
            loss.backward()
            opt.step()
        self.model.eval()

        with torch.no_grad():
            rec = self.model(seq_t).cpu().numpy()
        errs = ((rec - seq) ** 2).mean(axis=(1, 2))
        self.threshold = float(errs.mean() + settings.lstm_anomaly_sigma * errs.std())

    def check(self, desc: np.ndarray) -> tuple:
        """Returns (is_anomaly, reconstruction_error)."""
        self.buffer.append(desc)
        if self.model is None or len(self.buffer) < settings.lstm_seq_len:
            return False, 0.0
        seq = (np.array(self.buffer) - self.mean) / self.std
        t = torch.from_numpy(seq[None]).float().to(self.device)
        with torch.no_grad():
            rec = self.model(t).cpu().numpy()[0]
        err = float(((rec - seq) ** 2).mean())
        return err > self.threshold, err
