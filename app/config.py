"""
Application configuration via Pydantic BaseSettings.
All values overridable by environment variables with SOS_ prefix.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SOS_", env_file=".env", extra="ignore")

    # ---- Paths ----
    storage_dir: Path = Path("/data/jobs")
    model_dir: Path = Path("/data/models")

    # ---- HTTP ----
    max_upload_mb: int = 500
    cors_origins: List[str] = ["*"]

    # ---- Model ----
    model_name: str = "yolov8s.pt"          # swap to yolov8n.pt for lower VRAM
    imgsz: int = 960
    conf_thresh: float = 0.35
    iou_thresh: float = 0.6
    use_half_precision: bool = True

    # ---- Classes (COCO) ----
    person_cls: int = 0
    vehicle_cls: Tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck

    # ---- Vehicle speed thresholds ----
    speed_limit_kmh: float = 80.0
    fast_vehicle_count: int = 10
    speed_window_sec: float = 3.0
    min_track_frames: int = 6
    speed_ema_alpha: float = 0.4

    # ---- Crowd thresholds ----
    crowd_count: int = 10
    crowd_radius_m: float = 5.0
    crowd_speed_kmh: float = 8.0
    crowd_dispersion_thresh: float = 1.5
    lstm_anomaly_sigma: float = 2.5

    # ---- Environmental thresholds ----
    flow_magnitude_thresh: float = 6.0
    ssim_drop_thresh: float = 0.35
    background_displacement_m: float = 1.0

    # ---- Alert debouncing ----
    alert_sustain_frames: int = 8
    alert_cooldown_frames: int = 30

    # ---- LSTM warmup ----
    lstm_warmup_frames: int = 300
    lstm_hidden: int = 32
    lstm_seq_len: int = 10
    lstm_epochs: int = 30

    # ---- Runtime ----
    reader_queue_size: int = 16
    writer_queue_size: int = 16
    max_concurrent_jobs: int = 1   # 1 per GPU; raise only if you have multiple GPUs

    # ---- Default calibration (used when request omits it) ----
    default_src_points: List[List[float]] = Field(
        default_factory=lambda: [[520.0, 380.0], [760.0, 380.0], [1100.0, 680.0], [180.0, 680.0]]
    )
    default_real_width_m: float = 7.0
    default_real_height_m: float = 25.0
    default_bev_px_per_m: float = 20.0


settings = Settings()
