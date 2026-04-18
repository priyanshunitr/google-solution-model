"""
Pydantic request/response schemas.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class CalibrationInput(BaseModel):
    """Optional calibration payload sent with the video upload.

    Four image-space points (TL, TR, BR, BL) of a real-world rectangle on the
    ground plane, plus the rectangle's true dimensions in meters.
    """
    src_points: List[List[float]] = Field(
        ...,
        description="4 points in pixel space: [TL, TR, BR, BL], each [x, y]",
        min_length=4,
        max_length=4,
    )
    real_width_m: float = Field(..., gt=0, description="Real width of rectangle in meters")
    real_height_m: float = Field(..., gt=0, description="Real height of rectangle in meters")


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    poll_url: str


class AlertSummary(BaseModel):
    type: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    reason: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float = Field(..., ge=0.0, le=1.0)
    message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    n_alerts: int = 0
    fps_processed: Optional[float] = None
    video_url: Optional[str] = None
    alerts_url: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    worker_alive: bool
    active_jobs: int
    queued_jobs: int


class ServiceInfo(BaseModel):
    name: str
    version: str
    device: str
    model: str
    max_upload_mb: int
