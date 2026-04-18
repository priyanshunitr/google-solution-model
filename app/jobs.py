"""
Thread-safe in-memory job store.

For single-node deployments. If you scale horizontally, swap this for
Redis/Postgres — the interface is small and easy to reimplement.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class Job:
    job_id: str
    input_path: Path
    output_video_path: Path
    output_alerts_path: Path
    calibration: Optional[dict] = None
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    alerts_summary: List[dict] = field(default_factory=list)
    fps_processed: Optional[float] = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def create(
        self,
        job_id: str,
        input_path: Path,
        output_video_path: Path,
        output_alerts_path: Path,
        calibration: Optional[dict] = None,
    ) -> Job:
        with self._lock:
            job = Job(
                job_id=job_id,
                input_path=input_path,
                output_video_path=output_video_path,
                output_alerts_path=output_alerts_path,
                calibration=calibration,
            )
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                for k, v in kwargs.items():
                    setattr(job, k, v)

    def remove(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)

    def queued_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)


job_store = JobStore()
