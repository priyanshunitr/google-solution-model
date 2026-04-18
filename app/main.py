"""
FastAPI entrypoint for the SOS detection service.

Endpoints:
    POST   /jobs                  Upload a video, optionally pass calibration JSON.
    GET    /jobs/{job_id}         Poll job status.
    GET    /jobs/{job_id}/video   Download annotated output (only when done).
    GET    /jobs/{job_id}/alerts  Download structured alerts JSON.
    DELETE /jobs/{job_id}         Cancel / delete a job and its artifacts.
    GET    /health                Liveness probe.
    GET    /                      Service info.
"""
from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.jobs import JobStatus, job_store
from app.schemas import (
    CalibrationInput,
    HealthResponse,
    JobCreateResponse,
    JobStatusResponse,
    ServiceInfo,
)
from app.worker import worker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("sos.main")

app = FastAPI(
    title="SOS Detection Service",
    description="Vehicle speed, crowd anomaly, and environmental disturbance detection from video.",
    version="1.0.0",
)

# CORS: open by default; tighten allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    worker.start()
    log.info("Service started. storage=%s models=%s", settings.storage_dir, settings.model_dir)


@app.on_event("shutdown")
def _shutdown() -> None:
    worker.stop()
    log.info("Service stopped.")


@app.get("/", response_model=ServiceInfo)
def root() -> ServiceInfo:
    return ServiceInfo(
        name="SOS Detection Service",
        version="1.0.0",
        device=worker.device_name(),
        model=settings.model_name,
        max_upload_mb=settings.max_upload_mb,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        worker_alive=worker.is_alive(),
        active_jobs=job_store.active_count(),
        queued_jobs=job_store.queued_count(),
    )


@app.post("/jobs", response_model=JobCreateResponse, status_code=202)
async def create_job(
    video: UploadFile = File(..., description="Video file (mp4/mov/avi/mkv)"),
    calibration: Optional[str] = Form(
        None,
        description="JSON string matching CalibrationInput schema. If omitted, defaults used.",
    ),
) -> JobCreateResponse:
    # ---- Validate extension ----
    suffix = Path(video.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(400, f"Unsupported video extension: {suffix}")

    # ---- Parse optional calibration ----
    calib: Optional[CalibrationInput] = None
    if calibration:
        try:
            calib = CalibrationInput.model_validate_json(calibration)
        except Exception as e:
            raise HTTPException(400, f"Invalid calibration JSON: {e}")

    # ---- Create job dir, stream upload to disk ----
    job_id = uuid.uuid4().hex
    job_dir = settings.storage_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    input_path = job_dir / f"input{suffix}"

    size = 0
    max_bytes = settings.max_upload_mb * 1024 * 1024
    try:
        with input_path.open("wb") as f:
            while chunk := await video.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(413, f"Upload exceeds {settings.max_upload_mb} MB limit")
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, f"Upload failed: {e}")

    log.info("Accepted job %s (%d bytes)", job_id, size)

    # ---- Register + enqueue ----
    job_store.create(
        job_id=job_id,
        input_path=input_path,
        output_video_path=job_dir / "output.mp4",
        output_alerts_path=job_dir / "alerts.json",
        calibration=calib.model_dump() if calib else None,
    )
    worker.enqueue(job_id)

    return JobCreateResponse(
        job_id=job_id,
        status=JobStatus.QUEUED.value,
        poll_url=f"/jobs/{job_id}",
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str) -> JobStatusResponse:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        message=job.message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        n_alerts=len(job.alerts_summary) if job.alerts_summary else 0,
        fps_processed=job.fps_processed,
        video_url=f"/jobs/{job_id}/video" if job.status == JobStatus.DONE else None,
        alerts_url=f"/jobs/{job_id}/alerts" if job.status == JobStatus.DONE else None,
    )


@app.get("/jobs/{job_id}/video")
def download_video(job_id: str) -> FileResponse:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(409, f"Job not ready (status: {job.status.value})")
    if not job.output_video_path.exists():
        raise HTTPException(500, "Output video missing on disk")
    return FileResponse(
        path=job.output_video_path,
        media_type="video/mp4",
        filename=f"sos_{job_id}.mp4",
    )


@app.get("/jobs/{job_id}/alerts")
def download_alerts(job_id: str) -> JSONResponse:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.DONE:
        raise HTTPException(409, f"Job not ready (status: {job.status.value})")
    if not job.output_alerts_path.exists():
        raise HTTPException(500, "Alerts file missing on disk")
    with job.output_alerts_path.open() as f:
        return JSONResponse(content=json.load(f))


@app.delete("/jobs/{job_id}", status_code=204)
def delete_job(job_id: str) -> None:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job_dir = job.input_path.parent
    job_store.remove(job_id)
    shutil.rmtree(job_dir, ignore_errors=True)
    log.info("Deleted job %s", job_id)
