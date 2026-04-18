"""
Background worker thread that consumes job IDs from a queue and runs the
SOS detection pipeline for each.

One worker thread per process. For multi-GPU setups, run multiple processes
behind a load balancer rather than threading — CUDA + Python threading is a
known source of pain.
"""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from datetime import datetime
from typing import Optional

import torch

from app.config import settings
from app.jobs import JobStatus, job_store
from app.pipeline.runner import PipelineRunner

log = logging.getLogger("sos.worker")


class Worker:
    def __init__(self) -> None:
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._runner: Optional[PipelineRunner] = None

    # ---------- Lifecycle ----------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._runner = PipelineRunner()  # loads YOLO once, reused across jobs
        self._thread = threading.Thread(target=self._loop, name="sos-worker", daemon=True)
        self._thread.start()
        log.info("Worker started on device=%s", self.device_name())

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("__SHUTDOWN__")
        if self._thread:
            self._thread.join(timeout=5)
        log.info("Worker stopped")

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def device_name(self) -> str:
        if torch.cuda.is_available():
            return f"cuda:0 ({torch.cuda.get_device_name(0)})"
        return "cpu"

    # ---------- Queueing ----------

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    # ---------- Main loop ----------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job_id == "__SHUTDOWN__":
                break

            job = job_store.get(job_id)
            if not job:
                log.warning("Job %s vanished before processing", job_id)
                continue
            if job.status != JobStatus.QUEUED:
                log.info("Skipping job %s (status=%s)", job_id, job.status.value)
                continue

            self._run_job(job_id)

    def _run_job(self, job_id: str) -> None:
        log.info("Starting job %s", job_id)
        job_store.update(
            job_id,
            status=JobStatus.RUNNING,
            started_at=datetime.utcnow(),
            message="Processing started",
        )

        def progress_cb(frac: float, message: str) -> None:
            job_store.update(job_id, progress=frac, message=message)

        try:
            job = job_store.get(job_id)
            assert job is not None

            result = self._runner.run(
                input_path=job.input_path,
                output_video_path=job.output_video_path,
                output_alerts_path=job.output_alerts_path,
                calibration=job.calibration,
                progress_cb=progress_cb,
            )

            job_store.update(
                job_id,
                status=JobStatus.DONE,
                progress=1.0,
                finished_at=datetime.utcnow(),
                message=f"Done. {len(result.alerts)} alerts detected.",
                alerts_summary=result.alerts,
                fps_processed=result.fps_processed,
            )
            log.info("Finished job %s (%.1f fps, %d alerts)",
                     job_id, result.fps_processed, len(result.alerts))

        except Exception as e:
            tb = traceback.format_exc()
            log.error("Job %s failed: %s\n%s", job_id, e, tb)
            job_store.update(
                job_id,
                status=JobStatus.FAILED,
                finished_at=datetime.utcnow(),
                error=str(e),
                message="Job failed; see error field",
            )
        finally:
            # Free VRAM between jobs to avoid fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


worker = Worker()
