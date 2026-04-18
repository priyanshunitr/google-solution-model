"""
YOLO detector wrapper. Loads weights once, reused across jobs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

from app.config import settings

log = logging.getLogger("sos.detector")


class Detector:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half = settings.use_half_precision and self.device == "cuda"

        # Cache weights inside the model dir so container restarts don't re-download
        weights_path = settings.model_dir / settings.model_name
        if weights_path.exists():
            self.model = YOLO(str(weights_path))
        else:
            # Ultralytics will download to the current working dir by default; we move it
            self.model = YOLO(settings.model_name)
            try:
                src = Path(settings.model_name)
                if src.exists() and src.resolve() != weights_path.resolve():
                    src.replace(weights_path)
                    log.info("Cached weights to %s", weights_path)
            except Exception as e:
                log.warning("Could not cache weights: %s", e)

        self.model.to(self.device)
        if self.half:
            self.model.model.half()
        log.info("Detector ready: model=%s device=%s half=%s", settings.model_name, self.device, self.half)

    def infer(self, frame: np.ndarray) -> sv.Detections:
        res = self.model(
            frame,
            imgsz=settings.imgsz,
            conf=settings.conf_thresh,
            iou=settings.iou_thresh,
            half=self.half,
            verbose=False,
            device=self.device,
        )[0]
        return sv.Detections.from_ultralytics(res)
