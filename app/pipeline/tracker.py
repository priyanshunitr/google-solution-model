"""
Per-class tracker. Holds two independent ByteTrack instances so that vehicle
and person tracks never share IDs or pollute each other's state.
"""
from __future__ import annotations

import numpy as np
import supervision as sv

from app.config import settings


class PerClassTracker:
    def __init__(self, fps: float) -> None:
        self.vehicle = sv.ByteTrack(frame_rate=int(fps))
        self.person = sv.ByteTrack(frame_rate=int(fps))

    def update(self, det: sv.Detections):
        """Return (vehicles, persons) — both are tracked Detections."""
        if len(det) == 0:
            empty = det[:0]
            return empty, empty

        vmask = np.isin(det.class_id, settings.vehicle_cls)
        pmask = det.class_id == settings.person_cls
        vehicles = self.vehicle.update_with_detections(det[vmask]) if vmask.any() else det[:0]
        persons = self.person.update_with_detections(det[pmask]) if pmask.any() else det[:0]
        return vehicles, persons
