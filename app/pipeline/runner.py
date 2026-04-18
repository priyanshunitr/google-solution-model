"""
End-to-end pipeline runner. Wraps detector + tracker + speed + crowd +
environmental + alerts. Exposes a single `run()` method the worker calls.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np
import supervision as sv
import torch

from app.config import settings
from app.pipeline.alerts import AlertDebouncer
from app.pipeline.calibration import BEVCalibration
from app.pipeline.crowd import (
    CrowdLSTMConfirmer,
    CrowdRuleDetector,
    build_motion_descriptor,
)
from app.pipeline.detector import Detector
from app.pipeline.environmental import EnvironmentalMonitor
from app.pipeline.speed import FastVehicleWindow, SpeedEstimator
from app.pipeline.tracker import PerClassTracker

log = logging.getLogger("sos.runner")


@dataclass
class PipelineResult:
    alerts: List[dict] = field(default_factory=list)
    fps_processed: float = 0.0
    n_frames: int = 0
    calibration: dict = field(default_factory=dict)


class PipelineRunner:
    """Holds persistent state (YOLO weights); `run()` is called per job."""

    def __init__(self) -> None:
        self.detector = Detector()

    # ---------------- public API ----------------

    def run(
        self,
        input_path: Path,
        output_video_path: Path,
        output_alerts_path: Path,
        calibration: Optional[dict] = None,
        progress_cb: Optional[Callable[[float, str], None]] = None,
    ) -> PipelineResult:

        def _progress(frac: float, msg: str) -> None:
            if progress_cb:
                progress_cb(max(0.0, min(1.0, frac)), msg)

        # ---- Probe video ----
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        log.info("Video: %dx%d @ %.1f fps, %d frames", w, h, fps, n_frames)

        # ---- Calibration ----
        calib = BEVCalibration(
            src_points=calibration.get("src_points") if calibration else None,
            real_width_m=calibration.get("real_width_m") if calibration else None,
            real_height_m=calibration.get("real_height_m") if calibration else None,
        )

        # ---- State ----
        tracker = PerClassTracker(fps)
        speed_est = SpeedEstimator(fps)
        fast_window = FastVehicleWindow(fps, settings.speed_window_sec)
        env_monitor = EnvironmentalMonitor(calib)
        crowd_rules = CrowdRuleDetector(fps)
        lstm_confirmer = CrowdLSTMConfirmer(device=self.detector.device)

        # =====================================================
        # PASS 1: warmup -> collect descriptors, train LSTM-AE
        # =====================================================
        _progress(0.02, "Warmup: collecting motion baseline")
        warmup_descriptors = self._warmup_pass(
            input_path, fps, calib, n_frames, _progress
        )
        if len(warmup_descriptors) > 0:
            _progress(0.18, "Training LSTM baseline")
            lstm_confirmer.fit(warmup_descriptors)
            log.info("LSTM fit on %d warmup descriptors (threshold=%.4f)",
                     len(warmup_descriptors), lstm_confirmer.threshold)

        # =====================================================
        # PASS 2: main processing
        # =====================================================
        deb_hs = AlertDebouncer(settings.alert_sustain_frames, settings.alert_cooldown_frames)
        deb_crowd = AlertDebouncer(settings.alert_sustain_frames, settings.alert_cooldown_frames)
        deb_env = AlertDebouncer(settings.alert_sustain_frames, settings.alert_cooldown_frames)

        box_annot = sv.BoxAnnotator(thickness=2)
        label_annot = sv.LabelAnnotator(text_scale=0.45, text_thickness=1, text_padding=3)
        person_box = sv.BoxAnnotator(thickness=1, color=sv.Color(0, 200, 255))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        cap = cv2.VideoCapture(str(input_path))
        alerts_log: List[dict] = []
        t0 = time.time()
        fno = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                env = env_monitor.step(frame)
                det = self.detector.infer(frame)
                vehicles, persons = tracker.update(det)

                # Speed
                v_labels: List[str] = []
                active_vids = set()
                if len(vehicles):
                    img_pts = vehicles.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                    bev_m = calib.img_to_bev_m(img_pts)
                    for tid, xy in zip(vehicles.tracker_id, bev_m):
                        active_vids.add(int(tid))
                        sp = speed_est.step(int(tid), xy)
                        if sp is not None:
                            v_labels.append(f"#{tid} {int(sp)} km/h")
                            if sp > settings.speed_limit_kmh:
                                fast_window.add(fno, int(tid))
                        else:
                            v_labels.append(f"#{tid}")
                speed_est.evict(active_vids)

                n_fast = fast_window.unique_count(fno)
                raw_hs = n_fast >= settings.fast_vehicle_count

                # Crowd
                crowd_rule = {"triggered": False, "n_fast": 0, "dispersion": 0.0, "mean_speed_kmh": 0.0}
                if len(persons):
                    img_pts = persons.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                    p_bev_m = calib.img_to_bev_m(img_pts)
                    crowd_rule = crowd_rules.update(persons.tracker_id, p_bev_m)

                desc = build_motion_descriptor(
                    env["flow_mag"], env["ssim_drop"], env["bg_shift_m"],
                    len(persons), crowd_rule["mean_speed_kmh"],
                    crowd_rule["dispersion"], float(len(persons)), n_fast,
                )
                lstm_anom, lstm_err = lstm_confirmer.check(desc)
                raw_crowd = crowd_rule["triggered"] and lstm_anom
                raw_env = env["anomaly"]

                # Debounce + log
                e_hs = deb_hs.step(raw_hs, fno)
                e_cr = deb_crowd.step(raw_crowd, fno)
                e_en = deb_env.step(raw_env, fno)

                for name, e, reason in [
                    ("high_speed_group", e_hs, f"{n_fast} vehicles >{settings.speed_limit_kmh} km/h"),
                    ("crowd_anomaly", e_cr, f"rule+lstm (err={lstm_err:.3f})"),
                    ("environmental", e_en,
                     f"flow={env['flow_mag']:.1f} ssim_drop={env['ssim_drop']:.2f} "
                     f"bg_shift={env['bg_shift_m']:.2f}m"),
                ]:
                    if e["event"] == "start":
                        alerts_log.append({
                            "type": name,
                            "start_frame": e["start_frame"],
                            "start_time_s": e["start_frame"] / fps,
                            "reason": reason,
                            "end_frame": None,
                            "end_time_s": None,
                        })
                    elif e["event"] == "end":
                        for a in reversed(alerts_log):
                            if a["type"] == name and a["end_frame"] is None:
                                a["end_frame"] = fno
                                a["end_time_s"] = fno / fps
                                break

                # Annotate
                if len(vehicles):
                    frame = box_annot.annotate(frame, vehicles)
                    frame = label_annot.annotate(frame, vehicles, v_labels)
                if len(persons):
                    frame = person_box.annotate(frame, persons)

                y = 30
                cv2.putText(frame, f"Fast vehicles ({settings.speed_window_sec}s): {n_fast}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y += 25
                cv2.putText(frame,
                            f"Flow:{env['flow_mag']:.1f} SSIM drop:{env['ssim_drop']:.2f} "
                            f"BG shift:{env['bg_shift_m']:.2f}m",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 22
                cv2.putText(frame, f"LSTM err:{lstm_err:.3f}/thr:{lstm_confirmer.threshold:.3f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 25

                active = []
                if e_hs["active"]:
                    active.append("HIGH-SPEED GROUP")
                if e_cr["active"]:
                    active.append("CROWD ANOMALY")
                if e_en["active"]:
                    active.append("ENVIRONMENTAL")
                if active:
                    banner = "SOS: " + " | ".join(active)
                    (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (5, y - 5), (15 + tw, y + th + 10), (0, 0, 180), -1)
                    cv2.putText(frame, banner, (10, y + th + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                writer.write(frame)
                fno += 1

                if fno % 20 == 0:
                    frac = 0.2 + 0.8 * (fno / max(n_frames, 1))
                    _progress(frac, f"Processing frame {fno}/{n_frames}")

        finally:
            cap.release()
            writer.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Close any alerts still open at EOF
        for a in alerts_log:
            if a["end_frame"] is None:
                a["end_frame"] = fno - 1
                a["end_time_s"] = (fno - 1) / fps

        elapsed = time.time() - t0
        fps_proc = fno / elapsed if elapsed > 0 else 0.0

        out_json = {
            "video": input_path.name,
            "fps": fps,
            "n_frames": fno,
            "fps_processed": fps_proc,
            "calibration": calib.as_dict(),
            "lstm_threshold": lstm_confirmer.threshold,
            "alerts": alerts_log,
        }
        with output_alerts_path.open("w") as f:
            json.dump(out_json, f, indent=2)

        _progress(1.0, f"Done: {len(alerts_log)} alerts")
        return PipelineResult(
            alerts=alerts_log, fps_processed=fps_proc,
            n_frames=fno, calibration=calib.as_dict(),
        )

    # ---------------- warmup helper ----------------

    def _warmup_pass(
        self,
        input_path: Path,
        fps: float,
        calib: BEVCalibration,
        n_frames_total: int,
        progress_cb: Callable[[float, str], None],
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(input_path))
        env = EnvironmentalMonitor(calib)
        tracker = PerClassTracker(fps)
        speed_est = SpeedEstimator(fps)
        crowd_rules = CrowdRuleDetector(fps)

        n_warm = min(settings.lstm_warmup_frames, n_frames_total)
        descriptors: List[np.ndarray] = []
        try:
            for i in range(n_warm):
                ok, frame = cap.read()
                if not ok:
                    break
                e = env.step(frame)
                det = self.detector.infer(frame)
                vehicles, persons = tracker.update(det)

                fast_count = 0
                if len(vehicles):
                    img_pts = vehicles.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                    bev_m = calib.img_to_bev_m(img_pts)
                    for tid, xy in zip(vehicles.tracker_id, bev_m):
                        sp = speed_est.step(int(tid), xy)
                        if sp is not None and sp > settings.speed_limit_kmh:
                            fast_count += 1

                dispersion = 0.0
                mean_kmh = 0.0
                if len(persons):
                    img_pts = persons.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
                    p_bev_m = calib.img_to_bev_m(img_pts)
                    cr = crowd_rules.update(persons.tracker_id, p_bev_m)
                    dispersion = cr["dispersion"]
                    mean_kmh = cr["mean_speed_kmh"]

                descriptors.append(
                    build_motion_descriptor(
                        e["flow_mag"], e["ssim_drop"], e["bg_shift_m"],
                        len(persons), mean_kmh, dispersion,
                        float(len(persons)), fast_count,
                    )
                )

                if i % 30 == 0:
                    progress_cb(0.02 + 0.15 * (i / max(n_warm, 1)),
                                f"Warmup {i}/{n_warm}")
        finally:
            cap.release()

        return np.array(descriptors, dtype=np.float32) if descriptors else np.zeros((0, 8), dtype=np.float32)
