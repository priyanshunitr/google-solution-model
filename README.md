# SOS Detection Service

Production-grade FastAPI backend for multi-scenario SOS detection from video:
vehicle speed, high-speed vehicle groups, crowd anomaly (rules + LSTM), and
environmental disturbance.

## Architecture

```
Client (React Native / curl / anything)
        |
        | POST /jobs  (upload video, optional calibration)
        v
   FastAPI app  --->  Job Store (in-memory)
        |                   ^
        | enqueue            |  status updates
        v                   |
   Worker thread  --->  PipelineRunner
                           |
                           +-- Detector (YOLOv8s, FP16)
                           +-- PerClassTracker (ByteTrack x2)
                           +-- SpeedEstimator (Kalman + EMA)
                           +-- CrowdRuleDetector + LSTM confirmer
                           +-- EnvironmentalMonitor (flow + SSIM + LK)
                           +-- AlertDebouncer (3 independent)
                                 |
                                 v
                           output.mp4 + alerts.json
```

## Folder layout

```
sos-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI routes
│   ├── config.py            # Pydantic Settings
│   ├── schemas.py           # Request/response models
│   ├── jobs.py              # In-memory job store
│   ├── worker.py            # Background worker thread
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── detector.py      # YOLO wrapper
│   │   ├── tracker.py       # Per-class ByteTrack
│   │   ├── speed.py         # Kalman speed + sliding window
│   │   ├── crowd.py         # Rules + LSTM autoencoder
│   │   ├── environmental.py # Optical flow + SSIM + LK
│   │   ├── calibration.py   # BEV homography
│   │   ├── alerts.py        # Debouncer
│   │   └── runner.py        # Orchestrator
│   └── models/              # YOLO weights cached here
├── requirements.txt
├── Dockerfile               # GPU (CUDA 12.1)
├── Dockerfile.cpu           # CPU-only fallback
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .env.example
├── README.md
└── client_example/
    └── ReactNativeClient.tsx
```

## Memory & compute requirements

| Mode | VRAM | RAM | Throughput (1080p) | Notes |
|------|------|-----|--------------------|-------|
| GPU, YOLOv8s FP16 (default) | ~1.5–2.5 GB | ~3–4 GB | 25–30 fps on T4 | Recommended |
| GPU, YOLOv8s FP32 | ~2.5–3.5 GB | ~3–4 GB | 18–22 fps on T4 | If FP16 causes issues |
| GPU, YOLOv8n FP16 | ~0.8–1.2 GB | ~2–3 GB | 40–50 fps on T4 | Lower accuracy |
| CPU, YOLOv8n | 0 | ~4–6 GB | 2–5 fps | Any 4+ core machine |

**Deploy target recommendations:**
- **AWS:** `g4dn.xlarge` (T4 16 GB, 4 vCPU, 16 GB RAM) — fits comfortably
- **GCP:** `n1-standard-4` + 1× T4
- **Azure:** `Standard_NC4as_T4_v3`
- **Runpod / Lambda / Vast.ai:** any RTX 3060+ / A10 / L4
- **CPU-only (dev/testing):** any 8 GB RAM machine; expect 5–10× slowdown

**Disk:** plan ~3× input video size (input + output + intermediates).

**Concurrency:** one job per GPU. For higher throughput, run multiple
container replicas behind a load balancer.

## Configuration

All settings are env-overridable with `SOS_` prefix. See `.env.example`.
Key variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `SOS_MODEL_NAME` | `yolov8s.pt` | YOLO weights |
| `SOS_IMGSZ` | `960` | Inference image size |
| `SOS_USE_HALF_PRECISION` | `true` | FP16 on GPU |
| `SOS_MAX_UPLOAD_MB` | `500` | Upload size limit |
| `SOS_SPEED_LIMIT_KMH` | `80` | "Fast" threshold |
| `SOS_FAST_VEHICLE_COUNT` | `10` | Trigger count |
| `SOS_SPEED_WINDOW_SEC` | `3.0` | Sliding window duration |

## API

### `POST /jobs`
Multipart form:
- `video` *(file, required)*
- `calibration` *(str, optional)* — JSON matching `CalibrationInput`:
  ```json
  {
    "src_points": [[520,380],[760,380],[1100,680],[180,680]],
    "real_width_m": 7.0,
    "real_height_m": 25.0
  }
  ```

Returns `202`:
```json
{"job_id": "abc123...", "status": "queued", "poll_url": "/jobs/abc123..."}
```

### `GET /jobs/{job_id}`
Poll at ~2s intervals. Response:
```json
{
  "job_id": "...",
  "status": "running" | "done" | "failed" | "queued",
  "progress": 0.47,
  "message": "Processing frame 450/960",
  "n_alerts": 0,
  "fps_processed": null,
  "video_url": null,
  "alerts_url": null,
  "error": null
}
```

When `status == "done"`, `video_url` and `alerts_url` are set.

### `GET /jobs/{job_id}/video`
Returns `video/mp4` of annotated output.

### `GET /jobs/{job_id}/alerts`
Returns JSON:
```json
{
  "video": "input.mp4",
  "fps": 30.0,
  "n_frames": 900,
  "fps_processed": 27.4,
  "calibration": { ... },
  "lstm_threshold": 0.0234,
  "alerts": [
    {
      "type": "high_speed_group",
      "start_frame": 120, "end_frame": 180,
      "start_time_s": 4.0, "end_time_s": 6.0,
      "reason": "11 vehicles >80.0 km/h"
    }
  ]
}
```

### `DELETE /jobs/{job_id}`
Deletes job record and all artifacts on disk.

### `GET /health`
Liveness probe. Returns 200 if worker is alive.

## Local dev (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export SOS_STORAGE_DIR=./data/jobs
export SOS_MODEL_DIR=./data/models
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for Swagger UI.

## Docker (GPU)

Requires NVIDIA Container Toolkit on the host.

```bash
docker compose up --build -d
docker compose logs -f
```

Service listens on `:8000`, data volume persists in Docker volume `sos-data`.

## Docker (CPU)

Edit `docker-compose.yml`:
- Change `dockerfile: Dockerfile` → `dockerfile: Dockerfile.cpu`
- Remove the `deploy:` block
- Add env vars: `SOS_USE_HALF_PRECISION=false`, `SOS_MODEL_NAME=yolov8n.pt`

## Cloud deployment

### AWS ECS (GPU)
1. Build and push to ECR
2. Task definition: `g4dn.xlarge`, 1 GPU, 8 GB memory
3. EFS volume mounted at `/data`

### Any VPS with GPU
```bash
# On server
git clone <your-repo>
cd sos-backend
docker compose up -d
# Expose port 8000 or put nginx/caddy in front
```

### RunPod / Vast.ai
Use a PyTorch 2.4 CUDA 12.1 template, then `docker compose up` inside it.

## Calibration — why it matters

Speeds are computed from **BEV (bird's-eye-view) coordinates** obtained by a
perspective homography. Without knowing the real dimensions of a ground-plane
rectangle, speed values are arbitrary. The request defaults are placeholders;
to get accurate km/h values you **must** send a `calibration` field that maps
4 pixel points (a quadrilateral on the road) to known real distances.

Easiest way: pick a lane (common widths: 3.5m per lane), measure the distance
between two known markers (crosswalk, light poles) for the "height".

## What changed from the original Colab notebook

1. **Separate ByteTrack instances** for vehicles vs. persons (fixed ID
   collision bug).
2. **Kalman filter on BEV meters** for speed (was finite-difference on pixels).
3. **Sliding window + debouncer** for alerts (was `sos_flag = True` and never
   reset, spamming identical alerts every frame).
4. **Metric crowd clustering** (meters, not perspective-skewed pixels).
5. **LSTM autoencoder** trained per-video on warmup frames for crowd
   confirmation (was just a rule).
6. **HSV fire detector removed** (unreliable — fired on sunsets, brake lights).
7. **Production-grade job lifecycle** with progress, cancellation, artifact
   retention.
