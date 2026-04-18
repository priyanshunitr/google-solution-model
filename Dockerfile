# ============================================================
# SOS Detection Service - GPU-enabled Dockerfile
# Base image includes CUDA 12.1 runtime + Python 3.10.
# For CPU-only deployments, see Dockerfile.cpu (swap base image).
# ============================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        libgl1 libglib2.0-0 ffmpeg \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Install torch from CUDA index first for correct GPU binaries
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.4.1 torchvision==0.19.1

COPY requirements.txt .
# Remove torch lines from requirements (already installed above) and install the rest
RUN grep -vE '^(torch|torchvision)==' requirements.txt > /tmp/req-rest.txt && \
    pip install -r /tmp/req-rest.txt

COPY app ./app

# Persist storage/models on a volume
RUN mkdir -p /data/jobs /data/models
VOLUME ["/data"]

ENV SOS_STORAGE_DIR=/data/jobs \
    SOS_MODEL_DIR=/data/models

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
