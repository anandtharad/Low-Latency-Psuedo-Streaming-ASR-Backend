# syntax=docker/dockerfile:1
#
# Streaming ASR service.
#
# Two targets, because the CUDA image is ~8 GB and most deployments do not need
# it. Build the one you want:
#
#   docker build --target cpu -t streaming-asr:cpu .
#   docker build --target gpu -t streaming-asr:gpu .
#
# The model, LM and lexicon are NOT baked in. They are mounted at run time and
# located by ASR_* environment variables, so the same image serves any
# checkpoint and no image contains patient-adjacent data.

# ---------------------------------------------------------------------------
# Shared application layer
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # The SentencePiece vocabulary contains U+2581, which crashes a printer
    # running under the default POSIX locale.
    PYTHONIOENCODING=utf-8 \
    ASR_HOST=0.0.0.0 \
    ASR_PORT=8000

# libsndfile is required by soundfile; libgomp by onnxruntime.
#
# ffmpeg is the decoder fallback, and is not really optional: a browser
# recording via MediaRecorder is WebM/Opus and an iOS recording is m4a/AAC, and
# libsndfile decodes neither. Without it those uploads are rejected with a 415.
# Costs ~60 MB. Drop it only if every client is guaranteed to send WAV/FLAC.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first: this layer is cached across source edits.
COPY requirements-server.txt ./
RUN pip install --no-cache-dir -r requirements-server.txt

COPY streaming_asr/ ./streaming_asr/
COPY tools/ ./tools/

# Run as a non-root user; the service never needs to write to its own image.
RUN useradd --create-home --uid 10001 asr && chown -R asr:asr /app
USER asr

EXPOSE 8000

# Reports readiness only once the model is actually loaded and serving, not
# merely when the port is open.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:${ASR_PORT}/health || exit 1

ENTRYPOINT ["python", "-m", "streaming_asr.server.app"]

# ---------------------------------------------------------------------------
# CPU target
# ---------------------------------------------------------------------------
FROM base AS cpu

USER root
RUN pip install --no-cache-dir \
        "torch==2.5.1" "torchaudio==2.5.1" \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "onnxruntime>=1.19"
USER asr

ENV ASR_DEVICE=cpu

# ---------------------------------------------------------------------------
# GPU target
# ---------------------------------------------------------------------------
# Run with:  docker run --gpus all ...
#
# torch and onnxruntime-gpu MUST agree on the CUDA major version.
# onnxruntime-gpu does not ship the CUDA runtime -- it loads cuBLAS/cuDNN from
# the library path, which the CUDA torch wheel provides. A mismatch does not
# raise: get_available_providers() still lists CUDAExecutionProvider, the
# session silently falls back to CPU, and throughput quietly collapses.
# The pairing below is pinned for that reason. /health reports which providers
# actually loaded, so verify there after any bump.
FROM base AS gpu

USER root
RUN pip install --no-cache-dir \
        "torch==2.5.1" "torchaudio==2.5.1" \
        --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir "onnxruntime-gpu==1.20.2" \
    && pip install --no-cache-dir pynvml

# torch must be imported before onnxruntime for ORT to find the CUDA libraries;
# the engine forces that ordering in code. This makes them discoverable to the
# dynamic loader as well.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}
USER asr

ENV ASR_DEVICE=cuda

# ---------------------------------------------------------------------------
# GPU + KenLM target
# ---------------------------------------------------------------------------
# The reference final decoder is torchaudio's flashlight binding with KenLM.
# It has no prebuilt wheel and must be compiled, which is why it is a separate
# target: skip it and the service falls back to an LM-free beam search, which
# is materially worse. Build only if you are using --lm.
FROM gpu AS gpu-kenlm

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        libboost-all-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev \
    && pip install --no-cache-dir flashlight-text kenlm \
    && apt-get purge -y build-essential cmake git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
USER asr
