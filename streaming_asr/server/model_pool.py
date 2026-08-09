"""Process-wide model residency.

The whole point of running as a service: load once, serve many.

Three things are expensive to construct and must be shared across requests:

* the ONNX session (seconds to load; on GPU it also owns a CUDA arena, and a
  second session would double GPU memory for no benefit),
* the KenLM decoder (the LM binary is routinely gigabytes),
* the mel frontend's filterbank matrices.

Everything else is per-stream conversation state -- the rolling buffer, the
hypothesis tracker, the metrics -- and must **not** be shared, or two callers
would interleave into one transcript.

Thread-safety
-------------
``session.Run()`` is thread-safe in ONNX Runtime, so the engine is shared
directly. The final decoder is not assumed to be: flashlight's decoder holds
internal beam state, so calls are serialised behind a lock. That is an
acceptable trade because finalisation happens once per utterance, not per
chunk -- whereas cloning the decoder per request would mean re-reading the LM.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from streaming_asr.config import StreamingASRConfig
from streaming_asr.decoding.beam_ctc_lm import (
    BeamDecodeResult,
    FinalDecoder,
    build_final_decoder,
)
from streaming_asr.device import gpu_name, resolve_device
from streaming_asr.inference.onnx_engine import ONNXASREngine
from streaming_asr.pipeline import StreamingASRPipeline

logger = logging.getLogger(__name__)


class LockedFinalDecoder(FinalDecoder):
    """Serialises access to a decoder that is not known to be thread-safe."""

    def __init__(self, inner: FinalDecoder) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.name = inner.name
        self.used_lm = inner.used_lm

    def decode(self, logits: np.ndarray) -> BeamDecodeResult:
        with self._lock:
            return self._inner.decode(logits)


@dataclass
class PoolStatus:
    ready: bool
    model_path: str
    runtime: str
    frontend_device: str
    providers: list[str]
    gpu: Optional[str]
    zero_copy: bool
    decoder_backend: str
    used_lm: bool
    subsampling_factor: Optional[int]
    vocab_size: int
    stateless_graph: bool
    load_seconds: float
    active_streams: int
    max_concurrent_streams: int
    total_model_calls: int


class ModelPool:
    """Holds the loaded model for the lifetime of the process."""

    def __init__(self, config: StreamingASRConfig, max_concurrent_streams: int = 4) -> None:
        started = time.perf_counter()
        self.config = config
        self.max_concurrent_streams = max_concurrent_streams

        if not config.onnx_model_path:
            raise ValueError("onnx_model_path is required")
        logger.info("Loading model: %s", config.onnx_model_path)

        from streaming_asr_lite.factory import build_engine, describe_runtime

        if config.runtime == "lite":
            # No torch device to place: the frontend is an ONNX graph running on
            # the encoder's providers. Calling resolve_device() here would import
            # torch and defeat the point of this runtime.
            self.placement = None
            self.engine = build_engine(config)
        else:
            self.placement = resolve_device(config.device, config.providers)
            logger.info("Placement: %s", self.placement.describe())
            self.engine = build_engine(config, providers=self.placement.providers)

        logger.info("Runtime: %s", describe_runtime(config, self.engine))

        # Build the final decoder eagerly. Deferring it would move a
        # multi-second KenLM load onto the first user's request instead of
        # startup, and hide a missing-file error until then.
        self._final_decoder: Optional[FinalDecoder] = None
        if config.final_beam_decode:
            self._final_decoder = LockedFinalDecoder(build_final_decoder(config))
            logger.info(
                "Final decoder: %s (lm=%s)",
                self._final_decoder.name, self._final_decoder.used_lm,
            )

        self._active = 0
        self._active_lock = threading.Lock()
        self.load_seconds = time.perf_counter() - started

        self._warmup()
        logger.info("Model ready in %.2fs", self.load_seconds)

    def _warmup(self) -> None:
        """Run the model once so the first request is not an outlier.

        ORT allocates arenas and autotunes CUDA kernels lazily; without this the
        first caller pays for it and sees a latency spike that has nothing to do
        with their audio.
        """
        pipeline = self.new_pipeline()
        pipeline.warmup(iterations=2)

    # ---- per-stream state ------------------------------------------------

    def new_pipeline(self):
        """Create an isolated pipeline sharing the loaded model.

        The engine and decoder are shared; all per-stream state is not.
        """
        from streaming_asr_lite.factory import build_pipeline

        return build_pipeline(
            self.config, engine=self.engine, final_decoder=self._final_decoder
        )

    # ---- admission control ----------------------------------------------

    def try_acquire(self) -> bool:
        """Reserve a stream slot, or refuse.

        Concurrency has to be capped. Rolling-window inference reprocesses each
        sample ``buffer/chunk`` times -- 25 at the reference operating point --
        so N concurrent streams cost 25N times real time. Past the limit,
        accepting another stream does not slow one caller down politely; it
        pushes *every* caller past real time at once and they all fall
        permanently behind.
        """
        with self._active_lock:
            if self._active >= self.max_concurrent_streams:
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._active_lock:
            self._active = max(0, self._active - 1)

    @property
    def active_streams(self) -> int:
        with self._active_lock:
            return self._active

    # ---- introspection ---------------------------------------------------

    def status(self) -> PoolStatus:
        report = self.engine.graph_report
        decoder = self._final_decoder
        return PoolStatus(
            ready=True,
            model_path=str(self.config.onnx_model_path),
            runtime=self.config.runtime,
            frontend_device=(
                "onnx" if self.placement is None else self.placement.torch_device
            ),
            providers=self.engine.active_providers,
            gpu=(
                gpu_name(self.placement.device_id)
                if self.placement is not None and self.placement.on_cuda else None
            ),
            # Under the lite runtime both the frontend and the encoder are ONNX
            # sessions on the same providers, so features never leave the device.
            zero_copy=(
                self.engine.on_cuda if self.placement is None
                else (self.placement.zero_copy_possible and self.engine.on_cuda)
            ),
            decoder_backend=decoder.name if decoder else "disabled",
            used_lm=bool(decoder and decoder.used_lm),
            subsampling_factor=self.engine.subsampling_factor,
            vocab_size=len(self.config.ensure_blank_in_vocabulary()),
            stateless_graph=report.is_stateless,
            load_seconds=round(self.load_seconds, 3),
            active_streams=self.active_streams,
            max_concurrent_streams=self.max_concurrent_streams,
            total_model_calls=self.engine.call_count,
        )

    def close(self) -> None:
        self.engine.close()
