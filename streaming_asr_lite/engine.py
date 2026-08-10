"""ONNX Runtime engine without the torch CUDA preload.

Why not reuse :class:`streaming_asr.inference.ONNXASREngine`? Its constructor
used to import torch outright so that ONNX Runtime could find ``cublas`` and
``cudnn`` from ``torch/lib`` on Windows -- correct for the main package, and
self-defeating for a runtime that exists to remove torch.

Both now call :func:`streaming_asr_lite.execution.ensure_cuda_libraries`, which
registers that same directory **without importing the module**. So this engine
gets CUDA where it is available, at no import cost, and stops depending on the
CUDA toolkit or the container image being right.

Everything torch-free is reused -- the report dataclasses, the stateless-graph
detection, the subsampling-factor logic all come from the main package. Only
the constructor differs.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional, Sequence

import numpy as np

from streaming_asr.inference.onnx_engine import (
    _STATEFUL_HINTS,
    InferenceResult,
    ModelGraphReport,
    TensorSpec,
)
from streaming_asr_lite.execution import ensure_cuda_libraries, resolve_intra_op_threads

logger = logging.getLogger(__name__)


class LiteONNXEngine:
    """Loads the model once and runs it. No torch, at import or at runtime."""

    def __init__(
        self,
        model_path: str,
        providers: str | Sequence[str] = "auto",
        intra_op_threads: int = 0,
        max_concurrent_streams: int = 1,
    ) -> None:
        ensure_cuda_libraries()
        import onnxruntime as ort

        self.model_path = model_path
        self._ort = ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        resolved = self._resolve_providers(providers)
        threads, reason = resolve_intra_op_threads(
            intra_op_threads, resolved, max_concurrent_streams
        )
        if threads > 0:
            options.intra_op_num_threads = threads
            logger.info("intra_op_threads=%d (%s)", threads, reason)

        logger.info("Loading ONNX model %s with providers=%s", model_path, resolved)
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=resolved
        )

        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._features_input = self._pick_input(("audio_signal", "features", "input"), 0)
        self._length_input = self._pick_input(("length", "lengths", "input_length"), 1)

        self._active_providers = list(self.session.get_providers())
        self._graph_report = self._build_graph_report(self._active_providers)
        self._subsampling_factor: Optional[int] = None
        self._subsampling_probe_frames = 0
        self._stats_lock = threading.Lock()
        self._call_count = 0
        self._total_inference_time = 0.0

    # ---- setup ----------------------------------------------------------

    def _resolve_providers(self, providers: str | Sequence[str]) -> list[str]:
        available = self._ort.get_available_providers()
        if providers != "auto":
            requested = [providers] if isinstance(providers, str) else list(providers)
            missing = [p for p in requested if p not in available]
            if missing:
                raise ValueError(
                    f"Requested providers {missing} are unavailable. Available: {available}"
                )
            return requested
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _pick_input(self, candidates: Sequence[str], fallback_index: int) -> str:
        for name in candidates:
            if name in self._input_names:
                return name
        if fallback_index < len(self._input_names):
            chosen = self._input_names[fallback_index]
            logger.warning("No input named any of %s; using positional '%s'",
                           candidates, chosen)
            return chosen
        raise ValueError(f"cannot resolve any of {candidates} in {self._input_names}")

    def _build_graph_report(self, providers: list[str]) -> ModelGraphReport:
        def spec(meta: Any) -> TensorSpec:
            shape = tuple(d if isinstance(d, int) else (d or "?") for d in (meta.shape or ()))
            return TensorSpec(name=meta.name, shape=shape, dtype=str(meta.type))

        inputs = [spec(i) for i in self.session.get_inputs()]
        outputs = [spec(o) for o in self.session.get_outputs()]

        def stateful(names: Sequence[str], exclude: Sequence[str]) -> list[str]:
            return [n for n in names
                    if n not in exclude and any(h in n.lower() for h in _STATEFUL_HINTS)]

        known = (self._features_input, self._length_input)
        return ModelGraphReport(
            inputs=inputs, outputs=outputs, providers=providers,
            stateful_inputs=stateful([i.name for i in inputs], known),
            stateful_outputs=stateful([o.name for o in outputs], ()),
        )

    # ---- introspection --------------------------------------------------

    @property
    def graph_report(self) -> ModelGraphReport:
        return self._graph_report

    @property
    def active_providers(self) -> list[str]:
        return list(self._active_providers)

    @property
    def on_cuda(self) -> bool:
        return any("CUDA" in p or "Tensorrt" in p for p in self._active_providers)

    @property
    def subsampling_factor(self) -> Optional[int]:
        return self._subsampling_factor

    def ctc_frame_duration(self, hop_duration: float) -> float:
        return hop_duration * (self._subsampling_factor or 4)

    @property
    def call_count(self) -> int:
        return self._call_count

    # ---- inference ------------------------------------------------------

    def run(self, features: np.ndarray, feature_length: int | np.ndarray) -> InferenceResult:
        if features.dtype != np.float32:
            features = features.astype(np.float32, copy=False)
        features = np.ascontiguousarray(features)

        if isinstance(feature_length, np.ndarray):
            lengths = feature_length.astype(np.int64, copy=False)
        else:
            # Allocated per call: a shared buffer races when one session serves
            # several concurrent streams.
            lengths = np.array([int(feature_length)], dtype=np.int64)

        start = time.perf_counter()
        outputs = self.session.run(
            None, {self._features_input: features, self._length_input: lengths}
        )
        elapsed = time.perf_counter() - start

        logits = outputs[0]
        if logits.ndim != 3:
            raise ValueError(f"expected 3-D logits, got {logits.shape}")

        input_frames = int(features.shape[-1])
        output_frames = int(logits.shape[1])
        # Estimate from the largest input seen; a short warm-up window
        # under-reports the factor and would mis-scale every timestamp.
        if output_frames > 0 and input_frames > self._subsampling_probe_frames:
            self._subsampling_factor = max(1, int(round(input_frames / output_frames)))
            self._subsampling_probe_frames = input_frames

        with self._stats_lock:
            self._call_count += 1
            self._total_inference_time += elapsed

        return InferenceResult(
            logits=logits, input_frames=input_frames,
            output_frames=output_frames, inference_time=elapsed,
        )

    def warmup(self, n_mels: int, n_frames: int, iterations: int = 2) -> None:
        dummy = np.zeros((1, n_mels, n_frames), dtype=np.float32)
        with self._stats_lock:
            baseline_calls, baseline_time = self._call_count, self._total_inference_time
        for _ in range(iterations):
            self.run(dummy, n_frames)
        with self._stats_lock:
            self._call_count = baseline_calls
            self._total_inference_time = baseline_time

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]
