"""ONNX Runtime inference for the exported Conformer-CTC encoder+decoder.

This module does exactly one thing: turn features into CTC logits. It performs
no decoding and holds no hypothesis state -- keeping inference, decoding and
hypothesis tracking separate is what makes it possible to swap the greedy
decoder for beam search without touching the model path.

It also answers the question posed in section 30 of the design brief: *is this
graph actually stateless?* :meth:`ONNXASREngine.graph_report` inspects every
input and output for cache/state tensors before we commit to the rolling-window
strategy. If the export ever gains encoder caching, this is where it surfaces.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

#: Substrings that would indicate a stateful / cached streaming export.
_STATEFUL_HINTS = (
    "cache", "state", "mem", "hidden", "prev", "past", "carry", "context_buf",
)


def _preload_cuda_runtime() -> None:
    """Import torch before onnxruntime so ORT can find the CUDA libraries.

    On Windows this is the difference between a working GPU session and a
    silent fall back to CPU. ``onnxruntime-gpu`` does not ship the CUDA runtime;
    it loads ``cublas``/``cudnn`` from the DLL search path. A pip-installed
    CUDA build of torch does ship them, and registers its ``torch/lib``
    directory when imported -- but only if it is imported *first*.

    Verified on this machine: with torch imported first the session reports
    ``['CUDAExecutionProvider', 'CPUExecutionProvider']``; without it, the same
    call reports ``['CPUExecutionProvider']`` and gives no error, only a log
    line about a missing DLL.

    Harmless when torch is absent or CPU-only.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        pass


@dataclass
class TensorSpec:
    name: str
    shape: tuple[Any, ...]
    dtype: str

    def __str__(self) -> str:
        shape = "x".join(str(d) for d in self.shape)
        return f"{self.name}: {self.dtype}[{shape}]"


@dataclass
class ModelGraphReport:
    """Structural facts about the loaded ONNX graph."""

    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    providers: list[str]
    stateful_inputs: list[str] = field(default_factory=list)
    stateful_outputs: list[str] = field(default_factory=list)

    @property
    def is_stateless(self) -> bool:
        """True if no cache/state tensors were found in either direction."""
        return not self.stateful_inputs and not self.stateful_outputs

    @property
    def vocab_size(self) -> Optional[int]:
        """Output units, i.e. vocabulary + blank, if statically known."""
        if not self.outputs:
            return None
        last = self.outputs[0].shape[-1]
        return last if isinstance(last, int) else None

    def render(self) -> str:
        lines = ["ONNX graph report", "-" * 60]
        lines.append(f"providers: {', '.join(self.providers)}")
        lines.append("inputs:")
        lines += [f"  {spec}" for spec in self.inputs]
        lines.append("outputs:")
        lines += [f"  {spec}" for spec in self.outputs]
        if self.is_stateless:
            lines.append(
                "state: NONE FOUND -> model is stateless/full-context. "
                "Rolling-window inference is the correct strategy."
            )
        else:
            lines.append(
                f"state: FOUND -> inputs={self.stateful_inputs} outputs={self.stateful_outputs}. "
                "True incremental inference may be possible; investigate before "
                "paying the rolling-window recompute cost."
            )
        return "\n".join(lines)


@dataclass
class InferenceResult:
    """One model call."""

    logits: np.ndarray          # (B, T_out, V)
    input_frames: int           # T_in, feature frames fed in
    output_frames: int          # T_out, CTC frames produced
    inference_time: float       # seconds, model only
    preprocess_time: float = 0.0

    @property
    def subsampling_factor(self) -> float:
        if self.output_frames == 0:
            return 0.0
        return self.input_frames / self.output_frames


class ONNXASREngine:
    """Loads the ONNX model once and runs it on demand.

    Args:
        model_path: Path to the exported ``.onnx`` file.
        providers: ``"auto"`` prefers CUDA and falls back to CPU; otherwise an
            explicit list of ONNX Runtime provider names.
        intra_op_threads: 0 leaves the ORT default.
    """

    def __init__(
        self,
        model_path: str,
        providers: str | Sequence[str] = "auto",
        intra_op_threads: int = 0,
        device_id: int = 0,
    ) -> None:
        _preload_cuda_runtime()
        import onnxruntime as ort

        self.model_path = model_path
        self._ort = ort

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads > 0:
            sess_options.intra_op_num_threads = intra_op_threads

        resolved = self._resolve_providers(providers)
        logger.info("Loading ONNX model %s with providers=%s", model_path, resolved)
        self.session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=resolved
        )

        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._active_providers = list(self.session.get_providers())
        self._device_id = device_id
        # Zero-copy handoff is attempted only once; if it fails we fall back
        # permanently rather than raising per chunk.
        self._io_binding_failed = False

        # NeMo exports name these 'audio_signal' and 'length'. Resolve by name
        # where possible, fall back to position, so a differently-named export
        # still works.
        self._features_input = self._pick_input(("audio_signal", "features", "input"), 0)
        self._length_input = self._pick_input(("length", "lengths", "input_length"), 1)

        self._graph_report = self._build_graph_report(resolved)
        # Counters are mutated from every request thread in the server, so they
        # need a lock. ORT's own Run() is thread-safe; our bookkeeping is not.
        self._stats_lock = threading.Lock()

        # Populated on the first call; the ratio of feature frames to CTC
        # frames is the encoder's subsampling factor (4 for a standard
        # Conformer), which sets the resolution of every timestamp we produce.
        self._subsampling_factor: Optional[int] = None
        #: Width of the largest input the factor has been estimated from. The
        #: estimate is only trustworthy once this is comfortably larger than
        #: the encoder's receptive field.
        self._subsampling_probe_frames = 0
        self._call_count = 0
        self._total_inference_time = 0.0

    # ---- setup helpers ---------------------------------------------------

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
        if "TensorrtExecutionProvider" in available:
            return ["TensorrtExecutionProvider", "CPUExecutionProvider"]
        logger.warning(
            "CUDAExecutionProvider unavailable; falling back to CPU. Expect a "
            "much worse RTF -- the rolling-window strategy reprocesses each "
            "sample many times over."
        )
        return ["CPUExecutionProvider"]

    def _pick_input(self, candidates: Sequence[str], fallback_index: int) -> str:
        for name in candidates:
            if name in self._input_names:
                return name
        if fallback_index < len(self._input_names):
            chosen = self._input_names[fallback_index]
            logger.warning(
                "No input named any of %s; falling back to positional input '%s'",
                candidates, chosen,
            )
            return chosen
        raise ValueError(
            f"Model has {len(self._input_names)} inputs ({self._input_names}); "
            f"cannot resolve any of {candidates}"
        )

    def _build_graph_report(self, providers: list[str]) -> ModelGraphReport:
        def spec(meta: Any) -> TensorSpec:
            shape = tuple(d if isinstance(d, int) else (d or "?") for d in (meta.shape or ()))
            return TensorSpec(name=meta.name, shape=shape, dtype=str(meta.type))

        inputs = [spec(i) for i in self.session.get_inputs()]
        outputs = [spec(o) for o in self.session.get_outputs()]

        def stateful(names: Sequence[str], exclude: Sequence[str]) -> list[str]:
            return [
                n for n in names
                if n not in exclude and any(h in n.lower() for h in _STATEFUL_HINTS)
            ]

        # 'length' is a sequence length, not a state tensor - exclude it.
        known = (self._features_input, self._length_input)
        return ModelGraphReport(
            inputs=inputs,
            outputs=outputs,
            providers=providers,
            stateful_inputs=stateful([i.name for i in inputs], known),
            stateful_outputs=stateful([o.name for o in outputs], ()),
        )

    # ---- introspection ---------------------------------------------------

    @property
    def graph_report(self) -> ModelGraphReport:
        return self._graph_report

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    @property
    def subsampling_factor(self) -> Optional[int]:
        """Feature frames per CTC frame; known after the first call."""
        return self._subsampling_factor

    def ctc_frame_duration(self, hop_duration: float) -> float:
        """Seconds of audio per CTC output frame.

        40 ms for a 4x-subsampled Conformer at a 10 ms hop. This is the
        resolution of every token timestamp the pipeline produces.
        """
        factor = self._subsampling_factor or 4
        return hop_duration * factor

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_inference_time(self) -> float:
        return self._total_inference_time

    @property
    def average_inference_time(self) -> float:
        return self._total_inference_time / self._call_count if self._call_count else 0.0

    # ---- inference -------------------------------------------------------

    def run(self, features: np.ndarray, feature_length: int | np.ndarray) -> InferenceResult:
        """Run the model on one batch of features.

        Args:
            features: ``(B, n_mels, T)`` float32.
            feature_length: Valid frames per batch item.

        Returns:
            :class:`InferenceResult` with logits ``(B, T_out, V)``.
        """
        if features.dtype != np.float32:
            features = features.astype(np.float32, copy=False)
        features = np.ascontiguousarray(features)

        if isinstance(feature_length, np.ndarray):
            lengths = feature_length.astype(np.int64, copy=False)
        else:
            # Deliberately allocated per call rather than reused. A shared
            # scratch buffer races when one session serves several concurrent
            # streams: a second thread can overwrite the length before ORT has
            # read it, silently masking the first stream's features to the
            # wrong number of frames. An 8-byte allocation is free next to a
            # model call.
            lengths = np.array([int(feature_length)], dtype=np.int64)

        ort_inputs = {
            self._features_input: features,
            self._length_input: lengths,
        }

        start = time.perf_counter()
        outputs = self.session.run(None, ort_inputs)
        elapsed = time.perf_counter() - start

        return self._finish(outputs[0], int(features.shape[-1]), elapsed)

    def _finish(
        self, logits: np.ndarray, input_frames: int, elapsed: float
    ) -> InferenceResult:
        """Validate the output, learn the subsampling factor, record timing."""
        if logits.ndim != 3:
            raise ValueError(
                f"Expected 3-D logits (B, T, V) from output '{self._output_names[0]}', "
                f"got shape {logits.shape}"
            )

        output_frames = int(logits.shape[1])
        if output_frames > 0 and input_frames > self._subsampling_probe_frames:
            # Estimate from the LARGEST input seen, not merely the first.
            #
            # Convolutional subsampling pads at the edges, so the ratio is only
            # asymptotically the true factor. On a short warm-up window a 4x
            # encoder measures as 3x (17 feature frames -> 5 CTC frames = 3.4),
            # and latching that would scale every timestamp the pipeline
            # produces by 3/4 -- silently, since the values stay self-consistent
            # and the transcript still reads correctly.
            factor = max(1, int(round(input_frames / output_frames)))
            if factor != self._subsampling_factor:
                logger.info(
                    "Encoder subsampling factor: %d (%d feature frames -> %d CTC "
                    "frames); CTC frame period is %d x the feature hop",
                    factor, input_frames, output_frames, factor,
                )
            self._subsampling_factor = factor
            self._subsampling_probe_frames = input_frames

        with self._stats_lock:
            self._call_count += 1
            self._total_inference_time += elapsed

        return InferenceResult(
            logits=logits,
            input_frames=input_frames,
            output_frames=output_frames,
            inference_time=elapsed,
        )

    @property
    def active_providers(self) -> list[str]:
        """Providers the session actually loaded, not the ones requested."""
        return list(self._active_providers)

    @property
    def on_cuda(self) -> bool:
        return any("CUDA" in p or "Tensorrt" in p for p in self._active_providers)

    def run_torch(self, features: "Any", feature_length: int) -> InferenceResult:
        """Run the model on a torch tensor, avoiding a host round trip on GPU.

        When the frontend has already produced features in CUDA memory,
        converting them to NumPy would copy device -> host only for ONNX
        Runtime to copy them straight back. ``io_binding`` lets ORT read the
        existing device buffer instead.

        Falls back to the NumPy path for CPU tensors, for sessions without a
        CUDA provider, and permanently after any binding failure -- an
        optimisation must never be the reason inference stops working.
        """
        import torch

        if not (
            isinstance(features, torch.Tensor)
            and features.is_cuda
            and self.on_cuda
            and not self._io_binding_failed
        ):
            array = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) \
                else features
            return self.run(array, feature_length)

        tensor = features.contiguous().to(torch.float32)
        try:
            return self._run_io_binding(tensor, feature_length)
        except Exception as exc:  # pragma: no cover - needs a CUDA device
            self._io_binding_failed = True
            logger.warning(
                "CUDA zero-copy io_binding failed (%s); falling back to host "
                "transfers for the rest of this session.", exc,
            )
            return self.run(tensor.detach().cpu().numpy(), feature_length)

    def _run_io_binding(self, tensor: "Any", feature_length: int) -> InferenceResult:
        """Bind a CUDA tensor's buffer directly as the model input."""
        import numpy as _np

        binding = self.session.io_binding()
        binding.bind_input(
            name=self._features_input,
            device_type="cuda",
            device_id=self._device_id,
            element_type=_np.float32,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
        # The length input is a single int64; keeping it on the host is
        # cheaper than staging a device buffer for 8 bytes.
        binding.bind_cpu_input(
            self._length_input, _np.array([int(feature_length)], dtype=_np.int64)
        )
        for name in self._output_names:
            binding.bind_output(name, "cpu")

        start = time.perf_counter()
        # Ensure the frontend's kernels have completed before ORT reads the
        # buffer; torch and ORT do not share a stream.
        binding.synchronize_inputs()
        self.session.run_with_iobinding(binding)
        binding.synchronize_outputs()
        elapsed = time.perf_counter() - start

        logits = binding.copy_outputs_to_cpu()[0]
        return self._finish(logits, int(tensor.shape[-1]), elapsed)

    def warmup(self, n_mels: int, n_frames: int, iterations: int = 2) -> None:
        """Run throwaway inferences so the first real chunk is not an outlier.

        ONNX Runtime allocates arenas and (on CUDA) autotunes kernels lazily.
        Without this the first-partial-latency metric measures setup, not
        steady-state behaviour.

        **Startup only.** The engine is shared across every stream in the
        server, so this must not be called on a live one: it rewinds the
        counters, and any inference a concurrent stream records while it runs
        is discarded along with the warm-up's own. ``ModelPool`` calls it once,
        before serving.
        """
        with self._stats_lock:
            baseline_calls = self._call_count
            baseline_time = self._total_inference_time

        dummy = np.zeros((1, n_mels, n_frames), dtype=np.float32)
        for _ in range(iterations):
            self.run(dummy, n_frames)

        # Warm-up calls are not real work; keep them out of the metrics.
        # Restoring the baseline rather than zeroing preserves any history from
        # before, and the write is locked so it cannot tear against a
        # concurrent increment.
        with self._stats_lock:
            self._call_count = baseline_calls
            self._total_inference_time = baseline_time

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]
