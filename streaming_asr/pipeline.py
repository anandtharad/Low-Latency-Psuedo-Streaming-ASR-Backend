"""The streaming ASR pipeline.

This is *sliding-window / rolling-buffer pseudo-streaming inference*, not true
streaming Conformer inference. The distinction matters. The model has no
encoder cache and no recurrent state; we simply re-run an offline, full-context
model on overlapping windows of a rolling buffer and reconcile the results.

The cost of that choice is visible in one number:
``config.window_redundancy`` -- 25 at the reference operating point, meaning
every sample of audio is processed 25 times. The benefit is that no model
surgery, retraining or re-export is required.

Flow per chunk::

    chunk -> rolling buffer -> preprocessing -> ONNX -> CTC logits
                                                          |
                                       greedy decode (fast, per chunk)
                                                          |
                                       hypothesis tracker (stabilise)
                                                          |
                                       committed / partial -> ASREvent

and once, at the endpoint::

    retained audio -> preprocessing -> ONNX -> beam search + KenLM -> FINAL
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from streaming_asr.audio.base import AudioChunk, AudioSource
from streaming_asr.buffer.rolling_buffer import RollingAudioBuffer
from streaming_asr.config import StreamingASRConfig
from streaming_asr.decoding.beam_ctc_lm import FinalDecoder, build_final_decoder
from streaming_asr.decoding.greedy_ctc import GreedyCTCDecoder
from streaming_asr.device import RuntimePlacement, resolve_device
from streaming_asr.endpointing.endpoint import (
    CompositeEndpointDetector,
    EndpointDetector,
    ExplicitEndpointDetector,
    build_endpoint_detector,
)
from streaming_asr.events import ASREvent, ASREventType
from streaming_asr.hypothesis.aligner import HypothesisAligner, build_aligner
from streaming_asr.hypothesis.tracker import HypothesisTracker
from streaming_asr.inference.onnx_engine import ONNXASREngine
from streaming_asr.metrics import MetricsCollector
from streaming_asr.preprocessing.filterbank import Preprocessor
from streaming_asr.types import GreedyHypothesis

logger = logging.getLogger(__name__)

EventCallback = Callable[[ASREvent], None]


class StreamingASRPipeline:
    """Orchestrates buffering, inference, decoding and stabilisation.

    Every heavyweight component is constructed once, in ``__init__``, and
    reused for the life of the pipeline: the ONNX session, the preprocessing
    module and the KenLM decoder all have setup costs that would dominate if
    paid per chunk.

    Args:
        config: Fully-populated configuration.
        engine: Pre-built inference engine. Constructed from
            ``config.onnx_model_path`` when omitted.
        final_decoder: Pre-built final decoder. Built lazily on first use when
            omitted, so that a streaming-only session never pays to load KenLM.
        aligner / endpoint_detector: Overrides for the configured defaults.
    """

    def __init__(
        self,
        config: StreamingASRConfig,
        engine: Optional[ONNXASREngine] = None,
        final_decoder: Optional[FinalDecoder] = None,
        aligner: Optional[HypothesisAligner] = None,
        endpoint_detector: Optional[EndpointDetector] = None,
    ) -> None:
        self.config = config
        self.vocabulary = config.ensure_blank_in_vocabulary()

        # Frontend and model placement are decided together; see device.py.
        self.placement = resolve_device(config.device, config.providers)
        logger.info("Runtime placement: %s", self.placement.describe())

        if engine is None:
            if not config.onnx_model_path:
                raise ValueError("config.onnx_model_path is required when no engine is supplied")
            engine = ONNXASREngine(
                model_path=config.onnx_model_path,
                providers=self.placement.providers,
                intra_op_threads=config.intra_op_threads,
                device_id=self.placement.device_id,
            )
        self.engine = engine
        self.placement = self._reconcile_placement(self.placement, engine)

        self.preprocessor = Preprocessor(
            config.preprocessing, device=self.placement.torch_device
        )
        self.greedy = GreedyCTCDecoder(
            vocabulary=self.vocabulary,
            blank_id=config.resolved_blank_id,
        )
        self.tracker = HypothesisTracker(
            aligner=aligner or build_aligner(
                config.stability.aligner, config.stability.time_tolerance
            ),
            stability_window=config.stability.stability_window,
            min_stable_updates=config.stability.min_stable_updates,
            time_tolerance=config.stability.time_tolerance,
        )
        self.endpoint_detector = endpoint_detector or build_endpoint_detector(
            config.endpoint, config.sample_rate
        )
        self._final_decoder = final_decoder

        self.buffer = RollingAudioBuffer(
            buffer_samples=config.buffer_samples,
            chunk_samples=config.chunk_samples,
            sample_rate=config.sample_rate,
        )

        self._config_summary = {
            "chunk_duration": config.chunk_duration,
            "context_duration": config.context_duration,
            "buffer_duration": config.buffer_duration,
            "window_redundancy": config.window_redundancy,
            "aligner": config.stability.aligner,
            "stability_window": config.stability.stability_window,
            "min_stable_updates": config.stability.min_stable_updates,
        }
        self.metrics = self._new_metrics()

        self._history: list[np.ndarray] = []
        self._history_samples = 0
        self._max_history_samples = int(config.max_history * config.sample_rate)
        self._callbacks: list[EventCallback] = []
        self._ended = False
        self._finalized = False
        self._last_hypothesis: Optional[GreedyHypothesis] = None
        self._audio_time = 0.0

        self._validate_against_model()

        logger.info(
            "Pipeline ready: window=%.2fs (context=%.2fs + chunk=%.2fs), redundancy=%.1fx",
            config.buffer_duration, config.context_duration, config.chunk_duration,
            config.window_redundancy,
        )

    def _validate_against_model(self) -> None:
        """Check the config against the graph before any audio is processed.

        Both mismatches below are silent in the worst way. A wrong vocabulary
        size means every argmax indexes a different token table, so the output
        is fluent-looking nonsense rather than an error. A wrong mel count is
        caught by ONNX Runtime, but only on the first chunk and with a shape
        error that says nothing about which knob is wrong.

        Cheap to check once, here, where the message can name the fix.
        """
        report = self.engine.graph_report

        expected_mels = self.config.preprocessing.features
        graph_mels = report.inputs[0].shape[1] if report.inputs else None
        if isinstance(graph_mels, int) and graph_mels != expected_mels:
            raise ValueError(
                f"Model expects {graph_mels} mel bins but preprocessing is configured "
                f"for {expected_mels}. Set PreprocessingConfig(features={graph_mels}) "
                f"-- the frontend must match the one used at training/export."
            )

        vocab_size = report.vocab_size
        configured = len(self.vocabulary)
        if isinstance(vocab_size, int) and vocab_size != configured:
            raise ValueError(
                f"Model has {vocab_size} output units but the vocabulary has "
                f"{configured} entries (including the CTC blank).\n"
                f"This is almost always the wrong vocabulary for this checkpoint, "
                f"and it fails silently: decoding would index a different token "
                f"table and emit confident, wrong text.\n"
                f"Extract the right one with:\n"
                f"  python tools/extract_vocabulary.py --nemo <model>.nemo "
                f"--out vocabulary.txt --verify {self.config.onnx_model_path}\n"
                f"then pass --vocabulary vocabulary.txt"
            )

    @staticmethod
    def _reconcile_placement(placement, engine) -> "RuntimePlacement":
        """Correct the frontend device against what the session actually loaded.

        ``onnxruntime.get_available_providers()`` reports what the build was
        *compiled* with, not what can actually be created. A CUDA provider whose
        dependencies are missing -- a version mismatch between onnxruntime-gpu
        and the installed CUDA runtime is the common case -- is still listed,
        then silently falls back to CPU at session creation.

        Left unchecked that produces the exact split placement this module
        exists to avoid: frontend on GPU, model on CPU, and a device-to-host
        copy of the features on every single window. Trust
        ``session.get_providers()``, which reports reality.
        """
        engine_on_cuda = getattr(engine, "on_cuda", False)
        if not placement.on_cuda or engine_on_cuda:
            return placement

        active = getattr(engine, "active_providers", ["CPUExecutionProvider"])
        logger.warning(
            "CUDA was selected but the ONNX session actually loaded %s. Moving the "
            "mel frontend back to CPU to avoid a device-to-host copy every window. "
            "This usually means onnxruntime-gpu does not match the installed CUDA "
            "runtime; check the ORT startup log for the missing library.",
            ", ".join(active),
        )
        return replace(placement, torch_device="cpu", providers=list(active),
                       reason="downgraded: CUDA provider failed to load")

    def _new_metrics(self) -> MetricsCollector:
        metrics = MetricsCollector()
        metrics.config_summary = dict(self._config_summary)
        # Only report GPU figures when GPU work actually happened; a
        # CUDA-capable torch build on a CPU run would otherwise look busy.
        metrics.on_gpu = self.placement.on_cuda or self.engine.on_cuda
        metrics.gpu_device_id = self.placement.device_id
        return metrics

    # ---- setup -----------------------------------------------------------

    def warmup(self, iterations: int = 2) -> None:
        """Pre-run the model so first-chunk latency is not a setup artefact."""
        features, _ = self.preprocessor(
            np.zeros((1, self.config.buffer_samples), dtype=np.float32),
            n_samples=self.config.buffer_samples,
        )
        self.engine.warmup(
            n_mels=int(features.shape[1]), n_frames=int(features.shape[2]), iterations=iterations
        )

    @property
    def final_decoder(self) -> FinalDecoder:
        """The beam+LM decoder, built on first access."""
        if self._final_decoder is None:
            self._final_decoder = build_final_decoder(self.config)
        return self._final_decoder

    def on_event(self, callback: EventCallback) -> None:
        """Register a callback invoked for every emitted event."""
        self._callbacks.append(callback)

    def _emit(self, event: ASREvent) -> ASREvent:
        for callback in self._callbacks:
            callback(event)
        return event

    # ---- streaming -------------------------------------------------------

    def process_chunk(self, chunk: AudioChunk) -> list[ASREvent]:
        """Process one chunk of audio and return any resulting events."""
        if self._ended:
            logger.warning("Chunk ignored: the pipeline has already endpointed")
            return []

        self.metrics.mark_first_audio()
        events: list[ASREvent] = []

        self._retain(chunk.samples)
        self.buffer.push(chunk.samples)
        self._audio_time = self.buffer.current_time
        self.metrics.audio_duration = self._audio_time

        if self.config.greedy_decode:
            hypothesis, timings = self._infer_window()
            self._last_hypothesis = hypothesis

            tracker_start = time.perf_counter()
            state = self.tracker.update(hypothesis)
            tracker_time = time.perf_counter() - tracker_start

            self.metrics.record_window(
                preprocess_time=timings[0],
                inference_time=timings[1],
                decode_time=hypothesis.decode_time,
                tracker_time=tracker_time,
                chunk_capture_time=chunk.capture_time or None,
            )
            for word in state.newly_committed:
                self.metrics.record_commit(state.audio_time, word.end_time)

            if state.committed_text or state.partial_text:
                self.metrics.mark_first_partial()

            events.append(
                self._emit(
                    ASREvent(
                        type=ASREventType.PARTIAL,
                        timestamp=self._audio_time,
                        wall_time=time.perf_counter(),
                        committed_text=state.committed_text,
                        partial_text=state.partial_text,
                        full_hypothesis=state.full_hypothesis,
                        newly_committed=state.newly_committed,
                        confidence=_mean_posterior(hypothesis),
                        window_start=hypothesis.window_start_time,
                        window_end=hypothesis.window_end_time,
                        new_audio_start=hypothesis.new_audio_start_time,
                        new_audio_end=hypothesis.new_audio_end_time,
                    )
                )
            )

            if self.config.emit_metrics:
                events.append(
                    self._emit(
                        ASREvent(
                            type=ASREventType.METRICS,
                            timestamp=self._audio_time,
                            wall_time=time.perf_counter(),
                            metrics=self.metrics.snapshot(),
                        )
                    )
                )

        decision = self.endpoint_detector.update(chunk.samples, self._audio_time)
        if decision.is_endpoint:
            events.append(
                self._emit(
                    ASREvent(
                        type=ASREventType.ENDPOINT,
                        timestamp=self._audio_time,
                        wall_time=time.perf_counter(),
                        metrics={"reason": decision.reason},
                    )
                )
            )
            self._ended = True

        return events

    def _infer_window(self) -> tuple[GreedyHypothesis, tuple[float, float]]:
        """Preprocess, run the model and greedily decode the current window."""
        # While the buffer is still filling, feed only the real audio. Handing
        # the model seconds of digital silence it never saw in training makes
        # it hallucinate tokens into the padding, and skews the per-feature
        # normalisation statistics. ``pad_warmup_window`` restores the
        # reference behaviour.
        if self.config.pad_warmup_window or self.buffer.is_warm:
            window = self.buffer.window
            n_samples = self.config.buffer_samples
            window_start_time = self.buffer.window_start_time
        else:
            window = self.buffer.valid_window()
            n_samples = self.buffer.valid_samples
            window_start_time = self.buffer.valid_start_time

        pre_start = time.perf_counter()
        features, feature_lengths = self.preprocessor(window, n_samples=n_samples)
        length = int(feature_lengths[0])
        preprocess_time = time.perf_counter() - pre_start

        # run_torch keeps CUDA features in device memory instead of round
        # tripping through the host, and falls back to the NumPy path on CPU.
        result = self.engine.run_torch(features, length)

        hypothesis = self.greedy.decode(
            logits=result.logits,
            window_start_time=window_start_time,
            window_end_time=self.buffer.window_end_time,
            ctc_frame_duration=self.engine.ctc_frame_duration(self.preprocessor.hop_duration),
            new_audio_start_time=self.buffer.window_end_time - self.config.chunk_duration,
            new_audio_end_time=self.buffer.window_end_time,
            retain_frame_posteriors=self.config.retain_frame_posteriors,
        )
        return hypothesis, (preprocess_time, result.inference_time)

    def stream(self, source: AudioSource | Iterable[AudioChunk]) -> Iterator[ASREvent]:
        """Consume an audio source, yielding events, and finalise at the end."""
        chunks = source.stream() if isinstance(source, AudioSource) else source
        for chunk in chunks:
            for event in self.process_chunk(chunk):
                yield event
            if self._ended:
                break
        if not self._finalized:
            yield self.finalize()

    # ---- history ---------------------------------------------------------

    def _retain(self, samples: np.ndarray) -> None:
        """Keep audio for the final decode, bounded by ``max_history``."""
        self._history.append(np.asarray(samples, dtype=np.float32).copy())
        self._history_samples += len(samples)
        while self._history_samples > self._max_history_samples and len(self._history) > 1:
            dropped = self._history.pop(0)
            self._history_samples -= len(dropped)

    def retained_audio(self) -> np.ndarray:
        """All audio retained for finalisation, as one contiguous array."""
        if not self._history:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._history)

    # ---- endpoint & finalisation ----------------------------------------

    def end_of_speech(self) -> None:
        """Signal that the utterance is over. Further chunks are refused."""
        for detector in _explicit_detectors(self.endpoint_detector):
            detector.trigger()
        self._ended = True
        self.metrics.mark_endpoint()

    def finalize(self) -> ASREvent:
        """Produce the authoritative transcript.

        Re-runs the model over the whole retained utterance and decodes it with
        beam search + KenLM. This is not a merge of the streaming output: the
        streaming hypotheses were each produced from a truncated 4-second view
        of the audio, whereas this decode sees the entire utterance and has a
        language model. Concatenating provisional fragments would preserve
        their errors; re-decoding discards them.

        The streaming transcript is still reported, as
        ``ASREvent.provisional_text``, so the drift between the two can be
        measured.
        """
        if self._finalized:
            raise RuntimeError("finalize() has already been called")
        self._finalized = True
        self._ended = True

        if self.metrics.endpoint_wall is None:
            self.metrics.mark_endpoint()

        start = time.perf_counter()
        self.tracker.flush()
        provisional = self.tracker.committed_text

        text = provisional
        used_lm = False
        backend = "streaming"
        if self.config.final_beam_decode:
            audio = self.retained_audio()
            if audio.size:
                logits, inference_time = self._final_inference(audio)
                self.metrics.final_inference_time = inference_time
                decoder = self.final_decoder
                result = decoder.decode(logits)
                self.metrics.final_decode_time = result.decode_time
                text = result.text
                used_lm = result.used_lm
                backend = result.backend
                logger.info(
                    "Final decode via %s in %.3fs (inference %.3fs)",
                    result.backend, result.decode_time, inference_time,
                )

        self.metrics.finalize_time = time.perf_counter() - start
        self.metrics.mark_final()

        return self._emit(
            ASREvent(
                type=ASREventType.FINAL,
                timestamp=self._audio_time,
                wall_time=time.perf_counter(),
                text=text,
                provisional_text=provisional,
                committed_text=provisional,
                used_lm=used_lm,
                decoder=backend,
                metrics=self.metrics.snapshot(),
            )
        )

    def _final_inference(self, audio: np.ndarray) -> tuple[np.ndarray, float]:
        """Run the model over the full utterance, segmenting if it is long.

        The checkpoint was trained on utterances up to 11 s. A single pass over
        a 60 s recording is out of distribution, so past
        ``config.final_segment_duration`` the audio is cut into overlapping
        segments and the logits stitched, discarding half the overlap at each
        seam so no frame is taken from a segment edge.
        """
        sample_rate = self.config.sample_rate
        duration = len(audio) / sample_rate
        total_inference = 0.0

        if duration <= self.config.final_segment_duration:
            features, lengths = self.preprocessor(audio.reshape(1, -1), n_samples=len(audio))
            result = self.engine.run_torch(features, int(lengths[0]))
            return result.logits, result.inference_time

        logger.info(
            "Final audio is %.1fs; decoding in %.1fs segments with %.1fs overlap",
            duration, self.config.final_segment_duration, self.config.final_segment_overlap,
        )
        seg_samples = int(self.config.final_segment_duration * sample_rate)
        overlap_samples = int(self.config.final_segment_overlap * sample_rate)
        step = max(1, seg_samples - overlap_samples)

        frame_duration = self.engine.ctc_frame_duration(self.preprocessor.hop_duration)
        trim_frames = int(round((self.config.final_segment_overlap / 2) / frame_duration))

        pieces: list[np.ndarray] = []
        start = 0
        while start < len(audio):
            nominal_end = start + seg_samples
            if nominal_end >= len(audio):
                end = len(audio)
            else:
                # Optionally move the cut into a pause. Off by default: it did
                # not help when measured. See config.final_segment_snap.
                end = _snap_to_silence(
                    audio, nominal_end,
                    search=int(self.config.final_segment_snap * sample_rate),
                )

            segment = audio[start:end]
            features, lengths = self.preprocessor(segment.reshape(1, -1), n_samples=len(segment))
            result = self.engine.run_torch(features, int(lengths[0]))
            total_inference += result.inference_time

            logits = result.logits[0]
            left = 0 if start == 0 else trim_frames
            right = logits.shape[0] if end >= len(audio) else logits.shape[0] - trim_frames
            if right > left:
                pieces.append(logits[left:right])

            if end >= len(audio):
                break
            start = max(start + 1, end - overlap_samples)

        stitched = np.concatenate(pieces, axis=0)[None, ...]
        return stitched, total_inference

    # ---- lifecycle -------------------------------------------------------

    def reset(self) -> None:
        """Prepare for a new utterance, keeping all loaded models."""
        self.buffer.reset()
        self.tracker.reset()
        self.endpoint_detector.reset()
        self.metrics = self._new_metrics()
        self._history.clear()
        self._history_samples = 0
        self._ended = False
        self._finalized = False
        self._last_hypothesis = None
        self._audio_time = 0.0

    @property
    def is_finalized(self) -> bool:
        """True once :meth:`finalize` has run. Lets a caller interrupted
        mid-stream finalise without risking a second call."""
        return self._finalized

    @property
    def last_hypothesis(self) -> Optional[GreedyHypothesis]:
        """The most recent raw greedy hypothesis, for debugging."""
        return self._last_hypothesis


def _snap_to_silence(audio: np.ndarray, target: int, search: int, frame: int = 400) -> int:
    """Move a segment boundary to the quietest point within +/- ``search``.

    The idea is that cutting a long utterance at a fixed offset can land
    mid-word and truncate it on both sides of the seam. Measured on the
    synthetic long fixture it did not help, so ``final_segment_snap`` defaults
    to 0 and this is a no-op; see that field for the numbers.

    Args:
        audio: Full waveform.
        target: Nominal cut position, in samples.
        search: How far either side to look, in samples.
        frame: Energy analysis frame, in samples (25 ms at 16 kHz).

    Returns:
        The chosen cut position, clamped to the bounds of ``audio``.
    """
    if search <= 0:
        return min(target, len(audio))

    lo = max(frame, target - search)
    hi = min(len(audio) - frame, target + search)
    if hi <= lo:
        return min(target, len(audio))

    positions = np.arange(lo, hi, frame // 2, dtype=np.int64)
    if positions.size == 0:
        return min(target, len(audio))

    # RMS of the window centred on each candidate cut.
    offsets = np.arange(-frame // 2, frame // 2)
    windows = audio[positions[:, None] + offsets[None, :]]
    energy = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1))

    return int(positions[int(np.argmin(energy))])


def _mean_posterior(hypothesis: GreedyHypothesis) -> Optional[float]:
    if not hypothesis.token_spans:
        return None
    return sum(t.posterior for t in hypothesis.token_spans) / len(hypothesis.token_spans)


def _explicit_detectors(detector: EndpointDetector) -> list[ExplicitEndpointDetector]:
    """Find the explicit detector(s) inside a possibly-composite detector."""
    if isinstance(detector, ExplicitEndpointDetector):
        return [detector]
    if isinstance(detector, CompositeEndpointDetector):
        found: list[ExplicitEndpointDetector] = []
        for child in detector.detectors:
            found.extend(_explicit_detectors(child))
        return found
    return []
