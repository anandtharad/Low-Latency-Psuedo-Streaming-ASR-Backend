"""Pause-segmented pipeline, torch-free.

Behaviourally identical to :class:`streaming_asr.segmented.SegmentedASRPipeline`
-- same thresholds, same events, same bug fixes -- but it builds features with
:class:`~streaming_asr_lite.frontend.OnnxMelFrontend` instead of torchaudio.

Everything else is imported from ``streaming_asr`` rather than copied. Only
``segmented.py`` pulls torch at import time (via the preprocessor); the greedy
decoder, beam decoders, ONNX engine, config, metrics and event types are all
already torch-free and are reused directly.

.. note::
   The segmentation loop *is* duplicated from ``segmented.py``, because that
   module cannot be imported without torch. This is deliberate -- the working
   pipeline was to be left untouched -- but it is a drift risk: a fix applied
   to one must be applied to the other.

   The clean resolution, once this path has proven itself, is a small change to
   ``segmented.py`` making the preprocessor injectable. Both pipelines then
   collapse into one and this file disappears. Recorded in ``docs/TODO.md``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from streaming_asr.audio.base import AudioChunk, AudioSource
from streaming_asr.config import StreamingASRConfig
from streaming_asr.decoding.beam_ctc_lm import FinalDecoder, build_final_decoder
from streaming_asr.decoding.greedy_ctc import GreedyCTCDecoder
from streaming_asr.events import ASREvent, ASREventType
from streaming_asr.metrics import MetricsCollector
from streaming_asr_lite.engine import LiteONNXEngine
from streaming_asr_lite.frontend import OnnxMelFrontend

logger = logging.getLogger(__name__)

EventCallback = Callable[[ASREvent], None]


class SpeechDetector:
    """Per-chunk speech/silence classification with hysteresis.

    Speech is entered at ``threshold`` and only left at half of it, so a single
    dipped chunk mid-word cannot read as a pause and split the segment.
    """

    def __init__(self, threshold: float = 0.005) -> None:
        self.threshold = threshold
        self.release = threshold * 0.5
        self._in_speech = False

    def reset(self) -> None:
        self._in_speech = False

    def update(self, chunk: np.ndarray) -> tuple[bool, float]:
        if chunk.size == 0:
            return False, 0.0
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64))))
        if self._in_speech:
            self._in_speech = rms >= self.release
        else:
            self._in_speech = rms >= self.threshold
        return self._in_speech, rms


@dataclass
class SegmentResult:
    text: str
    start_time: float
    end_time: float
    decoder: str = ""
    used_lm: bool = False
    forced: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def _quietest_point(audio: np.ndarray, search_from: int, frame: int = 400) -> int:
    """Index of the lowest-energy frame at or after ``search_from``."""
    if len(audio) <= search_from + frame:
        return len(audio)
    positions = np.arange(search_from, len(audio) - frame, frame // 2, dtype=np.int64)
    if positions.size == 0:
        return len(audio)
    offsets = np.arange(frame)
    windows = audio[positions[:, None] + offsets[None, :]]
    energy = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1))
    return int(positions[int(np.argmin(energy))])


class LiteSegmentedPipeline:
    """Pause-segmented ASR with no torch dependency.

    Args:
        config: Shared configuration; ``config.segmentation`` holds thresholds.
        frontend_path: The exported ``frontend.onnx``.
        engine / final_decoder: Pre-built and shared, so a server loads once.
    """

    def __init__(
        self,
        config: StreamingASRConfig,
        frontend_path: str | Path,
        engine: Optional[LiteONNXEngine] = None,
        final_decoder: Optional[FinalDecoder] = None,
    ) -> None:
        self.config = config
        self.settings = config.segmentation
        self.vocabulary = config.ensure_blank_in_vocabulary()

        if engine is None:
            if not config.onnx_model_path:
                raise ValueError("config.onnx_model_path is required")
            engine = LiteONNXEngine(
                model_path=config.onnx_model_path,
                providers=config.providers,
                intra_op_threads=config.intra_op_threads,
            )
        self.engine = engine

        # Share the encoder's providers: on CUDA this keeps the features in
        # device memory between the two sessions.
        self.preprocessor = OnnxMelFrontend(
            frontend_path,
            providers=engine.active_providers,
            hop_duration=config.preprocessing.hop_duration,
        )
        self.greedy = GreedyCTCDecoder(
            vocabulary=self.vocabulary, blank_id=config.resolved_blank_id
        )
        self._final_decoder = final_decoder
        self.detector = SpeechDetector(threshold=self.settings.energy_threshold)

        self.metrics = MetricsCollector()
        self.metrics.segment_silence = self.settings.segment_silence
        self.metrics.config_summary = {
            "pipeline": "lite-segmented",
            "segment_silence": self.settings.segment_silence,
            "turn_silence": self.settings.turn_silence,
            "max_segment_duration": self.settings.max_segment_duration,
        }
        self._callbacks: list[EventCallback] = []
        self._reset_all()

        logger.info(
            "Lite pipeline ready (no torch): cut at %.2fs silence, turn ends at "
            "%.2fs, segments capped at %.1fs",
            self.settings.segment_silence, self.settings.turn_silence,
            self.settings.max_segment_duration,
        )
        if config.final_beam_decode and config.beam.backend == "pure_python":
            logger.warning(
                "final_beam_decode is on with the pure-Python beam decoder. It is "
                "O(frames x beam x tokens) in Python and takes seconds on a "
                "%.0fs segment -- the loop will stall at every segment boundary "
                "and a live microphone will fall behind. Use --no-final-beam, or "
                "a native backend (flashlight / pyctcdecode) once an LM exists.",
                self.settings.max_segment_duration,
            )

    # ---- state ----------------------------------------------------------

    def _reset_all(self) -> None:
        self._buffer: list[np.ndarray] = []
        self._buffer_samples = 0
        self._buffer_start_time = 0.0
        self._speech_samples = 0
        self._trailing_silence = 0.0
        self._turn_segments: list[SegmentResult] = []
        self._all_segments: list[SegmentResult] = []
        self._partial_text = ""
        self._audio_time = 0.0
        self._ended = False
        self._finalized = False
        self._pending_events: list[ASREvent] = []
        self._last_partial_audio_time = -1e9
        self._last_partial_cost = 0.0
        self.detector.reset()

    @property
    def final_decoder(self) -> FinalDecoder:
        if self._final_decoder is None:
            self._final_decoder = build_final_decoder(self.config)
        return self._final_decoder

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    @property
    def transcript(self) -> str:
        return " ".join(s.text for s in self._all_segments if s.text).strip()

    def on_event(self, callback: EventCallback) -> None:
        self._callbacks.append(callback)

    def _emit(self, event: ASREvent) -> ASREvent:
        for callback in self._callbacks:
            callback(event)
        return event

    def warmup(self, iterations: int = 2) -> None:
        samples = int(min(self.settings.max_segment_duration, 4.0)
                      * self.config.sample_rate)
        features, _ = self.preprocessor(np.zeros((1, samples), dtype=np.float32))
        self.engine.warmup(
            n_mels=int(features.shape[1]), n_frames=int(features.shape[2]),
            iterations=iterations,
        )

    # ---- audio intake ---------------------------------------------------

    def process_chunk(self, chunk: AudioChunk) -> list[ASREvent]:
        if self._ended:
            return []

        self.metrics.mark_first_audio()
        events: list[ASREvent] = []

        samples = np.asarray(chunk.samples, dtype=np.float32)
        if not self._buffer:
            self._buffer_start_time = chunk.start_time
        self._buffer.append(samples.copy())
        self._buffer_samples += samples.size
        self._audio_time = chunk.end_time
        self.metrics.audio_duration = self._audio_time

        is_speech, _ = self.detector.update(samples)
        if is_speech:
            self._speech_samples += samples.size
            self._trailing_silence = 0.0
        else:
            self._trailing_silence += samples.size / self.config.sample_rate
            self._trim_leading_silence()

        has_speech = self._speech_samples >= int(
            self.settings.min_segment_speech * self.config.sample_rate
        )

        if (
            has_speech
            and self._trailing_silence < self.settings.segment_silence
            and self._partial_is_due()
        ):
            events.append(self._emit_partial(chunk))

        if has_speech and self._buffer_duration() >= self.settings.max_segment_duration:
            events.extend(self._close_segment(forced=True))
        elif has_speech and self._trailing_silence >= self.settings.segment_silence:
            events.extend(self._close_segment(forced=False))

        if self._turn_segments and self._trailing_silence >= self.settings.turn_silence:
            events.append(self._emit_turn_final())

        return events

    def _partial_is_due(self) -> bool:
        """Rate-limit partials to what the machine can actually sustain.

        A partial re-decodes the whole open segment, so its cost grows as the
        segment does. Emitting one per chunk regardless means that on a long
        segment the loop takes longer than a chunk to process a chunk -- on a
        live microphone the input queue then overruns and *everything*,
        including segments, arrives late.

        The interval floor is the measured cost of the last partial, so the
        pipeline always retains headroom for the segment decode. When decoding
        is quick this is a no-op and partials arrive every chunk.

        Measured against **audio** time, not wall clock. The two are equivalent
        for a live source, but a file streamed at full speed delivers chunks
        with no wall-clock gap between them -- throttling on wall time there
        would suppress nearly every partial in batch mode.
        """
        interval = max(self.settings.min_partial_interval, self._last_partial_cost)
        if interval <= 0.0:
            return True
        return (self._audio_time - self._last_partial_audio_time) >= interval

    def _trim_leading_silence(self) -> None:
        """Bound pre-speech silence so it cannot inflate the next segment."""
        if self._speech_samples:
            return
        keep = int(self.settings.speech_pad * self.config.sample_rate)
        if self._buffer_samples <= keep:
            return
        retained = self._buffer_audio()[-keep:]
        self._buffer = [retained]
        self._buffer_samples = int(retained.size)
        self._buffer_start_time = (
            self._audio_time - retained.size / self.config.sample_rate
        )

    def _buffer_duration(self) -> float:
        return self._buffer_samples / self.config.sample_rate

    def _buffer_audio(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._buffer)

    # ---- decoding -------------------------------------------------------

    def _decode(self, audio: np.ndarray) -> tuple[np.ndarray, float, float]:
        pre_start = time.perf_counter()
        features, lengths = self.preprocessor(audio.reshape(1, -1))
        preprocess_time = time.perf_counter() - pre_start
        # engine.run(), not run_torch(): features are already NumPy and torch
        # is not installed in this runtime.
        result = self.engine.run(features, int(lengths[0]))
        return result.logits, preprocess_time, result.inference_time

    def _emit_partial(self, chunk: AudioChunk) -> ASREvent:
        started = time.perf_counter()
        audio = self._buffer_audio()
        logits, preprocess_time, inference_time = self._decode(audio)

        decode_start = time.perf_counter()
        self._partial_text = self.greedy.decode_text(logits)
        decode_time = time.perf_counter() - decode_start

        self._last_partial_cost = time.perf_counter() - started
        self._last_partial_audio_time = self._audio_time

        self.metrics.record_window(
            preprocess_time=preprocess_time, inference_time=inference_time,
            decode_time=decode_time, tracker_time=0.0,
            chunk_capture_time=chunk.capture_time or None,
        )
        if self._partial_text:
            self.metrics.mark_first_partial()

        confirmed = " ".join(s.text for s in self._turn_segments if s.text).strip()
        return self._emit(ASREvent(
            type=ASREventType.PARTIAL,
            timestamp=self._audio_time,
            wall_time=time.perf_counter(),
            committed_text=confirmed,
            partial_text=self._partial_text,
            full_hypothesis=" ".join(x for x in (confirmed, self._partial_text) if x),
            window_start=self._buffer_start_time,
            window_end=self._audio_time,
        ))

    def _close_segment(self, forced: bool) -> list[ASREvent]:
        audio = self._buffer_audio()
        keep_from = len(audio)

        if forced:
            cut = _quietest_point(audio, search_from=len(audio) // 2)
            keep_from = cut
            audio = audio[:cut]

        events: list[ASREvent] = []
        segment = self._decode_segment(audio, forced=forced)
        if segment is not None:
            self._turn_segments.append(segment)
            self._all_segments.append(segment)
            events.append(self._emit(ASREvent(
                type=ASREventType.SEGMENT,
                timestamp=segment.end_time,
                wall_time=time.perf_counter(),
                text=segment.text,
                committed_text=self.transcript,
                decoder=segment.decoder,
                used_lm=segment.used_lm,
                window_start=segment.start_time,
                window_end=segment.end_time,
                metrics={"forced": forced},
            )))

        tail = int(self.settings.speech_pad * self.config.sample_rate)
        full = self._buffer_audio()
        retained = full[max(keep_from - tail, keep_from):] if forced \
            else full[max(0, len(full) - tail):]

        self._buffer = [retained] if retained.size else []
        self._buffer_samples = int(retained.size)
        self._buffer_start_time = (
            self._audio_time - retained.size / self.config.sample_rate
        )
        self._speech_samples = int(retained.size) if forced else 0
        self._partial_text = ""
        if forced:
            self._trailing_silence = 0.0
        return events

    def _decode_segment(self, audio: np.ndarray, forced: bool) -> Optional[SegmentResult]:
        duration = len(audio) / self.config.sample_rate
        if duration < self.settings.min_segment_speech:
            return None

        start = time.perf_counter()
        logits, _, inference_time = self._decode(audio)

        if self.config.final_beam_decode:
            result = self.final_decoder.decode(logits)
            text, decoder, used_lm = result.text, result.backend, result.used_lm
        else:
            text, decoder, used_lm = self.greedy.decode_text(logits), "greedy", False

        elapsed = time.perf_counter() - start
        self.metrics.record_segment(elapsed)
        self.metrics.final_inference_time += inference_time

        end_time = self._audio_time if forced else self._audio_time - self._trailing_silence
        segment = SegmentResult(
            text=text.strip(), start_time=self._buffer_start_time, end_time=end_time,
            decoder=decoder, used_lm=used_lm, forced=forced,
        )
        logger.info("segment [%.2f-%.2f]%s: %r (%.3fs)",
                    segment.start_time, segment.end_time,
                    " forced" if forced else "", segment.text, elapsed)
        return segment if segment.text else None

    def _emit_turn_final(self) -> ASREvent:
        text = " ".join(s.text for s in self._turn_segments if s.text).strip()
        event = ASREvent(
            type=ASREventType.FINAL,
            timestamp=self._audio_time,
            wall_time=time.perf_counter(),
            text=text, provisional_text=text, committed_text=self.transcript,
            decoder=self._turn_segments[-1].decoder if self._turn_segments else "",
            used_lm=any(s.used_lm for s in self._turn_segments),
            metrics=self.metrics.snapshot(),
        )
        self._turn_segments = []
        self.metrics.mark_endpoint()
        self.metrics.mark_final()
        return self._emit(event)

    # ---- lifecycle ------------------------------------------------------

    def drain_pending(self) -> list[ASREvent]:
        """Events finalize() produced besides the FINAL. Emit before the final."""
        events, self._pending_events = self._pending_events, []
        return events

    def stream(self, source: AudioSource | Iterable[AudioChunk]) -> Iterator[ASREvent]:
        chunks = source.stream() if isinstance(source, AudioSource) else source
        for chunk in chunks:
            for event in self.process_chunk(chunk):
                yield event
        if not self._finalized:
            final = self.finalize()
            for event in self.drain_pending():
                yield event
            yield final

    def end_of_speech(self) -> None:
        self._ended = True
        self.metrics.mark_endpoint()

    def finalize(self) -> ASREvent:
        if self._finalized:
            raise RuntimeError("finalize() has already been called")
        self._finalized = True
        self._ended = True
        if self.metrics.endpoint_wall is None:
            self.metrics.mark_endpoint()

        # Stashed, not dropped: the last segment nearly always closes here, and
        # a caller building its transcript from SEGMENT events would lose it.
        if self._speech_samples >= int(
            self.settings.min_segment_speech * self.config.sample_rate
        ):
            self._pending_events.extend(self._close_segment(forced=False))

        text = " ".join(s.text for s in self._turn_segments if s.text).strip()
        self._turn_segments = []
        self.metrics.mark_final()

        return self._emit(ASREvent(
            type=ASREventType.FINAL,
            timestamp=self._audio_time,
            wall_time=time.perf_counter(),
            text=text, provisional_text=text, committed_text=self.transcript,
            used_lm=any(s.used_lm for s in self._all_segments),
            decoder=self._all_segments[-1].decoder if self._all_segments else "",
            metrics=self.metrics.snapshot(),
        ))

    def reset(self) -> None:
        self._reset_all()
        self.metrics = MetricsCollector()
