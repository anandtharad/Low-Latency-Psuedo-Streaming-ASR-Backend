"""Pause-segmented streaming ASR.

Why this exists alongside ``pipeline.py``
-----------------------------------------
The windowed pipeline commits words irreversibly as it goes: roughly six
decisions per second, each taken at an arbitrary point that frequently lands
mid-word. Every model quirk at such a point becomes a permanent defect, and
defects compound -- measured on a real Conformer as ``climb in in in`` and
``traverses the the the``, degrading further the longer a session ran.

Three successive repairs to that commitment logic each fixed the case in front
of them and broke another. The problem is structural, not a bug: nothing can
reliably decide "is this word new?" at a boundary chosen by a timer.

So this pipeline never commits mid-utterance. It cuts at **pauses**, where the
question does not arise, and re-decodes each segment whole:

* **partial** -- the current segment re-decoded from its start, every chunk.
  Revisable, replaced wholesale, never accumulated.
* **segment** -- emitted after a short pause. Authoritative for that span.
* **final** -- emitted after a longer silence. The turn's segments joined.

The evidence for this shape came from the model itself. On an 8 s utterance the
single-pass final decode *repaired* the streaming duplication outright
(``traverses the the the highest`` -> ``traverses the highest``). On 31 s cut
into fixed 10 s pieces it produced garbage. Short, pause-bounded, in-distribution
spans are where this checkpoint is strong -- and its training capped utterances
at 11 s, so segments are kept under that.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

import numpy as np

from streaming_asr.audio.base import AudioChunk, AudioSource
from streaming_asr.config import StreamingASRConfig
from streaming_asr.decoding.beam_ctc_lm import FinalDecoder, build_final_decoder
from streaming_asr.decoding.greedy_ctc import GreedyCTCDecoder
from streaming_asr.device import RuntimePlacement, resolve_device
from streaming_asr.events import ASREvent, ASREventType
from streaming_asr.inference.onnx_engine import ONNXASREngine
from streaming_asr.metrics import MetricsCollector
from streaming_asr.preprocessing.filterbank import Preprocessor

logger = logging.getLogger(__name__)

EventCallback = Callable[[ASREvent], None]


class SpeechDetector:
    """Per-chunk speech/silence classification with hysteresis.

    Hysteresis matters more here than in a plain endpoint detector: a single
    dipped chunk in the middle of a word must not read as a pause and split a
    segment mid-word, which is exactly the boundary this design exists to
    avoid. Speech is entered at ``threshold`` and only left at half of it.
    """

    def __init__(self, threshold: float = 0.005, sample_rate: int = 16000) -> None:
        self.threshold = threshold
        self.release = threshold * 0.5
        self.sample_rate = sample_rate
        self._in_speech = False

    def reset(self) -> None:
        self._in_speech = False

    def update(self, chunk: np.ndarray) -> tuple[bool, float]:
        """Return ``(is_speech, rms)`` for one chunk."""
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
    """One pause-bounded span, decoded whole."""

    text: str
    start_time: float
    end_time: float
    decoder: str = ""
    used_lm: bool = False
    forced: bool = False        # cut by the length cap rather than by a pause

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class SegmentedASRPipeline:
    """Streaming ASR that cuts at pauses and decodes each segment whole.

    Args:
        config: Shared configuration. ``config.segmentation`` holds the
            thresholds; everything else (model, device, decoders) is reused.
        engine / final_decoder: Pre-built and shared, as in the windowed
            pipeline, so a server loads them once.
    """

    def __init__(
        self,
        config: StreamingASRConfig,
        engine: Optional[ONNXASREngine] = None,
        final_decoder: Optional[FinalDecoder] = None,
    ) -> None:
        self.config = config
        self.settings = config.segmentation
        self.vocabulary = config.ensure_blank_in_vocabulary()

        self.placement: RuntimePlacement = resolve_device(config.device, config.providers)
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
        if self.placement.on_cuda and not engine.on_cuda:
            self.placement = RuntimePlacement(
                "cpu", engine.active_providers, self.placement.device_id,
                "downgraded: CUDA provider failed to load",
            )

        self.preprocessor = Preprocessor(
            config.preprocessing, device=self.placement.torch_device
        )
        self.greedy = GreedyCTCDecoder(
            vocabulary=self.vocabulary, blank_id=config.resolved_blank_id
        )
        self._final_decoder = final_decoder
        self.detector = SpeechDetector(
            threshold=self.settings.energy_threshold, sample_rate=config.sample_rate
        )

        self.metrics = MetricsCollector()
        self.metrics.on_gpu = self.placement.on_cuda or engine.on_cuda
        self.metrics.gpu_device_id = self.placement.device_id
        self.metrics.segment_silence = self.settings.segment_silence
        self.metrics.config_summary = {
            "pipeline": "segmented",
            "segment_silence": self.settings.segment_silence,
            "turn_silence": self.settings.turn_silence,
            "max_segment_duration": self.settings.max_segment_duration,
        }

        self._callbacks: list[EventCallback] = []
        self._reset_all()

        logger.info(
            "Segmented pipeline ready: cut at %.2fs silence, turn ends at %.2fs, "
            "segments capped at %.1fs",
            self.settings.segment_silence, self.settings.turn_silence,
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
        #: Events produced inside finalize(), which returns only the FINAL.
        #: Drained by callers so the last segment is not lost.
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
        """Everything decoded so far, across all turns."""
        return " ".join(s.text for s in self._all_segments if s.text).strip()

    def on_event(self, callback: EventCallback) -> None:
        self._callbacks.append(callback)

    def _emit(self, event: ASREvent) -> ASREvent:
        for callback in self._callbacks:
            callback(event)
        return event

    def warmup(self, iterations: int = 2) -> None:
        """Size the warm-up to a typical segment, not a chunk.

        The encoder's subsampling factor is measured from the inputs it sees,
        and a short input measures it wrong.
        """
        samples = int(min(self.settings.max_segment_duration, 4.0) * self.config.sample_rate)
        features, _ = self.preprocessor(
            np.zeros((1, samples), dtype=np.float32), n_samples=samples
        )
        self.engine.warmup(
            n_mels=int(features.shape[1]), n_frames=int(features.shape[2]),
            iterations=iterations,
        )

    # ---- audio intake ---------------------------------------------------

    def process_chunk(self, chunk: AudioChunk) -> list[ASREvent]:
        if self._ended:
            logger.warning("chunk ignored: the pipeline has already ended")
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
        chunk_duration = samples.size / self.config.sample_rate
        if is_speech:
            self._speech_samples += samples.size
            self._trailing_silence = 0.0
        else:
            self._trailing_silence += chunk_duration
            self._trim_leading_silence()

        has_speech = self._speech_samples >= int(
            self.settings.min_segment_speech * self.config.sample_rate
        )

        # A partial for the open segment, while it is still open.
        if (
            has_speech
            and self._trailing_silence < self.settings.segment_silence
            and self._partial_is_due()
        ):
            events.append(self._emit_partial(chunk))

        # The segment has run past what the model handles well with no pause in
        # sight; cut it at the quietest point rather than letting it grow.
        if has_speech and self._buffer_duration() >= self.settings.max_segment_duration:
            events.extend(self._close_segment(forced=True))

        # A pause: close the segment.
        elif has_speech and self._trailing_silence >= self.settings.segment_silence:
            events.extend(self._close_segment(forced=False))

        # A longer silence: the turn is over.
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
        pipeline always retains headroom for the segment decode. A no-op when
        decoding is quick.

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
        """Keep only a little audio ahead of speech that has not started.

        Without this the buffer grows through every gap between turns, so a
        segment reaches ``max_segment_duration`` while the speaker is still
        mid-phrase and gets a forced cut. Observed as a phrase split into
        "i have been having chest" + "for three days after eating", losing the
        word on the seam.
        """
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
        """Preprocess and run the model over one span. Returns logits + timings."""
        pre_start = time.perf_counter()
        features, lengths = self.preprocessor(audio.reshape(1, -1), n_samples=len(audio))
        preprocess_time = time.perf_counter() - pre_start

        result = self.engine.run_torch(features, int(lengths[0]))
        return result.logits, preprocess_time, result.inference_time

    def _emit_partial(self, chunk: AudioChunk) -> ASREvent:
        """Re-decode the open segment from its start.

        Deliberately a full re-decode rather than an incremental update. It is
        the whole point of this design: the partial is replaced outright each
        time, so no error it contains can survive into the transcript.
        """
        started = time.perf_counter()
        audio = self._buffer_audio()
        logits, preprocess_time, inference_time = self._decode(audio)

        decode_start = time.perf_counter()
        self._partial_text = self.greedy.decode_text(logits)
        decode_time = time.perf_counter() - decode_start

        self._last_partial_cost = time.perf_counter() - started
        self._last_partial_audio_time = self._audio_time

        self.metrics.record_window(
            preprocess_time=preprocess_time,
            inference_time=inference_time,
            decode_time=decode_time,
            tracker_time=0.0,
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
        """Decode the open segment properly and publish it."""
        audio = self._buffer_audio()
        keep_from = len(audio)

        if forced:
            # No pause available, so cut where the audio is quietest. Never in
            # the first half -- a cut there would leave almost nothing behind
            # and the next segment would immediately hit the cap again.
            cut = _quietest_point(
                audio, search_from=len(audio) // 2, sample_rate=self.config.sample_rate
            )
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

        # Retain a little audio so the next segment does not start clipped.
        tail = int(self.settings.speech_pad * self.config.sample_rate)
        retained = self._buffer_audio()[max(keep_from - tail, keep_from) :] if forced \
            else self._buffer_audio()[max(0, len(self._buffer_audio()) - tail):]

        self._buffer = [retained] if retained.size else []
        self._buffer_samples = int(retained.size)
        self._buffer_start_time = self._audio_time - retained.size / self.config.sample_rate
        self._speech_samples = 0
        self._partial_text = ""
        if forced:
            # A forced cut is not a pause; speech is still in progress.
            self._trailing_silence = 0.0
            self._speech_samples = int(retained.size)
        return events

    def _decode_segment(self, audio: np.ndarray, forced: bool) -> Optional[SegmentResult]:
        """Authoritative decode of one segment."""
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

        end_time = self._audio_time - self._trailing_silence if not forced else self._audio_time
        segment = SegmentResult(
            text=text.strip(),
            start_time=self._buffer_start_time,
            end_time=end_time,
            decoder=decoder,
            used_lm=used_lm,
            forced=forced,
        )
        logger.info(
            "segment [%.2f-%.2f]%s: %r (%.3fs)",
            segment.start_time, segment.end_time, " forced" if forced else "",
            segment.text, elapsed,
        )
        return segment if segment.text else None

    def _emit_turn_final(self) -> ASREvent:
        text = " ".join(s.text for s in self._turn_segments if s.text).strip()
        event = ASREvent(
            type=ASREventType.FINAL,
            timestamp=self._audio_time,
            wall_time=time.perf_counter(),
            text=text,
            provisional_text=text,
            committed_text=self.transcript,
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
        """Take the events finalize() produced besides the FINAL itself.

        Callers must emit these *before* the final, so a consumer sees the last
        segment in the same order it would have arrived mid-stream.
        """
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
        """Close whatever is open and emit the last turn."""
        if self._finalized:
            raise RuntimeError("finalize() has already been called")
        self._finalized = True
        self._ended = True

        if self.metrics.endpoint_wall is None:
            self.metrics.mark_endpoint()

        # Anything still buffered is a segment that never got its pause.
        # Its events are stashed rather than dropped: a caller that builds its
        # transcript from `segment` events would otherwise lose the tail of
        # every stream, since the last segment almost always closes here.
        if self._speech_samples >= int(
            self.settings.min_segment_speech * self.config.sample_rate
        ):
            self._pending_events.extend(self._close_segment(forced=False))

        text = " ".join(s.text for s in self._turn_segments if s.text).strip()
        self._turn_segments = []
        self.metrics.mark_final()

        # ``text`` is the turn still open at end of stream, which is empty when
        # the last turn already closed on silence. ``committed_text`` always
        # carries the whole session, so a caller has one field to read
        # regardless of where the audio happened to stop.
        return self._emit(ASREvent(
            type=ASREventType.FINAL,
            timestamp=self._audio_time,
            wall_time=time.perf_counter(),
            text=text,
            provisional_text=text,
            committed_text=self.transcript,
            used_lm=any(s.used_lm for s in self._all_segments),
            decoder=self._all_segments[-1].decoder if self._all_segments else "",
            metrics=self.metrics.snapshot(),
        ))

    def reset(self) -> None:
        self._reset_all()
        self.metrics = MetricsCollector()
        self.metrics.on_gpu = self.placement.on_cuda or self.engine.on_cuda
        self.metrics.gpu_device_id = self.placement.device_id


def _quietest_point(
    audio: np.ndarray, search_from: int, sample_rate: int, frame: int = 400
) -> int:
    """Index of the lowest-energy frame at or after ``search_from``.

    Used only when a segment hits the length cap without a pause. Cutting at
    the quietest available point is the least-bad option: it is the place least
    likely to fall inside a vowel.
    """
    if len(audio) <= search_from + frame:
        return len(audio)

    positions = np.arange(search_from, len(audio) - frame, frame // 2, dtype=np.int64)
    if positions.size == 0:
        return len(audio)

    offsets = np.arange(frame)
    windows = audio[positions[:, None] + offsets[None, :]]
    energy = np.sqrt(np.mean(np.square(windows, dtype=np.float64), axis=1))
    return int(positions[int(np.argmin(energy))])
