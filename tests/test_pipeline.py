"""End-to-end pipeline behaviour, exercised against a fake inference engine.

These tests use a stub engine rather than an ONNX session so they run
anywhere, deterministically, in milliseconds. The real-model path is covered by
``test_onnx_integration.py``, which is skipped when no fixture is present.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from streaming_asr.audio.wav_source import InMemorySource
from streaming_asr.config import EndpointConfig, StabilityConfig, StreamingASRConfig
from streaming_asr.decoding.beam_ctc_lm import BeamDecodeResult, FinalDecoder
from streaming_asr.events import ASREventType
from streaming_asr.inference.onnx_engine import (
    InferenceResult,
    ModelGraphReport,
    TensorSpec,
)
from streaming_asr.pipeline import StreamingASRPipeline

VOCAB = ["▁one", "▁two", "▁three", "▁four", "__"]
BLANK = 4


class FakeEngine:
    """Emits tokens according to where the window sits in the stream.

    Mimics a full-context model: the whole window is decoded each time, so as
    the window slides the leading tokens fall out of view -- the behaviour that
    breaks naive prefix-based stabilisation.
    """

    def __init__(self, word_times: list[tuple[int, float]], frame_duration: float = 0.04):
        self.word_times = word_times          # (token_id, absolute start time)
        self.frame_duration = frame_duration
        self.calls = 0
        self._subsampling = 4

    @property
    def subsampling_factor(self) -> int:
        return self._subsampling

    @property
    def on_cuda(self) -> bool:
        return False

    @property
    def active_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    @property
    def graph_report(self) -> ModelGraphReport:
        """Declare the shapes this fake actually honours.

        The pipeline validates its config against the graph at construction, so
        a double has to describe itself truthfully or it would either fail that
        check or let a genuine mismatch through.
        """
        return ModelGraphReport(
            inputs=[
                TensorSpec("audio_signal", (1, 80, "time"), "tensor(float)"),
                TensorSpec("length", (1,), "tensor(int64)"),
            ],
            outputs=[TensorSpec("logprobs", (1, "time_out", len(VOCAB)), "tensor(float)")],
            providers=["CPUExecutionProvider"],
        )

    def ctc_frame_duration(self, hop_duration: float) -> float:
        return self.frame_duration

    def warmup(self, n_mels: int, n_frames: int, iterations: int = 2) -> None:
        return None

    def run_torch(self, features, feature_length) -> InferenceResult:
        """Mirror the real engine's torch entry point, which the pipeline uses."""
        array = features.detach().cpu().numpy() if hasattr(features, "detach") else features
        return self.run(array, feature_length)

    def run(self, features: np.ndarray, feature_length) -> InferenceResult:
        self.calls += 1
        n_out = max(1, features.shape[-1] // self._subsampling)
        logits = np.full((1, n_out, len(VOCAB)), -20.0, dtype=np.float32)
        logits[0, :, BLANK] = 0.0

        # The caller stashes the window's absolute start time here so the fake
        # can decide which words are currently visible.
        window_start = getattr(self, "window_start", 0.0)
        for token_id, start_time in self.word_times:
            frame = int(round((start_time - window_start) / self.frame_duration))
            if 0 <= frame < n_out:
                logits[0, frame, token_id] = 5.0
                logits[0, frame, BLANK] = -20.0
        return InferenceResult(
            logits=logits, input_frames=features.shape[-1],
            output_frames=n_out, inference_time=0.0001,
        )


class WindowAwareEngine(FakeEngine):
    """Wraps the pipeline's buffer so the fake knows the window's origin."""

    def bind(self, pipeline: StreamingASRPipeline) -> None:
        self._pipeline = pipeline

    def run(self, features, feature_length):
        self.window_start = self._pipeline.buffer.window_start_time
        return super().run(features, feature_length)


class StubFinalDecoder(FinalDecoder):
    name = "stub"

    def __init__(self, text: str, used_lm: bool = True):
        self.text = text
        self.used_lm = used_lm
        self.calls = 0

    def decode(self, logits: np.ndarray) -> BeamDecodeResult:
        self.calls += 1
        return BeamDecodeResult(
            text=self.text, words=self.text.split(), decode_time=0.001,
            backend=self.name, used_lm=self.used_lm,
        )


def build_pipeline(
    final_text: str = "one two three four",
    stability_window: float = 0.4,
    min_stable_updates: int = 2,
    chunk_duration: float = 0.16,
    context_duration: float = 1.84,
    endpoint: str = "explicit",
) -> tuple[StreamingASRPipeline, WindowAwareEngine, StubFinalDecoder]:
    config = StreamingASRConfig(
        chunk_duration=chunk_duration,
        context_duration=context_duration,
        vocabulary=VOCAB,
        blank_id=BLANK,
        stability=StabilityConfig(
            stability_window=stability_window,
            min_stable_updates=min_stable_updates,
            time_tolerance=0.12,
        ),
        endpoint=EndpointConfig(detector=endpoint, energy_threshold=0.01,
                                silence_duration=0.4, min_speech_duration=0.2),
    )
    engine = WindowAwareEngine(word_times=[(0, 0.5), (1, 1.2), (2, 1.9), (3, 2.6)])
    decoder = StubFinalDecoder(final_text)
    pipeline = StreamingASRPipeline(config, engine=engine, final_decoder=decoder)
    engine.bind(pipeline)
    return pipeline, engine, decoder


def make_audio(duration: float = 4.0, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    return (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


# ---- streaming -----------------------------------------------------------


def test_pipeline_emits_partials_then_a_final():
    pipeline, _, _ = build_pipeline()
    source = InMemorySource(make_audio(), 16000, pipeline.config.chunk_samples)

    events = list(pipeline.stream(source))
    kinds = [e.type for e in events]

    assert ASREventType.PARTIAL in kinds
    assert kinds[-1] is ASREventType.FINAL
    assert kinds.count(ASREventType.FINAL) == 1


def test_streaming_commits_words_progressively():
    pipeline, _, _ = build_pipeline()
    source = InMemorySource(make_audio(), 16000, pipeline.config.chunk_samples)

    committed_over_time = [
        event.committed_text
        for event in pipeline.stream(source)
        if event.type is ASREventType.PARTIAL
    ]

    # Committed text only ever grows; it is never retracted.
    for earlier, later in zip(committed_over_time, committed_over_time[1:]):
        assert later.startswith(earlier)
    assert committed_over_time[-1] != ""


def test_model_is_called_once_per_chunk():
    pipeline, engine, _ = build_pipeline()
    audio = make_audio(2.0)
    source = InMemorySource(audio, 16000, pipeline.config.chunk_samples)

    list(pipeline.stream(source))
    # The trailing partial chunk is zero-padded rather than dropped, so the
    # count rounds up; +1 for the final full-utterance inference.
    expected = math.ceil(2.0 / pipeline.config.chunk_duration)
    assert engine.calls == expected + 1


def test_window_redundancy_is_reported():
    config = StreamingASRConfig(chunk_duration=0.16, context_duration=3.84)
    assert config.buffer_duration == pytest.approx(4.0)
    assert config.window_redundancy == pytest.approx(25.0)


# ---- finalisation --------------------------------------------------------


def test_final_transcript_overrides_the_streaming_one():
    """Section 28: the beam+LM result replaces the provisional transcript.

    It is not merged with it. Streaming hypotheses each came from a truncated
    view of the audio; concatenating them would preserve their errors.
    """
    pipeline, _, decoder = build_pipeline(final_text="one two three four five")
    source = InMemorySource(make_audio(), 16000, pipeline.config.chunk_samples)

    final = [e for e in pipeline.stream(source) if e.type is ASREventType.FINAL][0]

    assert decoder.calls == 1
    assert final.text == "one two three four five"
    assert final.provisional_text != final.text
    assert final.used_lm is True


def test_beam_decoder_runs_once_not_per_chunk():
    """The expensive decoder must not run on every streaming update."""
    pipeline, _, decoder = build_pipeline()
    source = InMemorySource(make_audio(3.0), 16000, pipeline.config.chunk_samples)
    list(pipeline.stream(source))
    assert decoder.calls == 1


def test_finalize_is_not_reentrant():
    pipeline, _, _ = build_pipeline()
    source = InMemorySource(make_audio(1.0), 16000, pipeline.config.chunk_samples)
    list(pipeline.stream(source))
    with pytest.raises(RuntimeError, match="already been called"):
        pipeline.finalize()


def test_final_beam_decode_can_be_disabled():
    pipeline, _, decoder = build_pipeline()
    pipeline.config.final_beam_decode = False
    source = InMemorySource(make_audio(1.5), 16000, pipeline.config.chunk_samples)

    final = [e for e in pipeline.stream(source) if e.type is ASREventType.FINAL][0]
    assert decoder.calls == 0
    assert final.text == final.provisional_text
    # "streaming" must be distinguishable from "beam ran but had no LM";
    # used_lm alone is False in both cases.
    assert final.decoder == "streaming"
    assert final.used_lm is False


def test_final_event_reports_which_decoder_ran():
    pipeline, _, _ = build_pipeline()
    source = InMemorySource(make_audio(1.5), 16000, pipeline.config.chunk_samples)

    final = [e for e in pipeline.stream(source) if e.type is ASREventType.FINAL][0]
    assert final.decoder == "stub"
    assert final.used_lm is True


def test_lm_free_beam_still_reports_its_backend():
    from streaming_asr.decoding.beam_ctc_lm import PurePythonBeamDecoder

    config = StreamingASRConfig(
        chunk_duration=0.16, context_duration=1.84, vocabulary=VOCAB, blank_id=BLANK,
        stability=StabilityConfig(stability_window=0.4, min_stable_updates=2),
    )
    engine = WindowAwareEngine(word_times=[(0, 0.5), (1, 1.2)])
    pipeline = StreamingASRPipeline(
        config, engine=engine,
        final_decoder=PurePythonBeamDecoder(VOCAB, blank_id=BLANK, beam_size=5),
    )
    engine.bind(pipeline)

    source = InMemorySource(make_audio(1.5), 16000, config.chunk_samples)
    final = [e for e in pipeline.stream(source) if e.type is ASREventType.FINAL][0]

    assert final.decoder == "pure_python"
    assert final.used_lm is False


# ---- endpointing ---------------------------------------------------------


def test_explicit_end_of_speech_stops_intake():
    pipeline, _, _ = build_pipeline()
    audio = make_audio(2.0)
    source = InMemorySource(audio, 16000, pipeline.config.chunk_samples)

    chunks = list(source.stream())
    for chunk in chunks[:4]:
        pipeline.process_chunk(chunk)

    pipeline.end_of_speech()
    assert pipeline.process_chunk(chunks[4]) == []

    final = pipeline.finalize()
    assert final.type is ASREventType.FINAL


def test_energy_vad_endpoints_on_silence():
    pipeline, _, _ = build_pipeline(endpoint="energy")
    speech = make_audio(1.0)
    silence = np.zeros(16000, dtype=np.float32)
    source = InMemorySource(
        np.concatenate([speech, silence]), 16000, pipeline.config.chunk_samples
    )

    kinds = [e.type for e in pipeline.stream(source)]
    assert ASREventType.ENDPOINT in kinds


def test_long_utterances_are_decoded_in_segments():
    """A 60 s recording must not be fed to an 11 s-trained encoder in one pass."""
    pipeline, engine, _ = build_pipeline()
    pipeline.config.final_segment_duration = 2.0
    pipeline.config.final_segment_overlap = 0.5

    audio = make_audio(9.0)
    logits, _ = pipeline._final_inference(audio)

    assert logits.ndim == 3 and logits.shape[0] == 1
    # ~9 s at a 40 ms CTC frame, minus the trimmed seams.
    assert 150 < logits.shape[1] < 260


def test_short_utterances_take_the_single_pass_path():
    pipeline, engine, _ = build_pipeline()
    pipeline.config.final_segment_duration = 20.0
    before = engine.calls

    pipeline._final_inference(make_audio(3.0))
    assert engine.calls == before + 1


def test_segment_snapping_is_off_by_default():
    """It was measured and did not help; see config.final_segment_snap."""
    assert StreamingASRConfig().final_segment_snap == 0.0


def test_snap_to_silence_finds_a_quiet_point():
    from streaming_asr.pipeline import _snap_to_silence

    audio = np.ones(16000, dtype=np.float32)
    audio[7000:7800] = 0.0                       # a pause near the target
    snapped = _snap_to_silence(audio, target=8000, search=1600)

    assert 7000 <= snapped <= 7800


def test_snap_to_silence_is_a_noop_when_disabled():
    from streaming_asr.pipeline import _snap_to_silence

    assert _snap_to_silence(np.ones(1000, dtype=np.float32), 500, search=0) == 500


# ---- history and metrics -------------------------------------------------


def test_retained_audio_matches_what_was_pushed():
    pipeline, _, _ = build_pipeline()
    audio = make_audio(1.6)
    source = InMemorySource(audio, 16000, pipeline.config.chunk_samples)
    for chunk in source.stream():
        pipeline.process_chunk(chunk)

    assert len(pipeline.retained_audio()) == len(audio)
    np.testing.assert_allclose(pipeline.retained_audio(), audio, atol=1e-6)


def test_max_history_bounds_memory():
    pipeline, _, _ = build_pipeline()
    pipeline._max_history_samples = 16000        # 1 s
    source = InMemorySource(make_audio(3.0), 16000, pipeline.config.chunk_samples)
    for chunk in source.stream():
        pipeline.process_chunk(chunk)

    assert len(pipeline.retained_audio()) <= 16000 + pipeline.config.chunk_samples


def test_metrics_are_populated():
    pipeline, _, _ = build_pipeline()
    source = InMemorySource(make_audio(2.0), 16000, pipeline.config.chunk_samples)
    list(pipeline.stream(source))

    snapshot = pipeline.metrics.snapshot()
    assert snapshot["model_calls"] > 0
    assert snapshot["inference"]["count"] > 0
    assert pipeline.metrics.first_partial_latency is not None
    assert pipeline.metrics.finalization_latency is not None
    assert snapshot["window_redundancy"] > 1


def test_reset_preserves_the_config_summary_in_metrics():
    """A benchmark row must stay self-describing after a reset."""
    pipeline, _, _ = build_pipeline()
    source = InMemorySource(make_audio(1.0), 16000, pipeline.config.chunk_samples)
    list(pipeline.stream(source))
    pipeline.reset()

    snapshot = pipeline.metrics.snapshot()
    assert snapshot["window_redundancy"] > 1
    assert snapshot["chunk_duration"] == pytest.approx(pipeline.config.chunk_duration)


class SpyPreprocessor:
    """Records the declared valid length of every window fed to the model."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.widths: list[int] = []

    def __call__(self, waveform, n_samples=None):
        self.widths.append(int(n_samples))
        return self._inner(waveform, n_samples)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _run_and_capture_widths(pipeline, duration: float) -> list[int]:
    spy = SpyPreprocessor(pipeline.preprocessor)
    pipeline.preprocessor = spy
    source = InMemorySource(make_audio(duration), 16000, pipeline.config.chunk_samples)
    for chunk in source.stream():
        pipeline.process_chunk(chunk)
    return spy.widths


def test_warmup_feeds_only_real_audio_by_default():
    """Handing the model seconds of digital silence makes it hallucinate.

    Until the buffer fills, only the audio actually received is sent.
    """
    pipeline, _, _ = build_pipeline()
    assert pipeline.config.pad_warmup_window is False

    widths = _run_and_capture_widths(pipeline, 1.0)

    # Early windows grow with the audio rather than jumping to the full buffer.
    assert widths[0] == pipeline.config.chunk_samples
    assert widths[1] == 2 * pipeline.config.chunk_samples
    assert max(widths) <= pipeline.config.buffer_samples


def test_pad_warmup_window_restores_reference_behaviour():
    pipeline, _, _ = build_pipeline()
    pipeline.config.pad_warmup_window = True

    widths = _run_and_capture_widths(pipeline, 0.5)
    assert set(widths) == {pipeline.config.buffer_samples}


def test_reset_allows_a_second_utterance():
    pipeline, _, _ = build_pipeline()
    source = InMemorySource(make_audio(1.0), 16000, pipeline.config.chunk_samples)
    list(pipeline.stream(source))

    pipeline.reset()
    assert pipeline.tracker.committed_text == ""

    source2 = InMemorySource(make_audio(1.0), 16000, pipeline.config.chunk_samples)
    events = list(pipeline.stream(source2))
    assert events[-1].type is ASREventType.FINAL


# ---- configuration guards ------------------------------------------------


def test_stability_window_must_fit_inside_the_context():
    """A word not committed before it leaves the buffer is lost forever."""
    with pytest.raises(ValueError, match="leave the rolling buffer"):
        StreamingASRConfig(
            context_duration=1.0,
            stability=StabilityConfig(stability_window=1.5),
        )


def test_config_geometry_is_derived_not_hardcoded():
    config = StreamingASRConfig(chunk_duration=0.32, context_duration=2.0)
    assert config.buffer_duration == pytest.approx(2.32)
    assert config.chunk_samples == 5120
    assert config.buffer_samples == 5120 + 32000
