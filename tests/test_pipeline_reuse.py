"""Reusing one pipeline across many utterances.

Two distinct risks when a process handles more than one utterance:

1. **State leaking between utterances.** Anything ``reset()`` forgets to clear
   silently contaminates the next transcript. Auditing the field list by hand
   is not proof, so the test here is behavioural: a reset pipeline must produce
   *exactly* what a freshly-constructed one produces.

2. **Unbounded growth.** A stream that never endpoints -- a WebSocket client
   that just keeps talking -- must not accumulate memory without limit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from streaming_asr.audio.wav_source import InMemorySource, load_wav
from streaming_asr.config import StabilityConfig, StreamingASRConfig, load_vocabulary
from streaming_asr.events import ASREventType
from streaming_asr.inference.onnx_engine import ONNXASREngine
from streaming_asr.metrics import BoundedSamples, MetricsCollector
from streaming_asr.pipeline import StreamingASRPipeline

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MODEL = FIXTURES / "synthetic_model.onnx"
AUDIO = FIXTURES / "synthetic.wav"
VOCAB = FIXTURES / "vocabulary.txt"


# ---- bounded accumulation (no fixture needed) ----------------------------


def test_sample_window_is_bounded():
    samples = BoundedSamples(maxlen=100)
    for i in range(10_000):
        samples.add(float(i))

    assert len(samples._window) == 100          # memory bounded
    assert samples.count == 10_000              # count stays exact


def test_aggregates_stay_exact_beyond_the_window():
    """Mean, sum and max must not drift once samples start being evicted."""
    samples = BoundedSamples(maxlen=10)
    values = [float(i) for i in range(1000)]
    for value in values:
        samples.add(value)

    assert samples.count == len(values)
    assert samples.total == pytest.approx(sum(values))
    assert samples.mean == pytest.approx(sum(values) / len(values))
    # The max lives far outside the retained window; it must survive eviction.
    assert samples.max == pytest.approx(999.0)
    assert samples.summary()["max"] == pytest.approx(999.0)


def test_empty_summary_is_well_formed():
    summary = BoundedSamples().summary()
    assert summary == {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}


def test_metrics_memory_does_not_grow_with_a_long_stream():
    """The failure this guards: one float per chunk per metric, forever.

    At a 160 ms step that is ~22k entries per metric per hour on a connection
    that never endpoints.
    """
    metrics = MetricsCollector()
    for _ in range(50_000):
        metrics.record_window(0.001, 0.002, 0.0001, 0.0001, chunk_capture_time=None)

    assert len(metrics.inference_times._window) <= 4096
    assert metrics.inference_times.count == 50_000
    # RTF must still be right: it uses the exact running total, not the window.
    assert metrics.inference_times.total == pytest.approx(50_000 * 0.002)


# ---- reset equivalence (needs the fixture) -------------------------------

pytestmark_fixture = pytest.mark.skipif(
    not (MODEL.exists() and AUDIO.exists() and VOCAB.exists()),
    reason="synthetic fixture not built; run tools/build_synthetic_fixture.py",
)


@pytest.fixture(scope="module")
def engine() -> ONNXASREngine:
    if not MODEL.exists():
        pytest.skip("fixture not built")
    return ONNXASREngine(str(MODEL), providers="auto")


def _config() -> StreamingASRConfig:
    vocab = load_vocabulary(VOCAB)
    return StreamingASRConfig(
        chunk_duration=0.16,
        context_duration=3.84,
        onnx_model_path=str(MODEL),
        vocabulary=vocab,
        blank_id=len(vocab) - 1,
        final_beam_decode=False,      # isolate the streaming path
        stability=StabilityConfig(stability_window=0.6, min_stable_updates=2),
    )


def _run(pipeline: StreamingASRPipeline, audio: np.ndarray) -> list[tuple[str, str]]:
    """The full committed/partial trail, not just the final text."""
    source = InMemorySource(audio, 16000, pipeline.config.chunk_samples)
    trail = []
    for event in pipeline.stream(source):
        if event.type is ASREventType.PARTIAL:
            trail.append((event.committed_text, event.partial_text))
    return trail


@pytestmark_fixture
def test_reset_leaves_no_trace_of_the_previous_utterance(engine):
    """A reset pipeline must behave identically to a brand new one.

    Compares the entire partial trail, not just the final transcript: state
    leaking through reset would most likely show up as a different
    stabilisation path even where the end result coincides.
    """
    config = _config()
    audio = load_wav(AUDIO, 16000)

    reference = _run(StreamingASRPipeline(config, engine=engine), audio)

    reused = StreamingASRPipeline(config, engine=engine)
    _run(reused, audio)          # first utterance
    reused.reset()
    after_reset = _run(reused, audio)

    assert after_reset == reference, "reset() left state behind"


@pytestmark_fixture
def test_three_consecutive_utterances_are_all_identical(engine):
    config = _config()
    audio = load_wav(AUDIO, 16000)

    pipeline = StreamingASRPipeline(config, engine=engine)
    transcripts = []
    for _ in range(3):
        _run(pipeline, audio)
        transcripts.append(pipeline.tracker.committed_text)
        pipeline.reset()

    assert len(set(transcripts)) == 1, f"drift across utterances: {transcripts}"
    assert transcripts[0].strip()


@pytestmark_fixture
def test_reset_clears_per_utterance_state_but_keeps_loaded_models(engine):
    config = _config()
    pipeline = StreamingASRPipeline(config, engine=engine)
    _run(pipeline, load_wav(AUDIO, 16000))

    preprocessor, greedy = pipeline.preprocessor, pipeline.greedy
    pipeline.reset()

    # Per-utterance state gone...
    assert pipeline.tracker.committed_text == ""
    assert pipeline.buffer.total_pushed == 0
    assert pipeline.retained_audio().size == 0
    assert pipeline.metrics.model_calls == 0
    assert not pipeline.is_finalized
    assert pipeline.last_hypothesis is None

    # ...expensive objects retained.
    assert pipeline.preprocessor is preprocessor
    assert pipeline.greedy is greedy
    assert pipeline.engine is engine

    # And the config echo survives, so a post-reset metrics row is still
    # self-describing.
    assert pipeline.metrics.snapshot()["window_redundancy"] > 1


@pytestmark_fixture
def test_engine_counters_accumulate_across_utterances(engine):
    """Engine counters are lifetime; per-utterance metrics are not.

    Conflating the two would make a long-lived process report a steadily worse
    RTF for every utterance after the first.
    """
    config = _config()
    pipeline = StreamingASRPipeline(config, engine=engine)
    audio = load_wav(AUDIO, 16000)[:32000]

    _run(pipeline, audio)
    first = pipeline.metrics.model_calls
    engine_after_first = engine.call_count

    pipeline.reset()
    _run(pipeline, audio)

    assert pipeline.metrics.model_calls == first          # per-utterance: reset
    assert engine.call_count > engine_after_first          # lifetime: keeps going
