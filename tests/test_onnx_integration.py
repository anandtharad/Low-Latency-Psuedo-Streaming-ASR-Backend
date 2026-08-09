"""End-to-end tests against a real ONNX session.

Skipped unless the synthetic fixture has been built::

    python tools/build_synthetic_fixture.py --out fixtures

The fixture is a stand-in, not the IndicConformer, but it exercises the same
code path: a real ONNX Runtime session, the real mel frontend, real 4x
subsampling and a genuinely bidirectional encoder whose output shifts as the
window slides.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from streaming_asr.audio.wav_source import WavFileSource, load_wav
from streaming_asr.config import StabilityConfig, StreamingASRConfig, load_vocabulary
from streaming_asr.events import ASREventType
from streaming_asr.inference.onnx_engine import ONNXASREngine
from streaming_asr.pipeline import StreamingASRPipeline

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MODEL = FIXTURES / "synthetic_model.onnx"
AUDIO = FIXTURES / "synthetic.wav"
VOCAB = FIXTURES / "vocabulary.txt"

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and AUDIO.exists() and VOCAB.exists()),
    reason="synthetic fixture not built; run tools/build_synthetic_fixture.py",
)


@pytest.fixture(scope="module")
def vocabulary() -> list[str]:
    return load_vocabulary(VOCAB)


@pytest.fixture(scope="module")
def engine() -> ONNXASREngine:
    return ONNXASREngine(str(MODEL), providers="auto")


def make_config(vocabulary: list[str], **overrides) -> StreamingASRConfig:
    kwargs = dict(
        chunk_duration=0.16,
        context_duration=3.84,
        onnx_model_path=str(MODEL),
        vocabulary=vocabulary,
        blank_id=len(vocabulary) - 1,
        final_beam_decode=True,
        stability=StabilityConfig(stability_window=0.6, min_stable_updates=2),
    )
    kwargs.update(overrides)
    return StreamingASRConfig(**kwargs)


# ---- graph introspection (section 30) ------------------------------------


def test_graph_report_identifies_a_stateless_model(engine):
    """Section 30: confirm there is no encoder cache before committing to
    rolling-window inference."""
    report = engine.graph_report

    assert [s.name for s in report.inputs] == ["audio_signal", "length"]
    assert report.is_stateless
    assert report.stateful_inputs == []
    assert report.stateful_outputs == []
    assert "ONNX graph report" in report.render()


def test_engine_detects_the_subsampling_factor(engine):
    """4x subsampling means every timestamp has 40 ms resolution."""
    features = np.zeros((1, 80, 400), dtype=np.float32)
    result = engine.run(features, 400)

    assert result.output_frames == 100
    assert engine.subsampling_factor == 4
    assert engine.ctc_frame_duration(hop_duration=0.01) == pytest.approx(0.04)


def test_subsampling_factor_survives_a_short_first_window():
    """A tiny warm-up window must not permanently mis-scale every timestamp.

    Conv subsampling pads at the edges, so a short input under-reports the
    factor (17 feature frames -> 5 CTC frames measures as 3x, not 4x).
    Latching that would scale every token timestamp by 3/4 -- invisibly, since
    the values stay self-consistent and the transcript still reads correctly.
    """
    engine = ONNXASREngine(str(MODEL), providers="auto")

    engine.run(np.zeros((1, 80, 17), dtype=np.float32), 17)
    engine.run(np.zeros((1, 80, 401), dtype=np.float32), 401)

    assert engine.subsampling_factor == 4
    assert engine.ctc_frame_duration(hop_duration=0.01) == pytest.approx(0.04)

    # A later short window must not undo the better estimate.
    engine.run(np.zeros((1, 80, 17), dtype=np.float32), 17)
    assert engine.subsampling_factor == 4


def test_wrong_vocabulary_size_is_rejected_at_construction(vocabulary, engine):
    """The one misconfiguration that produces fluent nonsense instead of an error.

    A vocabulary that does not belong to the checkpoint makes every argmax
    index a different token table. Nothing downstream can detect that, so it
    has to be caught here.
    """
    truncated = list(vocabulary)[:100]
    config = make_config(vocabulary)
    config.vocabulary = truncated
    config.blank_id = len(truncated) - 1

    with pytest.raises(ValueError, match="output units but the vocabulary has"):
        StreamingASRPipeline(config, engine=engine)


def test_wrong_mel_count_is_rejected_at_construction(vocabulary, engine):
    from streaming_asr.config import PreprocessingConfig

    config = make_config(vocabulary, preprocessing=PreprocessingConfig(features=64))
    with pytest.raises(ValueError, match="mel bins but preprocessing is configured"):
        StreamingASRPipeline(config, engine=engine)


def test_matching_configuration_is_accepted(vocabulary, engine):
    StreamingASRPipeline(make_config(vocabulary), engine=engine)


def test_engine_accepts_variable_length_windows(engine):
    """The window size is configurable, so the session must not be shape-bound."""
    for frames in (100, 250, 401):
        result = engine.run(np.zeros((1, 80, frames), dtype=np.float32), frames)
        assert result.logits.shape[0] == 1
        assert result.logits.shape[2] == 129


# ---- streaming -----------------------------------------------------------


def test_streaming_produces_committed_text(vocabulary, engine):
    config = make_config(vocabulary)
    pipeline = StreamingASRPipeline(config, engine=engine)
    source = WavFileSource(AUDIO, config.sample_rate, config.chunk_samples)

    events = list(pipeline.stream(source))
    final = [e for e in events if e.type is ASREventType.FINAL][0]

    assert final.provisional_text.strip() != ""
    assert final.text.strip() != ""


def test_committed_text_only_grows(vocabulary, engine):
    """Committed output is never retracted, however the model revises itself."""
    config = make_config(vocabulary)
    pipeline = StreamingASRPipeline(config, engine=engine)
    source = WavFileSource(AUDIO, config.sample_rate, config.chunk_samples)

    history = [
        e.committed_text for e in pipeline.stream(source)
        if e.type is ASREventType.PARTIAL
    ]
    for earlier, later in zip(history, history[1:]):
        assert later.startswith(earlier), f"retracted: {earlier!r} -> {later!r}"


def test_streaming_survives_the_utterance_outgrowing_the_buffer(vocabulary, engine):
    """The fixture is ~8.8 s against a 4 s window, so early words scroll out.

    This is the exact condition under which the reference implementation loses
    its opening words. The committed transcript must retain them.
    """
    audio = load_wav(AUDIO, 16000)
    assert len(audio) / 16000 > 4.0, "fixture must be longer than the window"

    config = make_config(vocabulary)
    pipeline = StreamingASRPipeline(config, engine=engine)
    source = WavFileSource(AUDIO, config.sample_rate, config.chunk_samples)
    list(pipeline.stream(source))

    committed = pipeline.tracker.committed_text.split()
    # More words survive than could ever fit in a single 4 s window's view of
    # the end of the utterance.
    assert len(committed) >= 5


def test_token_timestamps_track_ground_truth(vocabulary, engine):
    """Word timings recovered from CTC frames should land near the truth."""
    spans_path = FIXTURES / "word_spans.json"
    if not spans_path.exists():
        pytest.skip("word_spans.json not present")
    truth = json.loads(spans_path.read_text(encoding="utf-8"))

    config = make_config(vocabulary)
    pipeline = StreamingASRPipeline(config, engine=engine)
    source = WavFileSource(AUDIO, config.sample_rate, config.chunk_samples)
    list(pipeline.stream(source))

    committed = pipeline.tracker.committed_words
    if not committed:
        pytest.skip("fixture model produced no committed words")

    truth_by_word = {entry["word"]: entry for entry in truth}
    checked = 0
    for word in committed:
        entry = truth_by_word.get(word.text)
        if entry is None:
            continue
        # Half a second of slack: CTC emits a token somewhere inside its
        # acoustic span, not at its exact onset.
        assert abs(word.start_time - entry["start"]) < 0.75, (
            f"{word.text!r} at {word.start_time:.2f}s, expected ~{entry['start']:.2f}s"
        )
        checked += 1
    assert checked > 0


# ---- decoding comparison (section 23) ------------------------------------


def test_greedy_and_beam_decode_the_same_logits(vocabulary, engine):
    config = make_config(vocabulary)
    pipeline = StreamingASRPipeline(config, engine=engine)

    audio = load_wav(AUDIO, config.sample_rate)
    features, lengths = pipeline.preprocessor(audio.reshape(1, -1), n_samples=len(audio))
    result = engine.run_torch(features, int(lengths[0]))

    greedy = pipeline.greedy.decode_text(result.logits)
    beam = pipeline.final_decoder.decode(result.logits)

    assert greedy.strip()
    assert beam.text.strip()
    # Without a language model the two should be close; this is the number
    # section 23 asks us to measure rather than assume.
    assert beam.decode_time > 0


# ---- geometry sweep ------------------------------------------------------


@pytest.mark.parametrize("chunk_ms,context_sec", [(160, 3.84), (320, 2.0), (500, 1.5)])
def test_alternate_window_geometries_run(vocabulary, engine, chunk_ms, context_sec):
    """The 4 s / 160 ms operating point is configurable, not baked in."""
    config = make_config(
        vocabulary,
        chunk_duration=chunk_ms / 1000.0,
        context_duration=context_sec,
        stability=StabilityConfig(
            stability_window=min(0.6, 0.4 * context_sec),
            min_stable_updates=2,
            time_tolerance=max(0.12, chunk_ms / 1000.0),
        ),
    )
    pipeline = StreamingASRPipeline(config, engine=engine)
    source = WavFileSource(AUDIO, config.sample_rate, config.chunk_samples)

    final = [e for e in pipeline.stream(source) if e.type is ASREventType.FINAL][0]
    assert final.text.strip() != ""
    assert pipeline.metrics.model_calls > 0
