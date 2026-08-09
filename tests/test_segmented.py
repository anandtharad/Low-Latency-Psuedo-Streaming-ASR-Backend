"""Pause-segmented pipeline.

The properties asserted here are the ones the windowed pipeline could not hold:
a phrase repeated later must appear every time, and nothing may be duplicated,
however the model's output churns. Both come for free from cutting at pauses
instead of committing on a timer -- these tests exist to keep it that way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from streaming_asr.audio.wav_source import InMemorySource, load_wav
from streaming_asr.config import (
    BeamDecoderConfig,
    SegmentationConfig,
    StreamingASRConfig,
    load_vocabulary,
)
from streaming_asr.events import ASREventType
from streaming_asr.inference.onnx_engine import ONNXASREngine
from streaming_asr.segmented import SegmentedASRPipeline, SpeechDetector, _quietest_point

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MODEL = FIXTURES / "synthetic_model.onnx"
AUDIO = FIXTURES / "synthetic.wav"
VOCAB = FIXTURES / "vocabulary.txt"
SR = 16000


# ---- speech detection (no model needed) ----------------------------------


def test_speech_detector_uses_hysteresis():
    """A dip inside a word must not read as a pause.

    Without hysteresis a single quiet chunk mid-word closes the segment and
    cuts it in the middle -- the exact boundary this design exists to avoid.
    """
    detector = SpeechDetector(threshold=0.01)
    loud = np.full(1600, 0.5, dtype=np.float32)
    dip = np.full(1600, 0.006, dtype=np.float32)      # below onset, above release
    quiet = np.zeros(1600, dtype=np.float32)

    assert detector.update(loud)[0] is True
    assert detector.update(dip)[0] is True            # still speech
    assert detector.update(quiet)[0] is False


def test_speech_detector_ignores_sub_threshold_noise():
    detector = SpeechDetector(threshold=0.01)
    noise = np.full(1600, 0.004, dtype=np.float32)
    assert detector.update(noise)[0] is False


def test_quietest_point_avoids_the_first_half():
    audio = np.ones(16000, dtype=np.float32)
    audio[12000:12800] = 0.0
    cut = _quietest_point(audio, search_from=8000, sample_rate=SR)
    assert 12000 <= cut <= 12800


def test_quietest_point_handles_short_input():
    assert _quietest_point(np.ones(100, dtype=np.float32), 50, SR) == 100


# ---- configuration -------------------------------------------------------


def test_turn_silence_must_not_precede_segment_silence():
    with pytest.raises(ValueError, match="turn cannot end"):
        SegmentationConfig(segment_silence=1.0, turn_silence=0.5)


# ---- end to end ----------------------------------------------------------

pytestmark_fixture = pytest.mark.skipif(
    not (MODEL.exists() and AUDIO.exists() and VOCAB.exists()),
    reason="synthetic fixture not built; run tools/build_synthetic_fixture.py",
)


@pytest.fixture(scope="module")
def engine() -> ONNXASREngine:
    if not MODEL.exists():
        pytest.skip("fixture not built")
    return ONNXASREngine(str(MODEL), providers="auto")


def _config(**overrides) -> StreamingASRConfig:
    vocab = load_vocabulary(VOCAB)
    segmentation = SegmentationConfig(
        segment_silence=0.5, turn_silence=1.5, max_segment_duration=10.0,
        energy_threshold=0.01,
    )
    kwargs = dict(
        onnx_model_path=str(MODEL), vocabulary=vocab, blank_id=len(vocab) - 1,
        final_beam_decode=False,          # greedy segments: fast and deterministic
        beam=BeamDecoderConfig(backend="pure_python", beam_size=5),
        segmentation=segmentation,
    )
    kwargs.update(overrides)
    return StreamingASRConfig(**kwargs)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _run(pipeline: SegmentedASRPipeline, audio: np.ndarray):
    source = InMemorySource(audio, SR, pipeline.config.chunk_samples)
    return list(pipeline.stream(source))


@pytestmark_fixture
def test_single_phrase_produces_one_segment(engine):
    pipeline = SegmentedASRPipeline(_config(), engine=engine)
    events = _run(pipeline, load_wav(AUDIO, SR))

    segments = [e for e in events if e.type is ASREventType.SEGMENT]
    expected = (FIXTURES / "transcript.txt").read_text(encoding="utf-8").strip()

    assert len(segments) == 1
    assert segments[0].text == expected
    assert pipeline.transcript == expected


@pytestmark_fixture
def test_repeated_phrases_each_appear(engine):
    """The failure that broke the windowed design, from the opposite side.

    Saying the same thing again must produce it again -- the text-based
    duplicate strip used to swallow it -- while nothing may be emitted twice
    for one utterance.
    """
    phrase = load_wav(AUDIO, SR)
    audio = np.concatenate([phrase, _silence(1.0), phrase, _silence(1.0), phrase])

    pipeline = SegmentedASRPipeline(_config(), engine=engine)
    events = _run(pipeline, audio)

    segments = [e for e in events if e.type is ASREventType.SEGMENT]
    expected = (FIXTURES / "transcript.txt").read_text(encoding="utf-8").strip()

    assert len(segments) == 3, [s.text for s in segments]
    for segment in segments:
        assert segment.text == expected

    words = pipeline.transcript.split()
    assert words == expected.split() * 3


@pytestmark_fixture
def test_no_word_is_ever_duplicated(engine):
    """No committed word appears more times than it was spoken."""
    phrase = load_wav(AUDIO, SR)
    pipeline = SegmentedASRPipeline(_config(), engine=engine)
    _run(pipeline, np.concatenate([phrase, _silence(1.0), phrase]))

    expected = (FIXTURES / "transcript.txt").read_text(encoding="utf-8").strip().split()
    produced = pipeline.transcript.split()

    for word in set(expected):
        assert produced.count(word) == expected.count(word) * 2, (
            f"{word!r} x{produced.count(word)}, expected x{expected.count(word) * 2}"
        )


@pytestmark_fixture
def test_short_pause_segments_without_ending_the_turn(engine):
    """0.8s gap: a segment closes, the turn stays open."""
    phrase = load_wav(AUDIO, SR)
    # The fixture carries its own lead/tail silence, so trim to isolate the gap.
    trimmed = phrase[int(0.3 * SR): -int(0.5 * SR)]
    audio = np.concatenate([trimmed, _silence(0.8), trimmed, _silence(2.5)])

    pipeline = SegmentedASRPipeline(_config(), engine=engine)
    events = _run(pipeline, audio)

    segments = [e for e in events if e.type is ASREventType.SEGMENT]
    finals = [e for e in events if e.type is ASREventType.FINAL and e.text]

    assert len(segments) == 2
    # Both segments belong to one turn, so the turn final carries both.
    assert len(finals) == 1, [f.text for f in finals]
    assert finals[0].text == " ".join(s.text for s in segments)


@pytestmark_fixture
def test_long_silence_ends_the_turn(engine):
    phrase = load_wav(AUDIO, SR)
    audio = np.concatenate([phrase, _silence(2.5), phrase, _silence(2.5)])

    pipeline = SegmentedASRPipeline(_config(), engine=engine)
    events = _run(pipeline, audio)

    finals = [e for e in events if e.type is ASREventType.FINAL and e.text]
    assert len(finals) == 2, [f.text for f in finals]


@pytestmark_fixture
def test_silence_between_turns_does_not_inflate_the_next_segment(engine):
    """Regression: idle silence used to accumulate into the segment buffer.

    It pushed the *next* segment past max_segment_duration while the speaker
    was still mid-phrase, forcing a cut that split "…chest pain for…" and lost
    the word on the seam.
    """
    phrase = load_wav(AUDIO, SR)
    audio = np.concatenate([phrase, _silence(6.0), phrase])

    pipeline = SegmentedASRPipeline(_config(), engine=engine)
    events = _run(pipeline, audio)

    segments = [e for e in events if e.type is ASREventType.SEGMENT]
    assert not any(e.metrics.get("forced") for e in segments), "unexpected forced cut"
    expected = (FIXTURES / "transcript.txt").read_text(encoding="utf-8").strip()
    assert all(s.text == expected for s in segments)


@pytestmark_fixture
def test_partials_are_revisable_and_never_accumulate(engine):
    """A partial describes only the open segment, and is replaced wholesale."""
    pipeline = SegmentedASRPipeline(_config(), engine=engine)
    events = _run(pipeline, load_wav(AUDIO, SR))

    partials = [e for e in events if e.type is ASREventType.PARTIAL]
    assert len(partials) > 10

    expected_words = set(
        (FIXTURES / "transcript.txt").read_text(encoding="utf-8").split()
    )
    for event in partials:
        # A partial never grows beyond the phrase it is transcribing.
        assert len(event.partial_text.split()) <= len(expected_words) + 2


@pytestmark_fixture
def test_long_speech_without_a_pause_is_force_cut(engine):
    """A speaker who never pauses still gets bounded segments."""
    phrase = load_wav(AUDIO, SR)
    # Strip the internal silences so there is no pause to cut at.
    continuous = np.concatenate([phrase[int(0.3 * SR): -int(0.5 * SR)]] * 3)

    pipeline = SegmentedASRPipeline(
        _config(segmentation=SegmentationConfig(
            segment_silence=0.5, turn_silence=1.5, max_segment_duration=6.0,
            energy_threshold=0.01,
        )),
        engine=engine,
    )
    _run(pipeline, continuous)

    segments = [s for s in pipeline._all_segments]
    assert len(segments) >= 2
    assert any(s.forced for s in segments)
    for segment in segments:
        assert segment.duration <= 6.0 + 1.0      # cap plus retained padding


@pytestmark_fixture
def test_final_segment_is_published_not_swallowed(engine):
    """The last segment usually closes inside finalize(); it must still be seen.

    Regression: it was emitted internally but never handed to the caller, so a
    client building its transcript from `segment` events lost the tail of every
    stream. Only visible on audio whose last segment has no pause after it --
    which is most audio.
    """
    pipeline = SegmentedASRPipeline(_config(), engine=engine)
    # Trim the fixture's trailing silence so the segment cannot close early.
    audio = load_wav(AUDIO, SR)[: -int(0.5 * SR)]

    events = _run(pipeline, audio)
    segments = [e for e in events if e.type is ASREventType.SEGMENT]

    assert len(segments) == 1, "the final segment was dropped from the event stream"
    assert segments[0].text == pipeline.transcript

    # And it arrives before the final, matching mid-stream ordering.
    kinds = [e.type for e in events]
    assert kinds.index(ASREventType.SEGMENT) < len(kinds) - 1
    assert kinds[-1] is ASREventType.FINAL


@pytestmark_fixture
def test_segments_reconstruct_the_transcript_exactly(engine):
    """Appending every `segment` must equal the session transcript.

    This is the contract an application depends on.
    """
    phrase = load_wav(AUDIO, SR)
    pipeline = SegmentedASRPipeline(_config(), engine=engine)
    events = _run(pipeline, np.concatenate([phrase, _silence(1.0), phrase]))

    joined = " ".join(
        e.text for e in events if e.type is ASREventType.SEGMENT and e.text
    )
    assert joined == pipeline.transcript


@pytestmark_fixture
def test_reset_clears_everything(engine):
    pipeline = SegmentedASRPipeline(_config(), engine=engine)
    _run(pipeline, load_wav(AUDIO, SR))
    assert pipeline.transcript

    pipeline.reset()
    assert pipeline.transcript == ""
    assert not pipeline.is_finalized

    events = _run(pipeline, load_wav(AUDIO, SR))
    assert any(e.type is ASREventType.SEGMENT for e in events)
