"""Final beam decoding, audio sources and endpoint detection."""

from __future__ import annotations

import numpy as np
import pytest

from streaming_asr.audio.wav_source import InMemorySource
from streaming_asr.config import BeamDecoderConfig, EndpointConfig, StreamingASRConfig
from streaming_asr.decoding.beam_ctc_lm import (
    PurePythonBeamDecoder,
    build_final_decoder,
)
from streaming_asr.endpointing.endpoint import (
    CompositeEndpointDetector,
    EnergyVADEndpointDetector,
    ExplicitEndpointDetector,
    NullEndpointDetector,
    build_endpoint_detector,
)

VOCAB = ["▁one", "▁two", "▁three", "__"]
BLANK = 3


def log_probs(frame_ids: list[int]) -> np.ndarray:
    """Peaked log-probabilities that argmax to ``frame_ids``."""
    logits = np.full((1, len(frame_ids), len(VOCAB)), -8.0, dtype=np.float32)
    for t, token in enumerate(frame_ids):
        logits[0, t, token] = 0.0
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


# ---- pure-python beam decoder --------------------------------------------


def test_beam_decoder_matches_greedy_on_a_peaked_sequence():
    decoder = PurePythonBeamDecoder(VOCAB, blank_id=BLANK, beam_size=10)
    result = decoder.decode(log_probs([BLANK, 0, 0, BLANK, 1, BLANK, 2]))

    assert result.text == "one two three"
    assert result.words == ["one", "two", "three"]
    assert result.used_lm is False
    assert result.backend == "pure_python"


def test_beam_decoder_collapses_repeats_correctly():
    decoder = PurePythonBeamDecoder(VOCAB, blank_id=BLANK, beam_size=10)
    # one one (blank-separated) must stay two tokens
    assert decoder.decode(log_probs([0, 0, BLANK, 0])).text == "one one"
    assert decoder.decode(log_probs([0, 0, 0, 0])).text == "one"


def test_beam_decoder_handles_all_blank():
    decoder = PurePythonBeamDecoder(VOCAB, blank_id=BLANK, beam_size=5)
    assert decoder.decode(log_probs([BLANK] * 6)).text == ""


def test_build_final_decoder_falls_back_without_an_lm():
    """Falling back to an LM-free decoder must be explicit, not silent."""
    config = StreamingASRConfig(vocabulary=VOCAB, blank_id=BLANK)
    with pytest.warns(None) if False else _no_warning_context():
        decoder = build_final_decoder(config)
    assert decoder.name == "pure_python"
    assert decoder.used_lm is False


def _no_warning_context():
    import contextlib
    return contextlib.nullcontext()


def test_build_final_decoder_respects_explicit_backend():
    config = StreamingASRConfig(
        vocabulary=VOCAB, blank_id=BLANK,
        beam=BeamDecoderConfig(backend="pure_python"),
    )
    assert build_final_decoder(config).name == "pure_python"


def test_build_final_decoder_rejects_unknown_backend():
    config = StreamingASRConfig(
        vocabulary=VOCAB, blank_id=BLANK, beam=BeamDecoderConfig(backend="nope"),
    )
    with pytest.raises(ValueError, match="Unknown beam decoder backend"):
        build_final_decoder(config)


# ---- audio sources -------------------------------------------------------


def test_in_memory_source_chunking_and_timestamps():
    samples = np.arange(1000, dtype=np.float32)
    source = InMemorySource(samples, sample_rate=1000, chunk_samples=250)

    chunks = list(source.stream())
    assert len(chunks) == 4
    assert chunks[0].start_sample == 0
    assert chunks[1].start_time == pytest.approx(0.25)
    assert chunks[-1].is_last
    np.testing.assert_array_equal(np.concatenate([c.samples for c in chunks]), samples)


def test_trailing_partial_chunk_is_padded_not_dropped():
    """The reference iterator discards it, losing up to a chunk of speech."""
    samples = np.ones(300, dtype=np.float32)
    chunks = list(InMemorySource(samples, 1000, 250, pad_final_chunk=True).stream())

    assert len(chunks) == 2
    assert len(chunks[1].samples) == 250
    assert chunks[1].samples[:50].tolist() == [1.0] * 50
    assert chunks[1].samples[50:].tolist() == [0.0] * 200


def test_trailing_partial_chunk_can_be_dropped_for_reference_parity():
    samples = np.ones(300, dtype=np.float32)
    chunks = list(InMemorySource(samples, 1000, 250, pad_final_chunk=False).stream())
    assert len(chunks) == 1


def test_source_stop_halts_streaming():
    source = InMemorySource(np.ones(1000, dtype=np.float32), 1000, 100)
    collected = []
    for chunk in source.stream():
        collected.append(chunk)
        if len(collected) == 3:
            source.stop()
    assert len(collected) == 3


def test_chunks_carry_capture_time():
    source = InMemorySource(np.ones(500, dtype=np.float32), 1000, 250)
    assert all(chunk.capture_time > 0 for chunk in source.stream())


# ---- endpoint detection --------------------------------------------------


def test_explicit_detector_only_fires_when_triggered():
    detector = ExplicitEndpointDetector()
    silence = np.zeros(1600, dtype=np.float32)

    assert not detector.update(silence, 1.0).is_endpoint
    detector.trigger()
    assert detector.update(silence, 1.1).is_endpoint

    detector.reset()
    assert not detector.update(silence, 1.2).is_endpoint


def test_null_detector_never_fires():
    detector = NullEndpointDetector()
    assert not detector.update(np.zeros(1600, dtype=np.float32), 5.0).is_endpoint


def test_energy_detector_requires_speech_before_silence():
    """Silence before the speaker starts is not the end of an utterance."""
    detector = EnergyVADEndpointDetector(
        silence_duration=0.3, energy_threshold=0.01,
        min_speech_duration=0.2, sample_rate=16000,
    )
    silence = np.zeros(1600, dtype=np.float32)      # 100 ms

    for i in range(10):
        assert not detector.update(silence, i * 0.1).is_endpoint


def test_energy_detector_fires_after_speech_then_silence():
    detector = EnergyVADEndpointDetector(
        silence_duration=0.3, energy_threshold=0.01,
        min_speech_duration=0.2, sample_rate=16000,
    )
    speech = np.full(1600, 0.5, dtype=np.float32)
    silence = np.zeros(1600, dtype=np.float32)

    for i in range(4):
        assert not detector.update(speech, i * 0.1).is_endpoint

    fired = [detector.update(silence, 0.4 + i * 0.1).is_endpoint for i in range(4)]
    assert any(fired)


def test_energy_detector_resets_silence_on_renewed_speech():
    detector = EnergyVADEndpointDetector(
        silence_duration=0.3, energy_threshold=0.01,
        min_speech_duration=0.1, sample_rate=16000,
    )
    speech = np.full(1600, 0.5, dtype=np.float32)
    silence = np.zeros(1600, dtype=np.float32)

    detector.update(speech, 0.1)
    detector.update(silence, 0.2)
    detector.update(silence, 0.3)
    detector.update(speech, 0.4)                 # pause was mid-sentence
    assert not detector.update(silence, 0.5).is_endpoint


def test_composite_detector_fires_on_any_child():
    explicit = ExplicitEndpointDetector()
    composite = CompositeEndpointDetector(explicit, NullEndpointDetector())
    silence = np.zeros(1600, dtype=np.float32)

    assert not composite.update(silence, 1.0).is_endpoint
    explicit.trigger()
    decision = composite.update(silence, 1.1)
    assert decision.is_endpoint and "explicit" in decision.reason


def test_build_endpoint_detector_always_keeps_explicit_available():
    """end_of_speech() must work whatever automatic detector is configured."""
    detector = build_endpoint_detector(EndpointConfig(detector="energy"), 16000)
    assert isinstance(detector, CompositeEndpointDetector)
    assert any(isinstance(d, ExplicitEndpointDetector) for d in detector.detectors)


def test_build_endpoint_detector_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown endpoint detector"):
        build_endpoint_detector(EndpointConfig(detector="psychic"), 16000)


# ---- config round-tripping -----------------------------------------------


def test_config_round_trips_through_to_dict():
    """to_dict() output must reload; describe() may summarise instead."""
    config = StreamingASRConfig(
        chunk_duration=0.32, context_duration=2.0, vocabulary=VOCAB, blank_id=BLANK,
    )
    restored = StreamingASRConfig.from_dict(config.to_dict())

    assert restored.chunk_duration == pytest.approx(0.32)
    assert restored.buffer_duration == pytest.approx(2.32)
    assert list(restored.vocabulary) == VOCAB
    assert restored.stability.aligner == config.stability.aligner


def test_describe_summarises_the_vocabulary():
    config = StreamingASRConfig(vocabulary=VOCAB, blank_id=BLANK)
    assert "tokens>" in config.describe()
    assert "<4 tokens>" in config.to_dict(summarize_vocabulary=True)["vocabulary"]


def test_ensure_blank_in_vocabulary_appends_once():
    config = StreamingASRConfig(vocabulary=["a", "b"])
    assert config.ensure_blank_in_vocabulary() == ["a", "b", "__"]

    explicit = StreamingASRConfig(vocabulary=["a", "b", "__"], blank_id=2)
    assert explicit.ensure_blank_in_vocabulary() == ["a", "b", "__"]


def test_resolved_blank_id_defaults_to_last_entry():
    assert StreamingASRConfig(vocabulary=["a", "b", "__"]).resolved_blank_id == 2
    assert StreamingASRConfig(vocabulary=["a", "b"], blank_id=0).resolved_blank_id == 0
