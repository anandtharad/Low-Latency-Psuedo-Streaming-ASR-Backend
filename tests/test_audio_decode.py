"""Container/codec handling for uploaded audio.

Clients send whatever their platform produces and label it however they like,
so decoding is driven by content rather than by filename, and failures have to
be actionable.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from streaming_asr.audio.decode import (
    UnsupportedAudioError,
    decode,
    describe_support,
    ffmpeg_available,
    libsndfile_formats,
)

SAMPLE_RATE = 16000


@pytest.fixture(scope="module")
def tone() -> np.ndarray:
    t = np.arange(SAMPLE_RATE * 2, dtype=np.float32) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _encode(audio: np.ndarray, fmt: str, rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format=fmt)
    return buffer.getvalue()


# ---- formats -------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["WAV", "FLAC", "OGG"])
def test_common_containers_decode(tone, fmt):
    samples = decode(_encode(tone, fmt), target_sample_rate=SAMPLE_RATE)
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert abs(len(samples) - len(tone)) < 0.05 * SAMPLE_RATE


@pytest.mark.skipif("MP3" not in libsndfile_formats(), reason="libsndfile built without MP3")
def test_mp3_decodes(tone):
    """MP3 needs libsndfile >= 1.1; older builds fall through to ffmpeg."""
    samples = decode(_encode(tone, "MP3"), target_sample_rate=SAMPLE_RATE)
    # Lossy round-trip: check duration and energy, not sample equality.
    assert abs(len(samples) - len(tone)) < 0.1 * SAMPLE_RATE
    assert float(np.abs(samples).mean()) > 0.05


def test_stereo_is_downmixed(tone):
    stereo = np.stack([tone, tone * 0.5], axis=1)
    samples = decode(_encode(stereo, "WAV"), target_sample_rate=SAMPLE_RATE)
    assert samples.ndim == 1
    np.testing.assert_allclose(samples, tone * 0.75, atol=1e-4)


def test_non_target_sample_rate_is_resampled(tone):
    # 16000 samples declared at 8 kHz is 2 seconds of audio.
    narrowband = _encode(tone[:SAMPLE_RATE], "WAV", rate=8000)

    samples = decode(narrowband, target_sample_rate=SAMPLE_RATE)

    # Resampling changes the sample count, never the duration.
    assert abs(len(samples) / SAMPLE_RATE - 2.0) < 0.05
    assert len(samples) == pytest.approx(2 * SAMPLE_RATE, abs=0.05 * SAMPLE_RATE)


def test_extension_is_irrelevant(tmp_path, tone):
    """Content decides the format; clients mislabel uploads constantly."""
    mislabelled = tmp_path / "recording.wav"
    mislabelled.write_bytes(_encode(tone, "FLAC"))

    samples = decode(mislabelled, target_sample_rate=SAMPLE_RATE)
    assert abs(len(samples) - len(tone)) < 0.05 * SAMPLE_RATE


def test_path_and_bytes_agree(tmp_path, tone):
    path = tmp_path / "a.wav"
    path.write_bytes(_encode(tone, "WAV"))

    np.testing.assert_allclose(
        decode(path, SAMPLE_RATE), decode(path.read_bytes(), SAMPLE_RATE), atol=1e-6
    )


# ---- failures ------------------------------------------------------------


def test_non_audio_raises_an_actionable_error():
    with pytest.raises(UnsupportedAudioError) as excinfo:
        decode(b"this is a text file, not audio\n" * 20)

    message = str(excinfo.value)
    if ffmpeg_available():
        assert "ffmpeg could not decode" in message
    else:
        # Name what the deployment *can* take, so the caller can act.
        assert "WAV" in message and "ffmpeg" in message


def test_empty_upload_is_reported_clearly():
    with pytest.raises(UnsupportedAudioError, match="empty"):
        decode(b"")


def test_truncated_file_is_rejected(tone):
    with pytest.raises(UnsupportedAudioError):
        decode(_encode(tone, "WAV")[:16])


def test_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        decode(tmp_path / "nope.wav")


def test_oversized_audio_is_rejected(monkeypatch, tone):
    import streaming_asr.audio.decode as module

    monkeypatch.setattr(module, "MAX_DECODE_SECONDS", 0.5)
    with pytest.raises(UnsupportedAudioError, match="above the"):
        decode(_encode(tone, "WAV"), target_sample_rate=SAMPLE_RATE)


# ---- capability reporting ------------------------------------------------


def test_describe_support_is_honest_about_ffmpeg():
    support = describe_support()

    assert "WAV" in support["libsndfile_formats"]
    assert isinstance(support["ffmpeg_available"], bool)
    if support["ffmpeg_available"]:
        assert "m4a" in support["notes"]
    else:
        # Must say plainly that browser and iOS recordings will be refused.
        assert "WebM" in support["notes"] and "rejected" in support["notes"]
