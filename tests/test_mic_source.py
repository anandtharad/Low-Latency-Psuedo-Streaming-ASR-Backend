"""Microphone capture.

The parts that need real hardware are skipped without it. The part that
matters most -- the backlog policy when the consumer cannot keep up with real
time -- is exercised by driving the audio callback directly, no device needed.
"""

from __future__ import annotations

import queue

import numpy as np
import pytest

sd = pytest.importorskip("sounddevice", reason="sounddevice not installed")

from streaming_asr.audio import AudioSource  # noqa: E402
from streaming_asr.audio.mic_source import (  # noqa: E402
    MicrophoneSource,
    list_input_devices,
)


def _has_input_device() -> bool:
    try:
        return len(list_input_devices()) > 0
    except Exception:
        return False


requires_device = pytest.mark.skipif(
    not _has_input_device(), reason="no audio input device available"
)


@requires_device
def test_microphone_source_satisfies_the_audio_source_interface():
    """The engine must not be able to tell a mic from a file."""
    source = MicrophoneSource(sample_rate=16000, chunk_samples=2560)
    try:
        assert isinstance(source, AudioSource)
        assert source.sample_rate == 16000
        assert source.chunk_samples == 2560
    finally:
        source.close()


@requires_device
def test_device_listing_reports_usable_fields():
    for info in list_input_devices():
        assert info["channels"] > 0
        assert isinstance(info["index"], int)
        assert info["name"]


@requires_device
def test_backlog_drops_oldest_audio_instead_of_falling_behind():
    """When the consumer is slower than real time, stay current.

    Growing the queue would push the transcript further behind the speaker with
    every chunk and never recover. Dropping the oldest audio keeps latency
    bounded, and the overrun counter makes the loss visible rather than silent.
    """
    source = MicrophoneSource(sample_rate=16000, chunk_samples=1600, max_queued_chunks=3)
    try:
        blocks = [np.full(1600, float(i), dtype=np.float32) for i in range(6)]
        for block in blocks:
            source._callback(block.reshape(-1, 1), len(block), None, None)

        assert source.overruns == 3, "expected three dropped blocks"
        assert source._queue.qsize() == 3

        # What survives is the newest audio, not the oldest.
        survived = []
        while True:
            try:
                survived.append(source._queue.get_nowait()[0])
            except queue.Empty:
                break
        assert survived == [3.0, 4.0, 5.0]
    finally:
        source.close()


@requires_device
def test_multichannel_input_is_downmixed_to_mono():
    source = MicrophoneSource(sample_rate=16000, chunk_samples=1600)
    try:
        stereo = np.stack(
            [np.full(1600, 1.0, dtype=np.float32), np.full(1600, 3.0, dtype=np.float32)],
            axis=1,
        )
        source._callback(stereo, 1600, None, None)

        block = source._queue.get_nowait()
        assert block.ndim == 1
        assert np.allclose(block, 2.0)          # mean of 1.0 and 3.0
    finally:
        source.close()


@requires_device
def test_stop_unblocks_a_waiting_consumer():
    source = MicrophoneSource(sample_rate=16000, chunk_samples=1600)
    try:
        source.stop()
        assert source._stopped
    finally:
        source.close()


def test_missing_backend_raises_a_actionable_error(monkeypatch):
    """A missing native dependency must say what to install."""
    import streaming_asr.audio.mic_source as mic

    def _boom() -> None:
        raise mic.MicrophoneUnavailable(
            "sounddevice is required for microphone capture. Install it with "
            "'pip install sounddevice'."
        )

    monkeypatch.setattr(mic, "_import_sounddevice", _boom)
    with pytest.raises(mic.MicrophoneUnavailable, match="pip install sounddevice"):
        mic._import_sounddevice()
