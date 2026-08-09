"""Rolling buffer: shifting, warm-up and the offset->absolute-time mapping."""

from __future__ import annotations

import numpy as np
import pytest

from streaming_asr.buffer.rolling_buffer import RollingAudioBuffer


def test_shift_discards_oldest_and_appends_newest():
    """[A][B][C][D] + [E] -> [B][C][D][E]"""
    buffer = RollingAudioBuffer(buffer_samples=4, chunk_samples=1, sample_rate=4)
    for value in (1.0, 2.0, 3.0, 4.0):
        buffer.push(np.array([value], dtype=np.float32))
    assert buffer.window[0].tolist() == [1.0, 2.0, 3.0, 4.0]

    buffer.push(np.array([5.0], dtype=np.float32))
    assert buffer.window[0].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_warmup_left_pads_with_zeros():
    buffer = RollingAudioBuffer(buffer_samples=4, chunk_samples=2, sample_rate=4)
    buffer.push(np.array([7.0, 8.0], dtype=np.float32))

    assert buffer.window[0].tolist() == [0.0, 0.0, 7.0, 8.0]
    assert buffer.valid_samples == 2
    assert not buffer.is_warm

    buffer.push(np.array([9.0, 10.0], dtype=np.float32))
    assert buffer.is_warm
    assert buffer.valid_samples == 4


def test_valid_window_excludes_warmup_padding():
    buffer = RollingAudioBuffer(buffer_samples=4, chunk_samples=1, sample_rate=4)
    buffer.push(np.array([1.0], dtype=np.float32))
    buffer.push(np.array([2.0], dtype=np.float32))
    assert buffer.valid_window()[0].tolist() == [1.0, 2.0]


def test_multi_chunk_sizes_and_full_width_push():
    buffer = RollingAudioBuffer(buffer_samples=4, chunk_samples=2, sample_rate=4)
    buffer.push(np.arange(4, dtype=np.float32))
    assert buffer.window[0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert buffer.total_pushed == 4


def test_window_start_time_is_negative_during_warmup():
    """The absolute-time mapping must be correct from the very first window.

    Frame 0 of the first window is audio that precedes the stream, so its
    timestamp is negative. Clamping it to zero here would misdate every token
    in every warm-up window.
    """
    buffer = RollingAudioBuffer(buffer_samples=16, chunk_samples=4, sample_rate=4)
    buffer.push(np.ones(4, dtype=np.float32))

    assert buffer.window_start_sample == -12
    assert buffer.window_start_time == pytest.approx(-3.0)
    assert buffer.current_time == pytest.approx(1.0)
    assert buffer.valid_start_time == 0.0


def test_offset_to_absolute_time_after_sliding():
    buffer = RollingAudioBuffer(buffer_samples=8, chunk_samples=2, sample_rate=8)
    for _ in range(6):
        buffer.push(np.zeros(2, dtype=np.float32))

    # 12 samples pushed at 8 Hz = 1.5 s; the window starts 8 samples earlier.
    assert buffer.current_time == pytest.approx(1.5)
    assert buffer.window_start_time == pytest.approx(0.5)
    assert buffer.offset_to_absolute_time(0) == pytest.approx(0.5)
    assert buffer.offset_to_absolute_time(8) == pytest.approx(1.5)


def test_push_is_in_place_without_reallocating():
    """The buffer must be reused, not reallocated 6.25 times a second."""
    buffer = RollingAudioBuffer(buffer_samples=8, chunk_samples=2, sample_rate=8)
    identity = id(buffer.window)
    for _ in range(10):
        buffer.push(np.ones(2, dtype=np.float32))
    assert id(buffer.window) == identity


def test_reset_returns_to_prestream_state():
    buffer = RollingAudioBuffer(buffer_samples=4, chunk_samples=2, sample_rate=4)
    buffer.push(np.ones(2, dtype=np.float32))
    buffer.reset()
    assert buffer.total_pushed == 0
    assert buffer.window[0].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_rejects_oversized_chunk():
    buffer = RollingAudioBuffer(buffer_samples=4, chunk_samples=2, sample_rate=4)
    with pytest.raises(ValueError, match="exceeds buffer"):
        buffer.push(np.zeros(5, dtype=np.float32))


def test_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        RollingAudioBuffer(buffer_samples=0, chunk_samples=1)
    with pytest.raises(ValueError):
        RollingAudioBuffer(buffer_samples=4, chunk_samples=5)
