"""WAV-file and in-memory audio sources.

``WavFileSource`` simulates a live microphone by slicing a file into
fixed-size chunks. This is what development and benchmarking run against.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from streaming_asr.audio.base import AudioChunk, AudioSource
from streaming_asr.audio.decode import decode

logger = logging.getLogger(__name__)


def load_wav(path: str | Path, target_sample_rate: int = 16000) -> np.ndarray:
    """Load an audio file as mono float32 at ``target_sample_rate``.

    Despite the name, any container :mod:`streaming_asr.audio.decode` supports
    works -- WAV, FLAC, OGG, MP3, and more via ffmpeg. The format is detected
    from the file's content, not its extension.
    """
    return decode(path, target_sample_rate=target_sample_rate)


class InMemorySource(AudioSource):
    """Streams a NumPy waveform that is already in memory.

    Args:
        samples: 1-D float32 waveform.
        sample_rate: Sample rate of ``samples``.
        chunk_samples: Samples per emitted chunk.
        real_time: If True, sleep between chunks so that chunks arrive at
            wall-clock speed. Leave False for benchmarking, where we want to
            measure compute rather than wait on the clock.
        pad_final_chunk: If True, zero-pad the trailing partial chunk so no
            audio is dropped. The reference ``AudioChunkIterator`` discarded
            it, silently truncating up to one chunk of speech.
    """

    def __init__(
        self,
        samples: np.ndarray,
        sample_rate: int,
        chunk_samples: int,
        real_time: bool = False,
        pad_final_chunk: bool = True,
    ) -> None:
        self._samples = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)
        self._sample_rate = sample_rate
        self._chunk_samples = int(chunk_samples)
        self._real_time = real_time
        self._pad_final_chunk = pad_final_chunk
        self._stopped = False

        if self._chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    @property
    def total_samples(self) -> int:
        return len(self._samples)

    @property
    def duration(self) -> float:
        return len(self._samples) / self._sample_rate

    def stop(self) -> None:
        """Ask the stream to terminate at the next chunk boundary."""
        self._stopped = True

    def stream(self) -> Iterator[AudioChunk]:
        n = len(self._samples)
        step = self._chunk_samples
        chunk_period = step / self._sample_rate
        stream_start = time.perf_counter()

        index = 0
        emitted = 0
        while index < n and not self._stopped:
            end = index + step
            if end <= n:
                block = self._samples[index:end]
                is_last = end >= n
            else:
                if not self._pad_final_chunk:
                    break
                block = np.zeros(step, dtype=np.float32)
                block[: n - index] = self._samples[index:]
                is_last = True

            if self._real_time:
                # Sleep until this chunk would actually have arrived.
                target = stream_start + (emitted + 1) * chunk_period
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

            yield AudioChunk(
                samples=block,
                start_sample=index,
                sample_rate=self._sample_rate,
                is_last=is_last,
                capture_time=time.perf_counter(),
            )
            index = end
            emitted += 1


class WavFileSource(InMemorySource):
    """Simulates a live stream from a WAV file on disk."""

    def __init__(
        self,
        path: str | Path,
        sample_rate: int,
        chunk_samples: int,
        real_time: bool = False,
        pad_final_chunk: bool = True,
        max_duration: Optional[float] = None,
    ) -> None:
        samples = load_wav(path, target_sample_rate=sample_rate)
        if max_duration is not None:
            samples = samples[: int(max_duration * sample_rate)]
        self.path = Path(path)
        super().__init__(
            samples=samples,
            sample_rate=sample_rate,
            chunk_samples=chunk_samples,
            real_time=real_time,
            pad_final_chunk=pad_final_chunk,
        )

    @property
    def waveform(self) -> np.ndarray:
        """The full decoded waveform (used by the final full-utterance decode)."""
        return self._samples
