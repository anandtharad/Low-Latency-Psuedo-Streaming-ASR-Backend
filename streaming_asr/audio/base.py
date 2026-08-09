"""Audio source interface.

The ASR engine consumes ``AudioChunk`` objects and knows nothing about where
they came from. Swapping a WAV file for PyAudio, sounddevice, WebRTC, RTP or a
gRPC stream means writing a new ``AudioSource`` -- no change to the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class AudioChunk:
    """A fixed-size block of mono float32 PCM.

    Attributes:
        samples: 1-D float32 array in [-1, 1].
        start_sample: Index of the first sample within the overall stream.
        sample_rate: Sample rate in Hz.
        is_last: True if this is the final chunk of the stream.
        capture_time: ``time.perf_counter()`` at which the chunk became
            available. For a live source this is the true arrival time; for the
            WAV simulator it is the simulated arrival time. Latency metrics are
            measured against this, so it must be set by the source.
    """

    samples: np.ndarray
    start_sample: int
    sample_rate: int
    is_last: bool = False
    capture_time: float = 0.0

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def start_time(self) -> float:
        return self.start_sample / self.sample_rate

    @property
    def end_time(self) -> float:
        return (self.start_sample + len(self.samples)) / self.sample_rate


class AudioSource(ABC):
    """Produces a stream of equally-sized audio chunks."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @property
    @abstractmethod
    def chunk_samples(self) -> int:
        ...

    @abstractmethod
    def stream(self) -> Iterator[AudioChunk]:
        """Yield chunks until the source is exhausted or stopped."""

    def close(self) -> None:  # pragma: no cover - trivial default
        """Release any underlying device or file handle."""
        return None

    def __enter__(self) -> "AudioSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
