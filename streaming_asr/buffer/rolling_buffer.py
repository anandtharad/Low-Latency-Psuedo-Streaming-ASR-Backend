"""Fixed-size rolling audio buffer.

The buffer holds ``context_samples`` of history plus the newest chunk. Each
push shifts the contents left by one chunk and writes the new audio at the
right edge::

    Before:  [A][B][C][D]
    Push:              [E]
    After:   [B][C][D][E]

The array is allocated once and mutated in place. No per-chunk concatenation,
no growing lists -- at the reference operating point this runs 6.25 times per
second of audio, so allocation churn here would be felt.

The buffer also owns the mapping from buffer offsets to absolute stream time,
which is what makes cross-window hypothesis merging tractable.
"""

from __future__ import annotations

import numpy as np


class RollingAudioBuffer:
    """A pre-allocated sliding window over a stream of audio chunks.

    Args:
        buffer_samples: Total window size handed to the model.
        chunk_samples: Nominal push size. Pushes may be shorter (the final
            chunk) but never longer than ``buffer_samples``.
        sample_rate: Used for the sample <-> time conversions.
        dtype: Buffer dtype. float32 matches what the preprocessor wants and
            avoids a copy at the torch boundary.
    """

    def __init__(
        self,
        buffer_samples: int,
        chunk_samples: int,
        sample_rate: int = 16000,
        dtype: np.dtype | type = np.float32,
    ) -> None:
        if buffer_samples <= 0:
            raise ValueError("buffer_samples must be positive")
        if not 0 < chunk_samples <= buffer_samples:
            raise ValueError("chunk_samples must be in (0, buffer_samples]")

        self.buffer_samples = int(buffer_samples)
        self.chunk_samples = int(chunk_samples)
        self.sample_rate = int(sample_rate)

        # Shape (1, N) mirrors the reference layout and the batch dimension the
        # preprocessor expects, avoiding a reshape on every window.
        self._buffer = np.zeros((1, self.buffer_samples), dtype=dtype)
        self._total_pushed = 0

    # ---- state ----------------------------------------------------------

    @property
    def total_pushed(self) -> int:
        """Total samples ever pushed, i.e. the absolute stream position."""
        return self._total_pushed

    @property
    def valid_samples(self) -> int:
        """Real (non-warm-up-padding) samples currently in the buffer."""
        return min(self._total_pushed, self.buffer_samples)

    @property
    def is_warm(self) -> bool:
        """True once the buffer holds no warm-up zero padding."""
        return self._total_pushed >= self.buffer_samples

    @property
    def current_sample(self) -> int:
        """Absolute index one past the newest sample in the buffer."""
        return self._total_pushed

    @property
    def current_time(self) -> float:
        """Absolute timestamp of the right edge of the window, in seconds."""
        return self._total_pushed / self.sample_rate

    @property
    def window_start_sample(self) -> int:
        """Absolute index of ``buffer[0]``.

        Negative during warm-up, when the left of the buffer is zero padding
        that precedes the start of the stream. Keeping it signed means the
        offset-to-absolute-time mapping stays correct from the very first
        window instead of needing a special case.
        """
        return self._total_pushed - self.buffer_samples

    @property
    def window_start_time(self) -> float:
        return self.window_start_sample / self.sample_rate

    @property
    def window_end_time(self) -> float:
        return self.current_time

    @property
    def valid_start_time(self) -> float:
        """Absolute time of the first *real* audio sample in the window."""
        return max(0.0, self.window_start_time)

    # ---- mutation --------------------------------------------------------

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Shift in a new chunk and return the current window.

        Returns:
            The internal buffer, shape ``(1, buffer_samples)``. This is a live
            view, not a copy -- treat it as read-only and consume it before the
            next push. Use :meth:`window_copy` if you need to retain it.
        """
        flat = np.asarray(chunk, dtype=self._buffer.dtype).reshape(-1)
        n = flat.size
        if n == 0:
            return self._buffer
        if n > self.buffer_samples:
            raise ValueError(
                f"chunk of {n} samples exceeds buffer of {self.buffer_samples}"
            )

        if n == self.buffer_samples:
            self._buffer[0, :] = flat
        else:
            # In-place memmove of the retained context, then write the new tail.
            self._buffer[0, :-n] = self._buffer[0, n:]
            self._buffer[0, -n:] = flat

        self._total_pushed += n
        return self._buffer

    def reset(self) -> None:
        """Return to the pre-stream state."""
        self._buffer.fill(0.0)
        self._total_pushed = 0

    # ---- access ----------------------------------------------------------

    @property
    def window(self) -> np.ndarray:
        """Current window, shape ``(1, buffer_samples)``. Live view."""
        return self._buffer

    def window_copy(self) -> np.ndarray:
        return self._buffer.copy()

    def valid_window(self) -> np.ndarray:
        """Only the real audio, dropping warm-up padding. Shape ``(1, k)``."""
        return self._buffer[:, self.buffer_samples - self.valid_samples :]

    # ---- time mapping ----------------------------------------------------

    def offset_to_absolute_time(self, offset_samples: int) -> float:
        """Map an offset within the window to absolute stream time."""
        return (self.window_start_sample + offset_samples) / self.sample_rate

    def describe(self) -> str:
        return (
            f"window=[{self.window_start_time:.2f}s, {self.window_end_time:.2f}s] "
            f"valid={self.valid_samples}/{self.buffer_samples} "
            f"warm={self.is_warm}"
        )
