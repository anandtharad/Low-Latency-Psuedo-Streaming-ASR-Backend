"""Fast greedy CTC decoding.

Greedy decoding exists here for one reason: latency. It runs on every window
(6.25 times per second of audio at the reference operating point), so it must
be cheap -- argmax plus a single pass over the frames, no beam, no language
model. The expensive beam+KenLM decoder runs once, at the endpoint.

Unlike a typical greedy decoder this one does not throw the frame indices away.
Token timing is what allows hypotheses from overlapping windows to be merged by
*when* a word was spoken rather than by string comparison.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from streaming_asr.types import (
    GreedyHypothesis,
    TimedToken,
    group_tokens_into_words,
    tokens_to_text,
)


def ctc_collapse(
    frame_ids: Sequence[int] | np.ndarray, blank_id: int
) -> list[tuple[int, int, int]]:
    """Collapse a frame-level CTC path into ``(token_id, start, end)`` spans.

    Standard CTC collapse: merge runs of identical labels, then drop blanks.
    A blank between two identical labels separates them, so
    ``A A blank A`` yields two ``A`` tokens rather than one::

        blank blank A A blank B B  ->  [(A, 2, 3), (B, 5, 6)]

    ``end`` is inclusive.
    """
    spans: list[tuple[int, int, int]] = []
    previous = -1
    for frame, raw in enumerate(frame_ids):
        token = int(raw)
        if token != previous:
            if token != blank_id:
                spans.append((token, frame, frame))
        elif token != blank_id and spans:
            # Same non-blank label repeating: extend the current span.
            token_id, start, _ = spans[-1]
            spans[-1] = (token_id, start, frame)
        previous = token
    return spans


class GreedyCTCDecoder:
    """Argmax + collapse, with timing recovery.

    Args:
        vocabulary: Token strings, indexed by class id.
        blank_id: CTC blank index. Defaults to the last entry, matching the
            reference, which appends ``"__"`` to the model vocabulary.
        compute_posteriors: Also compute per-token confidence. Costs one extra
            reduction over the logits.
        logits_are_log_probs: The NeMo export emits log-probabilities. When
            True, posteriors are ``exp(mean log p)``; when False the frames are
            softmaxed first. ``None`` auto-detects on the first call.
    """

    def __init__(
        self,
        vocabulary: Sequence[str],
        blank_id: Optional[int] = None,
        compute_posteriors: bool = True,
        logits_are_log_probs: Optional[bool] = None,
    ) -> None:
        self.vocabulary = list(vocabulary)
        self.blank_id = blank_id if blank_id is not None else len(self.vocabulary) - 1
        self.compute_posteriors = compute_posteriors
        self._log_probs = logits_are_log_probs

        if not 0 <= self.blank_id < len(self.vocabulary):
            raise ValueError(
                f"blank_id {self.blank_id} outside vocabulary of {len(self.vocabulary)}"
            )

    # ---- helpers ---------------------------------------------------------

    def _detect_log_probs(self, logits: np.ndarray) -> bool:
        """Log-probability rows sum (in exp space) to ~1; raw logits do not."""
        probe = logits[0, : min(8, logits.shape[1])].astype(np.float64)
        if probe.size == 0:
            return True
        total = np.exp(probe).sum(axis=-1)
        return bool(np.allclose(total, 1.0, atol=0.05))

    def _frame_posteriors(self, window: np.ndarray, top: np.ndarray) -> np.ndarray:
        if self._log_probs:
            return np.exp(top)
        # Softmax denominator only; the numerator is already the top logit.
        shifted = window - window.max(axis=-1, keepdims=True)
        denom = np.log(np.exp(shifted).sum(axis=-1)) + window.max(axis=-1)
        return np.exp(top - denom)

    # ---- main entry point ------------------------------------------------

    def decode(
        self,
        logits: np.ndarray,
        window_start_time: float,
        window_end_time: float,
        ctc_frame_duration: float,
        new_audio_start_time: Optional[float] = None,
        new_audio_end_time: Optional[float] = None,
        valid_frames: Optional[int] = None,
        retain_frame_posteriors: bool = False,
    ) -> GreedyHypothesis:
        """Decode one window's logits into a timed hypothesis.

        Args:
            logits: ``(B, T, V)``; only batch item 0 is decoded.
            window_start_time: Absolute stream time of CTC frame 0. May be
                negative while the rolling buffer is still warming up.
            ctc_frame_duration: Seconds per CTC frame (hop x subsampling).
            valid_frames: Truncate to this many frames before decoding.

        Returns:
            A :class:`GreedyHypothesis` whose token timestamps are absolute.
        """
        start = time.perf_counter()

        window = logits[0]
        if valid_frames is not None:
            window = window[:valid_frames]

        if self._log_probs is None:
            self._log_probs = self._detect_log_probs(logits)

        frame_ids = window.argmax(axis=-1)
        spans = ctc_collapse(frame_ids, self.blank_id)

        posteriors: Optional[np.ndarray] = None
        if self.compute_posteriors and spans:
            top = np.take_along_axis(window, frame_ids[:, None], axis=-1)[:, 0]
            posteriors = self._frame_posteriors(window, top)

        token_spans: list[TimedToken] = []
        for token_id, start_frame, end_frame in spans:
            if posteriors is not None:
                confidence = float(posteriors[start_frame : end_frame + 1].mean())
            else:
                confidence = 1.0
            token_spans.append(
                TimedToken(
                    token_id=token_id,
                    token=self.vocabulary[token_id],
                    start_frame=start_frame,
                    end_frame=end_frame,
                    # Frame k covers [k*d, (k+1)*d) within the window; shift by
                    # the window's absolute origin to get stream time.
                    start_time=window_start_time + start_frame * ctc_frame_duration,
                    end_time=window_start_time + (end_frame + 1) * ctc_frame_duration,
                    posterior=confidence,
                )
            )

        tokens = [t.token for t in token_spans]
        return GreedyHypothesis(
            text=tokens_to_text(tokens),
            token_ids=[t.token_id for t in token_spans],
            tokens=tokens,
            frame_indices=[t.start_frame for t in token_spans],
            token_spans=token_spans,
            words=group_tokens_into_words(token_spans),
            window_start_time=window_start_time,
            window_end_time=window_end_time,
            new_audio_start_time=(
                new_audio_start_time if new_audio_start_time is not None else window_end_time
            ),
            new_audio_end_time=(
                new_audio_end_time if new_audio_end_time is not None else window_end_time
            ),
            ctc_frame_duration=ctc_frame_duration,
            decode_time=time.perf_counter() - start,
            frame_posteriors=(
                posteriors if (retain_frame_posteriors and posteriors is not None) else None
            ),
        )

    def decode_text(self, logits: np.ndarray, valid_frames: Optional[int] = None) -> str:
        """Text-only convenience path, for offline greedy-vs-beam comparison."""
        window = logits[0]
        if valid_frames is not None:
            window = window[:valid_frames]
        spans = ctc_collapse(window.argmax(axis=-1), self.blank_id)
        return tokens_to_text([self.vocabulary[t] for t, _, _ in spans])
