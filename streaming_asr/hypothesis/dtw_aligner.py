"""Optional DTW alignment backend.

Deliberately opt-in (``--enable-dtw`` / ``stability.aligner="dtw"``). Section 14
of the brief is explicit that DTW should not be adopted merely because it is a
well-known sequence-alignment algorithm, and the escalation order is:

    known window offset -> token/frame timing -> simple sequence alignment
    -> edit-distance alignment -> DTW, if measurement demands it

The first four are implemented in ``aligner.py`` and are the default path. DTW
is here so the question can be *tested* rather than argued about.

Where DTW genuinely earns its place is :meth:`DTWHypothesisAligner.align_posteriors`:
aligning the frame-level CTC posterior sequences of two overlapping windows.
That is a continuous, monotonic warping problem with no natural edit
operations, which is exactly what DTW is for -- unlike word-sequence alignment,
where insertions and deletions are real and edit distance models them better.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from streaming_asr.hypothesis.aligner import Alignment, AlignmentOp, HypothesisAligner
from streaming_asr.types import TimedWord


class DTWHypothesisAligner(HypothesisAligner):
    """Dynamic time warping over word sequences.

    Note the structural mismatch this carries: DTW produces a monotonic
    many-to-many warping in which *every* element is matched to something. Real
    hypothesis revision inserts and deletes words, which DTW must express as
    degenerate many-to-one segments. Edit distance models that directly. Expect
    this backend to underperform :class:`~streaming_asr.hypothesis.aligner.TimeAwareAligner`
    on word sequences; it is provided for measurement.

    Args:
        time_tolerance: Start-time agreement required to call a pair a match.
        band: Sakoe-Chiba band half-width in elements. ``None`` disables it.
        time_weight: Relative weight of timing versus token identity in the
            local cost.
    """

    name = "dtw"

    def __init__(
        self,
        time_tolerance: float = 0.12,
        band: Optional[int] = 8,
        time_weight: float = 1.0,
    ) -> None:
        self.time_tolerance = time_tolerance
        self.band = band
        self.time_weight = time_weight

    def _cost_matrix(
        self, previous: Sequence[TimedWord], current: Sequence[TimedWord]
    ) -> np.ndarray:
        n, m = len(previous), len(current)
        prev_starts = np.fromiter((w.start_time for w in previous), dtype=np.float64, count=n)
        cur_starts = np.fromiter((w.start_time for w in current), dtype=np.float64, count=m)

        # Timing term: normalised absolute start-time difference.
        time_cost = np.abs(prev_starts[:, None] - cur_starts[None, :]) / max(
            self.time_tolerance, 1e-6
        )
        # Identity term: 0 for the same word, 1 otherwise.
        identity = np.ones((n, m), dtype=np.float64)
        for i, pw in enumerate(previous):
            for j, cw in enumerate(current):
                if pw.text == cw.text:
                    identity[i, j] = 0.0
        return identity + self.time_weight * time_cost

    def align(
        self, previous: Sequence[TimedWord], current: Sequence[TimedWord]
    ) -> Alignment:
        n, m = len(previous), len(current)
        if n == 0 or m == 0:
            ops = [AlignmentOp("delete", i, None) for i in range(n)]
            ops += [AlignmentOp("insert", None, j) for j in range(m)]
            return Alignment(ops=ops, distance=float(n + m))

        local = self._cost_matrix(previous, current)
        path, distance = _dtw_path(local, band=self.band)

        matched_prev: set[int] = set()
        matched_cur: set[int] = set()
        ops: list[AlignmentOp] = []
        for i, j in path:
            if previous[i].text == current[j].text and \
                    abs(previous[i].start_time - current[j].start_time) <= self.time_tolerance:
                if i not in matched_prev and j not in matched_cur:
                    ops.append(AlignmentOp("match", i, j))
                    matched_prev.add(i)
                    matched_cur.add(j)

        # Everything the warping path could not pair off is an edit.
        for i in range(n):
            if i not in matched_prev:
                ops.append(AlignmentOp("delete", i, None))
        for j in range(m):
            if j not in matched_cur:
                ops.append(AlignmentOp("insert", None, j))

        ops.sort(key=lambda op: (op.cur_index if op.cur_index is not None else 1e9,
                                 op.prev_index if op.prev_index is not None else 1e9))
        return Alignment(ops=ops, distance=float(distance))

    def align_posteriors(
        self, previous: np.ndarray, current: np.ndarray, band: Optional[int] = None
    ) -> tuple[list[tuple[int, int]], float]:
        """Align two frame-level CTC posterior sequences.

        This is the use DTW is actually suited to. Given windows N and N+1,
        the overlapping audio should produce near-identical posterior
        trajectories offset by exactly one chunk; the warping path measures how
        far the model's interpretation of that shared audio actually moved.

        Args:
            previous: ``(T1, V)`` posteriors or log-posteriors.
            current: ``(T2, V)`` posteriors.

        Returns:
            The warping path and its normalised cost.
        """
        if previous.ndim != 2 or current.ndim != 2:
            raise ValueError("posterior sequences must be 2-D (T, V)")

        # Cosine distance is scale-invariant, so it works on probabilities and
        # log-probabilities alike.
        a = previous / (np.linalg.norm(previous, axis=1, keepdims=True) + 1e-9)
        b = current / (np.linalg.norm(current, axis=1, keepdims=True) + 1e-9)
        local = 1.0 - a @ b.T

        path, distance = _dtw_path(local, band=band if band is not None else self.band)
        return path, float(distance / max(len(path), 1))


def _dtw_path(
    local: np.ndarray, band: Optional[int] = None
) -> tuple[list[tuple[int, int]], float]:
    """Standard DTW with an optional Sakoe-Chiba band.

    Returns the warping path in ascending order and the accumulated cost.
    """
    n, m = local.shape
    inf = np.inf
    acc = np.full((n + 1, m + 1), inf, dtype=np.float64)
    acc[0, 0] = 0.0

    for i in range(1, n + 1):
        if band is not None:
            # Band around the diagonal, scaled for non-square inputs.
            center = int(round((i - 1) * m / max(n, 1)))
            lo, hi = max(1, center - band + 1), min(m, center + band + 1)
        else:
            lo, hi = 1, m
        for j in range(lo, hi + 1):
            best = min(acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
            if best < inf:
                acc[i, j] = local[i - 1, j - 1] + best

    if not np.isfinite(acc[n, m]):
        # The band was too tight to reach the corner; retry unbanded.
        if band is not None:
            return _dtw_path(local, band=None)
        return [], float("inf")

    path: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = (acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1])
        step = int(np.argmin(candidates))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1

    path.reverse()
    return path, float(acc[n, m])
