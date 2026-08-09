"""Aligning successive hypotheses against one another.

Why this is not just string comparison
--------------------------------------
Section 13 of the brief is right to warn against it, and the reference notebook
proves the point. Once the utterance outgrows the rolling buffer, successive
hypotheses are not prefixes of each other -- the front falls off as the window
slides::

    chunk 31: "india versus pakistan world cup final"
    chunk 33: "versus pakistan world cup final"
    chunk 35: "pakistan world cup final"

A longest-common-prefix comparison sees those as total disagreement and would
never commit anything. Meanwhile real recognition churn ("I have chest pain" ->
"I've been having chest pain") reorders and re-segments words in place.

The ordering the brief prescribes is followed here: exploit the *known window
offset* first, fall back to sequence alignment, and only reach for DTW if
measurement shows it is needed.

:class:`TimeAwareAligner` is the default. Because every word carries an
absolute timestamp, and consecutive windows are shifted by a known amount, two
observations of the same spoken word land at nearly the same absolute time. The
aligner therefore only permits word pairs to align when they are temporally
plausible, which collapses the ambiguity that pure text alignment suffers from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

from streaming_asr.types import TimedWord

_INF = float("inf")


@dataclass(frozen=True)
class AlignmentOp:
    """One edit operation relating a previous item to a current item."""

    op: str                        # "match" | "substitute" | "insert" | "delete"
    prev_index: Optional[int]
    cur_index: Optional[int]


@dataclass
class Alignment:
    """The relationship between two consecutive hypotheses."""

    ops: list[AlignmentOp] = field(default_factory=list)
    distance: float = 0.0

    @property
    def matched_pairs(self) -> list[tuple[int, int]]:
        """``(prev_index, cur_index)`` for items judged identical."""
        return [
            (op.prev_index, op.cur_index)
            for op in self.ops
            if op.op == "match" and op.prev_index is not None and op.cur_index is not None
        ]

    def previous_for(self, cur_index: int) -> Optional[int]:
        """The previous-hypothesis index matching ``cur_index``, if any."""
        for op in self.ops:
            if op.cur_index == cur_index and op.op == "match":
                return op.prev_index
        return None

    @property
    def inserted_indices(self) -> list[int]:
        """Current-hypothesis indices with no counterpart in the previous one."""
        return [op.cur_index for op in self.ops if op.op == "insert" and op.cur_index is not None]

    def new_suffix_indices(self) -> list[int]:
        """Current indices appearing after the last matched item.

        For ``"I have chest"`` -> ``"I have chest pain"`` this is the index of
        ``"pain"``: the genuinely new region at the right edge.
        """
        matched = [c for _, c in self.matched_pairs]
        last = max(matched) if matched else -1
        return [op.cur_index for op in self.ops
                if op.cur_index is not None and op.cur_index > last]

    @property
    def is_stable(self) -> bool:
        """True when nothing was substituted or deleted -- pure extension."""
        return all(op.op in ("match", "insert") for op in self.ops)


class HypothesisAligner(ABC):
    """Compares the word sequences of two successive hypotheses."""

    name: str = "base"

    @abstractmethod
    def align(
        self, previous: Sequence[TimedWord], current: Sequence[TimedWord]
    ) -> Alignment:
        """Align ``previous`` to ``current``, both in time order."""

    @staticmethod
    def _texts(words: Sequence[TimedWord]) -> list[str]:
        return [w.text for w in words]


class PrefixAligner(HypothesisAligner):
    """Longest common prefix.

    The simplest possible strategy, and the one most streaming demos use. It is
    correct only while the utterance is shorter than the rolling buffer; after
    that the window slides and the shared region is a *suffix*/*infix*, not a
    prefix. Kept because it is the cheapest baseline to measure against.
    """

    name = "prefix"

    def align(
        self, previous: Sequence[TimedWord], current: Sequence[TimedWord]
    ) -> Alignment:
        prev_texts = self._texts(previous)
        cur_texts = self._texts(current)

        ops: list[AlignmentOp] = []
        i = 0
        while i < len(prev_texts) and i < len(cur_texts) and prev_texts[i] == cur_texts[i]:
            ops.append(AlignmentOp("match", i, i))
            i += 1

        distance = 0.0
        for j in range(i, len(prev_texts)):
            ops.append(AlignmentOp("delete", j, None))
            distance += 1
        for j in range(i, len(cur_texts)):
            ops.append(AlignmentOp("insert", None, j))
            distance += 1

        return Alignment(ops=ops, distance=distance)


class LevenshteinAligner(HypothesisAligner):
    """Token-level edit-distance alignment with backtrace.

    Handles substitution and re-segmentation, which the prefix aligner cannot,
    at O(n*m). Hypothesis lengths here are tens of words, so the cost is
    irrelevant next to a model call.
    """

    name = "levenshtein"

    def __init__(
        self,
        substitution_cost: float = 1.0,
        insertion_cost: float = 1.0,
        deletion_cost: float = 1.0,
    ) -> None:
        self.substitution_cost = substitution_cost
        self.insertion_cost = insertion_cost
        self.deletion_cost = deletion_cost

    def _pair_cost(
        self, prev: TimedWord, cur: TimedWord
    ) -> tuple[float, bool]:
        """Cost of aligning two items, and whether they count as a match."""
        if prev.text == cur.text:
            return 0.0, True
        return self.substitution_cost, False

    def align(
        self, previous: Sequence[TimedWord], current: Sequence[TimedWord]
    ) -> Alignment:
        n, m = len(previous), len(current)
        if n == 0 or m == 0:
            ops = [AlignmentOp("delete", i, None) for i in range(n)]
            ops += [AlignmentOp("insert", None, j) for j in range(m)]
            return Alignment(ops=ops, distance=float(n + m))

        # dp[i][j] = best cost aligning previous[:i] to current[:j]
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] + self.deletion_cost
        for j in range(1, m + 1):
            dp[0][j] = dp[0][j - 1] + self.insertion_cost

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                pair, _ = self._pair_cost(previous[i - 1], current[j - 1])
                dp[i][j] = min(
                    dp[i - 1][j - 1] + pair,
                    dp[i - 1][j] + self.deletion_cost,
                    dp[i][j - 1] + self.insertion_cost,
                )

        # Backtrace. Among equal-cost paths, prefer the one that yields the
        # most genuine matches: take the diagonal only when it is a real match,
        # and fall back to substitution last.
        #
        # Preferring the diagonal unconditionally silently loses matches. For
        # ["see", "it"] against ["it", "any", "more"] there are two paths of
        # cost 3, and the substitution-first one reports no matches at all --
        # hiding the fact that "it" is the same word in both. Consumers that
        # ask "which of these words have I already seen?" then get the wrong
        # answer.
        ops: list[AlignmentOp] = []
        i, j = n, m
        while i > 0 or j > 0:
            if i > 0 and j > 0:
                pair, is_match = self._pair_cost(previous[i - 1], current[j - 1])
                if is_match and abs(dp[i][j] - (dp[i - 1][j - 1] + pair)) < 1e-9:
                    ops.append(AlignmentOp("match", i - 1, j - 1))
                    i, j = i - 1, j - 1
                    continue
            if i > 0 and abs(dp[i][j] - (dp[i - 1][j] + self.deletion_cost)) < 1e-9:
                ops.append(AlignmentOp("delete", i - 1, None))
                i -= 1
                continue
            if j > 0 and abs(dp[i][j] - (dp[i][j - 1] + self.insertion_cost)) < 1e-9:
                ops.append(AlignmentOp("insert", None, j - 1))
                j -= 1
                continue
            if i > 0 and j > 0:
                ops.append(AlignmentOp("substitute", i - 1, j - 1))
                i, j = i - 1, j - 1
            elif i > 0:                       # pragma: no cover - defensive
                ops.append(AlignmentOp("delete", i - 1, None))
                i -= 1
            else:                             # pragma: no cover - defensive
                ops.append(AlignmentOp("insert", None, j - 1))
                j -= 1

        ops.reverse()
        return Alignment(ops=ops, distance=dp[n][m])


class TimeAwareAligner(LevenshteinAligner):
    """Edit-distance alignment constrained by absolute word timing.

    This is the default. Successive windows are offset by exactly one chunk, so
    the same spoken word should reappear at nearly the same absolute timestamp.
    Two words are only allowed to align when their start times agree to within
    ``time_tolerance``; beyond ``max_drift`` the pairing is forbidden outright.

    That constraint is what makes repeated words tractable. In "pain pain",
    pure text alignment cannot tell which occurrence is which; the timestamps
    can.
    """

    name = "time"

    def __init__(
        self,
        time_tolerance: float = 0.12,
        max_drift: Optional[float] = None,
        substitution_cost: float = 1.0,
        insertion_cost: float = 1.0,
        deletion_cost: float = 1.0,
        time_penalty_weight: float = 0.5,
    ) -> None:
        super().__init__(substitution_cost, insertion_cost, deletion_cost)
        self.time_tolerance = time_tolerance
        # Past this separation, two words cannot be the same observation.
        self.max_drift = max_drift if max_drift is not None else max(4 * time_tolerance, 0.4)
        self.time_penalty_weight = time_penalty_weight

    def _pair_cost(self, prev: TimedWord, cur: TimedWord) -> tuple[float, bool]:
        drift = abs(prev.start_time - cur.start_time)
        if drift > self.max_drift:
            # Temporally impossible: force the aligner to use insert+delete.
            return _INF, False
        if prev.text == cur.text and drift <= self.time_tolerance:
            return 0.0, True
        if prev.text == cur.text:
            # Same word, but drifted more than tolerance. Treat as a weak match
            # so it still aligns, without resetting the stability streak.
            return self.time_penalty_weight * (drift / self.max_drift), False
        return self.substitution_cost + self.time_penalty_weight * (drift / self.max_drift), False


def build_aligner(name: str, time_tolerance: float = 0.12) -> HypothesisAligner:
    """Factory used by the config layer."""
    name = name.lower()
    if name == "prefix":
        return PrefixAligner()
    if name == "levenshtein":
        return LevenshteinAligner()
    if name == "time":
        return TimeAwareAligner(time_tolerance=time_tolerance)
    if name == "dtw":
        from streaming_asr.hypothesis.dtw_aligner import DTWHypothesisAligner

        return DTWHypothesisAligner(time_tolerance=time_tolerance)
    raise ValueError(
        f"Unknown aligner {name!r}; expected one of prefix, levenshtein, time, dtw"
    )
