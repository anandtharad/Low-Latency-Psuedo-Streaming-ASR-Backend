"""Turning a stream of unstable window hypotheses into committed text.

The tracker does no ASR. It observes successive greedy hypotheses and decides
which parts have settled enough to be published as committed.

Design principle: **compute frequently, commit conservatively.**

Two independent conditions must both hold before a word is committed:

1. *Temporal maturity* -- the newest audio must be at least
   ``stability_window`` seconds past the word's end. The model is full-context
   and bidirectional, so its reading of audio at time t keeps changing as more
   right-context arrives. A word is not final merely because it sits outside
   the newest 160 ms of the window (section 15).
2. *Repeated agreement* -- ``min_stable_updates`` consecutive windows must have
   produced the same word at the same time. Agreement is established by the
   aligner, not by string equality of whole transcripts.

There is a hard deadline hiding in this design, and it is why
``StabilityConfig.validate_against`` exists: a word must be committed before it
scrolls out of the rolling buffer. Past that point no future window can see it
and the word is gone. The reference notebook has exactly this bug -- its
transcript degrades from "india versus pakistan world cup final" to "pakistan
world cup final" as the window slides past the opening words, with nothing
retaining them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence

from streaming_asr.hypothesis.aligner import (
    Alignment,
    HypothesisAligner,
    LevenshteinAligner,
    build_aligner,
)
from streaming_asr.types import GreedyHypothesis, TimedWord

logger = logging.getLogger(__name__)


@dataclass
class TrackerState:
    """A snapshot of the tracker after one update."""

    committed_text: str
    partial_text: str
    committed_words: list[TimedWord] = field(default_factory=list)
    partial_words: list[TimedWord] = field(default_factory=list)
    newly_committed: list[TimedWord] = field(default_factory=list)
    alignment: Optional[Alignment] = None
    #: Wall-clock audio time of the window that produced this state.
    audio_time: float = 0.0

    @property
    def full_hypothesis(self) -> str:
        parts = [p for p in (self.committed_text, self.partial_text) if p]
        return " ".join(parts)


class HypothesisTracker:
    """Maintains committed vs. partial transcript across sliding windows.

    Args:
        aligner: Comparison strategy. Defaults to the time-aware aligner.
        stability_window: Right-context required before a word may commit.
        min_stable_updates: Consecutive agreeing observations required.
        time_tolerance: Timestamp slack when deciding two observations refer to
            the same spoken word.
    """

    def __init__(
        self,
        aligner: Optional[HypothesisAligner] = None,
        stability_window: float = 0.6,
        min_stable_updates: int = 2,
        time_tolerance: float = 0.12,
    ) -> None:
        self.aligner = aligner or build_aligner("time", time_tolerance)
        #: Fixed, text-only aligner used purely to detect which part of a
        #: hypothesis re-transcribes already-committed audio. Independent of
        #: ``aligner`` because it must work when timestamps are unreliable --
        #: that is the situation it exists for.
        self._dedup_aligner = LevenshteinAligner()
        self.stability_window = stability_window
        self.min_stable_updates = max(1, int(min_stable_updates))
        self.time_tolerance = time_tolerance

        self._committed: list[TimedWord] = []
        self._committed_until: float = 0.0
        self._prev_words: list[TimedWord] = []
        self._prev_streaks: list[int] = []
        self._partial_words: list[TimedWord] = []
        self._update_count = 0

    # ---- public state ----------------------------------------------------

    @property
    def committed_words(self) -> list[TimedWord]:
        return list(self._committed)

    @property
    def committed_text(self) -> str:
        return " ".join(w.text for w in self._committed)

    @property
    def partial_text(self) -> str:
        return " ".join(w.text for w in self._partial_words)

    @property
    def full_hypothesis(self) -> str:
        parts = [p for p in (self.committed_text, self.partial_text) if p]
        return " ".join(parts)

    @property
    def committed_until(self) -> float:
        """End time of the last committed word."""
        return self._committed_until

    def reset(self) -> None:
        self._committed.clear()
        self._committed_until = 0.0
        self._prev_words = []
        self._prev_streaks = []
        self._partial_words = []
        self._update_count = 0

    # ---- core update -----------------------------------------------------

    def update(self, hypothesis: GreedyHypothesis) -> TrackerState:
        """Fold one window hypothesis into the tracked state."""
        self._update_count += 1
        audio_time = hypothesis.window_end_time

        candidates = self._eligible_words(hypothesis)
        alignment = self.aligner.align(self._prev_words, candidates)
        streaks = self._propagate_streaks(alignment, candidates)

        newly_committed = self._commit(candidates, streaks, audio_time)

        # Whatever was not committed is the partial tail.
        committed_texts_end = self._committed_until
        self._partial_words = [
            w for w in candidates if _center(w) > committed_texts_end
        ]

        self._prev_words = candidates
        self._prev_streaks = streaks

        return TrackerState(
            committed_text=self.committed_text,
            partial_text=self.partial_text,
            committed_words=list(self._committed),
            partial_words=list(self._partial_words),
            newly_committed=newly_committed,
            alignment=alignment,
            audio_time=audio_time,
        )

    # ---- internals -------------------------------------------------------

    def _eligible_words(self, hypothesis: GreedyHypothesis) -> list[TimedWord]:
        """Drop words this window cannot speak reliably about.

        Two exclusions:

        * ``truncated_start`` -- the window began mid-word, so the leading word
          is a fragment of something whose beginning is no longer visible.
        * Words already covered by committed audio. Committed text is never
          revised; re-recognitions of it are noise.
        """
        words = [
            word for i, word in enumerate(hypothesis.words)
            if not (i == 0 and word.truncated_start)
        ]

        # Order matters. The overlap strip must see the window's *complete*
        # transcription to count the repeated run correctly; filtering by time
        # first would remove part of that run and leave the strip to remove the
        # rest, deleting a genuine repetition twice over.
        words = self._strip_committed_overlap(words)

        # Backstop for anything the strip could not match -- typically a
        # committed region the model has since re-transcribed differently.
        #
        # Keyed on the midpoint: a word counts as new when more than half of it
        # lies past the committed boundary.
        #
        # Keying on the word's *start* instead is tempting -- it would catch a
        # model that merges across the boundary ("is" + "closed" -> "isclosed"
        # after "is" is committed, duplicating "is"). But measured against the
        # simulation it drops real words whenever timestamp drift exceeds the
        # gap between adjacent words, which in fast speech is routine. Losing
        # speech is far worse than duplicating it: a duplicate is visible and
        # can be repaired downstream, dropped audio is unrecoverable. The rare
        # merge-across-boundary duplicate is the accepted cost.
        return [w for w in words if _center(w) > self._committed_until]

    def _strip_committed_overlap(self, words: list[TimedWord]) -> list[TimedWord]:
        """Drop the leading run that re-transcribes already-committed audio.

        The timestamp cutoff alone is not enough. A word's CTC spike moves as
        the window slides, so a committed word can reappear a frame or two
        later, cross ``_committed_until``, and be committed again. Seen on a
        real Conformer as ``climb in`` becoming ``climb in in in``.

        Requiring the committed tail to match the hypothesis prefix *exactly*
        does not work either, and fails in the worst possible way. The model
        re-segments freely between windows -- "war craft" one window, "warcraft"
        the next -- and a single disagreement anywhere in the overlap breaks an
        equality test completely, stripping nothing. Then the first duplicate
        commits, the committed tail is now itself wrong, and no correct
        hypothesis will ever match it again: the corruption is self-sustaining,
        which is exactly the ``in in in`` / ``the the the`` cascade.

        So the overlap is found by **alignment**, which tolerates mismatches,
        and the walk stops at the first genuinely new word. Stopping early is
        safe -- the timestamp filter is still behind it -- whereas stripping too
        far would delete real speech irrecoverably.
        """
        if not self._committed or not words:
            return words

        # A margin beyond len(words): the committed tail may hold words the
        # current window no longer sees, or spurious ones from an earlier slip.
        tail = self._committed[-(len(words) + 8):]

        # Text-only alignment, deliberately. This runs *because* timestamps
        # drifted, so it must not depend on them; and it is a correctness
        # mechanism rather than a tunable, so it does not follow the
        # configured aligner.
        alignment = self._dedup_aligner.align(tail, words)

        # Boundary = the last confirmed match, stopping at the first word that
        # is genuinely new.
        #
        # Text agreement alone is NOT sufficient to call a word "already
        # committed", and relying on it loses speech. A speaker who says
        # "one two three four" and then says it again half a minute later
        # produces a hypothesis that matches the committed tail perfectly --
        # and the second utterance gets stripped and never appears. Observed
        # live, exactly that way.
        #
        # So a match only counts when the two words describe the *same stretch
        # of audio*. Re-transcriptions of committed audio overlap it in time
        # (drift is tens of milliseconds); a fresh utterance of the same words
        # is disjoint from it by seconds.
        #
        # Only matches move the boundary. A substitution near the trailing edge
        # is usually a stale committed word paired against genuinely new speech
        # ("in" against "colorado"), and trusting it would delete that speech
        # permanently. Substitutions *inside* the overlap are stepped over --
        # re-spellings like "war craft" -> "warcraft" must not halt the walk,
        # or the matches after them are never reached.
        last_match = -1
        for op in alignment.ops:
            if op.cur_index is None:
                continue                     # committed word not seen this window
            if op.prev_index is None:
                break                        # no counterpart at all: new speech
            if not self._same_audio(tail[op.prev_index], words[op.cur_index]):
                break                        # same words, different moment: new speech
            if op.op == "match":
                last_match = op.cur_index

        strip_until = last_match + 1
        if strip_until:
            logger.debug(
                "dropping %d already-committed word(s): %s",
                strip_until, [w.text for w in words[:strip_until]],
            )
        return words[strip_until:]

    def _same_audio(self, committed: TimedWord, candidate: TimedWord) -> bool:
        """Do these two words describe the same stretch of audio?

        Two observations of one spoken word overlap in time -- the CTC spike
        moves by tens of milliseconds between windows, far less than a word
        lasts. Two separate utterances of the same word occupy disjoint spans.
        A small tolerance absorbs drift at the edges without spanning the gap
        between distinct utterances.
        """
        overlap = (
            min(committed.end_time, candidate.end_time)
            - max(committed.start_time, candidate.start_time)
        )
        return overlap > -self.time_tolerance

    def _propagate_streaks(
        self, alignment: Alignment, candidates: Sequence[TimedWord]
    ) -> list[int]:
        """Carry agreement counts forward through the alignment.

        A word matched to a previous observation inherits its streak plus one;
        anything else starts over at one. Carrying counts *through the
        alignment* rather than keying them on text or time is what keeps the
        bookkeeping correct when words shift position between windows.
        """
        streaks = [1] * len(candidates)
        for prev_index, cur_index in alignment.matched_pairs:
            if 0 <= prev_index < len(self._prev_streaks):
                streaks[cur_index] = self._prev_streaks[prev_index] + 1
        return streaks

    def _commit(
        self,
        candidates: Sequence[TimedWord],
        streaks: Sequence[int],
        audio_time: float,
    ) -> list[TimedWord]:
        """Commit the longest stable prefix of ``candidates``.

        Commitment is prefix-only and in time order. Committing a word while
        leaving an earlier unstable word uncommitted would produce a transcript
        with a hole in it, and there is no way to fill that hole later without
        retracting text the caller has already been given.
        """
        deadline = audio_time - self.stability_window
        newly_committed: list[TimedWord] = []

        for word, streak in zip(candidates, streaks):
            if word.end_time > deadline:
                break                       # not enough right context yet
            if streak < self.min_stable_updates:
                break                       # has not agreed often enough
            if _center(word) <= self._committed_until:
                continue                    # already covered
            self._committed.append(word)
            self._committed_until = max(self._committed_until, word.end_time)
            newly_committed.append(word)

        if newly_committed:
            logger.debug(
                "committed %r (audio_time=%.2fs)",
                " ".join(w.text for w in newly_committed), audio_time,
            )
        return newly_committed

    # ---- finalisation ----------------------------------------------------

    def flush(self) -> list[TimedWord]:
        """Commit everything still pending, ignoring the stability rules.

        Called at the endpoint: there is no more audio coming, so waiting for
        additional right-context or agreement can never resolve anything. The
        result is still only the *provisional* streaming transcript -- the beam
        + LM decode overrides it.
        """
        flushed: list[TimedWord] = []
        for word in self._partial_words:
            if _center(word) > self._committed_until:
                self._committed.append(word)
                self._committed_until = max(self._committed_until, word.end_time)
                flushed.append(word)
        self._partial_words = []
        return flushed


def _center(word: TimedWord) -> float:
    """Midpoint of a word, used for boundary comparisons.

    Comparing midpoints rather than start or end times keeps a word that
    straddles the committed boundary from being either double-counted or
    dropped.
    """
    return 0.5 * (word.start_time + word.end_time)
