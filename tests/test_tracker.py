"""Hypothesis stabilisation: what gets committed, when, and what never does."""

from __future__ import annotations

import pytest

from streaming_asr.hypothesis.aligner import PrefixAligner, TimeAwareAligner
from streaming_asr.hypothesis.tracker import HypothesisTracker
from streaming_asr.types import GreedyHypothesis, TimedToken, TimedWord

WORD_STEP = 0.5
WORD_LEN = 0.4


def make_hypothesis(
    texts: list[str],
    audio_time: float,
    window_duration: float = 4.0,
    truncate_first: bool = False,
    offset: float = 0.0,
) -> GreedyHypothesis:
    """Build a hypothesis whose words sit on a regular grid from ``offset``."""
    words: list[TimedWord] = []
    for i, text in enumerate(texts):
        start = offset + i * WORD_STEP
        token = TimedToken(
            token_id=i, token=("" if (i == 0 and truncate_first) else "▁") + text,
            start_frame=i, end_frame=i, start_time=start, end_time=start + WORD_LEN,
        )
        words.append(TimedWord(
            text=text, tokens=(token,), start_time=start, end_time=start + WORD_LEN,
            truncated_start=(i == 0 and truncate_first),
        ))
    return GreedyHypothesis(
        text=" ".join(texts), token_ids=[], tokens=[], frame_indices=[],
        token_spans=[t for w in words for t in w.tokens], words=words,
        window_start_time=audio_time - window_duration, window_end_time=audio_time,
    )


def test_stable_prefix_commits_before_unstable_tail():
    """Section 28: given repeated "I have" then "I have chest",
    "I have" must commit before "chest" does."""
    tracker = HypothesisTracker(stability_window=0.2, min_stable_updates=2)

    tracker.update(make_hypothesis(["i", "have"], audio_time=1.5))
    state = tracker.update(make_hypothesis(["i", "have"], audio_time=2.0))

    assert state.committed_text == "i have"

    state = tracker.update(make_hypothesis(["i", "have", "chest"], audio_time=2.5))
    assert state.committed_text == "i have"      # "chest" seen only once
    assert "chest" in state.partial_text

    state = tracker.update(make_hypothesis(["i", "have", "chest"], audio_time=3.0))
    assert state.committed_text == "i have chest"


def test_single_observation_never_commits():
    tracker = HypothesisTracker(stability_window=0.1, min_stable_updates=2)
    state = tracker.update(make_hypothesis(["i", "have"], audio_time=5.0))
    assert state.committed_text == ""
    assert state.partial_text == "i have"


def test_min_stable_updates_is_honoured():
    tracker = HypothesisTracker(stability_window=0.1, min_stable_updates=3)
    for _ in range(2):
        state = tracker.update(make_hypothesis(["i", "have"], audio_time=5.0))
        assert state.committed_text == ""
    state = tracker.update(make_hypothesis(["i", "have"], audio_time=5.0))
    assert state.committed_text == "i have"


def test_stability_window_withholds_recent_words():
    """A word must not commit until enough right-context has arrived.

    The model is full-context, so its reading of recent audio is still moving.
    """
    tracker = HypothesisTracker(stability_window=1.0, min_stable_updates=1)

    # "have" ends at 0.9 s; at audio_time 1.5 s only 0.6 s of right-context
    # exists, which is short of the 1.0 s required.
    state = tracker.update(make_hypothesis(["i", "have"], audio_time=1.5))
    assert state.committed_text == "i"

    state = tracker.update(make_hypothesis(["i", "have"], audio_time=2.0))
    assert state.committed_text == "i have"


def test_commitment_is_prefix_only_no_holes():
    """A stable later word must not jump ahead of an unstable earlier one.

    Committing out of order would leave a gap in the transcript that can never
    be filled without retracting text already handed to the caller.
    """
    tracker = HypothesisTracker(stability_window=0.2, min_stable_updates=2)

    tracker.update(make_hypothesis(["i", "have", "chest"], audio_time=3.0))
    # "have" is re-recognised as "had": its streak resets, so nothing after it
    # may commit either.
    state = tracker.update(make_hypothesis(["i", "had", "chest"], audio_time=3.5))

    assert state.committed_text == "i"
    assert "chest" not in state.committed_text


def test_truncated_first_word_is_never_committed():
    """A word clipped by the left edge of the window is a fragment."""
    tracker = HypothesisTracker(stability_window=0.1, min_stable_updates=1)
    state = tracker.update(
        make_hypothesis(["ve", "chest", "pain"], audio_time=5.0, truncate_first=True)
    )
    assert not state.committed_text.startswith("ve")
    assert "chest" in state.committed_text


def test_committed_text_survives_the_window_sliding_past_it():
    """The failure the reference implementation exhibits.

    Its transcript decays from "india versus pakistan world cup final" to
    "pakistan world cup final" as the rolling buffer scrolls past the opening
    words. Committed text must accumulate independently of what any single
    window can still see.
    """
    tracker = HypothesisTracker(stability_window=0.3, min_stable_updates=2)

    for audio_time in (2.0, 2.5):
        tracker.update(make_hypothesis(["india", "versus", "pakistan"], audio_time=audio_time))
    assert "india" in tracker.committed_text

    # The window has now slid; "india" is gone from the model's view entirely.
    for audio_time in (4.0, 4.5):
        tracker.update(
            make_hypothesis(["versus", "pakistan", "world"], audio_time=audio_time, offset=0.5)
        )

    assert tracker.committed_text.startswith("india versus pakistan")
    assert "world" in tracker.full_hypothesis


def _hypothesis_at(specs: list[tuple[str, float, float]], audio_time: float) -> GreedyHypothesis:
    """Build a hypothesis from explicit ``(text, start, end)`` word spans."""
    words = []
    for i, (text, start, end) in enumerate(specs):
        token = TimedToken(
            token_id=i, token="▁" + text, start_frame=i, end_frame=i,
            start_time=start, end_time=end,
        )
        words.append(TimedWord(text=text, tokens=(token,), start_time=start, end_time=end))
    return GreedyHypothesis(
        text=" ".join(t for t, _, _ in specs), token_ids=[], tokens=[],
        frame_indices=[], token_spans=[t for w in words for t in w.tokens],
        words=words, window_start_time=audio_time - 4.0, window_end_time=audio_time,
    )


def test_drifted_word_is_not_committed_twice():
    """Regression: a real Conformer duplicated words as its spikes drifted.

    Observed with stt_en_conformer_ctc_large:
        "...wish to see it"  +  "it any more..."  ->  "...see it it any more"

    A committed word reappears in the next window a frame or two later,
    crosses the timestamp cutoff, and commits again. The timestamp comparison
    alone cannot catch this; the text overlap can.
    """
    tracker = HypothesisTracker(stability_window=0.2, min_stable_updates=2)

    for audio_time in (2.0, 2.2):
        tracker.update(_hypothesis_at(
            [("see", 1.00, 1.30), ("it", 1.35, 1.50)], audio_time
        ))
    assert tracker.committed_text == "see it"

    # Same "it", re-emitted 0.10s later -- enough for its centre to clear the
    # end of the committed one.
    for audio_time in (2.6, 2.8):
        tracker.update(_hypothesis_at(
            [("it", 1.45, 1.60), ("any", 1.70, 1.95), ("more", 2.00, 2.30)],
            audio_time,
        ))

    assert tracker.committed_text == "see it any more", tracker.committed_text


def test_multi_word_overlap_is_stripped():
    """The repeated run can be longer than one word."""
    tracker = HypothesisTracker(stability_window=0.2, min_stable_updates=2)

    for audio_time in (2.0, 2.2):
        tracker.update(_hypothesis_at(
            [("very", 1.00, 1.30), ("like", 1.35, 1.60)], audio_time
        ))
    assert tracker.committed_text == "very like"

    for audio_time in (2.6, 2.8):
        tracker.update(_hypothesis_at(
            [("very", 1.05, 1.35), ("like", 1.42, 1.68),
             ("the", 1.75, 1.95), ("old", 2.00, 2.30)],
            audio_time,
        ))

    assert tracker.committed_text == "very like the old", tracker.committed_text


def test_resegmentation_does_not_break_duplicate_removal():
    """The model re-spells words between windows; the strip must survive it.

    Real case: "a war craft exclusive ol bear like" committed, then the next
    window transcribed the same audio as "a warcraft exclusive ol bear like
    creature". An exact suffix/prefix test finds no match at all because of the
    one re-segmented word, strips nothing, and commits "like" twice.
    """
    tracker = HypothesisTracker(stability_window=0.2, min_stable_updates=2)

    committed = [("a", 1.00, 1.15), ("war", 1.20, 1.45), ("craft", 1.50, 1.80),
                 ("exclusive", 1.90, 2.40), ("ol", 2.50, 2.65),
                 ("bear", 2.70, 2.95), ("like", 3.00, 3.25)]
    for audio_time in (3.6, 3.8):
        tracker.update(_hypothesis_at(committed, audio_time))
    assert tracker.committed_text.endswith("bear like")

    # Same audio, "war craft" now one word, plus new content and drift.
    respelled = [("a", 1.02, 1.17), ("warcraft", 1.22, 1.82),
                 ("exclusive", 1.92, 2.42), ("ol", 2.52, 2.67),
                 ("bear", 2.72, 2.97), ("like", 3.05, 3.30),
                 ("creature", 3.40, 3.85)]
    for audio_time in (4.2, 4.4):
        tracker.update(_hypothesis_at(respelled, audio_time))

    assert tracker.committed_text.count("like") == 1, tracker.committed_text
    assert tracker.committed_text.endswith("bear like creature"), tracker.committed_text


def test_tracker_resyncs_after_a_duplicate_has_been_committed():
    """A corrupted committed tail must not block all future stripping.

    This is what made the bug cascade: once "climb in in" was committed, no
    correct hypothesis could ever match the committed tail exactly again, so
    every later window added another duplicate -- "in in in", "the the the".
    Alignment tolerates the extra words and re-synchronises.
    """
    tracker = HypothesisTracker(stability_window=0.2, min_stable_updates=2)

    # Seed a transcript that already contains a spurious repeat.
    for text, start, end in [("climb", 1.0, 1.3), ("in", 1.4, 1.5),
                             ("in", 1.55, 1.65)]:
        token = TimedToken(0, "▁" + text, 0, 0, start, end)
        tracker._committed.append(TimedWord(text, (token,), start, end))
    tracker._committed_until = 1.65

    # A correct hypothesis: one "in", then new content.
    clean = [("climb", 1.02, 1.32), ("in", 1.42, 1.52),
             ("colorado", 1.80, 2.40), ("which", 2.50, 2.80)]
    for audio_time in (3.0, 3.2):
        tracker.update(_hypothesis_at(clean, audio_time))

    # No *third* "in": the damage is not compounded.
    assert tracker.committed_text.count("in") == 2, tracker.committed_text
    assert tracker.committed_text.endswith("colorado which"), tracker.committed_text


def test_genuinely_repeated_words_are_preserved():
    """Stitching must not swallow a real repetition.

    The window re-transcribes both occurrences, so the overlap is one word and
    the second "the" survives.
    """
    tracker = HypothesisTracker(stability_window=0.2, min_stable_updates=2)

    for audio_time in (2.0, 2.2):
        tracker.update(_hypothesis_at([("the", 1.00, 1.20)], audio_time))
    assert tracker.committed_text == "the"

    for audio_time in (2.6, 2.8):
        tracker.update(_hypothesis_at(
            [("the", 1.00, 1.20), ("the", 1.30, 1.50), ("cat", 1.60, 1.90)],
            audio_time,
        ))

    assert tracker.committed_text == "the the cat", tracker.committed_text


def test_committed_words_are_not_duplicated_when_rerecognised():
    tracker = HypothesisTracker(stability_window=0.2, min_stable_updates=2)
    for audio_time in (1.5, 2.0, 2.5, 3.0):
        tracker.update(make_hypothesis(["i", "have"], audio_time=audio_time))
    assert tracker.committed_text == "i have"


def test_flush_commits_the_pending_tail():
    tracker = HypothesisTracker(stability_window=2.0, min_stable_updates=1)
    tracker.update(make_hypothesis(["i", "have", "chest"], audio_time=1.5))
    assert tracker.committed_text == ""

    flushed = tracker.flush()
    assert [w.text for w in flushed] == ["i", "have", "chest"]
    assert tracker.committed_text == "i have chest"


def test_full_hypothesis_joins_committed_and_partial():
    tracker = HypothesisTracker(stability_window=0.2, min_stable_updates=2)
    tracker.update(make_hypothesis(["i", "have", "chest"], audio_time=2.0))
    # "pain" is new this window, so it stays provisional while the rest commits.
    state = tracker.update(make_hypothesis(["i", "have", "chest", "pain"], audio_time=2.2))

    assert state.committed_text == "i have chest"
    assert state.partial_text == "pain"
    assert state.full_hypothesis == "i have chest pain"


def test_reset_clears_all_state():
    tracker = HypothesisTracker(stability_window=0.1, min_stable_updates=1)
    tracker.update(make_hypothesis(["i", "have"], audio_time=5.0))
    tracker.reset()
    assert tracker.committed_text == ""
    assert tracker.committed_until == 0.0


def test_prefix_aligner_backend_still_works_for_growing_hypotheses():
    tracker = HypothesisTracker(
        aligner=PrefixAligner(), stability_window=0.2, min_stable_updates=2
    )
    tracker.update(make_hypothesis(["i", "have"], audio_time=2.0))
    state = tracker.update(make_hypothesis(["i", "have"], audio_time=2.2))
    assert state.committed_text == "i have"


def test_newly_committed_reports_only_the_promotion():
    tracker = HypothesisTracker(stability_window=0.2, min_stable_updates=2)
    tracker.update(make_hypothesis(["i", "have"], audio_time=2.0))
    state = tracker.update(make_hypothesis(["i", "have"], audio_time=2.2))
    assert [w.text for w in state.newly_committed] == ["i", "have"]

    state = tracker.update(make_hypothesis(["i", "have"], audio_time=2.4))
    assert state.newly_committed == []


@pytest.mark.parametrize("aligner", [TimeAwareAligner(), PrefixAligner()])
def test_no_commitment_from_empty_hypotheses(aligner):
    tracker = HypothesisTracker(aligner=aligner, stability_window=0.1, min_stable_updates=1)
    for _ in range(3):
        state = tracker.update(make_hypothesis([], audio_time=1.0))
    assert state.committed_text == ""
