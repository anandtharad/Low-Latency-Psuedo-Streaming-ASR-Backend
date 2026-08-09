"""Drift measurement used by tools/check_alignment_fidelity.py.

The tool's job is to decide whether ``aligner='time'`` is safe for a given
checkpoint, so its own measurement has to be trustworthy. The original version
was not: it keyed a lookup on word text, which collapses repeated words and
manufactures drift samples of several seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from check_alignment_fidelity import (  # noqa: E402
    _percentile,
    drift_samples,
    suggest_tolerance,
    summarise,
)

from streaming_asr.hypothesis.aligner import LevenshteinAligner  # noqa: E402
from streaming_asr.types import TimedToken, TimedWord  # noqa: E402

FRAME = 0.04
CHUNK = 0.16


def words(*specs: tuple[str, float]) -> list[TimedWord]:
    """Build words from ``(text, start)``; each lasts 0.2s."""
    out = []
    for i, (text, start) in enumerate(specs):
        token = TimedToken(i, "▁" + text, i, i, start, start + 0.2)
        out.append(TimedWord(text, (token,), start, start + 0.2))
    return out


def test_repeated_words_do_not_manufacture_drift():
    """The defect this test exists for.

    "it" appears twice, 3.4s apart. Pairing by text kept only one of them and
    reported their separation as drift; monotonic alignment pairs first-with-
    first and second-with-second.
    """
    previous = words(("see", 2.00), ("it", 2.20), ("eyes", 5.40), ("it", 5.60))
    current = words(("see", 2.04), ("it", 2.24), ("eyes", 5.44), ("it", 5.64))

    drifts = drift_samples(previous, current, LevenshteinAligner())

    assert len(drifts) == 4
    # Every pair drifted by exactly one frame; nothing near the 3.4s separation.
    assert all(abs(d - FRAME) < 1e-6 for d in drifts), drifts
    assert max(abs(d) for d in drifts) < 0.1


def test_sliding_window_drops_leading_words_without_mispairing():
    """The front of the window falls away as it slides; that is not drift."""
    previous = words(("alpha", 1.00), ("beta", 1.40), ("gamma", 1.80))
    current = words(("beta", 1.40), ("gamma", 1.80), ("delta", 2.20))

    drifts = drift_samples(previous, current, LevenshteinAligner())

    assert len(drifts) == 2                    # beta, gamma
    assert all(abs(d) < 1e-6 for d in drifts)  # they did not move


def test_no_pairs_when_nothing_is_shared():
    drifts = drift_samples(
        words(("alpha", 1.0)), words(("omega", 9.0)), LevenshteinAligner()
    )
    assert drifts == []


def test_empty_inputs_are_safe():
    aligner = LevenshteinAligner()
    assert drift_samples([], words(("a", 1.0)), aligner) == []
    assert drift_samples(words(("a", 1.0)), [], aligner) == []


# ---- statistics ----------------------------------------------------------


def test_percentiles():
    values = [float(i) for i in range(101)]
    assert _percentile(values, 50) == pytest.approx(50.0)
    assert _percentile(values, 95) == pytest.approx(95.0)
    assert _percentile([], 90) == 0.0


def test_mad_resists_outliers_that_would_wreck_a_stdev():
    """One bad pairing must not dominate the spread."""
    clean = [FRAME] * 50
    contaminated = clean + [3.4]          # a mis-paired repeated word

    stats = summarise(contaminated, windows=50, unmatched=0, total=50)

    assert stats["median_drift"] == pytest.approx(FRAME)
    assert stats["mad_drift"] < 0.01      # unmoved by the outlier
    assert stats["max_abs"] == pytest.approx(3.4)


# ---- tolerance suggestion ------------------------------------------------


def test_very_stable_timestamps_earn_a_tighter_tolerance_than_the_default():
    """Every word drifting exactly one frame supports 0.08s, not 0.12s.

    A tighter tolerance is the better answer here: it still clears the observed
    drift twice over, while leaving less room to conflate two distinct
    utterances of the same word.
    """
    stats = summarise([FRAME] * 100, windows=100, unmatched=0, total=100)
    tolerance, note = suggest_tolerance(stats, FRAME, CHUNK)

    assert tolerance == pytest.approx(0.08)      # p95 (40ms) + one frame
    assert "default" in note                     # says how it compares


def test_suggestion_lands_on_the_default_for_typical_drift():
    """p95 of two frames -> 0.12s, which is the shipped default."""
    stats = summarise(
        [FRAME] * 90 + [2 * FRAME] * 10, windows=100, unmatched=0, total=100
    )
    tolerance, note = suggest_tolerance(stats, FRAME, CHUNK)

    assert tolerance == pytest.approx(0.12)
    assert note == "matches the default"


def test_suggestion_is_capped_below_the_repeat_confusion_band():
    """Never suggest a tolerance wide enough to conflate distinct repetitions.

    The old formula (3x an outlier-inflated stdev) returned 0.37s -- over two
    chunks, which reintroduces the duplicate-commit bug it was meant to avoid.
    """
    stats = summarise([1.5] * 100, windows=100, unmatched=0, total=100)
    tolerance, note = suggest_tolerance(stats, FRAME, CHUNK)

    assert tolerance <= 0.25
    assert "levenshtein" in note


def test_suggestion_never_goes_below_two_frames():
    stats = summarise([0.0] * 100, windows=100, unmatched=0, total=100)
    tolerance, _ = suggest_tolerance(stats, FRAME, CHUNK)
    assert tolerance >= 2 * FRAME


def test_summarise_handles_no_samples():
    assert summarise([], windows=5, unmatched=0, total=0)["samples"] == 0
