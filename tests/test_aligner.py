"""Hypothesis alignment across successive windows."""

from __future__ import annotations

import pytest

from streaming_asr.hypothesis.aligner import (
    LevenshteinAligner,
    PrefixAligner,
    TimeAwareAligner,
    build_aligner,
)
from streaming_asr.hypothesis.dtw_aligner import DTWHypothesisAligner
from streaming_asr.types import TimedToken, TimedWord


def words(*specs: tuple[str, float, float]) -> list[TimedWord]:
    """Build TimedWords from ``(text, start, end)`` triples."""
    result = []
    for i, (text, start, end) in enumerate(specs):
        token = TimedToken(
            token_id=i, token="▁" + text, start_frame=i, end_frame=i,
            start_time=start, end_time=end,
        )
        result.append(
            TimedWord(text=text, tokens=(token,), start_time=start, end_time=end)
        )
    return result


def sequential(*texts: str, step: float = 0.5, offset: float = 0.0) -> list[TimedWord]:
    """Evenly-spaced words, as a steadily-advancing utterance produces."""
    return words(*[
        (text, offset + i * step, offset + i * step + step * 0.8)
        for i, text in enumerate(texts)
    ])


ALL_ALIGNERS = [PrefixAligner(), LevenshteinAligner(), TimeAwareAligner(), DTWHypothesisAligner()]


@pytest.mark.parametrize("aligner", ALL_ALIGNERS, ids=lambda a: a.name)
def test_pure_extension_identifies_new_suffix(aligner):
    """"I have chest" -> "I have chest pain": the new region is "pain"."""
    previous = sequential("i", "have", "chest")
    current = sequential("i", "have", "chest", "pain")

    alignment = aligner.align(previous, current)
    new_indices = alignment.new_suffix_indices()

    assert [current[i].text for i in new_indices] == ["pain"]
    assert len(alignment.matched_pairs) == 3


@pytest.mark.parametrize("aligner", ALL_ALIGNERS, ids=lambda a: a.name)
def test_identical_hypotheses_align_completely(aligner):
    previous = sequential("i", "have", "chest")
    alignment = aligner.align(previous, list(previous))
    assert len(alignment.matched_pairs) == 3
    assert alignment.new_suffix_indices() == []


def test_prefix_aligner_fails_once_the_window_slides():
    """Documents exactly why the prefix aligner cannot be the default.

    When the buffer scrolls past the opening words -- the behaviour visible in
    the reference notebook, where "india versus pakistan world cup final"
    decays to "pakistan world cup final" -- prefix comparison sees total
    disagreement and would never commit anything.
    """
    previous = sequential("india", "versus", "pakistan", "world")
    current = sequential("versus", "pakistan", "world", "cup", offset=0.5)

    prefix = PrefixAligner().align(previous, current)
    assert prefix.matched_pairs == []          # nothing recognised as shared

    time_aware = TimeAwareAligner().align(previous, current)
    matched = [(previous[p].text, current[c].text) for p, c in time_aware.matched_pairs]
    assert matched == [("versus", "versus"), ("pakistan", "pakistan"), ("world", "world")]


def test_time_aware_aligner_rejects_temporally_impossible_pairs():
    """The same word spoken twice, far apart, must not be conflated."""
    previous = words(("pain", 1.0, 1.4))
    current = words(("pain", 9.0, 9.4))

    alignment = TimeAwareAligner(time_tolerance=0.12).align(previous, current)
    assert alignment.matched_pairs == []


def test_time_aware_aligner_tolerates_small_drift():
    previous = words(("chest", 2.00, 2.40))
    current = words(("chest", 2.06, 2.46))
    alignment = TimeAwareAligner(time_tolerance=0.12).align(previous, current)
    assert len(alignment.matched_pairs) == 1


def test_levenshtein_handles_substitution_in_place():
    previous = sequential("i", "have", "chest", "pain")
    current = sequential("i", "have", "chest", "pains")

    alignment = LevenshteinAligner().align(previous, current)
    ops = [op.op for op in alignment.ops]
    assert ops.count("match") == 3
    assert "substitute" in ops
    assert alignment.distance == pytest.approx(1.0)


def test_levenshtein_handles_resegmentation():
    """"I have chest pain" -> "I've been having chest pain"."""
    previous = sequential("i", "have", "chest", "pain")
    current = sequential("ive", "been", "having", "chest", "pain")

    alignment = LevenshteinAligner().align(previous, current)
    matched = [current[c].text for _, c in alignment.matched_pairs]
    assert "chest" in matched and "pain" in matched


def test_empty_sequences():
    aligner = TimeAwareAligner()
    assert aligner.align([], []).ops == []
    assert len(aligner.align([], sequential("a", "b")).inserted_indices) == 2
    assert all(op.op == "delete" for op in aligner.align(sequential("a"), []).ops)


def test_is_stable_flags_pure_extension():
    previous = sequential("i", "have")
    assert TimeAwareAligner().align(previous, sequential("i", "have", "chest")).is_stable
    assert not TimeAwareAligner().align(previous, sequential("i", "had")).is_stable


def test_build_aligner_factory():
    assert build_aligner("prefix").name == "prefix"
    assert build_aligner("levenshtein").name == "levenshtein"
    assert build_aligner("time").name == "time"
    assert build_aligner("dtw").name == "dtw"
    with pytest.raises(ValueError, match="Unknown aligner"):
        build_aligner("nope")


def test_dtw_posterior_alignment_recovers_a_known_shift():
    """DTW over frame posteriors should track a known window offset."""
    import numpy as np

    rng = np.random.default_rng(0)
    base = rng.random((40, 6))
    # Window N+1 sees the same acoustics shifted by 4 frames.
    previous, current = base[:36], base[4:]

    path, cost = DTWHypothesisAligner().align_posteriors(previous, current)
    # Cost stays low but is not zero: the path's endpoints are forced to the
    # corners, so the first and last few frames must align against
    # non-corresponding content.
    assert cost < 0.1
    # The path should map previous frame i to current frame i-4.
    mid = [(i, j) for i, j in path if 10 <= i <= 25]
    offsets = [i - j for i, j in mid]
    assert max(set(offsets), key=offsets.count) == 4
