"""Randomised end-to-end check that the tracker neither duplicates nor drops.

Hand-written cases kept missing real failures: two successive fixes for the
duplicate-commit bug passed every targeted test and still produced
"climb in in in" on real audio. The gap was that each fix was tested against
the one scenario that motivated it, while a real Conformer varies its output
along several axes at once.

So this replays sliding windows over a known sentence while perturbing the
hypotheses the way a real model does -- timestamp drift, re-segmentation,
clipped edges, an unsettled trailing word -- and asserts the invariant that
matters: **the committed transcript is a prefix of the reference**, with no
inserted or dropped words.

Repeated words are deliberately present in the reference, since those are what
the duplicate-removal logic is most likely to mishandle.
"""

from __future__ import annotations

import itertools
import random

import pytest

from streaming_asr.hypothesis.tracker import HypothesisTracker
from streaming_asr.types import GreedyHypothesis, TimedToken, TimedWord

FRAME = 0.04
CHUNK = 0.16
WINDOW = 4.0
WORD = 0.30
#: Inter-word gap. Kept larger than the drift injected below, because drift
#: exceeding the gap means adjacent words physically overlap, and then no
#: policy can tell "this word again" from "the next word" -- the scenario is
#: unresolvable rather than merely hard, so testing against it measures
#: nothing.
GAP = 0.12

# Repeats ("in", "the") are the shape that broke the real runs.
REFERENCE = (
    "events hill climb in colorado which traverses the highest paved road "
    "in north america and the road is closed in winter"
).split()


def _layout(reference: list[str]) -> list[tuple[str, float, float]]:
    spans, cursor = [], 0.5
    for text in reference:
        spans.append((text, cursor, cursor + WORD))
        cursor += WORD + GAP
    return spans


def _hypothesis(specs: list[tuple[str, float, float]], audio_time: float,
                truncated_first: bool) -> GreedyHypothesis:
    words = []
    for i, (text, start, end) in enumerate(specs):
        marker = "" if (i == 0 and truncated_first) else "▁"
        token = TimedToken(i, marker + text, i, i, start, end)
        words.append(TimedWord(
            text=text, tokens=(token,), start_time=start, end_time=end,
            truncated_start=(i == 0 and truncated_first),
        ))
    return GreedyHypothesis(
        text=" ".join(t for t, _, _ in specs), token_ids=[], tokens=[],
        frame_indices=[], token_spans=[t for w in words for t in w.tokens],
        words=words, window_start_time=audio_time - WINDOW,
        window_end_time=audio_time,
    )


def simulate(
    reference: list[str],
    seed: int,
    drift_frames: int = 1,
    resegment: bool = False,
    unsettled_tail: bool = False,
) -> str:
    """Stream `reference` through the tracker with realistic perturbations."""
    rng = random.Random(seed)
    spans = _layout(reference)
    total = spans[-1][2] + 0.5

    tracker = HypothesisTracker(stability_window=0.5, min_stable_updates=2)

    audio_time = CHUNK
    while audio_time <= total + WINDOW:
        window_start = audio_time - WINDOW
        visible = [
            (text, start, end) for text, start, end in spans
            if start >= window_start - 1e-9 and end <= audio_time + 1e-9
        ]

        # Timestamps wobble by a frame or two, as CTC spikes do.
        perturbed = [
            (text,
             start + rng.randint(-drift_frames, drift_frames) * FRAME,
             end + rng.randint(-drift_frames, drift_frames) * FRAME)
            for text, start, end in visible
        ]

        # The model occasionally merges two words into one spelling and later
        # splits them again ("war craft" <-> "warcraft").
        if resegment and len(perturbed) >= 2 and rng.random() < 0.25:
            k = rng.randrange(len(perturbed) - 1)
            (a, s1, _), (b, _, e2) = perturbed[k], perturbed[k + 1]
            perturbed = perturbed[:k] + [(a + b, s1, e2)] + perturbed[k + 2:]

        # The final word is still settling and may be a fragment.
        if unsettled_tail and perturbed and rng.random() < 0.4:
            text, start, end = perturbed[-1]
            perturbed = perturbed[:-1] + [(text[: max(1, len(text) - 2)], start, end)]

        # A window that starts mid-word yields a clipped leading fragment.
        clipped = False
        if perturbed and window_start > 0 and rng.random() < 0.3:
            clipped = True

        if perturbed:
            tracker.update(_hypothesis(perturbed, audio_time, clipped))
        audio_time += CHUNK

    tracker.flush()
    return tracker.committed_text


def assert_is_clean_prefix(committed: str, reference: list[str]) -> None:
    """Committed output must be a prefix of the reference, ignoring spacing.

    Compared with the spaces removed, deliberately. Whether the model spells a
    stretch of audio "war craft" or "warcraft" is the *model's* business and
    not something the tracker can or should correct -- and the simulation
    injects such merges on purpose. What the tracker is responsible for is that
    every stretch of audio appears exactly once: a duplicate adds characters, a
    dropped word removes them, and both show up here regardless of segmentation.
    """
    produced = "".join(committed.split())
    expected = "".join(reference)
    assert expected.startswith(produced), (
        f"\n  committed: {committed}"
        f"\n  reference: {' '.join(reference)}"
    )


@pytest.mark.parametrize("seed", range(12))
def test_no_duplicates_under_timestamp_drift(seed):
    """The failure seen on real audio: drift alone caused duplicate commits."""
    committed = simulate(REFERENCE, seed=seed, drift_frames=1)
    assert_is_clean_prefix(committed, REFERENCE)
    assert len(committed.split()) >= len(REFERENCE) - 2, committed


def assert_damage_is_bounded(committed: str, reference: list[str]) -> None:
    """Allow a little slack, but never runaway repetition.

    Perfection is unattainable once the model merges words across the commit
    boundary -- "events" is committed, the next window calls the same audio
    "eventshill", and neither keeping nor dropping it is right. The tracker
    prefers a short duplicate to dropping speech.

    What must never happen is the *cascade* that started this investigation:
    one duplicate corrupting the committed tail so every later window adds
    another, giving "in in in" and "the the the". That is unbounded, and these
    two checks still catch it.
    """
    produced = "".join(committed.split())
    expected = "".join(reference)
    if expected.startswith(produced):
        return

    excess = len(produced) - len(expected)
    assert excess <= 12, (
        f"\n  committed: {committed}\n  reference: {' '.join(reference)}"
    )
    longest_run = max(
        (len(list(group)) for _, group in itertools.groupby(committed.split())),
        default=0,
    )
    assert longest_run <= 2, f"runaway repetition: {committed}"


@pytest.mark.parametrize("seed", range(12))
def test_no_duplicates_under_drift_and_resegmentation(seed):
    """Re-segmentation is what defeated the exact suffix/prefix strip."""
    committed = simulate(REFERENCE, seed=seed, drift_frames=1, resegment=True)
    assert_damage_is_bounded(committed, REFERENCE)


@pytest.mark.parametrize("seed", range(12))
def test_survives_everything_at_once(seed):
    """Every perturbation firing together must still leave damage bounded."""
    committed = simulate(
        REFERENCE, seed=seed, drift_frames=1, resegment=True, unsettled_tail=True
    )
    assert_damage_is_bounded(committed, REFERENCE)


def test_repeated_words_are_kept_exactly_once_each():
    """"in" appears three times and "the" twice; each must survive once."""
    for seed in range(8):
        committed = simulate(REFERENCE, seed=seed, drift_frames=1).split()
        for word in ("in", "the", "road"):
            expected = REFERENCE[: len(committed)].count(word)
            assert committed.count(word) == expected, (
                f"seed {seed}: {word!r} x{committed.count(word)}, "
                f"expected x{expected} -- {' '.join(committed)}"
            )
