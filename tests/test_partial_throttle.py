"""Partial rate-limiting.

A partial re-decodes the whole open segment, so its cost grows with the
segment. Without a limit the loop takes longer than a chunk to process a chunk
on a long segment; on a live microphone the input queue then overruns and
segments arrive seconds late. Observed exactly that way on a real Conformer.

The limit is measured against **audio** time rather than wall clock -- see
``test_batch_mode_is_not_throttled`` for why that distinction matters.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from streaming_asr.config import SegmentationConfig


class _Throttle:
    """The shared rule, isolated from the pipelines that implement it."""

    def __init__(self, settings: SegmentationConfig) -> None:
        self.settings = settings
        self._audio_time = 0.0
        self._last_partial_audio_time = -1e9
        self._last_partial_cost = 0.0

    def _partial_is_due(self) -> bool:
        interval = max(self.settings.min_partial_interval, self._last_partial_cost)
        if interval <= 0.0:
            return True
        return (self._audio_time - self._last_partial_audio_time) >= interval


def test_cheap_decodes_are_not_throttled():
    throttle = _Throttle(SegmentationConfig())
    assert throttle._partial_is_due()

    throttle._last_partial_cost = 0.0
    throttle._last_partial_audio_time = throttle._audio_time
    assert throttle._partial_is_due()


def test_expensive_decodes_are_throttled():
    """A decode costing 0.5s must not re-run until 0.5s of audio has passed."""
    throttle = _Throttle(SegmentationConfig())
    throttle._audio_time = 10.0
    throttle._last_partial_cost = 0.5
    throttle._last_partial_audio_time = 10.0

    throttle._audio_time = 10.16                      # one chunk later
    assert not throttle._partial_is_due()

    throttle._audio_time = 10.5
    assert throttle._partial_is_due()


def test_batch_mode_is_not_throttled_by_wall_clock():
    """Why the rule uses audio time.

    A file streamed at full speed delivers chunks with no wall-clock gap. A
    wall-clock limiter would see zero elapsed time and suppress nearly every
    partial; an audio-time limiter advances with the stream regardless of how
    fast it is fed.
    """
    throttle = _Throttle(SegmentationConfig())
    throttle._last_partial_cost = 0.01                # fast decode

    emitted = 0
    for chunk in range(1, 40):                        # 39 chunks, no delay at all
        throttle._audio_time = chunk * 0.16
        if throttle._partial_is_due():
            emitted += 1
            throttle._last_partial_audio_time = throttle._audio_time

    assert emitted == 39, "batch mode should emit a partial for every chunk"


def test_explicit_minimum_interval_is_honoured():
    throttle = _Throttle(SegmentationConfig(min_partial_interval=0.5))
    throttle._audio_time = 5.0
    throttle._last_partial_cost = 0.0                 # decoding is free
    throttle._last_partial_audio_time = 5.0

    throttle._audio_time = 5.16
    assert not throttle._partial_is_due()             # still limited by config
    throttle._audio_time = 5.5
    assert throttle._partial_is_due()


def _logic(method) -> list[str]:
    """Source lines of a function body, excluding its docstring."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    body = tree.body[0].body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return [ast.dump(node) for node in body]


def test_the_limiter_has_exactly_one_implementation():
    """The rule was duplicated across the two pipelines; it is now inherited.

    The duplication existed because ``segmented.py`` could not be imported
    without torch, so the lite pipeline restated the loop rather than reusing
    it. It can now, and the lite pipeline subclasses it.

    Asserting function *identity* rather than comparing two sources catches a
    re-split at the moment it is introduced, instead of once the two copies
    have already drifted.
    """
    from streaming_asr.segmented import SegmentedASRPipeline
    from streaming_asr_lite.pipeline import LiteSegmentedPipeline

    assert LiteSegmentedPipeline._partial_is_due is SegmentedASRPipeline._partial_is_due, \
        "the lite pipeline has its own copy of the limiter again"
    assert _logic(_Throttle._partial_is_due) == \
        _logic(SegmentedASRPipeline._partial_is_due), \
        "this test's copy of the rule has drifted from the implementation"
