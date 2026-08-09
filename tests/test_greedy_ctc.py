"""CTC collapse, detokenisation, word grouping and timestamp recovery."""

from __future__ import annotations

import numpy as np
import pytest

from streaming_asr.decoding.greedy_ctc import GreedyCTCDecoder, ctc_collapse
from streaming_asr.types import TimedToken, group_tokens_into_words, tokens_to_text

# index: 0='A', 1='B', 2='▁hello', 3='▁wor', 4='ld', 5=blank
VOCAB = ["A", "B", "▁hello", "▁wor", "ld", "__"]
BLANK = 5


def test_collapse_removes_blanks_and_duplicate_runs():
    """blank blank A A blank B B -> A B"""
    frames = [BLANK, BLANK, 0, 0, BLANK, 1, 1]
    assert ctc_collapse(frames, BLANK) == [(0, 2, 3), (1, 5, 6)]


def test_blank_separates_repeated_labels():
    """A A blank A must yield two A tokens, not one."""
    assert ctc_collapse([0, 0, BLANK, 0], BLANK) == [(0, 0, 1), (0, 3, 3)]


def test_adjacent_repeats_collapse_into_one_token():
    assert ctc_collapse([0, 0, 0, 0], BLANK) == [(0, 0, 3)]


def test_all_blank_yields_nothing():
    assert ctc_collapse([BLANK] * 10, BLANK) == []


def test_empty_input():
    assert ctc_collapse([], BLANK) == []


def _one_hot(frame_ids: list[int], vocab_size: int = len(VOCAB)) -> np.ndarray:
    """Build log-probabilities that argmax to ``frame_ids``."""
    logits = np.full((1, len(frame_ids), vocab_size), -10.0, dtype=np.float32)
    for t, token in enumerate(frame_ids):
        logits[0, t, token] = 0.0
    return logits


def test_decode_produces_absolute_timestamps():
    """Token times must be absolute stream time, not window-relative frames."""
    decoder = GreedyCTCDecoder(VOCAB, blank_id=BLANK)
    # frames: blank, ▁hello, ▁hello, blank, ▁wor, ld
    logits = _one_hot([BLANK, 2, 2, BLANK, 3, 4])

    hypothesis = decoder.decode(
        logits,
        window_start_time=10.0,       # window begins 10 s into the stream
        window_end_time=10.24,
        ctc_frame_duration=0.04,
    )

    assert hypothesis.text == "hello world"
    assert [t.token for t in hypothesis.token_spans] == ["▁hello", "▁wor", "ld"]

    hello = hypothesis.token_spans[0]
    assert hello.start_frame == 1 and hello.end_frame == 2
    # frame 1 starts one frame-duration into a window that begins at 10.0 s
    assert hello.start_time == pytest.approx(10.04)
    assert hello.end_time == pytest.approx(10.12)


def test_decode_recovers_frame_indices():
    decoder = GreedyCTCDecoder(VOCAB, blank_id=BLANK)
    logits = _one_hot([BLANK, BLANK, 0, BLANK, 1])
    hypothesis = decoder.decode(logits, 0.0, 0.2, 0.04)
    assert hypothesis.frame_indices == [2, 4]
    assert hypothesis.token_ids == [0, 1]


def test_decode_groups_tokens_into_words():
    decoder = GreedyCTCDecoder(VOCAB, blank_id=BLANK)
    logits = _one_hot([2, BLANK, 3, 4])
    hypothesis = decoder.decode(logits, 0.0, 0.16, 0.04)

    assert [w.text for w in hypothesis.words] == ["hello", "world"]
    assert hypothesis.words[1].start_time == pytest.approx(0.08)
    assert hypothesis.words[1].end_time == pytest.approx(0.16)


def test_valid_frames_truncates():
    decoder = GreedyCTCDecoder(VOCAB, blank_id=BLANK)
    logits = _one_hot([2, 3, 4])
    hypothesis = decoder.decode(logits, 0.0, 0.12, 0.04, valid_frames=1)
    assert hypothesis.text == "hello"


def test_posteriors_are_probabilities():
    decoder = GreedyCTCDecoder(VOCAB, blank_id=BLANK)
    logits = np.log(np.array([[[0.1, 0.1, 0.7, 0.04, 0.03, 0.03]]], dtype=np.float32))
    hypothesis = decoder.decode(logits, 0.0, 0.04, 0.04)
    assert hypothesis.token_spans[0].posterior == pytest.approx(0.7, abs=1e-3)


# ---- detokenisation / word grouping --------------------------------------


def test_tokens_to_text_handles_sentencepiece_boundary():
    assert tokens_to_text(["▁ma", "a", "▁", "j", "o", "ng"]) == "maa jong"


def _token(text: str, start: float, end: float, index: int = 0) -> TimedToken:
    return TimedToken(
        token_id=index, token=text, start_frame=index, end_frame=index,
        start_time=start, end_time=end,
    )


def test_group_tokens_into_words_matches_reference_behaviour():
    tokens = [
        _token("▁ma", 0.0, 0.1, 0), _token("a", 0.1, 0.2, 1),
        _token("▁", 0.2, 0.3, 2), _token("j", 0.3, 0.4, 3),
        _token("o", 0.4, 0.5, 4), _token("ng", 0.5, 0.6, 5),
    ]
    words = group_tokens_into_words(tokens)
    assert [w.text for w in words] == ["maa", "jong"]
    assert words[0].start_time == pytest.approx(0.0)
    assert words[1].end_time == pytest.approx(0.6)


def test_first_word_flagged_when_window_starts_mid_word():
    """A window that begins mid-word yields a fragment, not a word.

    Committing such a fragment would put a truncated word into the transcript
    permanently, so the tracker needs this flag to refuse it.
    """
    tokens = [_token("ve", 0.0, 0.1, 0), _token("▁chest", 0.1, 0.3, 1)]
    words = group_tokens_into_words(tokens)
    assert words[0].truncated_start is True
    assert words[1].truncated_start is False
