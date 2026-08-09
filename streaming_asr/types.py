"""Shared value types.

These sit in their own module because they cross the decoding <-> hypothesis
boundary in both directions; importing them from either side would create a
cycle.

The central design commitment lives here: **every token carries an absolute
stream timestamp**, not just a frame index within its window. Frame indices are
window-relative and therefore meaningless across windows -- frame 50 of window
N and frame 50 of window N+1 describe audio 160 ms apart. Absolute times are
comparable across windows, which is what lets the hypothesis tracker recognise
"this is the same word I saw last time" without resorting to string matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

#: SentencePiece word-boundary marker used by the reference vocabulary.
WORD_BOUNDARY = "▁"  # '▁'


@dataclass(frozen=True)
class TimedToken:
    """One CTC-collapsed token with window-relative frames and absolute time."""

    token_id: int
    token: str
    start_frame: int              # CTC frame index within the emitting window
    end_frame: int                # inclusive
    start_time: float             # absolute stream time, seconds
    end_time: float               # absolute stream time, seconds
    posterior: float = 1.0        # mean frame posterior over the token's span

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TimedToken({self.token!r}, frames={self.start_frame}-{self.end_frame}, "
            f"t={self.start_time:.2f}-{self.end_time:.2f}, p={self.posterior:.2f})"
        )


@dataclass(frozen=True)
class TimedWord:
    """A whole word assembled from its constituent BPE tokens.

    Commitment happens at word granularity -- emitting half a word downstream
    is never useful -- while token- and frame-level detail is retained for
    alignment work.
    """

    text: str
    tokens: tuple[TimedToken, ...]
    start_time: float
    end_time: float
    #: True when the word's first token lacks the word-boundary marker, i.e.
    #: the window began mid-word and this word is probably truncated. Such
    #: words must never be committed from this window.
    truncated_start: bool = False

    @property
    def posterior(self) -> float:
        if not self.tokens:
            return 0.0
        return sum(t.posterior for t in self.tokens) / len(self.tokens)

    @property
    def start_frame(self) -> int:
        return self.tokens[0].start_frame if self.tokens else -1

    @property
    def end_frame(self) -> int:
        return self.tokens[-1].end_frame if self.tokens else -1

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        flag = "~" if self.truncated_start else ""
        return f"TimedWord({flag}{self.text!r}, {self.start_time:.2f}-{self.end_time:.2f})"


def tokens_to_text(tokens: Sequence[str]) -> str:
    """Detokenise SentencePiece pieces into plain text."""
    return "".join(tokens).replace(WORD_BOUNDARY, " ").strip()


def group_tokens_into_words(tokens: Sequence[TimedToken]) -> list[TimedWord]:
    """Group CTC tokens into words on the SentencePiece boundary marker.

    ``['▁ma', 'a', '▁', 'j', 'o', 'ng']`` -> ``['maa', 'jong']``.

    A word that begins with a token carrying no boundary marker started before
    this window did, so it is flagged ``truncated_start``.
    """
    words: list[TimedWord] = []
    current: list[TimedToken] = []
    current_truncated = False

    def flush() -> None:
        nonlocal current, current_truncated
        if not current:
            return
        text = "".join(t.token for t in current).replace(WORD_BOUNDARY, "")
        if text:
            words.append(
                TimedWord(
                    text=text,
                    tokens=tuple(current),
                    start_time=current[0].start_time,
                    end_time=current[-1].end_time,
                    truncated_start=current_truncated,
                )
            )
        current = []
        current_truncated = False

    for i, tok in enumerate(tokens):
        starts_word = tok.token.startswith(WORD_BOUNDARY)
        if starts_word:
            flush()
            current_truncated = False
        elif i == 0:
            # First token of the window with no boundary marker: this word ran
            # off the left edge of the window.
            current_truncated = True
        current.append(tok)

    flush()
    return words


@dataclass
class GreedyHypothesis:
    """Result of greedy CTC decoding over a single window.

    Frame-level detail is deliberately preserved (section 10 of the brief):
    once thrown away it cannot be recovered, and temporal alignment across
    overlapping windows is the whole basis of the stabilisation strategy.
    """

    text: str
    token_ids: list[int]
    tokens: list[str]
    frame_indices: list[int]           # start frame of each emitted token
    token_spans: list[TimedToken]
    words: list[TimedWord] = field(default_factory=list)

    # Provenance of the window that produced this hypothesis.
    window_start_time: float = 0.0
    window_end_time: float = 0.0
    new_audio_start_time: float = 0.0  # start of the freshly-arrived chunk
    new_audio_end_time: float = 0.0
    ctc_frame_duration: float = 0.04
    decode_time: float = 0.0
    #: Per-frame top-class posteriors, retained only when explicitly requested.
    frame_posteriors: Optional[object] = None

    @property
    def is_empty(self) -> bool:
        return not self.token_spans

    def word_texts(self) -> list[str]:
        return [w.text for w in self.words]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"GreedyHypothesis({self.text!r}, "
            f"window={self.window_start_time:.2f}-{self.window_end_time:.2f})"
        )
