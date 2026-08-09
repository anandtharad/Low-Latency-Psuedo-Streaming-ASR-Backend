"""Per-window trace logging, for making hypothesis instability visible.

During development the interesting question is not "what did it transcribe" but
"how much did it change its mind, and where". A trace records, for every
window: the raw greedy hypothesis, the committed/partial split, the window and
new-audio spans, and which words were promoted.

Reading a trace tells you directly whether ``stability_window`` is set
sensibly. Words that churn several times before settling mean it is too
aggressive; words that sit in the partial region long after they have stopped
changing mean it is too conservative and latency is being wasted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, TextIO

from streaming_asr.events import ASREvent, ASREventType
from streaming_asr.types import GreedyHypothesis


class HypothesisTracer:
    """Records the evolution of hypotheses across windows.

    Args:
        path: Destination file. ``.jsonl`` produces one JSON object per window
            for programmatic analysis; anything else produces a human-readable
            log.
        include_tokens: Also record token-level timings. Verbose but necessary
            when debugging alignment rather than stability.
    """

    def __init__(self, path: str | Path, include_tokens: bool = False) -> None:
        self.path = Path(path)
        self.include_tokens = include_tokens
        self.jsonl = self.path.suffix == ".jsonl"
        self._handle: Optional[TextIO] = None
        self._records: list[dict[str, Any]] = []
        self._previous_greedy = ""

    def __enter__(self) -> "HypothesisTracer":
        self._handle = self.path.open("w", encoding="utf-8")
        if not self.jsonl:
            self._handle.write(
                "# streaming hypothesis trace\n"
                "# window=[start,end]  new=[start,end]  greedy / committed / partial\n\n"
            )
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def record(self, event: ASREvent, hypothesis: Optional[GreedyHypothesis] = None) -> None:
        """Record one partial event and the greedy hypothesis behind it."""
        if event.type is not ASREventType.PARTIAL:
            return

        greedy_text = hypothesis.text if hypothesis is not None else ""
        record: dict[str, Any] = {
            "audio_time": round(event.timestamp, 3),
            "window_start": round(event.window_start or 0.0, 3),
            "window_end": round(event.window_end or 0.0, 3),
            "new_audio_start": round(event.new_audio_start or 0.0, 3),
            "new_audio_end": round(event.new_audio_end or 0.0, 3),
            "greedy": greedy_text,
            "committed": event.committed_text,
            "partial": event.partial_text,
            "newly_committed": [w.text for w in event.newly_committed],
            "greedy_changed": greedy_text != self._previous_greedy,
        }
        if self.include_tokens and hypothesis is not None:
            record["tokens"] = [
                {
                    "token": t.token,
                    "start": round(t.start_time, 3),
                    "end": round(t.end_time, 3),
                    "p": round(t.posterior, 3),
                }
                for t in hypothesis.token_spans
            ]
        self._previous_greedy = greedy_text
        self._records.append(record)

        if self._handle is None:
            return
        if self.jsonl:
            self._handle.write(json.dumps(record) + "\n")
        else:
            self._handle.write(self._render(record))
        self._handle.flush()

    @staticmethod
    def _render(record: dict[str, Any]) -> str:
        lines = [
            f"[{record['audio_time']:6.2f}s] "
            f"window=[{record['window_start']:6.2f},{record['window_end']:6.2f}] "
            f"new=[{record['new_audio_start']:6.2f},{record['new_audio_end']:6.2f}]"
        ]
        marker = "*" if record["greedy_changed"] else " "
        lines.append(f"        {marker} greedy   : {record['greedy']!r}")
        if record["newly_committed"]:
            lines.append(f"        + COMMIT  : {' '.join(record['newly_committed'])!r}")
        lines.append(f"          committed: {record['committed']!r}")
        lines.append(f"          partial  : {record['partial']!r}")
        return "\n".join(lines) + "\n\n"

    # ---- analysis --------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Quantify instability across the session."""
        if not self._records:
            return {}

        changes = sum(1 for r in self._records if r["greedy_changed"])
        committing = sum(1 for r in self._records if r["newly_committed"])

        # How many distinct greedy readings each committed word passed through
        # before it settled: the direct measure of churn.
        first_seen: dict[str, int] = {}
        committed_at: dict[str, int] = {}
        for index, record in enumerate(self._records):
            for word in record["partial"].split():
                first_seen.setdefault(word, index)
            for word in record["newly_committed"]:
                committed_at.setdefault(word, index)

        windows_pending = [
            committed_at[w] - first_seen[w] for w in committed_at if w in first_seen
        ]
        return {
            "windows": len(self._records),
            "greedy_changes": changes,
            "greedy_change_rate": round(changes / len(self._records), 3),
            "windows_that_committed": committing,
            "mean_windows_pending_before_commit": (
                round(sum(windows_pending) / len(windows_pending), 2)
                if windows_pending else 0.0
            ),
        }
