"""Event types emitted by the pipeline.

The pipeline is event-driven rather than return-value-driven because a caller
needs incremental output *during* an utterance -- consuming a generator of
events is the natural shape for that, and it keeps downstream consumers from
having to poll.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from streaming_asr.types import TimedWord


class ASREventType(str, Enum):
    PARTIAL = "partial"
    #: A pause-bounded span, decoded whole and authoritative for that span.
    #: Emitted well before the turn ends, so a latency-sensitive consumer does
    #: not have to wait for the speaker to stop. Several per turn.
    SEGMENT = "segment"
    FINAL = "final"
    METRICS = "metrics"
    ENDPOINT = "endpoint"


@dataclass
class ASREvent:
    """A single observable output of the streaming pipeline.

    ``PARTIAL`` events carry the committed/partial split and are provisional.
    ``FINAL`` carries the authoritative beam+LM transcript. ``METRICS`` carries
    benchmarking data and never transcript text.
    """

    type: ASREventType
    timestamp: float                       # audio-stream time, seconds
    wall_time: float = 0.0                 # perf_counter at emission

    # --- partial ---
    committed_text: str = ""
    partial_text: str = ""
    full_hypothesis: str = ""
    newly_committed: list[TimedWord] = field(default_factory=list)
    confidence: Optional[float] = None

    # --- window provenance, for debugging hypothesis instability ---
    window_start: Optional[float] = None
    window_end: Optional[float] = None
    new_audio_start: Optional[float] = None
    new_audio_end: Optional[float] = None

    # --- final ---
    text: str = ""
    #: The streaming transcript at the moment of finalisation, kept so callers
    #: can measure how far the provisional output drifted from the truth.
    provisional_text: str = ""
    used_lm: bool = False
    #: Which decoder produced ``text``: a backend name ("flashlight",
    #: "pyctcdecode", "pure_python"), or "streaming" when final beam decoding
    #: was disabled and the provisional transcript was returned as-is.
    #: Downstream consumers should not have to infer this from ``used_lm``,
    #: which cannot distinguish "beam without an LM" from "no beam at all".
    decoder: str = ""

    # --- metrics ---
    metrics: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - logging aid
        if self.type is ASREventType.PARTIAL:
            return (
                f"[{self.timestamp:6.2f}s] committed: {self.committed_text!r}\n"
                f"           partial: {self.partial_text!r}"
            )
        if self.type is ASREventType.FINAL:
            return f"[{self.timestamp:6.2f}s] FINAL: {self.text!r}"
        if self.type is ASREventType.ENDPOINT:
            return f"[{self.timestamp:6.2f}s] END OF SPEECH"
        return f"[{self.timestamp:6.2f}s] metrics: {self.metrics}"
