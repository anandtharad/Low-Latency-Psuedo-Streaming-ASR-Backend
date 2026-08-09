"""Per-stream records, aggregation and result serialisation.

Latency definitions
===================
The hard part of measuring a streaming recogniser is deciding what "latency"
means. Every definition below is stated in terms of things the client can
actually observe, and where an exact audio-to-result alignment is not
obtainable from the wire protocol that is said so rather than papered over.

The anchor is :meth:`SendLog.wall_time_for_audio_time` -- the wall-clock instant
at which the client had finished transmitting the audio up to a given
stream-time ``t``. The server derives its own ``t`` from cumulative samples
received, so the two clocks refer to the same point in the audio, and the
difference between them is real elapsed time and nothing else.

``connection_latency``
    ``connect()`` returning, until the server's ``ready`` event arrives. Covers
    TCP, the WebSocket handshake, admission control and per-stream pipeline
    construction.

``first_partial_latency``
    From when the audio the partial describes had been fully sent, to when the
    partial arrived. A partial carries ``t`` = the stream time of the last
    chunk folded into it, so this is exact.

``segment_latency`` (server processing)
    From when the endpoint condition could first have been detected, to when
    the ``segment`` event arrived. The condition is ``segment_silence`` of
    silence *after* the segment's ``end``, so the anchor is
    ``end + segment_silence``. For a forced cut (``forced: true``) there is no
    silence involved and the anchor is ``end`` itself.

    **Limitation:** this needs ``segment_silence``, which is server
    configuration and not part of the event. It is read from ``/info`` at
    startup. If that read fails the value is unknown, and only
    ``segment_response_latency`` below is reported.

``segment_response_latency`` (speaker-perceived)
    From the end of the segment's speech to the event arriving -- i.e.
    ``segment_latency`` plus the silence the server deliberately waits out.
    This is what a human experiences, and it is the number to quote in a
    product conversation. It cannot go below ``segment_silence`` by design.

``final_latency``
    From sending ``{"type":"end"}`` to receiving the ``final`` event carrying
    ``end_of_stream``. Exact: both ends are client-observed.

``rtf`` (real-time factor)
    Compute seconds per audio second. Taken from the **server's own** metrics,
    which are attached to the final event -- the client cannot see inference
    time. ``wall_rtf`` (wall-clock duration over audio duration) is also
    recorded, but it is only meaningful in maximum-throughput mode; under
    real-time pacing it is ~1.0 by construction and measures nothing.

None of these names mention CTC, beam search or any decoder concept. A
different model family behind the same protocol is scored by the same code.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

#: Percentiles reported for every latency series.
PERCENTILES = (50, 90, 95, 99)


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. ``q`` in 0..100."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (q / 100.0) * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: Sequence[float]) -> dict[str, float]:
    """count / mean / min / max / percentiles, in milliseconds.

    Milliseconds throughout: every latency in this system is in the tens of
    milliseconds to low seconds range, and mixing units between the JSON and
    the console table is how a 0.5 gets read as 500.
    """
    if not values:
        return {"count": 0}
    summary: dict[str, float] = {
        "count": len(values),
        "mean_ms": round(1000 * sum(values) / len(values), 2),
        "min_ms": round(1000 * min(values), 2),
        "max_ms": round(1000 * max(values), 2),
    }
    for q in PERCENTILES:
        summary[f"p{q}_ms"] = round(1000 * percentile(values, q), 2)
    return summary


# ---------------------------------------------------------------------------
# send log
# ---------------------------------------------------------------------------


class SendLog:
    """Maps a point in the audio to the wall-clock instant it finished sending.

    Every latency that claims to start "when the audio was available" resolves
    through here. Without it the only honest measurement would be
    request-to-response round trips, which say nothing about whether a
    recogniser is keeping up.
    """

    __slots__ = ("_audio_time", "_wall_time")

    def __init__(self) -> None:
        self._audio_time: list[float] = []
        self._wall_time: list[float] = []

    def record(self, cumulative_audio_time: float, wall_time: float) -> None:
        self._audio_time.append(cumulative_audio_time)
        self._wall_time.append(wall_time)

    @property
    def audio_duration(self) -> float:
        return self._audio_time[-1] if self._audio_time else 0.0

    def wall_time_for_audio_time(self, t: float) -> Optional[float]:
        """When audio up to stream time ``t`` had been fully transmitted.

        Returns ``None`` for a ``t`` past everything sent -- which happens
        legitimately: the server zero-pads the tail on ``end``, so the final
        segment can carry an ``end`` slightly beyond the audio that existed.
        Callers treat that as "not measurable" rather than inventing a value.
        """
        if not self._audio_time:
            return None
        index = bisect.bisect_left(self._audio_time, t)
        if index >= len(self._audio_time):
            return None
        return self._wall_time[index]


# ---------------------------------------------------------------------------
# per-stream result
# ---------------------------------------------------------------------------


@dataclass
class StreamResult:
    """One simulated user's session.

    ``status`` distinguishes outcomes that a success rate must not blur:

    ``ok``          the stream ran to a final event
    ``rejected``    the server refused admission (at capacity) -- working as
                    designed, not a failure of the server, but not a success
                    for the caller either
    ``error``       protocol error, server error event, or an exception
    ``timeout``     the client gave up waiting
    """

    stream_id: int
    status: str = "ok"
    error_kind: str = ""
    error: str = ""

    audio_fixture: str = ""
    audio_duration: float = 0.0
    chunk_duration: float = 0.0
    mode: str = ""

    connection_latency: Optional[float] = None
    wall_clock_duration: float = 0.0
    send_duration: float = 0.0

    first_partial_latency: Optional[float] = None
    first_segment_latency: Optional[float] = None
    final_latency: Optional[float] = None

    partial_latencies: list[float] = field(default_factory=list)
    segment_latencies: list[float] = field(default_factory=list)
    segment_response_latencies: list[float] = field(default_factory=list)
    turn_final_latencies: list[float] = field(default_factory=list)

    #: Event type -> count. Deliberately a free-form map: a model family that
    #: emits an event this project has never seen still gets counted.
    event_counts: dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    forced_segments: int = 0

    transcript: str = ""
    #: The server's own metrics snapshot, attached to the final event.
    server_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    @property
    def server_rtf(self) -> Optional[float]:
        value = self.server_metrics.get("rtf")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def wall_rtf(self) -> Optional[float]:
        if self.audio_duration <= 0:
            return None
        return self.wall_clock_duration / self.audio_duration

    def to_row(self) -> dict[str, Any]:
        """Flat CSV row. Additive-only: new fields go on the end."""
        return {
            "stream_id": self.stream_id,
            "status": self.status,
            "error_kind": self.error_kind,
            "audio_fixture": self.audio_fixture,
            "audio_duration_s": round(self.audio_duration, 3),
            "chunk_duration_ms": round(1000 * self.chunk_duration, 1),
            "mode": self.mode,
            "wall_time_s": round(self.wall_clock_duration, 3),
            "connection_latency_ms": _ms(self.connection_latency),
            "first_partial_latency_ms": _ms(self.first_partial_latency),
            "first_segment_latency_ms": _ms(self.first_segment_latency),
            "final_latency_ms": _ms(self.final_latency),
            "segment_latency_p50_ms": _ms(_p(self.segment_latencies, 50)),
            "segment_latency_p95_ms": _ms(_p(self.segment_latencies, 95)),
            "segment_response_p95_ms": _ms(_p(self.segment_response_latencies, 95)),
            "partials": self.event_counts.get("partial", 0),
            "segments": self.event_counts.get("segment", 0),
            "finals": self.event_counts.get("final", 0),
            "forced_segments": self.forced_segments,
            "errors": self.error_count,
            "server_rtf": _round(self.server_rtf, 4),
            "wall_rtf": _round(self.wall_rtf, 4),
            "transcript_chars": len(self.transcript),
        }


def _ms(value: Optional[float]) -> Optional[float]:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) \
        else round(1000 * value, 2)


def _p(values: Sequence[float], q: float) -> Optional[float]:
    return percentile(values, q) if values else None


def _round(value: Optional[float], digits: int) -> Optional[float]:
    return None if value is None else round(value, digits)


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


@dataclass
class LevelSummary:
    """Everything measured at one concurrency level."""

    concurrency: int
    mode: str
    chunk_duration: float
    pool_size: Optional[int] = None
    decoder_backend: str = ""
    used_lm: bool = False

    streams: int = 0
    successful: int = 0
    rejected: int = 0
    failed: int = 0
    timed_out: int = 0

    total_audio_duration: float = 0.0
    total_wall_time: float = 0.0

    latencies: dict[str, dict[str, float]] = field(default_factory=dict)
    rtf: dict[str, float] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    harness: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successful / self.streams if self.streams else 0.0

    @property
    def error_rate(self) -> float:
        return (self.failed + self.timed_out) / self.streams if self.streams else 0.0

    @property
    def aggregate_audio_throughput(self) -> float:
        """Audio seconds handled per wall second across all streams.

        Above 1.0 the service is processing faster than audio arrives in
        aggregate. Under real-time pacing this is bounded by the concurrency,
        so it is a *utilisation* reading there; under maximum-throughput mode
        it is genuine capacity.
        """
        return (self.total_audio_duration / self.total_wall_time
                if self.total_wall_time > 0 else 0.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update({
            "success_rate": round(self.success_rate, 4),
            "error_rate": round(self.error_rate, 4),
            "aggregate_audio_throughput": round(self.aggregate_audio_throughput, 3),
        })
        return data


def summarize_level(
    concurrency: int,
    mode: str,
    chunk_duration: float,
    results: Sequence[StreamResult],
    wall_time: float,
    pool_size: Optional[int] = None,
    decoder_backend: str = "",
    used_lm: bool = False,
) -> LevelSummary:
    ok = [r for r in results if r.status == "ok"]
    summary = LevelSummary(
        concurrency=concurrency, mode=mode, chunk_duration=chunk_duration,
        pool_size=pool_size, decoder_backend=decoder_backend, used_lm=used_lm,
        streams=len(results),
        successful=len(ok),
        rejected=sum(1 for r in results if r.status == "rejected"),
        failed=sum(1 for r in results if r.status == "error"),
        timed_out=sum(1 for r in results if r.status == "timeout"),
        total_audio_duration=round(sum(r.audio_duration for r in ok), 2),
        total_wall_time=round(wall_time, 3),
    )

    def collect(attribute: str) -> list[float]:
        values: list[float] = []
        for result in ok:
            value = getattr(result, attribute)
            if isinstance(value, list):
                values.extend(value)
            elif value is not None:
                values.append(value)
        return values

    summary.latencies = {
        name: summarize(collect(name))
        for name in (
            "connection_latency",
            "first_partial_latency",
            "first_segment_latency",
            "final_latency",
            "partial_latencies",
            "segment_latencies",
            "segment_response_latencies",
            "turn_final_latencies",
        )
    }

    server_rtfs = [r.server_rtf for r in ok if r.server_rtf is not None]
    wall_rtfs = [r.wall_rtf for r in ok if r.wall_rtf is not None]
    summary.rtf = {
        "server_p50": round(percentile(server_rtfs, 50), 4) if server_rtfs else None,
        "server_p95": round(percentile(server_rtfs, 95), 4) if server_rtfs else None,
        "server_max": round(max(server_rtfs), 4) if server_rtfs else None,
        "wall_p50": round(percentile(wall_rtfs, 50), 4) if wall_rtfs else None,
        "samples": len(server_rtfs),
    }
    summary.errors = sorted({
        f"{r.error_kind}: {r.error}" for r in results if r.status != "ok" and r.error
    })
    return summary


# ---------------------------------------------------------------------------
# thresholds
# ---------------------------------------------------------------------------


@dataclass
class Thresholds:
    """Acceptability limits, supplied by whoever is sizing the deployment.

    Deliberately all-optional and with no defaults baked in. What counts as
    acceptable p95 latency is a product decision that depends on the
    conversation being built, not something a benchmark gets to assert. With
    none supplied the sweep still reports every measurement and simply declines
    to name a capacity limit.
    """

    max_p95_ms: Optional[float] = None
    max_rtf: Optional[float] = None
    min_success_rate: Optional[float] = None
    max_error_rate: Optional[float] = None
    #: Which latency series ``max_p95_ms`` applies to.
    latency_metric: str = "segment_response_latencies"

    @property
    def configured(self) -> bool:
        return any(v is not None for v in (
            self.max_p95_ms, self.max_rtf, self.min_success_rate, self.max_error_rate
        ))

    def evaluate(self, summary: LevelSummary) -> list[str]:
        """Reasons this level is unacceptable. Empty means it passed."""
        breaches: list[str] = []
        if self.min_success_rate is not None and \
                summary.success_rate < self.min_success_rate:
            breaches.append(
                f"success rate {summary.success_rate:.1%} < "
                f"{self.min_success_rate:.1%}")
        if self.max_error_rate is not None and summary.error_rate > self.max_error_rate:
            breaches.append(
                f"error rate {summary.error_rate:.1%} > {self.max_error_rate:.1%}")
        if self.max_p95_ms is not None:
            series = summary.latencies.get(self.latency_metric, {})
            observed = series.get("p95_ms")
            if observed is not None and observed > self.max_p95_ms:
                breaches.append(
                    f"{self.latency_metric} p95 {observed:.0f} ms > "
                    f"{self.max_p95_ms:.0f} ms")
        if self.max_rtf is not None:
            observed = summary.rtf.get("server_p95")
            if observed is not None and observed > self.max_rtf:
                breaches.append(f"server RTF p95 {observed:.3f} > {self.max_rtf:.3f}")
        return breaches


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

#: Columns written for every stream, in every run. Extended by appending only,
#: so a results file from an older build still parses.
CSV_COLUMNS = [
    "run_id", "timestamp", "concurrency", "pool_size", "mode", "chunk_duration_ms",
    "audio_fixture", "model_family", "decoder_backend", "used_lm",
    "stream_id", "status", "error_kind", "audio_duration_s", "wall_time_s",
    "connection_latency_ms", "first_partial_latency_ms", "first_segment_latency_ms",
    "final_latency_ms", "segment_latency_p50_ms", "segment_latency_p95_ms",
    "segment_response_p95_ms", "partials", "segments", "finals", "forced_segments",
    "errors", "server_rtf", "wall_rtf", "transcript_chars",
]


def write_results(
    directory: str | Path,
    run_id: str,
    metadata: dict[str, Any],
    levels: Iterable[tuple[LevelSummary, Sequence[StreamResult]]],
) -> tuple[Path, Path]:
    """Write ``load_test_<run_id>.json`` and ``.csv``. Returns both paths.

    JSON holds everything including metadata and nested summaries; CSV holds one
    row per stream for loading into anything that reads tables. Both are written
    because they answer different questions and neither subsumes the other.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    levels = list(levels)

    json_path = directory / f"load_test_{run_id}.json"
    json_path.write_text(json.dumps({
        "run_id": run_id,
        "metadata": metadata,
        "levels": [
            {"summary": summary.to_dict(),
             "streams": [asdict(stream) for stream in streams]}
            for summary, streams in levels
        ],
    }, indent=2, default=str), encoding="utf-8")

    csv_path = directory / f"load_test_{run_id}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for summary, streams in levels:
            shared = {
                "run_id": run_id,
                "timestamp": metadata.get("timestamp", ""),
                "concurrency": summary.concurrency,
                "pool_size": summary.pool_size,
                "mode": summary.mode,
                "model_family": metadata.get("model", {}).get("family", ""),
                "decoder_backend": summary.decoder_backend,
                "used_lm": summary.used_lm,
            }
            for stream in streams:
                writer.writerow({**shared, **stream.to_row()})

    return json_path, csv_path
