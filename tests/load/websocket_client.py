"""One simulated streaming ASR user.

Deliberately knows nothing about the model behind the socket. It speaks the
wire protocol -- binary PCM up, JSON events down -- and scores whatever comes
back as ``partial`` / ``segment`` / ``final`` / ``error``. A CTC model, an RNNT
model or a cache-aware variant are all measured by this same client, which is
the point: the load framework must outlive the current decoder.

Two structural decisions worth knowing about:

**Send and receive run as separate tasks.** A single loop that sent a chunk and
then waited for events would couple the pacing to the server's response time,
so a slow server would make the client send slowly and the measurement would
quietly become self-fulfilling. The sender keeps to its schedule regardless of
what the receiver is doing.

**Pacing uses absolute deadlines.** ``await sleep(chunk_duration)`` after each
send accumulates the send cost into the schedule and drifts; at 160 ms chunks
over a minute that is a visible error. The sender sleeps until
``start + (n+1) * chunk_duration`` instead, so drift cannot accumulate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from tests.load.metrics import SendLog, StreamResult

logger = logging.getLogger(__name__)

#: The service's own default chunk. Used only if the server does not advertise
#: one in its ``ready`` message.
FALLBACK_CHUNK_SECONDS = 0.160

#: Event types this project currently defines. Anything else is counted and
#: passed over rather than rejected, so a future model family emitting a new
#: event does not fail the load test.
KNOWN_EVENTS = frozenset({"ready", "partial", "segment", "final", "endpoint", "error"})


class ProtocolViolation(Exception):
    """The server said something the protocol does not allow."""


class StartGate:
    """Holds connected clients until the level is ready to begin together.

    A plain ``asyncio.Barrier`` would be the obvious choice and is the wrong
    one: at high concurrency some clients are *refused admission* and return
    before ever reaching the gate, and a strict barrier would then hold the
    survivors until they timed out. This gate only counts arrivals; the runner
    decides when to release, so a client that never arrives costs nothing.
    """

    __slots__ = ("_event", "arrived")

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.arrived = 0

    async def wait(self) -> None:
        self.arrived += 1
        await self._event.wait()

    def release(self) -> None:
        self._event.set()

    @property
    def released(self) -> bool:
        return self._event.is_set()


@dataclass
class ClientConfig:
    """Everything that changes how a simulated user behaves."""

    url: str = "http://localhost:8000"
    #: ``None`` means "use whatever the server advertises in ``ready``", which
    #: also makes the client's chunk boundaries line up with the server's and
    #: keeps the audio-time to wall-time mapping exact.
    chunk_duration: Optional[float] = None
    #: True paces audio at wall-clock speed (a real caller). False sends flat
    #: out, which measures throughput and must never be called a latency result.
    real_time: bool = True
    wire_format: str = "int16"
    sample_rate: int = 16000
    #: Silence appended after the audio, so the last segment closes on a pause
    #: the way it would in conversation instead of on the ``end`` control.
    tail_silence: float = 0.0
    #: Give up if no event arrives for this long.
    idle_timeout: float = 60.0
    #: Give up on the whole stream after this long. ``None`` derives it from
    #: the audio duration.
    total_timeout: Optional[float] = None
    #: The server's ``segment_silence``, read from ``/info``. Needed to anchor
    #: ``segment_latency`` to the endpoint condition; without it only
    #: ``segment_response_latency`` is reported.
    segment_silence: Optional[float] = None
    #: Fail the stream on a protocol violation instead of merely counting it.
    strict_protocol: bool = True


def to_wire(audio: np.ndarray, wire_format: str) -> bytes:
    if wire_format == "int16":
        return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return np.ascontiguousarray(audio, dtype="<f4").tobytes()


def websocket_url(url: str) -> str:
    return (url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
            + "/ws/transcribe")


def _connect(url: str, **kwargs: Any):
    try:  # websockets >= 14
        from websockets.asyncio.client import connect
    except ImportError:  # pragma: no cover - older releases
        from websockets import connect  # type: ignore[no-redef]
    return connect(url, **kwargs)


class ProtocolChecker:
    """Validates the event stream as it arrives.

    Checks only invariants the protocol actually promises. Over-constraining it
    would make the load test fail whenever the recogniser is *allowed* to
    behave differently -- for example an RNNT model emitting far more partials,
    or a cache-aware model emitting segments at different times.
    """

    def __init__(self) -> None:
        self.violations: list[str] = []
        #: Last stream time seen, **per event kind**. Not one shared clock: a
        #: segment is retrospective -- it carries the end of the span it
        #: describes, which is behind the partials already emitted for audio
        #: arriving since. Comparing the two would flag correct behaviour.
        self._last_t: dict[str, float] = {}
        self._last_segment_end = -1.0
        self._saw_ready = False
        self._saw_end_of_stream = False

    def check(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if not isinstance(kind, str):
            self.violations.append(f"event without a string 'type': {event!r:.120}")
            return

        if kind == "ready":
            self._saw_ready = True
            return
        if not self._saw_ready and kind != "error":
            self.violations.append(f"{kind!r} arrived before 'ready'")

        t = event.get("t")
        if isinstance(t, (int, float)):
            previous = self._last_t.get(kind, -1.0)
            if t < previous - 1e-6:
                self.violations.append(
                    f"stream time went backwards on {kind!r}: {t} after {previous}")
            self._last_t[kind] = max(previous, float(t))

        if kind == "segment":
            start, end = event.get("start"), event.get("end")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                if end < start - 1e-6:
                    self.violations.append(f"segment ends before it starts: {start}-{end}")
                # Spans may overlap slightly -- the pipeline retains a little
                # audio across a cut so the next span does not start clipped --
                # but they must advance.
                if end < self._last_segment_end - 1e-6:
                    self.violations.append(
                        f"segment ends at {end}, before the previous segment's "
                        f"{self._last_segment_end}")
                self._last_segment_end = max(self._last_segment_end, float(end))
            if "transcript" not in event:
                self.violations.append("segment without a 'transcript' field")
        elif kind == "final":
            if "transcript" not in event:
                self.violations.append("final without a 'transcript' field")
            if event.get("end_of_stream"):
                if self._saw_end_of_stream:
                    self.violations.append("more than one end_of_stream final")
                self._saw_end_of_stream = True

    def finish(self, expect_final: bool) -> None:
        if expect_final and not self._saw_end_of_stream:
            self.violations.append("stream closed without an end_of_stream final")


@dataclass
class _Session:
    """Mutable per-stream state shared between the sender and receiver tasks."""

    result: StreamResult
    send_log: SendLog = field(default_factory=SendLog)
    checker: ProtocolChecker = field(default_factory=ProtocolChecker)
    end_sent_wall: Optional[float] = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    rejected: bool = False


async def run_stream(
    audio: np.ndarray,
    config: ClientConfig,
    stream_id: int = 0,
    fixture_name: str = "",
    start_barrier: Optional[StartGate] = None,
) -> StreamResult:
    """Stream one recording and return the measurements.

    Never raises for a server-side problem: a stream that is refused, dropped or
    times out comes back as a :class:`StreamResult` with a ``status`` saying so.
    One failing user must not take down a load test of thirty-two of them.
    """
    result = StreamResult(
        stream_id=stream_id,
        audio_fixture=fixture_name,
        chunk_duration=config.chunk_duration or FALLBACK_CHUNK_SECONDS,
        mode="realtime" if config.real_time else "throughput",
    )
    session = _Session(result=result)
    started = time.perf_counter()

    try:
        async with _connect(websocket_url(config.url), max_size=None,
                            open_timeout=30, close_timeout=5) as socket:
            ready = await asyncio.wait_for(socket.recv(), timeout=config.idle_timeout)
            event = _parse(ready)
            session.checker.check(event)

            if event.get("type") == "error":
                detail = str(event.get("detail", ""))
                result.status = "rejected" if "capacity" in detail.lower() else "error"
                result.error_kind = "admission" if result.status == "rejected" else "server"
                result.error = detail
                result.connection_latency = time.perf_counter() - started
                result.wall_clock_duration = result.connection_latency
                return result
            if event.get("type") != "ready":
                raise ProtocolViolation(
                    f"expected 'ready' first, got {event.get('type')!r}")

            result.connection_latency = time.perf_counter() - started
            chunk_seconds = _chunk_seconds(config, event)
            result.chunk_duration = chunk_seconds
            sample_rate = int(event.get("sample_rate") or config.sample_rate)

            await socket.send(json.dumps({"type": "config",
                                          "format": config.wire_format}))

            if start_barrier is not None:
                await start_barrier.wait()

            receiver = asyncio.create_task(_receive(socket, session, config))
            sender = asyncio.create_task(
                _send(socket, session, config, audio, sample_rate, chunk_seconds))

            budget = config.total_timeout or _default_budget(
                len(audio) / sample_rate, config)
            try:
                await asyncio.wait_for(session.done.wait(), timeout=budget)
            except asyncio.TimeoutError:
                result.status = "timeout"
                result.error_kind = "client_timeout"
                result.error = f"no end_of_stream final within {budget:.0f}s"
            finally:
                for task in (sender, receiver):
                    task.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)

    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - every failure is a datum, not a crash
        _record_failure(result, exc)

    result.wall_clock_duration = time.perf_counter() - started
    result.audio_duration = session.send_log.audio_duration

    session.checker.finish(expect_final=result.status == "ok")
    if session.checker.violations:
        result.error_count += len(session.checker.violations)
        if config.strict_protocol and result.status == "ok":
            result.status = "error"
            result.error_kind = "protocol"
            result.error = "; ".join(session.checker.violations[:3])
    return result


# ---------------------------------------------------------------------------


def _parse(message: Any) -> dict[str, Any]:
    if isinstance(message, (bytes, bytearray)):
        raise ProtocolViolation("server sent a binary frame; only JSON is defined")
    try:
        event = json.loads(message)
    except ValueError as exc:
        raise ProtocolViolation(f"server sent invalid JSON: {message!r:.120}") from exc
    if not isinstance(event, dict):
        raise ProtocolViolation(f"server sent a non-object event: {event!r:.120}")
    return event


def _chunk_seconds(config: ClientConfig, ready: dict[str, Any]) -> float:
    """Prefer the server's own chunk size.

    Matching it makes the client's chunk boundaries coincide with the server's,
    so the audio-time to wall-time mapping every latency depends on is exact
    rather than rounded to the nearest client frame.
    """
    if config.chunk_duration:
        return config.chunk_duration
    advertised = ready.get("chunk_ms")
    if isinstance(advertised, (int, float)) and advertised > 0:
        return float(advertised) / 1000.0
    return FALLBACK_CHUNK_SECONDS


def _default_budget(audio_seconds: float, config: ClientConfig) -> float:
    """How long the whole stream may take before the client gives up.

    Generous on purpose: the interesting result at high concurrency is *how
    late* things are, and a tight budget would convert that measurement into a
    pile of timeouts that say only "it was slow".
    """
    base = audio_seconds + config.tail_silence
    return (base * 3 + 120.0) if config.real_time else (base * 6 + 120.0)


def _record_failure(result: StreamResult, exc: BaseException) -> None:
    name = type(exc).__name__
    if isinstance(exc, ProtocolViolation):
        kind = "protocol"
    elif isinstance(exc, (ConnectionRefusedError, OSError)) and "Invalid" not in name:
        kind = "connect"
    elif "ConnectionClosed" in name:
        kind = "closed"
    elif isinstance(exc, asyncio.TimeoutError):
        kind = "client_timeout"
    else:
        kind = name
    result.status = "timeout" if kind == "client_timeout" else "error"
    result.error_kind = kind
    result.error = f"{name}: {exc}"[:300]
    result.error_count += 1


async def _send(
    socket: Any,
    session: _Session,
    config: ClientConfig,
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
) -> None:
    """Push audio, then close the utterance."""
    step = max(1, int(round(chunk_seconds * sample_rate)))
    if config.tail_silence > 0:
        audio = np.concatenate([
            audio, np.zeros(int(config.tail_silence * sample_rate), dtype=np.float32)
        ])

    origin = time.perf_counter()
    sent_samples = 0
    try:
        for index, start in enumerate(range(0, len(audio), step)):
            if config.real_time:
                delay = origin + (index + 1) * chunk_seconds - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            block = audio[start:start + step]
            await socket.send(to_wire(block, config.wire_format))
            sent_samples += block.size
            session.send_log.record(sent_samples / sample_rate, time.perf_counter())

        session.result.send_duration = time.perf_counter() - origin
        session.end_sent_wall = time.perf_counter()
        await socket.send(json.dumps({"type": "end"}))
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        # A drop mid-send is a real outcome; let the receiver finish reporting.
        if session.result.status == "ok":
            _record_failure(session.result, exc)
        session.done.set()


async def _receive(socket: Any, session: _Session, config: ClientConfig) -> None:
    """Consume events until the end-of-stream final, recording the timings."""
    result = session.result
    try:
        while True:
            raw = await asyncio.wait_for(socket.recv(), timeout=config.idle_timeout)
            arrived = time.perf_counter()
            event = _parse(raw)
            session.checker.check(event)

            kind = str(event.get("type", "unknown"))
            result.event_counts[kind] = result.event_counts.get(kind, 0) + 1
            if kind not in KNOWN_EVENTS:
                logger.debug("stream %d: unknown event %r (counted, ignored)",
                             result.stream_id, kind)

            if kind == "partial":
                _record_partial(session, event, arrived)
            elif kind == "segment":
                _record_segment(session, event, arrived, config)
            elif kind == "final":
                if _record_final(session, event, arrived, config):
                    session.done.set()
                    return
            elif kind == "error":
                result.error_count += 1
                detail = str(event.get("detail", ""))
                if "capacity" in detail.lower():
                    result.status, result.error_kind = "rejected", "admission"
                elif result.status == "ok":
                    result.status, result.error_kind = "error", "server"
                result.error = detail
                session.done.set()
                return
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        if result.status == "ok":
            result.status, result.error_kind = "timeout", "idle_timeout"
            result.error = f"no event for {config.idle_timeout:.0f}s"
        session.done.set()
    except Exception as exc:  # noqa: BLE001
        if result.status == "ok":
            _record_failure(result, exc)
        session.done.set()


def _record_partial(session: _Session, event: dict[str, Any], arrived: float) -> None:
    latency = _latency_from_audio_time(session, event.get("t"), arrived)
    if latency is None:
        return
    session.result.partial_latencies.append(latency)
    if session.result.first_partial_latency is None:
        session.result.first_partial_latency = latency


def _record_segment(
    session: _Session, event: dict[str, Any], arrived: float, config: ClientConfig
) -> None:
    result = session.result
    end = event.get("end")
    if not isinstance(end, (int, float)):
        return
    if event.get("forced"):
        result.forced_segments += 1

    # Speaker-perceived: from the end of the speech in this segment.
    response = _latency_from_audio_time(session, end, arrived)
    if response is not None:
        result.segment_response_latencies.append(response)
        if result.first_segment_latency is None:
            result.first_segment_latency = response

    # Server processing: from the earliest instant the endpoint condition could
    # have been detected. A forced cut has no silence in front of it.
    silence = 0.0 if event.get("forced") else (config.segment_silence or 0.0)
    if config.segment_silence is not None or event.get("forced"):
        processing = _latency_from_audio_time(session, end + silence, arrived)
        if processing is not None:
            result.segment_latencies.append(max(0.0, processing))


def _record_final(
    session: _Session, event: dict[str, Any], arrived: float, config: ClientConfig
) -> bool:
    """Record a final. Returns True if this one ends the stream."""
    result = session.result
    if event.get("end_of_stream"):
        result.transcript = str(event.get("transcript") or "")
        metrics = event.get("metrics")
        if isinstance(metrics, dict):
            result.server_metrics = metrics
        if session.end_sent_wall is not None:
            result.final_latency = arrived - session.end_sent_wall
        return True

    # A turn ended mid-stream on silence. Anchored like a segment: the turn
    # closes once turn_silence has elapsed, but the client only knows
    # segment_silence, so this is the speaker-perceived figure.
    latency = _latency_from_audio_time(session, event.get("t"), arrived)
    if latency is not None:
        result.turn_final_latencies.append(latency)
    return False


def _latency_from_audio_time(
    session: _Session, t: Any, arrived: float
) -> Optional[float]:
    if not isinstance(t, (int, float)):
        return None
    sent_at = session.send_log.wall_time_for_audio_time(float(t))
    if sent_at is None:
        return None
    return max(0.0, arrived - sent_at)


# ---------------------------------------------------------------------------
# fixtures and server introspection
# ---------------------------------------------------------------------------


def load_fixture(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Decode an audio fixture to mono float32 at the service's sample rate.

    Reuses the runtime's own torch-free decoder rather than reimplementing
    resampling, so the load test cannot accidentally feed the server audio it
    would have rejected or resampled differently.
    """
    from streaming_asr_lite.audio import decode_audio

    return decode_audio(Path(path), sample_rate)


async def describe_server(url: str, timeout: float = 15.0) -> dict[str, Any]:
    """Read ``/health`` and ``/info``. Never fatal: benchmarks still run blind.

    What comes back matters for reproducibility -- which providers *actually*
    loaded, which decoder backend is live, whether an LM is attached, and the
    segmentation thresholds the latency anchors depend on.
    """
    import httpx

    facts: dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(base_url=url.rstrip("/"), timeout=timeout) as client:
            health = await client.get("/health")
            if health.status_code == 200:
                facts["health"] = health.json()
            info = await client.get("/info")
            if info.status_code == 200:
                body = info.json()
                facts["graph"] = body.get("graph", {})
                config = body.get("config", {})
                facts["config"] = {
                    key: config.get(key)
                    for key in ("sample_rate", "chunk_duration", "pipeline", "runtime",
                                "final_beam_decode", "greedy_decode", "device")
                }
                facts["segmentation"] = config.get("segmentation", {})
    except Exception as exc:  # noqa: BLE001
        facts["error"] = f"{type(exc).__name__}: {exc}"
    return facts
