"""Run N simultaneous streaming ASR users against a live service.

    python -m tests.load.load_test --audio sample.wav --concurrency 8

Two modes, and conflating them is the single easiest way to publish a wrong
number:

``--mode realtime`` (default)
    Each client sends a chunk, waits a chunk's worth of wall-clock time, sends
    the next. This is what a phone call does. Latency measured here is what a
    person experiences, and the level at which it degrades is the service's
    real capacity.

``--mode throughput``
    Each client sends as fast as the socket accepts. This measures how much
    audio the machine can chew through, which is the right question for offline
    batch work and the wrong one for a conversation. It is **not** a real-time
    capacity figure and is never labelled as one, because feeding a GPU flat out
    keeps it clocked up: this project has measured ~1.5x optimistic RTF from
    exactly that effect.

The harness measures itself as well. If the client event loop cannot keep to
its own schedule, the "latency" it reports is partly its own -- so loop lag is
sampled throughout and reported, and a run where the harness is plausibly the
bottleneck says so instead of quietly producing a bad number.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

if __package__ in (None, ""):  # pragma: no cover - direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.load import environment
from tests.load.metrics import (  # noqa: E402
    LevelSummary,
    StreamResult,
    percentile,
    summarize_level,
    write_results,
)
from tests.load.monitors import ResourceSampler, render_unavailable  # noqa: E402
from tests.load.websocket_client import (  # noqa: E402
    ClientConfig,
    StartGate,
    describe_server,
    load_fixture,
    run_stream,
)

logger = logging.getLogger(__name__)

#: How often the loop-lag probe wakes. Small enough to see stalls shorter than
#: a chunk, large enough not to become a load source itself.
LAG_PROBE_INTERVAL = 0.05

#: Longest the runner waits for every client to finish its handshake before
#: starting the level anyway. On loopback this expires never; it exists so a
#: hung connection cannot stall a whole sweep.
CONNECT_GRACE = 5.0


async def _measure_loop_lag(stop: asyncio.Event, samples: list[float]) -> None:
    """Record how late the event loop is running its own callbacks.

    A client that cannot wake on time cannot pace audio on time either. This is
    the difference between "the server got slow" and "the load generator ran out
    of CPU", and without it the two are indistinguishable in the results.
    """
    while not stop.is_set():
        expected = time.perf_counter() + LAG_PROBE_INTERVAL
        try:
            await asyncio.wait_for(stop.wait(), timeout=LAG_PROBE_INTERVAL)
            return
        except asyncio.TimeoutError:
            samples.append(max(0.0, time.perf_counter() - expected))


async def _release_when_connected(gate: StartGate, tasks: list[asyncio.Task]) -> None:
    """Start every client together, once they have all finished connecting.

    Releasing immediately would let each client start as soon as its own
    handshake completed, staggering the level by however long the last
    connection took -- which is exactly the interval that grows as the service
    gets busy, so the measurement would flatter it.

    Counts finished tasks as well as arrivals: a client refused admission
    returns without ever reaching the gate, and waiting for it would hang the
    whole level at precisely the concurrency worth measuring.
    """
    deadline = time.perf_counter() + CONNECT_GRACE
    while time.perf_counter() < deadline:
        settled = gate.arrived + sum(1 for task in tasks if task.done())
        if settled >= len(tasks):
            break
        await asyncio.sleep(0.002)
    gate.release()


async def run_level(
    fixtures: Sequence[tuple[str, np.ndarray]],
    client_config: ClientConfig,
    concurrency: int,
    repeat: int = 1,
) -> tuple[list[StreamResult], float, dict[str, Any]]:
    """Launch ``concurrency`` independent streams, ``repeat`` times over.

    Every client gets its own connection and its own timing state, and they are
    released together by a barrier so the level really is *simultaneous* -- a
    staggered start would understate contention, which is the entire point of
    the measurement.
    """
    results: list[StreamResult] = []
    lag_samples: list[float] = []
    stop_probe = asyncio.Event()
    probe = asyncio.create_task(_measure_loop_lag(stop_probe, lag_samples))

    started = time.perf_counter()
    try:
        for round_index in range(repeat):
            gate = StartGate()
            tasks = []
            for index in range(concurrency):
                name, audio = fixtures[index % len(fixtures)]
                tasks.append(asyncio.create_task(run_stream(
                    audio=audio,
                    config=client_config,
                    stream_id=round_index * concurrency + index,
                    fixture_name=name,
                    start_barrier=gate,
                )))
            await _release_when_connected(gate, tasks)

            # return_exceptions: a client that raises despite its own guards is
            # recorded, not allowed to cancel the other thirty-one.
            for index, outcome in enumerate(
                await asyncio.gather(*tasks, return_exceptions=True)
            ):
                if isinstance(outcome, StreamResult):
                    results.append(outcome)
                else:
                    logger.error("stream %d raised: %r", index, outcome)
                    results.append(StreamResult(
                        stream_id=round_index * concurrency + index,
                        status="error", error_kind="harness",
                        error=f"{type(outcome).__name__}: {outcome}",
                        error_count=1,
                    ))
    finally:
        stop_probe.set()
        await asyncio.gather(probe, return_exceptions=True)

    wall_time = time.perf_counter() - started
    harness = {
        "loop_lag_p95_ms": round(1000 * percentile(lag_samples, 95), 2)
        if lag_samples else None,
        "loop_lag_max_ms": round(1000 * max(lag_samples), 2) if lag_samples else None,
        "loop_lag_samples": len(lag_samples),
    }
    return results, wall_time, harness


def harness_warning(harness: dict[str, Any], chunk_duration: float) -> Optional[str]:
    """Flag a run whose numbers may be the load generator's own fault."""
    lag = harness.get("loop_lag_p95_ms")
    if lag is None:
        return None
    budget = 1000 * chunk_duration * 0.25
    if lag > budget:
        return (
            f"client event-loop lag p95 is {lag:.0f} ms against a "
            f"{1000 * chunk_duration:.0f} ms chunk. The load generator itself may "
            f"be the bottleneck -- treat these latencies as an upper bound and "
            f"re-run with fewer clients per process."
        )
    return None


async def run_once(
    args: argparse.Namespace,
    fixtures: Sequence[tuple[str, np.ndarray]],
    server_facts: dict[str, Any],
) -> tuple[LevelSummary, list[StreamResult]]:
    """One concurrency level, with resource sampling around it."""
    segmentation = server_facts.get("segmentation") or {}
    client_config = ClientConfig(
        url=args.url,
        chunk_duration=(args.chunk_ms / 1000.0) if args.chunk_ms else None,
        real_time=args.mode == "realtime",
        wire_format=args.format,
        tail_silence=args.tail_silence,
        idle_timeout=args.idle_timeout,
        segment_silence=segmentation.get("segment_silence"),
        strict_protocol=not args.lenient_protocol,
    )

    with ResourceSampler(interval=args.monitor_interval,
                         device_index=args.gpu_index) as sampler:
        results, wall_time, harness = await run_level(
            fixtures, client_config, args.concurrency, repeat=args.repeat
        )
        resources = sampler.summary()

    health = server_facts.get("health", {}) or {}
    summary = summarize_level(
        concurrency=args.concurrency,
        mode="realtime" if client_config.real_time else "throughput",
        chunk_duration=results[0].chunk_duration if results
        else (args.chunk_ms or 160) / 1000.0,
        results=results,
        wall_time=wall_time,
        pool_size=health.get("max_concurrent_streams"),
        decoder_backend=str(health.get("decoder_backend", "")),
        used_lm=bool(health.get("used_lm")),
    )
    summary.resources = resources
    summary.harness = harness
    return summary, results


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

_TABLE_HEADER = (
    f"{'Concurrency':<13}{'Success':>9}{'RTF p50':>10}{'RTF p95':>10}"
    f"{'Resp p50':>11}{'Resp p95':>11}{'Final p95':>11}{'Errors':>8}{'Rej':>6}"
)


def render_table(summaries: Sequence[LevelSummary]) -> str:
    """The curve, one row per concurrency level.

    ``Resp`` is speaker-perceived segment latency -- from the end of the speech
    to the text existing. It includes the service's deliberate silence wait, so
    it can never fall below ``segment_silence``; that floor is the product
    decision, everything above it is the machine.
    """
    lines = [_TABLE_HEADER, "-" * len(_TABLE_HEADER)]
    for s in summaries:
        response = s.latencies.get("segment_response_latencies", {})
        final = s.latencies.get("final_latency", {})
        lines.append(
            f"{s.concurrency:<13d}"
            f"{s.success_rate * 100:>8.0f}%"
            f"{_fmt(s.rtf.get('server_p50'), '{:.3f}'):>10}"
            f"{_fmt(s.rtf.get('server_p95'), '{:.3f}'):>10}"
            f"{_fmt(response.get('p50_ms'), '{:.0f} ms'):>11}"
            f"{_fmt(response.get('p95_ms'), '{:.0f} ms'):>11}"
            f"{_fmt(final.get('p95_ms'), '{:.0f} ms'):>11}"
            f"{s.failed + s.timed_out:>8d}"
            f"{s.rejected:>6d}"
        )
    return "\n".join(lines)


def _fmt(value: Optional[float], spec: str) -> str:
    return "-" if value is None else spec.format(value)


def render_level_detail(summary: LevelSummary) -> str:
    """Everything measured at one level, for a single-level run."""
    lines = [
        f"  streams              : {summary.streams} "
        f"({summary.successful} ok, {summary.rejected} rejected, "
        f"{summary.failed} failed, {summary.timed_out} timed out)",
        f"  success rate         : {summary.success_rate:.1%}",
        f"  total audio          : {summary.total_audio_duration:.1f}s "
        f"in {summary.total_wall_time:.1f}s wall",
        f"  aggregate throughput : {summary.aggregate_audio_throughput:.2f}x "
        f"real time",
    ]
    if summary.rtf.get("samples"):
        lines.append(
            f"  server RTF           : p50 {summary.rtf['server_p50']:.3f}  "
            f"p95 {summary.rtf['server_p95']:.3f}  max {summary.rtf['server_max']:.3f}"
        )
    lines.append("")
    lines.append(f"  {'latency':<30}{'count':>7}{'p50':>10}{'p95':>10}"
                 f"{'p99':>10}{'max':>10}")
    lines.append("  " + "-" * 77)
    for name, series in summary.latencies.items():
        if not series.get("count"):
            continue
        lines.append(
            f"  {name:<30}{series['count']:>7d}"
            f"{series['p50_ms']:>9.0f}m{series['p95_ms']:>9.0f}m"
            f"{series['p99_ms']:>9.0f}m{series['max_ms']:>9.0f}m"
        )
    return "\n".join(lines)


def render_resources(resources: dict[str, Any]) -> str:
    lines = []
    if resources.get("samples"):
        parts = []
        for key in ("cpu_percent", "gpu_percent", "gpu_memory_used_mb", "ram_used_mb"):
            if f"{key}_mean" in resources:
                parts.append(f"{key} mean {resources[f'{key}_mean']:g} / "
                             f"max {resources[f'{key}_max']:g}")
        if parts:
            lines.append("  " + "\n  ".join(parts))
    lines.extend("  " + note for note in render_unavailable(resources))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by the single-level runner and the sweep."""
    parser.add_argument("--url", default="http://localhost:8000",
                        help="base URL of a running ASR service")
    parser.add_argument("--audio", nargs="+", required=True,
                        help="one or more recordings; clients cycle through them")
    parser.add_argument("--mode", default="realtime",
                        choices=["realtime", "throughput"],
                        help="realtime paces at wall-clock speed (a caller); "
                             "throughput sends flat out (batch capacity)")
    parser.add_argument("--chunk-ms", type=float, default=None,
                        help="chunk size to send; default follows the server's "
                             "advertised chunk so timings align exactly")
    parser.add_argument("--format", default="int16", choices=["int16", "float32"])
    parser.add_argument("--tail-silence", type=float, default=0.0,
                        help="silence appended after the audio so the last "
                             "segment closes on a pause rather than on 'end'")
    parser.add_argument("--repeat", type=int, default=1,
                        help="rounds at each concurrency level; results pooled")
    parser.add_argument("--idle-timeout", type=float, default=60.0)
    parser.add_argument("--lenient-protocol", action="store_true",
                        help="count protocol violations without failing the stream")
    parser.add_argument("--results-dir", default="results",
                        help="where the JSON and CSV go; '-' writes nothing")
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--require-backend", default=None,
                        help="skip the run unless the live decoder backend "
                             "matches (e.g. flashlight)")
    parser.add_argument("--verbose", action="store_true")


def configure_output(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_fixtures(paths: Sequence[str], sample_rate: int) -> list[tuple[str, np.ndarray]]:
    fixtures = []
    for path in paths:
        audio = load_fixture(path, sample_rate)
        fixtures.append((Path(path).name, audio))
    return fixtures


def backend_skip_reason(
    server_facts: dict[str, Any], required: Optional[str]
) -> Optional[str]:
    """``None`` to proceed, otherwise why this run cannot mean anything.

    An absent optional decoder is not a failure of the load test. Reporting it
    as one would make the suite red on every machine without a KenLM build,
    which is every Windows machine this project has run on.
    """
    if not required:
        return None
    live = str((server_facts.get("health") or {}).get("decoder_backend", ""))
    if live == required:
        return None
    return (f"optional backend unavailable: requested {required!r}, "
            f"service is running {live or 'unknown'!r}")


async def _main(args: argparse.Namespace) -> int:
    server_facts = await describe_server(args.url)
    if server_facts.get("error"):
        print(f"could not reach {args.url}: {server_facts['error']}", file=sys.stderr)
        print("Start the service first, e.g.:\n"
              "  python -m streaming_asr.server.app", file=sys.stderr)
        return 2

    skip = backend_skip_reason(server_facts, args.require_backend)
    if skip:
        print(f"SKIPPED: {skip}")
        return 0

    sample_rate = int((server_facts.get("config") or {}).get("sample_rate") or 16000)
    fixtures = load_fixtures(args.audio, sample_rate)
    total_audio = sum(len(audio) for _, audio in fixtures) / sample_rate

    metadata = environment.collect(
        server_url=args.url,
        server_facts=server_facts,
        audio_fixture=", ".join(name for name, _ in fixtures),
        audio_duration=total_audio / len(fixtures),
        chunk_duration=(args.chunk_ms or 160) / 1000.0,
        sample_rate=sample_rate,
        mode=args.mode,
        concurrency=[args.concurrency],
        pool_size=None,
    )

    print("Streaming ASR Load Test")
    print("=" * 78)
    print(environment.render(metadata))
    print(f"  mode        : {args.mode}   concurrency: {args.concurrency}"
          f"   repeat: {args.repeat}")
    print()

    summary, results = await run_once(args, fixtures, server_facts)

    print(render_level_detail(summary))
    resources = render_resources(summary.resources)
    if resources:
        print("\n  resources")
        print(resources)
    warning = harness_warning(summary.harness, summary.chunk_duration)
    if warning:
        print(f"\n  [!] {warning}")
    if summary.errors:
        print("\n  errors")
        for error in summary.errors[:10]:
            print(f"    {error}")

    if args.results_dir != "-":
        run = environment.run_id()
        json_path, csv_path = write_results(
            args.results_dir, run, metadata, [(summary, results)]
        )
        print(f"\nWrote {json_path}\n      {csv_path}")

    # Exit non-zero only for a harness/transport failure, never for a slow
    # service: "too slow" is a measurement, and the sweep decides what it means.
    return 1 if summary.failed or summary.timed_out else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_arguments(parser)
    parser.add_argument("--concurrency", type=int, default=1,
                        help="simultaneous streams; any positive value")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    configure_output(args.verbose)
    return asyncio.run(_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
