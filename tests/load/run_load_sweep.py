"""Sweep concurrency (and optionally pool size) and print the whole curve.

    python -m tests.load.run_load_sweep --audio sample.wav --levels 1,2,4,8,16,32

The curve is the deliverable, not a single pass/fail. A run stops early only
when explicitly told to (``--stop-on-failure``); otherwise every level is
measured even after the service has clearly given up, because *how* it degrades
past its limit -- gracefully refusing, or accepting everyone and making them all
late -- is the thing worth knowing before it happens in production.

Thresholds are optional and have no defaults. What counts as an acceptable p95
depends on the conversation being built, and a benchmark does not get to assert
that on your behalf. Supply them and the sweep names a capacity limit; leave
them out and it reports measurements only::

    python -m tests.load.run_load_sweep --audio sample.wav \\
        --levels 1,2,4,8,16,32 \\
        --max-p95-ms 1500 --max-rtf 1.0 --min-success-rate 0.99

Pool size
---------
``ASR_MAX_CONCURRENT_STREAMS`` is fixed when the service starts, so sweeping it
means restarting the service; ``--pool-sizes`` does that with ``--spawn-server``.

Be clear about what it is: an **admission cap**, not a worker pool. The ONNX
session is shared and ``Run()`` is re-entrant, so raising the cap does not add
parallelism -- it admits more concurrent callers onto the same device. Whether
that helps or simply spreads the same throughput over more unhappy users is
exactly what the sweep shows.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

if __package__ in (None, ""):  # pragma: no cover - direct-script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.load import environment  # noqa: E402
from tests.load.load_test import (  # noqa: E402
    add_common_arguments,
    backend_skip_reason,
    configure_output,
    harness_warning,
    load_fixtures,
    render_resources,
    render_table,
    run_once,
)
from tests.load.metrics import LevelSummary, StreamResult, Thresholds, write_results  # noqa: E402
from tests.load.websocket_client import describe_server  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_LEVELS = "1,2,4,8,16,32"


def parse_levels(raw: str) -> list[int]:
    """Accept any sequence, not just powers of two.

    ``--levels 12`` is a legitimate question; nothing about the service makes
    powers of two special, and hard-coding them would hide the knee of the
    curve whenever it falls between two of them.
    """
    levels = []
    for part in raw.replace(" ", "").split(","):
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise argparse.ArgumentTypeError(f"concurrency must be >= 1, got {value}")
        levels.append(value)
    if not levels:
        raise argparse.ArgumentTypeError("no concurrency levels given")
    return levels


async def sweep(
    args: argparse.Namespace,
    url: str,
    levels: Sequence[int],
    pool_size: Optional[int] = None,
    facts_out: Optional[dict[str, Any]] = None,
) -> list[tuple[LevelSummary, list[StreamResult]]]:
    """Measure every level against one service instance.

    ``facts_out`` receives the ``/health`` and ``/info`` readings. The metadata
    block has to describe the service that was *actually measured*, and with
    ``--spawn-server`` that is a process on an ephemeral port which no longer
    exists by the time the results are written -- so the facts are captured
    here, while it is up, rather than re-read afterwards.
    """
    server_facts = await describe_server(url)
    if server_facts.get("error"):
        raise ConnectionError(f"could not reach {url}: {server_facts['error']}")
    if facts_out is not None and not facts_out:
        facts_out.update(server_facts)

    skip = backend_skip_reason(server_facts, args.require_backend)
    if skip:
        print(f"SKIPPED: {skip}")
        return []

    sample_rate = int((server_facts.get("config") or {}).get("sample_rate") or 16000)
    fixtures = load_fixtures(args.audio, sample_rate)
    thresholds = build_thresholds(args)

    collected: list[tuple[LevelSummary, list[StreamResult]]] = []
    for level in levels:
        level_args = argparse.Namespace(**{**vars(args), "url": url,
                                           "concurrency": level})
        print(f"--- concurrency {level}"
              + (f", pool {pool_size}" if pool_size else "")
              + " " + "-" * 40)
        summary, results = await run_once(level_args, fixtures, server_facts)
        if pool_size is not None:
            summary.pool_size = pool_size
        collected.append((summary, results))

        breaches = thresholds.evaluate(summary)
        _print_level_line(summary, breaches)

        warning = harness_warning(summary.harness, summary.chunk_duration)
        if warning:
            print(f"  [!] {warning}")

        if breaches and args.stop_on_failure:
            print(f"  stopping: --stop-on-failure and this level breached "
                  f"{len(breaches)} threshold(s)")
            break
    return collected


def _print_level_line(summary: LevelSummary, breaches: Sequence[str]) -> None:
    response = summary.latencies.get("segment_response_latencies", {})
    print(f"  success {summary.success_rate:>6.1%}   "
          f"RTF p95 {_or_dash(summary.rtf.get('server_p95'), '{:.3f}')}   "
          f"response p95 {_or_dash(response.get('p95_ms'), '{:.0f} ms')}   "
          f"errors {summary.failed + summary.timed_out}   "
          f"rejected {summary.rejected}")
    resources = render_resources(summary.resources)
    if resources:
        print(resources)
    for breach in breaches:
        print(f"  FAIL {breach}")
    print()


def _or_dash(value: Optional[float], spec: str) -> str:
    return "-" if value is None else spec.format(value)


def build_thresholds(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        max_p95_ms=args.max_p95_ms,
        max_rtf=args.max_rtf,
        min_success_rate=args.min_success_rate,
        max_error_rate=args.max_error_rate,
        latency_metric=args.latency_metric,
    )


def render_verdict(
    summaries: Sequence[LevelSummary], thresholds: Thresholds
) -> list[str]:
    """Name the capacity limit, but only if someone defined what "acceptable" is."""
    if not thresholds.configured:
        return [
            "No thresholds supplied, so no capacity limit is claimed. Re-run with",
            "--max-p95-ms / --max-rtf / --min-success-rate to have the sweep pick one.",
        ]
    passing = [s.concurrency for s in summaries if not thresholds.evaluate(s)]
    if not passing:
        return ["No concurrency level met the thresholds, including the lowest tested."]

    highest = max(passing)
    lines = [f"Highest concurrency meeting every threshold: {highest}"]
    failed = [s.concurrency for s in summaries if thresholds.evaluate(s)]
    if failed and min(failed) < highest:
        lines.append(
            f"Note: level(s) {sorted(set(failed))} failed while {highest} passed -- "
            f"the curve is not monotonic, which usually means the run was noisy. "
            f"Re-run with --repeat 3 before trusting it."
        )
    return lines


async def _main(args: argparse.Namespace) -> int:
    levels = parse_levels(args.levels)
    thresholds = build_thresholds(args)
    all_levels: list[tuple[LevelSummary, list[StreamResult]]] = []

    print("Streaming ASR Load Sweep")
    print("=" * 78)

    server_facts: dict[str, Any] = {}

    if args.pool_sizes:
        from tests.load.server_process import spawned_server

        if not args.spawn_server:
            print("--pool-sizes needs --spawn-server: the admission cap is fixed at "
                  "startup, so measuring several values means restarting the "
                  "service.", file=sys.stderr)
            return 2
        for pool in parse_levels(args.pool_sizes):
            print(f"\n=== pool size {pool} " + "=" * 50)
            with spawned_server(pool, log_path=args.server_log) as url:
                all_levels.extend(await sweep(args, url, levels, pool_size=pool,
                                              facts_out=server_facts))
    elif args.spawn_server:
        from tests.load.server_process import spawned_server

        with spawned_server(args.max_concurrent_streams,
                            log_path=args.server_log) as url:
            all_levels.extend(await sweep(args, url, levels, facts_out=server_facts))
    else:
        all_levels.extend(await sweep(args, args.url, levels, facts_out=server_facts))

    if not all_levels:
        return 0

    summaries = [summary for summary, _ in all_levels]
    metadata = environment.collect(
        server_url=args.url,
        server_facts=server_facts,
        audio_fixture=", ".join(Path(p).name for p in args.audio),
        audio_duration=summaries[0].total_audio_duration
        / max(1, summaries[0].successful),
        chunk_duration=summaries[0].chunk_duration,
        sample_rate=16000,
        mode=args.mode,
        concurrency=levels,
        pool_size=summaries[0].pool_size,
        extra={"thresholds": vars(thresholds) if thresholds.configured else None},
    )

    print("\nStreaming ASR Load Test")
    print("=" * 78)
    print(environment.render(metadata))
    print(f"\nMode: {args.mode}")
    print(f"Chunk: {1000 * summaries[0].chunk_duration:.0f} ms\n")
    print(render_table(summaries))
    print()
    for line in render_verdict(summaries, thresholds):
        print(line)

    print("\nRTF below 1 is necessary for real-time work but not sufficient: the "
          "\nresponse latency a caller feels also contains the service's "
          "deliberate\nsilence wait before it will declare a segment closed.")

    if args.results_dir != "-":
        run = environment.run_id()
        json_path, csv_path = write_results(args.results_dir, run, metadata, all_levels)
        print(f"\nWrote {json_path}\n      {csv_path}")

        if args.plots:
            from tests.load.plots import plot_sweep

            written = plot_sweep(summaries, Path(args.results_dir), run)
            if written:
                print("      " + "\n      ".join(str(p) for p in written))
            else:
                print("      plots skipped: matplotlib is not installed "
                      "(pip install matplotlib)")

    failed_levels = [s.concurrency for s in summaries if thresholds.evaluate(s)]
    return 1 if (thresholds.configured and failed_levels and args.fail_on_breach) else 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_arguments(parser)
    parser.add_argument("--levels", default=DEFAULT_LEVELS,
                        help=f"comma-separated concurrency levels "
                             f"(default {DEFAULT_LEVELS}); any values, not just "
                             f"powers of two")
    parser.add_argument("--pool-sizes", default=None,
                        help="also sweep ASR_MAX_CONCURRENT_STREAMS; needs "
                             "--spawn-server because it is fixed at startup")
    parser.add_argument("--spawn-server", action="store_true",
                        help="start the service from the current ASR_* environment "
                             "instead of using --url")
    parser.add_argument("--max-concurrent-streams", type=int, default=32,
                        help="admission cap for a spawned server when not "
                             "sweeping pool sizes")
    parser.add_argument("--server-log", default=None,
                        help="file to capture a spawned server's output")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="stop at the first level that breaches a threshold; "
                             "off by default so the whole curve is visible")
    parser.add_argument("--fail-on-breach", action="store_true",
                        help="exit non-zero if any level breached a threshold")
    parser.add_argument("--plots", action="store_true",
                        help="also write PNG charts (needs matplotlib)")
    parser.add_argument("--max-p95-ms", type=float, default=None)
    parser.add_argument("--max-rtf", type=float, default=None)
    parser.add_argument("--min-success-rate", type=float, default=None)
    parser.add_argument("--max-error-rate", type=float, default=None)
    parser.add_argument("--latency-metric", default="segment_response_latencies",
                        choices=["segment_response_latencies", "segment_latencies",
                                 "final_latency", "first_partial_latency"],
                        help="which series --max-p95-ms applies to")
    args = parser.parse_args(argv)
    configure_output(args.verbose)
    try:
        return asyncio.run(_main(args))
    except ConnectionError as exc:
        print(str(exc), file=sys.stderr)
        print("Start the service first, e.g.:\n"
              "  python -m streaming_asr.server.app", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
