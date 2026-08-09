"""Does the load harness measure what it claims to?

These run against :mod:`tests.load.fake_server` -- a scripted server with **no
model**. That is the entire point: a fake can be told to take exactly 400 ms
over a segment, so the latency the client reports has a known correct answer.
Against a real recogniser every number is whatever it happens to be, which
tests the recogniser and not the ruler.

**Nothing here is an ASR performance result.** The benchmark lives in
``load_test.py`` / ``run_load_sweep.py`` and needs a real service.
"""

from __future__ import annotations

import asyncio
import csv
import json
import time

import pytest

from tests.load.fake_server import silence
from tests.load.load_test import render_table, run_level
from tests.load.metrics import (
    LevelSummary,
    SendLog,
    StreamResult,
    Thresholds,
    percentile,
    summarize,
    summarize_level,
    write_results,
)
from tests.load.monitors import ResourceSampler, render_unavailable
from tests.load.websocket_client import ClientConfig, describe_server, run_stream


def _config(server, **overrides) -> ClientConfig:
    defaults = dict(url=server.url, real_time=False, idle_timeout=15.0,
                    segment_silence=server.config.segment_silence)
    defaults.update(overrides)
    return ClientConfig(**defaults)


# ---------------------------------------------------------------------------
# pure measurement units
# ---------------------------------------------------------------------------


def test_percentiles_interpolate_rather_than_round_to_a_sample():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 50) == 2.0
    assert percentile(values, 0) == 0.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 75) == 3.0
    # A p95 that silently collapsed onto the max would hide the tail.
    assert 3.0 < percentile(values, 95) < 4.0


def test_summarize_reports_milliseconds_and_an_exact_count():
    summary = summarize([0.010, 0.020, 0.030])
    assert summary["count"] == 3
    assert summary["mean_ms"] == pytest.approx(20.0)
    assert summary["max_ms"] == pytest.approx(30.0)
    assert summarize([]) == {"count": 0}


def test_send_log_maps_a_point_in_the_audio_to_when_it_was_sent():
    """Every latency in this package resolves through this mapping."""
    log = SendLog()
    log.record(0.16, 100.0)
    log.record(0.32, 100.16)
    log.record(0.48, 100.32)

    # An exact boundary resolves to the send that completed it.
    assert log.wall_time_for_audio_time(0.32) == 100.16
    # A point inside a chunk resolves to the send that completed that chunk.
    assert log.wall_time_for_audio_time(0.20) == 100.16
    # Past everything sent is unmeasurable, not zero and not the last value.
    assert log.wall_time_for_audio_time(9.0) is None
    assert log.audio_duration == pytest.approx(0.48)


def test_summary_keeps_rejections_apart_from_failures():
    """A refused stream is not a crashed one, and must not be scored as either."""
    results = [
        StreamResult(stream_id=0, status="ok", audio_duration=2.0),
        StreamResult(stream_id=1, status="rejected"),
        StreamResult(stream_id=2, status="error", error="boom"),
        StreamResult(stream_id=3, status="timeout"),
    ]
    summary = summarize_level(4, "realtime", 0.16, results, wall_time=2.0)

    assert (summary.successful, summary.rejected, summary.failed,
            summary.timed_out) == (1, 1, 1, 1)
    assert summary.success_rate == 0.25
    # Rejections are the service working as designed, so they are not errors.
    assert summary.error_rate == 0.5


def test_thresholds_are_silent_until_someone_sets_them():
    summary = LevelSummary(concurrency=8, mode="realtime", chunk_duration=0.16)
    summary.streams = summary.successful = 8
    summary.latencies = {"segment_response_latencies": {"count": 8, "p95_ms": 900.0}}
    summary.rtf = {"server_p95": 0.4}

    assert Thresholds().configured is False
    assert Thresholds().evaluate(summary) == []

    breaches = Thresholds(max_p95_ms=500.0, max_rtf=0.2,
                          min_success_rate=1.0).evaluate(summary)
    assert len(breaches) == 2
    assert any("p95" in b for b in breaches)
    assert any("RTF" in b for b in breaches)


def test_results_are_written_as_both_json_and_csv(tmp_path):
    results = [StreamResult(stream_id=0, status="ok", audio_duration=2.0,
                            final_latency=0.12,
                            event_counts={"partial": 12, "segment": 3, "final": 1})]
    summary = summarize_level(1, "realtime", 0.16, results, wall_time=2.1)
    metadata = {"timestamp": "2026-01-01T00:00:00+00:00",
                "model": {"family": "ctc"}}

    json_path, csv_path = write_results(tmp_path, "testrun", metadata,
                                        [(summary, results)])

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["model"]["family"] == "ctc"
    assert payload["levels"][0]["summary"]["successful"] == 1

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows[0]["concurrency"] == "1"
    # The schema names an event family, never a decoder concept, so an RNNT run
    # lands in the same columns.
    assert rows[0]["model_family"] == "ctc"
    assert "ctc" not in "".join(rows[0].keys())


def test_resource_sampler_never_breaks_a_run_when_nothing_is_available():
    class Broken:
        name = "gpu"

        def start(self): raise RuntimeError("no")
        def sample(self): raise RuntimeError("no")
        def stop(self): raise RuntimeError("no")

        @property
        def unavailable_reason(self): return "simulated: no GPU on this host"

    with ResourceSampler(monitors=[Broken()], interval=0.02) as sampler:
        time.sleep(0.1)
    summary = sampler.summary()

    assert summary["samples"] == 0
    assert render_unavailable(summary) == [
        "GPU metrics unavailable: simulated: no GPU on this host"
    ]


# ---------------------------------------------------------------------------
# against the fake server
# ---------------------------------------------------------------------------


def test_a_stream_records_every_event_type_and_a_transcript(fake_server):
    result = asyncio.run(run_stream(silence(2.0), _config(fake_server),
                                    fixture_name="silence-2s"))

    assert result.status == "ok", result.error
    assert result.event_counts["partial"] > 5
    assert result.event_counts["segment"] >= 2
    assert result.event_counts["final"] == 1
    assert result.transcript
    assert result.audio_duration == pytest.approx(2.0, abs=0.05)
    assert result.connection_latency is not None
    assert result.final_latency is not None


def test_segment_latency_recovers_a_known_server_delay(fake_server):
    """The load test's central claim, checked against a known answer.

    The fake sleeps 400 ms before each segment and otherwise responds
    immediately, so the endpoint-anchored latency must come back at ~400 ms. If
    this measured the round trip, or anchored to the wrong point in the audio,
    the number would be wrong in a way no real-service run could reveal.

    Paced in real time on purpose: sent flat out, the fake's per-segment sleeps
    queue behind each other and every segment after the first inherits the
    backlog. That is a true reading of a server falling behind -- which is
    exactly what throughput mode is for -- but it is not the fixed delay this
    test is checking the arithmetic against.
    """
    fake_server.config.segment_delay = 0.4
    result = asyncio.run(run_stream(silence(3.0),
                                    _config(fake_server, real_time=True)))

    assert result.status == "ok", result.error
    assert result.segment_latencies, "no endpoint-anchored segment latencies"
    observed = percentile(result.segment_latencies, 50)
    assert observed == pytest.approx(0.4, abs=0.2), result.segment_latencies

    # The speaker-perceived figure must be larger by the silence the server
    # waits out before it will call a segment closed.
    perceived = percentile(result.segment_response_latencies, 50)
    assert perceived > observed
    assert perceived == pytest.approx(observed + fake_server.config.segment_silence,
                                      abs=0.25)


def test_real_time_pacing_takes_about_as_long_as_the_audio(fake_server):
    audio = silence(2.0)

    started = time.perf_counter()
    paced = asyncio.run(run_stream(audio, _config(fake_server, real_time=True)))
    paced_wall = time.perf_counter() - started

    started = time.perf_counter()
    flat_out = asyncio.run(run_stream(audio, _config(fake_server, real_time=False)))
    flat_wall = time.perf_counter() - started

    assert paced.status == flat_out.status == "ok"
    assert paced.mode == "realtime" and flat_out.mode == "throughput"
    assert paced_wall == pytest.approx(2.0, abs=0.8)
    # The whole reason the two modes are reported separately.
    assert flat_wall < paced_wall / 2


def test_clients_are_not_serialised_by_the_harness(fake_server):
    """Four two-second users must take about two seconds, not eight.

    A harness that awaited each client in turn would still produce plausible
    per-stream latencies while measuring nothing about contention. This is the
    guard against that, and it is why the runner uses a start barrier and
    ``gather`` rather than a loop.
    """
    async def scenario():
        return await run_level(
            fixtures=[("silence-2s", silence(2.0))],
            client_config=_config(fake_server, real_time=True),
            concurrency=4,
        )

    results, wall_time, harness = asyncio.run(scenario())

    assert len(results) == 4
    assert all(r.status == "ok" for r in results), [r.error for r in results]
    assert wall_time < 4.0, f"clients appear to have run one after another ({wall_time:.1f}s)"
    assert harness["loop_lag_samples"] > 0


def test_repeat_pools_rounds_into_one_level(fake_server):
    async def scenario():
        return await run_level(
            fixtures=[("silence-1s", silence(1.0))],
            client_config=_config(fake_server),
            concurrency=2, repeat=3,
        )

    results, _, _ = asyncio.run(scenario())
    assert len(results) == 6
    assert len({r.stream_id for r in results}) == 6


def test_the_client_follows_the_servers_advertised_chunk_size(fake_server):
    """Matching the server's blocking is what keeps the timing anchors exact."""
    fake_server.config.chunk_seconds = 0.32
    result = asyncio.run(run_stream(silence(1.6), _config(fake_server)))

    assert result.chunk_duration == pytest.approx(0.32)
    assert result.event_counts["partial"] == 5


def test_an_explicit_chunk_size_overrides_the_server(fake_server):
    result = asyncio.run(run_stream(
        silence(1.6), _config(fake_server, chunk_duration=0.08)))

    assert result.chunk_duration == pytest.approx(0.08)
    assert result.status == "ok", result.error


def test_server_introspection_captures_what_actually_loaded(fake_server):
    facts = asyncio.run(describe_server(fake_server.url))

    assert facts["health"]["ready"] is True
    assert facts["health"]["providers"] == ["FakeExecutionProvider"]
    assert facts["segmentation"]["segment_silence"] == fake_server.config.segment_silence


@pytest.fixture
def wav_fixture(tmp_path):
    """A real file on disk, because the CLIs take paths, not arrays."""
    import soundfile as sf

    path = tmp_path / "harness.wav"
    sf.write(path, silence(2.0), 16000, subtype="PCM_16")
    return path


def test_the_single_level_cli_runs_end_to_end(fake_server, wav_fixture, tmp_path):
    from tests.load import load_test

    code = load_test.main([
        "--url", fake_server.url, "--audio", str(wav_fixture),
        "--concurrency", "2", "--mode", "throughput",
        "--results-dir", str(tmp_path / "results"),
    ])

    assert code == 0
    written = list((tmp_path / "results").glob("load_test_*"))
    assert {p.suffix for p in written} == {".json", ".csv"}


def test_the_sweep_cli_walks_every_level_and_writes_one_file(
    fake_server, wav_fixture, tmp_path
):
    from tests.load import run_load_sweep

    code = run_load_sweep.main([
        "--url", fake_server.url, "--audio", str(wav_fixture),
        "--levels", "1,3", "--mode", "throughput",
        "--results-dir", str(tmp_path / "results"),
    ])

    assert code == 0
    payload = json.loads(
        next((tmp_path / "results").glob("load_test_*.json")).read_text("utf-8"))
    assert [level["summary"]["concurrency"] for level in payload["levels"]] == [1, 3]
    assert payload["metadata"]["workload"]["mode"] == "throughput"


def test_the_sweep_reports_a_capacity_limit_only_when_told_what_acceptable_means(
    fake_server, wav_fixture, tmp_path
):
    """Without thresholds it must report measurements and claim nothing."""
    from tests.load import run_load_sweep

    code = run_load_sweep.main([
        "--url", fake_server.url, "--audio", str(wav_fixture),
        "--levels", "1", "--mode", "throughput",
        "--results-dir", str(tmp_path / "r1"),
        # An impossible threshold, plus the flag that makes a breach fatal.
        "--max-p95-ms", "0.0001", "--fail-on-breach",
    ])
    assert code == 1

    code = run_load_sweep.main([
        "--url", fake_server.url, "--audio", str(wav_fixture),
        "--levels", "1", "--mode", "throughput",
        "--results-dir", str(tmp_path / "r2"),
    ])
    assert code == 0, "a run with no thresholds must never fail on latency"


def test_an_unreachable_service_exits_with_a_diagnostic_not_a_traceback(
    wav_fixture, tmp_path
):
    from tests.load import load_test
    from tests.load.server_process import free_port

    code = load_test.main([
        "--url", f"http://127.0.0.1:{free_port()}", "--audio", str(wav_fixture),
        "--concurrency", "1", "--results-dir", "-",
    ])
    assert code == 2


def test_plotting_is_optional_and_never_breaks_a_run(tmp_path):
    """With matplotlib absent this must return nothing, not raise.

    Asserted as "a list of files that all exist" rather than "empty", so the
    test means the same thing on a machine that does have matplotlib.
    """
    from tests.load.plots import plot_sweep

    summary = summarize_level(
        1, "realtime", 0.16,
        [StreamResult(stream_id=0, status="ok", audio_duration=2.0,
                      segment_response_latencies=[0.6, 0.7])],
        wall_time=2.0,
    )
    written = plot_sweep([summary], tmp_path, "testrun")

    assert isinstance(written, list)
    assert all(path.exists() for path in written)


def test_the_table_renders_missing_measurements_as_dashes():
    """A level where nothing succeeded must print, not raise."""
    empty = summarize_level(32, "realtime", 0.16,
                            [StreamResult(stream_id=0, status="error")],
                            wall_time=1.0)
    table = render_table([empty])
    assert "32" in table
    assert "-" in table
