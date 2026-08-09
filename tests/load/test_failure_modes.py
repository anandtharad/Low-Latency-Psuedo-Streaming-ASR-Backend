"""What the harness does when the service misbehaves.

A load test is most valuable exactly when things are going wrong, so it has to
survive every way a stream can fail without losing the run or the data. The
governing rule, checked repeatedly below: **one broken client must never take
down the other thirty-one**, and every failure must come back as a labelled
result rather than an exception.

Faults are injected into the scripted fake server. Reproducing a mid-stream
disconnect or a malformed frame against the real service would mean breaking
the real service; the point here is the client's response to them, which is
identical either way.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.load.fake_server import FakeConfig, FakeServer, silence
from tests.load.load_test import run_level
from tests.load.metrics import summarize_level
from tests.load.server_process import free_port
from tests.load.websocket_client import ClientConfig, describe_server, run_stream


def _config(server, **overrides) -> ClientConfig:
    defaults = dict(url=server.url, real_time=False, idle_timeout=3.0,
                    segment_silence=server.config.segment_silence)
    defaults.update(overrides)
    return ClientConfig(**defaults)


# ---------------------------------------------------------------------------
# the service is not there
# ---------------------------------------------------------------------------


def test_a_server_that_is_not_listening_is_a_result_not_a_crash():
    config = ClientConfig(url=f"http://127.0.0.1:{free_port()}", real_time=False)
    result = asyncio.run(run_stream(silence(1.0), config))

    assert result.status == "error"
    assert result.error_kind == "connect"
    assert result.error
    # And the run can still be summarised, which is what a sweep does next.
    summary = summarize_level(1, "throughput", 0.16, [result], wall_time=0.1)
    assert summary.success_rate == 0.0
    assert summary.failed == 1


def test_introspection_of_a_dead_server_reports_the_reason_and_returns():
    facts = asyncio.run(describe_server(f"http://127.0.0.1:{free_port()}", timeout=2.0))
    assert "error" in facts
    assert facts.get("health") is None


# ---------------------------------------------------------------------------
# admission control
# ---------------------------------------------------------------------------


def test_a_refused_connection_is_recorded_as_rejected_not_failed():
    """Capacity refusal is the service working. It must not read as an error.

    Conflating the two would make a correctly-behaving service look broken at
    exactly the concurrency where its admission cap starts doing its job, and
    would hide the far more serious case of a service that accepts everyone and
    makes them all late.
    """
    with FakeServer(FakeConfig(max_concurrent=1, segment_seconds=0.5)) as server:
        async def scenario():
            return await run_level(
                fixtures=[("silence-2s", silence(2.0))],
                client_config=_config(server, real_time=True, idle_timeout=10.0),
                concurrency=4,
            )

        results, wall_time, _ = asyncio.run(scenario())

    summary = summarize_level(4, "realtime", 0.16, results, wall_time=wall_time)
    assert summary.rejected >= 1, [r.status for r in results]
    assert summary.successful >= 1
    assert summary.failed == 0, [r.error for r in results if r.status == "error"]
    # Rejections do not count towards the error rate.
    assert summary.error_rate == 0.0
    assert all(r.error_kind == "admission" for r in results if r.status == "rejected")


def test_pool_exhaustion_still_yields_a_complete_measurement():
    """Every stream comes back, whatever the outcome, so the curve has a point."""
    with FakeServer(FakeConfig(max_concurrent=2, segment_seconds=0.5)) as server:
        async def scenario():
            return await run_level(
                fixtures=[("silence-1s", silence(1.0))],
                client_config=_config(server, real_time=True, idle_timeout=10.0),
                concurrency=8,
            )

        results, wall_time, _ = asyncio.run(scenario())

    assert len(results) == 8, "a stream went missing"
    summary = summarize_level(8, "realtime", 0.16, results, wall_time=wall_time)
    assert summary.streams == 8
    assert summary.successful + summary.rejected + summary.failed + \
        summary.timed_out == 8


# ---------------------------------------------------------------------------
# the connection breaks
# ---------------------------------------------------------------------------


def test_a_connection_dropped_mid_stream_is_labelled():
    with FakeServer(FakeConfig(fail_mode="drop", fail_after_chunks=2,
                               segment_seconds=0.5)) as server:
        result = asyncio.run(run_stream(silence(3.0), _config(server)))

    assert result.status == "error"
    assert result.error_kind in ("closed", "protocol", "ConnectionClosedError")
    # Whatever arrived before the drop is still recorded and still usable.
    assert result.event_counts.get("partial", 0) >= 1


def test_a_malformed_event_is_a_protocol_error_not_an_unhandled_exception():
    with FakeServer(FakeConfig(fail_mode="malformed", fail_after_chunks=2,
                               segment_seconds=0.5)) as server:
        result = asyncio.run(run_stream(silence(3.0), _config(server)))

    assert result.status == "error"
    assert result.error_kind == "protocol"
    assert "JSON" in result.error


def test_a_binary_frame_from_the_server_is_a_protocol_error():
    """Only JSON is defined downstream. Anything else is the server's bug."""
    with FakeServer(FakeConfig(fail_mode="binary", fail_after_chunks=2,
                               segment_seconds=0.5)) as server:
        result = asyncio.run(run_stream(silence(3.0), _config(server)))

    assert result.status == "error"
    assert result.error_kind == "protocol"


def test_a_server_error_event_is_attributed_to_the_server():
    with FakeServer(FakeConfig(fail_mode="error_event", fail_after_chunks=2,
                               segment_seconds=0.5)) as server:
        result = asyncio.run(run_stream(silence(3.0), _config(server)))

    assert result.status == "error"
    assert result.error_kind == "server"
    assert "simulated server failure" in result.error


# ---------------------------------------------------------------------------
# timeouts
# ---------------------------------------------------------------------------


def test_a_server_that_stops_responding_times_out_rather_than_hanging():
    """The idle timeout is what stops one wedged stream stalling the sweep."""
    with FakeServer(FakeConfig(fail_mode="silent", fail_after_chunks=2,
                               segment_seconds=0.5)) as server:
        result = asyncio.run(run_stream(
            silence(3.0), _config(server, idle_timeout=1.0)))

    assert result.status == "timeout"
    assert result.error_kind == "idle_timeout"
    assert result.wall_clock_duration < 15.0


def test_a_missing_final_is_a_timeout_with_the_partial_data_kept():
    with FakeServer(FakeConfig(fail_mode="no_final", segment_seconds=0.5)) as server:
        result = asyncio.run(run_stream(
            silence(2.0), _config(server, idle_timeout=1.5)))

    assert result.status == "timeout"
    assert result.event_counts.get("segment", 0) >= 1
    assert result.final_latency is None
    assert result.transcript == ""


def test_the_total_budget_bounds_a_stream_whose_events_never_stop():
    """An idle timeout alone is not enough: a chatty stuck server never idles."""
    with FakeServer(FakeConfig(fail_mode="no_final", segment_seconds=0.5)) as server:
        result = asyncio.run(run_stream(
            silence(1.0), _config(server, idle_timeout=30.0, total_timeout=2.0)))

    assert result.status == "timeout"
    assert result.wall_clock_duration < 10.0


# ---------------------------------------------------------------------------
# isolation
# ---------------------------------------------------------------------------


def test_one_failing_client_does_not_take_down_the_others():
    """The single most important property of the whole harness.

    Stream index 1 is dropped mid-stream; the other three must complete
    normally and their measurements must survive intact. A harness that
    propagated the first failure would report a total outage every time one
    connection flaked.
    """
    with FakeServer(FakeConfig(fail_mode="drop", fail_after_chunks=2,
                               fail_streams={1}, segment_seconds=0.5)) as server:
        async def scenario():
            return await run_level(
                fixtures=[("silence-2s", silence(2.0))],
                client_config=_config(server, real_time=True, idle_timeout=10.0),
                concurrency=4,
            )

        results, wall_time, _ = asyncio.run(scenario())

    summary = summarize_level(4, "realtime", 0.16, results, wall_time=wall_time)
    assert summary.streams == 4
    assert summary.failed == 1
    assert summary.successful == 3
    assert summary.success_rate == 0.75
    # The survivors are fully measured, not merely counted.
    healthy = [r for r in results if r.status == "ok"]
    assert all(r.final_latency is not None for r in healthy)
    assert all(r.event_counts.get("segment", 0) >= 1 for r in healthy)


def test_a_client_raising_despite_its_guards_is_still_a_row():
    """Belt and braces: ``run_level`` gathers with ``return_exceptions``."""
    import tests.load.load_test as load_test

    async def scenario():
        original = load_test.run_stream

        async def exploding(*args, **kwargs):
            if kwargs.get("stream_id") == 0:
                raise RuntimeError("simulated harness bug")
            return await original(*args, **kwargs)

        load_test.run_stream = exploding
        try:
            with FakeServer(FakeConfig(segment_seconds=0.5)) as server:
                return await run_level(
                    fixtures=[("silence-1s", silence(1.0))],
                    client_config=_config(server, idle_timeout=10.0),
                    concurrency=2,
                )
        finally:
            load_test.run_stream = original

    results, _, _ = asyncio.run(scenario())

    assert len(results) == 2
    harness_failures = [r for r in results if r.error_kind == "harness"]
    assert len(harness_failures) == 1
    assert "simulated harness bug" in harness_failures[0].error
    assert any(r.status == "ok" for r in results)


@pytest.mark.parametrize("concurrency", [1, 3, 7])
def test_arbitrary_concurrency_values_are_supported(concurrency):
    """Not just powers of two -- the knee of the curve rarely lands on one."""
    with FakeServer(FakeConfig(segment_seconds=0.5)) as server:
        async def scenario():
            return await run_level(
                fixtures=[("silence-1s", silence(1.0))],
                client_config=_config(server, idle_timeout=10.0),
                concurrency=concurrency,
            )

        results, _, _ = asyncio.run(scenario())

    assert len(results) == concurrency
    assert all(r.status == "ok" for r in results), [r.error for r in results]
    assert len({r.stream_id for r in results}) == concurrency
