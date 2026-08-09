"""Concurrency against the real service, with a real model behind it.

The fake server in ``test_load_harness.py`` proves the *ruler* is accurate. It
cannot prove anything about the recogniser, because there isn't one. These
tests run the actual ``streaming_asr.server.app`` in a subprocess, with a real
ONNX session, and ask the questions only a real model can answer:

* Do concurrent streams contaminate each other? One session serves every
  caller, so the failure mode is a transcript that is wrong *only* under load --
  no exception, nothing in the log. ``tests/test_engine_concurrency.py`` covers
  this at the engine; this covers it end to end through the socket, which is
  where per-stream pipeline state also lives.
* Does the admission cap actually cap?
* Does a client dying mid-stream corrupt the survivors' transcripts?

By default they use the synthetic fixture, which is small enough to load in
seconds and is the same code path a real checkpoint takes. Point them at the
real thing to make the same assertions about it::

    set ASR_LOAD_TEST_MODEL=stt_en_cconformer_ctc_large-averaged.onnx
    set ASR_LOAD_TEST_VOCAB=vocab.txt
    set ASR_LOAD_TEST_AUDIO=2086-149220-0033.wav
    python -m pytest tests/load/test_real_service.py -q

These are still **not** a performance benchmark: a handful of streams on a
loaded developer machine measures nothing about capacity. Use
``run_load_sweep.py`` for that.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import numpy as np
import pytest

from tests.load.load_test import run_level
from tests.load.metrics import summarize_level
from tests.load.websocket_client import ClientConfig, describe_server, run_stream

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures"

MODEL = Path(os.environ.get("ASR_LOAD_TEST_MODEL")
             or FIXTURES / "synthetic_model.onnx")
VOCAB = Path(os.environ.get("ASR_LOAD_TEST_VOCAB") or FIXTURES / "vocabulary.txt")
AUDIO = Path(os.environ.get("ASR_LOAD_TEST_AUDIO") or FIXTURES / "synthetic.wav")

pytestmark = [
    pytest.mark.skipif(
        not (MODEL.exists() and VOCAB.exists() and AUDIO.exists()),
        reason=("no model/vocab/audio available; build the synthetic fixture with "
                "tools/build_synthetic_fixture.py or set ASR_LOAD_TEST_MODEL"),
    ),
    pytest.mark.slow,
]

#: Admission cap the spawned service runs with. Small on purpose: the capacity
#: test needs a limit it can actually reach without opening thirty sockets.
MAX_STREAMS = 4


@pytest.fixture(scope="module")
def real_service():
    """The actual service, loaded once for the module."""
    from tests.load.server_process import ServerStartupError, spawned_server

    overrides = {
        "ASR_MODEL_PATH": str(MODEL),
        "ASR_VOCAB_PATH": str(VOCAB),
        "ASR_DEVICE": os.environ.get("ASR_LOAD_TEST_DEVICE", "cpu"),
        # Pinned to CPU unless asked otherwise, and pinned *explicitly* rather
        # than via ASR_DEVICE: provider selection is "auto" by default and will
        # take a GPU it finds. These tests assert behaviour, not speed, and a
        # test suite that quietly claims the GPU will corrupt any benchmark
        # running beside it -- which is how this line came to exist.
        "ASR_PROVIDERS": os.environ.get(
            "ASR_LOAD_TEST_PROVIDERS", "CPUExecutionProvider"),
        # Greedy only. The pure-Python beam is seconds per segment and would
        # turn every assertion below into a timeout that says nothing about
        # concurrency.
        "ASR_FINAL_BEAM": "false",
        "ASR_LOG_LEVEL": "warning",
    }
    try:
        with spawned_server(MAX_STREAMS, env_overrides=overrides,
                            startup_timeout=600.0) as url:
            yield url
    except ServerStartupError as exc:  # pragma: no cover - environment problem
        pytest.skip(f"real service would not start: {exc}")


@pytest.fixture(scope="module")
def audio() -> np.ndarray:
    from streaming_asr_lite.audio import decode_audio

    return decode_audio(AUDIO, 16000)


def _config(url: str, facts: dict, **overrides) -> ClientConfig:
    defaults = dict(
        url=url, real_time=False, idle_timeout=120.0,
        segment_silence=(facts.get("segmentation") or {}).get("segment_silence"),
    )
    defaults.update(overrides)
    return ClientConfig(**defaults)


@pytest.fixture(scope="module")
def facts(real_service) -> dict:
    return asyncio.run(describe_server(real_service))


# ---------------------------------------------------------------------------


def test_the_service_reports_what_actually_loaded(facts):
    """Recorded in every benchmark, so it had better be there."""
    health = facts["health"]
    assert health["ready"] is True
    assert health["max_concurrent_streams"] == MAX_STREAMS
    assert health["providers"], "no execution providers reported"
    assert facts["segmentation"]["segment_silence"] > 0


def test_one_stream_produces_a_transcript_and_server_metrics(real_service, facts, audio):
    result = asyncio.run(run_stream(audio, _config(real_service, facts),
                                    fixture_name=AUDIO.name))

    assert result.status == "ok", result.error
    assert result.transcript.strip(), "the real model produced no text"
    assert result.event_counts.get("segment", 0) >= 1
    # The server attaches its own metrics to the final; every RTF the load test
    # reports comes from there, so an empty dict would silently blank the
    # column rather than fail.
    assert result.server_metrics.get("rtf") is not None
    assert result.server_metrics.get("audio_duration") == pytest.approx(
        result.audio_duration, abs=0.5)


def test_concurrent_streams_do_not_contaminate_each_others_transcripts(
    real_service, facts, audio
):
    """The bug class this whole exercise exists for.

    Every client sends byte-identical audio, so every transcript must be
    byte-identical too. One ONNX session and one set of shared scratch buffers
    serve all of them; if any per-stream state leaked, the symptom would be a
    transcript that is subtly wrong under load only -- no exception, nothing in
    the log, and invisible to any single-stream test.

    Compared against a serially-produced reference rather than against each
    other, so a fault that corrupted *every* concurrent stream identically
    would still be caught.
    """
    reference = asyncio.run(run_stream(audio, _config(real_service, facts)))
    assert reference.status == "ok", reference.error
    assert reference.transcript.strip()

    async def scenario():
        return await run_level(
            fixtures=[(AUDIO.name, audio)],
            client_config=_config(real_service, facts),
            concurrency=MAX_STREAMS,
        )

    results, wall_time, _ = asyncio.run(scenario())
    ok = [r for r in results if r.status == "ok"]
    assert len(ok) == MAX_STREAMS, [r.error for r in results if r.status != "ok"]

    for result in ok:
        assert result.transcript == reference.transcript, (
            f"stream {result.stream_id} diverged under concurrency.\n"
            f"  serial     : {reference.transcript!r}\n"
            f"  concurrent : {result.transcript!r}"
        )

    summary = summarize_level(MAX_STREAMS, "throughput", 0.16, results, wall_time)
    assert summary.success_rate == 1.0
    assert summary.total_audio_duration > 0


def test_the_admission_cap_is_enforced_under_real_load(real_service, facts, audio):
    """Past the limit the service must refuse, not admit everyone and degrade."""
    async def scenario():
        return await run_level(
            fixtures=[(AUDIO.name, audio)],
            # Paced, so the streams genuinely overlap instead of each finishing
            # before the next one opens its socket.
            client_config=_config(real_service, facts, real_time=True),
            concurrency=MAX_STREAMS * 2,
        )

    results, wall_time, _ = asyncio.run(scenario())
    summary = summarize_level(MAX_STREAMS * 2, "realtime", 0.16, results, wall_time)

    assert summary.streams == MAX_STREAMS * 2
    assert summary.rejected >= 1, (
        "no stream was refused past the cap; the service admitted more than "
        "max_concurrent_streams")
    assert summary.failed == 0, [r.error for r in results if r.status == "error"]
    assert summary.successful >= 1


def test_capacity_is_returned_after_a_burst(real_service, facts, audio):
    """A refused burst must not leak slots, or the service 503s forever after."""
    async def scenario():
        return await run_level(
            fixtures=[(AUDIO.name, audio)],
            client_config=_config(real_service, facts, real_time=True),
            concurrency=MAX_STREAMS * 2,
        )

    asyncio.run(scenario())
    health = asyncio.run(describe_server(real_service))["health"]
    assert health["active_streams"] == 0, (
        "capacity slots were not returned; every later caller would be refused")

    result = asyncio.run(run_stream(audio, _config(real_service, facts)))
    assert result.status == "ok", result.error


def test_a_client_vanishing_mid_stream_leaves_the_others_correct(
    real_service, facts, audio
):
    """A dropped caller is routine. It must not corrupt anyone else's text."""
    reference = asyncio.run(run_stream(audio, _config(real_service, facts)))
    assert reference.status == "ok", reference.error

    async def scenario():
        healthy = [
            asyncio.create_task(run_stream(audio, _config(real_service, facts),
                                           stream_id=index))
            for index in range(2)
        ]
        # A third stream that opens, sends a little, and disappears without an
        # 'end' -- exactly what a closed browser tab looks like.
        abandoned = asyncio.create_task(run_stream(
            audio, _config(real_service, facts, total_timeout=0.4), stream_id=99))
        await asyncio.sleep(0.3)
        abandoned.cancel()
        results = await asyncio.gather(*healthy, return_exceptions=True)
        await asyncio.gather(abandoned, return_exceptions=True)
        return results

    results = asyncio.run(scenario())
    for result in results:
        assert not isinstance(result, BaseException), result
        assert result.status == "ok", result.error
        assert result.transcript == reference.transcript
