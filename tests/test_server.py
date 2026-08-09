"""Service layer: model residency, endpoints, and the streaming protocol.

Runs against the synthetic fixture with a real ONNX session, exercising the
same code path a container would.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from streaming_asr.audio.wav_source import load_wav  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MODEL = FIXTURES / "synthetic_model.onnx"
AUDIO = FIXTURES / "synthetic.wav"
VOCAB = FIXTURES / "vocabulary.txt"

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and AUDIO.exists() and VOCAB.exists()),
    reason="synthetic fixture not built; run tools/build_synthetic_fixture.py",
)


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    import os

    os.environ.update({
        "ASR_MODEL_PATH": str(MODEL),
        "ASR_VOCAB_PATH": str(VOCAB),
        "ASR_DEVICE": "cpu",
        "ASR_BEAM_BACKEND": "pure_python",
        "ASR_BEAM_SIZE": "5",          # keep the LM-free beam quick in tests
        "ASR_MAX_CONCURRENT_STREAMS": "2",
    })
    from streaming_asr.server.app import app

    with TestClient(app) as test_client:
        yield test_client


# ---- introspection -------------------------------------------------------


def test_health_reports_what_actually_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["ready"] is True
    assert body["stateless_graph"] is True
    assert body["subsampling_factor"] == 4
    assert body["vocab_size"] == 129
    # Reports reality, not the request: a silent GPU->CPU fallback is visible.
    assert "CPUExecutionProvider" in body["providers"]
    assert body["max_concurrent_streams"] == 2


def test_info_exposes_config_and_graph(client):
    body = client.get("/info").json()
    assert body["graph"]["stateless"] is True
    assert "audio_signal" in body["graph"]["inputs"][0]
    assert body["config"]["chunk_duration"] == pytest.approx(0.16)
    # The vocabulary is summarised rather than dumped in full.
    assert "tokens>" in body["config"]["vocabulary"]


# ---- batch ---------------------------------------------------------------


def test_transcribe_returns_the_expected_text(client):
    expected = (FIXTURES / "transcript.txt").read_text(encoding="utf-8").strip()
    with open(AUDIO, "rb") as fh:
        response = client.post("/transcribe", files={"file": ("a.wav", fh, "audio/wav")})

    assert response.status_code == 200
    body = response.json()
    assert body["text"].strip() == expected
    assert body["decoder"] == "pure_python"
    assert body["used_lm"] is False
    assert body["metrics"]["model_calls"] > 0


def test_transcribe_can_return_the_partial_trail(client):
    with open(AUDIO, "rb") as fh:
        response = client.post(
            "/transcribe?stream_events=true",
            files={"file": ("a.wav", fh, "audio/wav")},
        )
    partials = response.json()["partials"]
    assert len(partials) > 10
    # Committed text only ever grows.
    committed = [p["committed"] for p in partials]
    for earlier, later in zip(committed, committed[1:]):
        assert later.startswith(earlier)


def test_non_audio_upload_returns_415_not_500(client):
    """A client error must not present as a server error.

    A 500 tells the caller "the service is broken" and gives them nothing to
    act on; 415 with the supported formats tells them what to send instead.
    """
    response = client.post(
        "/transcribe",
        files={"file": ("notes.txt", b"this is not audio\n" * 20, "text/plain")},
    )
    assert response.status_code == 415
    assert "decode" in response.json()["detail"].lower()


def test_empty_upload_is_rejected_clearly(client):
    response = client.post("/transcribe", files={"file": ("a.wav", b"", "audio/wav")})
    assert response.status_code == 415
    assert "empty" in response.json()["detail"]


def test_flac_upload_is_accepted(client):
    """Uploads are not restricted to WAV."""
    import io

    import soundfile as sf

    from streaming_asr.audio.wav_source import load_wav

    audio = load_wav(AUDIO, 16000)
    buffer = io.BytesIO()
    sf.write(buffer, audio, 16000, format="FLAC")

    response = client.post(
        "/transcribe", files={"file": ("a.flac", buffer.getvalue(), "audio/flac")}
    )
    assert response.status_code == 200
    expected = (FIXTURES / "transcript.txt").read_text(encoding="utf-8").strip()
    assert response.json()["text"].strip() == expected


def test_info_advertises_decodable_formats(client):
    formats = client.get("/info").json()["audio_formats"]
    assert "WAV" in formats["libsndfile_formats"]
    assert isinstance(formats["ffmpeg_available"], bool)


def test_model_is_loaded_once_across_requests(client):
    """The point of a service: no reload per request."""
    before = client.get("/health").json()
    with open(AUDIO, "rb") as fh:
        client.post("/transcribe", files={"file": ("a.wav", fh, "audio/wav")})
    after = client.get("/health").json()

    assert after["load_seconds"] == before["load_seconds"]     # same pool
    assert after["total_model_calls"] > before["total_model_calls"]
    assert after["active_streams"] == 0                        # slot released


# ---- streaming -----------------------------------------------------------


def _pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def test_websocket_streams_partials_then_a_final(client):
    audio = load_wav(AUDIO, 16000)
    expected = (FIXTURES / "transcript.txt").read_text(encoding="utf-8").strip()

    with client.websocket_connect("/ws/transcribe") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["sample_rate"] == 16000

        ws.send_json({"type": "config", "format": "int16"})
        # Deliberately not chunk-aligned: the server must re-block.
        frame = 4000
        for start in range(0, len(audio), frame):
            ws.send_bytes(_pcm16(audio[start:start + frame]))
        ws.send_json({"type": "end"})

        partials, segments, final = [], [], None
        while final is None:
            event = ws.receive_json()
            if event["type"] == "partial":
                partials.append(event)
            elif event["type"] == "segment":
                segments.append(event)
            elif event["type"] == "final" and event.get("end_of_stream"):
                final = event
            elif event["type"] == "error":
                pytest.fail(f"server error: {event['detail']}")

    assert len(partials) > 10
    # Segments are the authoritative units, published before the turn ends.
    assert len(segments) >= 1
    assert segments[0]["text"].strip() == expected
    # 'transcript' is the whole session, whatever the turn boundaries were.
    assert final["transcript"].strip() == expected


def test_websocket_accepts_float32_frames(client):
    audio = load_wav(AUDIO, 16000)[:32000]
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        ws.send_bytes(audio.astype("<f4").tobytes())
        ws.send_json({"type": "end"})

        final = None
        while final is None:
            event = ws.receive_json()
            if event["type"] == "final" and event.get("end_of_stream"):
                final = event
    assert "transcript" in final


def test_websocket_reset_starts_a_new_utterance(client):
    audio = load_wav(AUDIO, 16000)
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        ws.send_json({"type": "config", "format": "int16"})
        ws.send_bytes(_pcm16(audio[:48000]))
        # Drain whatever partials arrived before resetting.
        ws.send_json({"type": "reset"})

        event = ws.receive_json()
        while event["type"] != "ready":
            event = ws.receive_json()
        assert event["reset"] is True


def test_capacity_is_enforced_rather_than_oversubscribed(client):
    """Past the limit, refuse -- do not degrade every existing stream.

    Each stream reprocesses every sample buffer/chunk times, so admitting an
    extra one does not slow a single caller down politely; it pushes all of
    them past real time together.
    """
    with client.websocket_connect("/ws/transcribe") as first:
        assert first.receive_json()["type"] == "ready"
        with client.websocket_connect("/ws/transcribe") as second:
            assert second.receive_json()["type"] == "ready"

            # The pool was configured with max_concurrent_streams=2.
            with client.websocket_connect("/ws/transcribe") as third:
                event = third.receive_json()
                assert event["type"] == "error"
                assert "capacity" in event["detail"]


def test_capacity_slots_are_returned_after_a_stream_ends(client):
    for _ in range(3):
        with client.websocket_connect("/ws/transcribe") as ws:
            assert ws.receive_json()["type"] == "ready"
    assert client.get("/health").json()["active_streams"] == 0


def test_capacity_slot_is_released_when_setup_fails(client, monkeypatch):
    """A slot taken before setup must still be returned if setup throws.

    Anything between acquiring the slot and entering the try/finally is a leak
    path: pipeline construction, or the opening 'ready' send to a client that
    already went away. Each aborted connection would burn a slot permanently,
    and after max_concurrent_streams of them the service 503s every caller
    until it is restarted -- a hard outage triggered by nothing worse than
    flaky clients or an aggressive load-balancer health probe.
    """
    from streaming_asr.server import app as app_module

    pool = app_module.POOL
    assert pool.active_streams == 0

    def _explode() -> None:
        raise RuntimeError("simulated setup failure")

    monkeypatch.setattr(pool, "new_pipeline", _explode)

    for _ in range(pool.max_concurrent_streams + 2):
        try:
            with client.websocket_connect("/ws/transcribe") as ws:
                ws.receive_json()
        except Exception:
            pass  # the failure itself is expected; the leak is what matters

    monkeypatch.undo()
    assert pool.active_streams == 0, (
        "capacity slots leaked on setup failure; the service would refuse all "
        "traffic after a handful of aborted connections"
    )

    # And the service must still accept new work.
    with client.websocket_connect("/ws/transcribe") as ws:
        assert ws.receive_json()["type"] == "ready"


def test_websocket_rejects_unknown_control_messages(client):
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        ws.send_json({"type": "nonsense"})
        assert ws.receive_json()["type"] == "error"


def test_websocket_reports_invalid_json(client):
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.receive_json()
        ws.send_text("{not json")
        assert ws.receive_json()["type"] == "error"


# ---- settings ------------------------------------------------------------


def test_missing_model_path_fails_fast(monkeypatch):
    """A container must not boot into a state where every request 500s."""
    from streaming_asr.server.settings import config_from_env

    monkeypatch.delenv("ASR_MODEL_PATH", raising=False)
    monkeypatch.delenv("ASR_CONFIG", raising=False)
    with pytest.raises(ValueError, match="ASR_MODEL_PATH is required"):
        config_from_env()


def test_missing_file_names_the_path(monkeypatch):
    from streaming_asr.server.settings import config_from_env

    monkeypatch.setenv("ASR_MODEL_PATH", "/nope/model.onnx")
    with pytest.raises(FileNotFoundError, match="ASR_MODEL_PATH not found"):
        config_from_env()


def test_env_overrides_are_applied(monkeypatch):
    from streaming_asr.server.settings import config_from_env

    monkeypatch.setenv("ASR_MODEL_PATH", str(MODEL))
    monkeypatch.setenv("ASR_CHUNK_MS", "320")
    monkeypatch.setenv("ASR_CONTEXT_SEC", "2.0")
    monkeypatch.setenv("ASR_ALIGNER", "levenshtein")
    monkeypatch.setenv("ASR_FINAL_BEAM", "false")

    config = config_from_env()
    assert config.chunk_duration == pytest.approx(0.32)
    assert config.context_duration == pytest.approx(2.0)
    assert config.stability.aligner == "levenshtein"
    assert config.final_beam_decode is False
