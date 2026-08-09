"""A scripted stand-in for the ASR service, with no model behind it.

Exists to test the **load harness**, not the recogniser. It speaks the same wire
protocol and emits events on a schedule derived from the audio it receives, so
every latency the client computes has a known correct answer -- which is the
only way to check that the measurement code is right. A real service cannot do
that: its latencies are whatever they happen to be.

Nothing measured against this server says anything about ASR performance, and
the tests that use it say so in their names. Keeping the two apart matters:
a benchmark that silently ran against a fake would look excellent.

It also injects failures on demand. Reproducing a mid-stream disconnect or a
malformed event against a real service means breaking the real service; here it
is a flag.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
# Module level, not inside build_app(): this file uses postponed annotation
# evaluation, so FastAPI resolves the endpoint's ``WebSocket`` annotation
# against module globals. Imported inside the factory it is invisible there,
# and the route silently fails to build -- every connection then gets a bare
# HTTP 403 with nothing in the log.
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.16


@dataclass
class FakeConfig:
    """Behaviour knobs. Mutated by tests between connections."""

    sample_rate: int = SAMPLE_RATE
    chunk_seconds: float = CHUNK_SECONDS
    #: Audio seconds per emitted ``segment``.
    segment_seconds: float = 2.0
    #: Silence the fake pretends to wait out before declaring a segment closed,
    #: mirroring the real service so the client's latency anchoring is exercised.
    segment_silence: float = 0.5
    #: Artificial processing delay applied before a segment is sent.
    segment_delay: float = 0.0
    max_concurrent: int = 1000
    decoder_backend: str = "greedy"
    used_lm: bool = False

    #: "none", "reject", "drop", "malformed", "binary", "error_event", "silent",
    #: "no_final"
    fail_mode: str = "none"
    #: Chunks to accept before ``fail_mode`` takes effect.
    fail_after_chunks: int = 3
    #: Apply the failure only to streams whose index is in this set. Empty means
    #: every stream, which is what "the server is broken" looks like; a subset
    #: is how "one client fails while the others carry on" is tested.
    fail_streams: set[int] = field(default_factory=set)

    def should_fail(self, stream_index: int) -> bool:
        if self.fail_mode == "none":
            return False
        return not self.fail_streams or stream_index in self.fail_streams


def build_app(config: FakeConfig) -> FastAPI:
    app = FastAPI()
    app.state.config = config
    app.state.active = 0
    app.state.accepted = 0
    app.state.lock = threading.Lock()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        settings: FakeConfig = app.state.config
        return {
            "ready": True,
            "model_path": "<fake: no model loaded>",
            "runtime": "fake",
            "frontend_device": "none",
            "providers": ["FakeExecutionProvider"],
            "gpu": None,
            "zero_copy": False,
            "decoder_backend": settings.decoder_backend,
            "used_lm": settings.used_lm,
            "subsampling_factor": 4,
            "vocab_size": 129,
            "stateless_graph": True,
            "load_seconds": 0.0,
            "active_streams": app.state.active,
            "max_concurrent_streams": settings.max_concurrent,
            "total_model_calls": 0,
        }

    @app.get("/info")
    async def info() -> dict[str, Any]:
        settings: FakeConfig = app.state.config
        return {
            "config": {
                "sample_rate": settings.sample_rate,
                "chunk_duration": settings.chunk_seconds,
                "pipeline": "segmented",
                "runtime": "fake",
                "device": "cpu",
                "greedy_decode": True,
                "final_beam_decode": False,
                "segmentation": {
                    "segment_silence": settings.segment_silence,
                    "turn_silence": 1.5,
                    "max_segment_duration": 10.0,
                },
            },
            "graph": {"stateless": True, "inputs": [], "outputs": [],
                      "providers": ["FakeExecutionProvider"],
                      "subsampling_factor": 4},
            "audio_formats": {"libsndfile_formats": ["WAV"], "ffmpeg_available": False},
        }

    @app.websocket("/ws/transcribe")
    async def transcribe(websocket: WebSocket) -> None:
        settings: FakeConfig = app.state.config
        await websocket.accept()

        with app.state.lock:
            index = app.state.accepted
            app.state.accepted += 1
            admitted = app.state.active < settings.max_concurrent
            if admitted:
                app.state.active += 1

        if not admitted or (settings.fail_mode == "reject" and settings.should_fail(index)):
            await websocket.send_json({
                "type": "error",
                "detail": f"at capacity ({settings.max_concurrent} concurrent streams)",
            })
            await websocket.close(code=1013)
            if admitted:
                with app.state.lock:
                    app.state.active -= 1
            return

        failing = settings.should_fail(index)
        try:
            await websocket.send_json({
                "type": "ready",
                "sample_rate": settings.sample_rate,
                "chunk_samples": int(settings.chunk_seconds * settings.sample_rate),
                "chunk_ms": round(1000 * settings.chunk_seconds, 1),
                "format": "float32 or int16 PCM, mono, little-endian",
            })
            await _serve(websocket, settings, failing)
        except WebSocketDisconnect:
            pass
        except Exception:  # noqa: BLE001 - a fake must not spew tracebacks
            pass
        finally:
            with app.state.lock:
                app.state.active = max(0, app.state.active - 1)
            try:
                await websocket.close()
            except Exception:  # noqa: BLE001
                pass

    return app


async def _serve(websocket: Any, settings: FakeConfig, failing: bool) -> None:
    """Consume audio and emit events on a deterministic schedule."""
    samples = 0
    chunks = 0
    dtype = "float32"
    next_segment_at = settings.segment_seconds
    segment_start = 0.0

    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return

        if (text := message.get("text")) is not None:
            control = json.loads(text)
            if control.get("type") == "config":
                dtype = control.get("format", dtype)
                continue
            if control.get("type") == "end":
                if failing and settings.fail_mode in ("no_final", "silent"):
                    # Hang rather than close. Closing would surface at the
                    # client as a dropped connection, which is a *different*
                    # failure with a different correct response; what is being
                    # modelled here is a server that has stopped answering
                    # while the socket stays up, and only a timeout catches it.
                    continue
                await _send_final(websocket, samples / settings.sample_rate,
                                  end_of_stream=True)
                return
            continue

        data = message.get("bytes")
        if not data:
            continue
        samples += len(data) // (2 if dtype == "int16" else 4)
        chunks += 1
        audio_time = samples / settings.sample_rate

        if failing and chunks > settings.fail_after_chunks:
            if settings.fail_mode == "drop":
                await websocket.close(code=1011)
                return
            if settings.fail_mode == "malformed":
                await websocket.send_text("{this is not json")
                return
            if settings.fail_mode == "binary":
                await websocket.send_bytes(b"\x00\x01\x02\x03")
                return
            if settings.fail_mode == "error_event":
                await websocket.send_json({"type": "error",
                                           "detail": "simulated server failure"})
                return
            if settings.fail_mode == "silent":
                continue

        await websocket.send_json({
            "type": "partial",
            "t": round(audio_time, 3),
            "committed": "",
            "partial": f"partial at {audio_time:.2f}",
            "text": f"partial at {audio_time:.2f}",
            "newly_committed": [],
            "confidence": None,
        })

        # A segment becomes emittable once its speech has ended *and* the
        # notional silence has elapsed -- the same shape as the real service,
        # so the client's endpoint-anchored latency has something to anchor to.
        while audio_time >= next_segment_at + settings.segment_silence:
            if settings.segment_delay:
                await asyncio.sleep(settings.segment_delay)
            await websocket.send_json({
                "type": "segment",
                "t": round(next_segment_at, 3),
                "start": round(segment_start, 3),
                "end": round(next_segment_at, 3),
                "text": f"segment {segment_start:.1f}-{next_segment_at:.1f}",
                "transcript": f"transcript through {next_segment_at:.1f}",
                "decoder": settings.decoder_backend,
                "used_lm": settings.used_lm,
                "forced": False,
            })
            segment_start = next_segment_at
            next_segment_at += settings.segment_seconds


async def _send_final(websocket: Any, audio_time: float, end_of_stream: bool) -> None:
    await websocket.send_json({
        "type": "final",
        "t": round(audio_time, 3),
        "text": "final",
        "transcript": f"transcript through {audio_time:.1f}",
        "provisional_text": "",
        "decoder": "greedy",
        "used_lm": False,
        "end_of_stream": end_of_stream,
        "metrics": {
            "rtf": 0.05, "streaming_rtf": 0.04, "audio_duration": round(audio_time, 3),
            "model_calls": 0, "chunks_processed": 0,
            "note": "fake server: these are constants, not measurements",
        },
    })


class FakeServer:
    """Runs :func:`build_app` on an ephemeral port for the life of a test."""

    def __init__(self, config: Optional[FakeConfig] = None) -> None:
        self.config = config or FakeConfig()
        self.port = 0
        self._server: Any = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "FakeServer":
        import uvicorn

        from tests.load.server_process import free_port

        self.port = free_port()
        config = uvicorn.Config(build_app(self.config), host="127.0.0.1",
                                port=self.port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True,
                                        name="fake-asr-server")
        self._thread.start()
        self._wait_until_serving()
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)

    def _wait_until_serving(self, timeout: float = 20.0) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if getattr(self._server, "started", False):
                return
            time.sleep(0.02)
        raise RuntimeError("fake server did not start")


def silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Audio for harness tests. Content is irrelevant -- the fake never looks."""
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)
