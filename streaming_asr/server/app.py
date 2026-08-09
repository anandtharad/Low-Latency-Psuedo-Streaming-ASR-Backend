"""HTTP + WebSocket service around the streaming ASR pipeline.

Endpoints
---------
``GET  /health``            liveness, plus what actually loaded
``GET  /info``              full config and ONNX graph report
``POST /transcribe``        one-shot: upload audio, get the final transcript
``WS   /ws/transcribe``     streaming: send PCM, receive partials then a final

The WebSocket endpoint is the one that matters -- a one-shot POST cannot
express incremental results, which is the entire point of this pipeline.

Threading
---------
Inference is blocking and CPU/GPU-bound. Running it inline in an async handler
would stall the event loop and, with it, *every other connection* on the
worker -- one slow stream would stop all of them. So every call into the
pipeline is dispatched to a worker thread, and the event loop only ever moves
bytes and JSON.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from streaming_asr.audio.base import AudioChunk
from streaming_asr.audio.decode import UnsupportedAudioError, decode, describe_support
from streaming_asr.events import ASREvent, ASREventType
from streaming_asr.server.model_pool import ModelPool
from streaming_asr.server.settings import config_from_env, server_settings

logger = logging.getLogger(__name__)

#: Populated at startup. Module-level so the endpoints can reach it.
POOL: Optional[ModelPool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model before the first request, release it on shutdown.

    Loading here rather than lazily means a misconfigured container fails to
    start instead of accepting traffic it cannot serve.
    """
    global POOL
    settings = server_settings()
    config = config_from_env()
    POOL = ModelPool(config, max_concurrent_streams=settings["max_concurrent_streams"])
    logger.info(
        "Service ready on %s:%s (%d concurrent streams max)",
        settings["host"], settings["port"], settings["max_concurrent_streams"],
    )
    yield
    if POOL is not None:
        POOL.close()
        POOL = None


app = FastAPI(
    title="Streaming ASR",
    description="Rolling-buffer pseudo-streaming ASR over an offline Conformer-CTC ONNX model",
    version="0.1.0",
    lifespan=lifespan,
)


def _pool() -> ModelPool:
    if POOL is None:  # pragma: no cover - only before startup completes
        raise HTTPException(status_code=503, detail="model still loading")
    return POOL


# ---------------------------------------------------------------------------
# Introspection
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness plus the facts worth alerting on.

    Reports what *actually* loaded, not what was requested -- a service that
    silently fell back from GPU to CPU is still 'healthy' but will miss its
    latency budget, and this is where that shows up.
    """
    if POOL is None:
        return JSONResponse({"ready": False, "detail": "loading"}, status_code=503)
    return JSONResponse({"ready": True, **asdict(POOL.status())})


@app.get("/info")
async def info() -> dict[str, Any]:
    pool = _pool()
    report = pool.engine.graph_report
    return {
        "config": pool.config.to_dict(summarize_vocabulary=True),
        "graph": {
            "inputs": [str(s) for s in report.inputs],
            "outputs": [str(s) for s in report.outputs],
            "stateless": report.is_stateless,
            "providers": pool.engine.active_providers,
            "subsampling_factor": pool.engine.subsampling_factor,
        },
        "runtime": asdict(pool.status()),
        # So a client can discover what it may upload instead of finding out
        # by getting a 415.
        "audio_formats": describe_support(),
    }


# ---------------------------------------------------------------------------
# Batch transcription
# ---------------------------------------------------------------------------


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="audio file (wav/flac/ogg)"),
    stream_events: bool = Query(False, description="also return the partial trail"),
) -> dict[str, Any]:
    """Transcribe a complete file.

    Runs the same streaming path as the WebSocket endpoint rather than a
    separate offline one, so a caller comparing the two gets comparable
    results and the streaming trail stays inspectable.
    """
    pool = _pool()
    if not pool.try_acquire():
        raise HTTPException(
            status_code=503,
            detail=f"at capacity ({pool.max_concurrent_streams} concurrent streams)",
        )
    try:
        raw = await file.read()
        try:
            audio = await asyncio.to_thread(decode, raw, pool.config.sample_rate)
        except UnsupportedAudioError as exc:
            # A client sending an undecodable file has made a client error.
            # Letting the decoder raise would surface it as a 500, which reads
            # as "the server is broken" and tells them nothing actionable.
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        max_seconds = server_settings()["max_upload_seconds"]
        if len(audio) / pool.config.sample_rate > max_seconds:
            raise HTTPException(
                status_code=413,
                detail=f"audio exceeds ASR_MAX_UPLOAD_SEC ({max_seconds}s)",
            )
        return await asyncio.to_thread(_run_batch, pool, audio, stream_events)
    finally:
        pool.release()


def _run_batch(pool: ModelPool, audio: np.ndarray, stream_events: bool) -> dict[str, Any]:
    from streaming_asr.audio.wav_source import InMemorySource

    pipeline = pool.new_pipeline()
    source = InMemorySource(audio, pool.config.sample_rate, pool.config.chunk_samples)

    partials: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    final: Optional[ASREvent] = None
    for event in pipeline.stream(source):
        if event.type is ASREventType.FINAL:
            final = event
        elif event.type is ASREventType.SEGMENT:
            segments.append({
                "start": round(event.window_start or 0.0, 3),
                "end": round(event.window_end or 0.0, 3),
                "text": event.text,
                "forced": bool(event.metrics.get("forced")),
            })
        elif stream_events and event.type is ASREventType.PARTIAL:
            partials.append({
                "t": round(event.timestamp, 3),
                "committed": event.committed_text,
                "partial": event.partial_text,
            })

    if final is None:  # pragma: no cover - stream() always finalises
        raise HTTPException(status_code=500, detail="no final transcript produced")

    # For the segmented pipeline a FINAL marks a turn, so the file's transcript
    # is the accumulation of every segment rather than the last event.
    text = getattr(pipeline, "transcript", None) or final.text

    payload: dict[str, Any] = {
        "text": text,
        "provisional_text": final.provisional_text,
        "decoder": final.decoder,
        "used_lm": final.used_lm,
        "duration": round(len(audio) / pool.config.sample_rate, 3),
        "metrics": final.metrics,
    }
    if segments:
        payload["segments"] = segments
    if stream_events:
        payload["partials"] = partials
    return payload


# ---------------------------------------------------------------------------
# Streaming transcription
# ---------------------------------------------------------------------------


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket) -> None:
    """Bidirectional streaming ASR.

    Protocol:

    * client -> server: **binary** frames of PCM at the configured sample rate,
      either float32 or int16 (set ``format`` in the optional opening message).
    * client -> server: **text** JSON control messages,
      ``{"type": "end"}`` to close the utterance and get the final transcript,
      ``{"type": "reset"}`` to abandon it and start over.
    * server -> client: JSON events -- ``ready``, ``partial``, ``endpoint``,
      ``final``, ``error``.

    Audio need not arrive in chunk-sized frames; it is re-blocked here, so a
    client can send whatever its capture stack produces.
    """
    await websocket.accept()

    if POOL is None:
        # HTTPException is meaningless on a WebSocket; speak the socket's own
        # protocol instead.
        await websocket.send_json({"type": "error", "detail": "model still loading"})
        await websocket.close(code=1013)
        return
    pool = POOL

    if not pool.try_acquire():
        await websocket.send_json({
            "type": "error",
            "detail": f"at capacity ({pool.max_concurrent_streams} concurrent streams)",
        })
        await websocket.close(code=1013)  # try again later
        return

    # Everything from here on MUST be inside the try, so that the finally
    # returns the capacity slot. Constructing the pipeline can fail, and the
    # opening 'ready' send fails whenever a client disconnects immediately --
    # a load-balancer probe is enough. Leaking a slot on those paths would
    # exhaust capacity permanently and 503 every subsequent caller.
    try:
        pipeline = pool.new_pipeline()
        chunk_samples = pool.config.chunk_samples
        sample_rate = pool.config.sample_rate
        pending = np.zeros(0, dtype=np.float32)
        dtype = "float32"
        next_sample = 0
        finalized = False

        await websocket.send_json({
            "type": "ready",
            "sample_rate": sample_rate,
            "chunk_samples": chunk_samples,
            "chunk_ms": round(1000 * pool.config.chunk_duration, 1),
            "format": "float32 or int16 PCM, mono, little-endian",
        })

        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if (text := message.get("text")) is not None:
                import json

                try:
                    control = json.loads(text)
                except ValueError:
                    await websocket.send_json({"type": "error", "detail": "invalid JSON"})
                    continue

                action = control.get("type")
                if action == "config":
                    dtype = control.get("format", dtype)
                    continue
                if action == "reset":
                    pipeline = pool.new_pipeline()
                    pending = np.zeros(0, dtype=np.float32)
                    next_sample = 0
                    finalized = False
                    await websocket.send_json({"type": "ready", "reset": True})
                    continue
                if action == "end":
                    # Flush whatever is left, zero-padded to a full chunk, so
                    # the tail of the utterance is not silently discarded.
                    if pending.size:
                        block = np.zeros(chunk_samples, dtype=np.float32)
                        block[: pending.size] = pending[:chunk_samples]
                        await _feed(websocket, pipeline, block, next_sample, sample_rate)
                        pending = np.zeros(0, dtype=np.float32)
                    await _finalize(websocket, pipeline)
                    finalized = True
                    break
                await websocket.send_json({
                    "type": "error", "detail": f"unknown control message {action!r}",
                })
                continue

            data = message.get("bytes")
            if not data:
                continue

            samples = _pcm_to_float32(data, dtype)
            pending = np.concatenate([pending, samples]) if pending.size else samples

            # Re-block into exactly chunk-sized frames.
            while pending.size >= chunk_samples:
                block, pending = pending[:chunk_samples], pending[chunk_samples:]
                endpointed = await _feed(
                    websocket, pipeline, block, next_sample, sample_rate
                )
                next_sample += chunk_samples
                if endpointed:
                    await _finalize(websocket, pipeline)
                    finalized = True
                    break
            if finalized:
                break

    except WebSocketDisconnect:
        logger.info("client disconnected before end of utterance")
    except Exception as exc:  # noqa: BLE001 - report, never leak a 500 into the socket
        logger.exception("websocket stream failed")
        try:
            await websocket.send_json({"type": "error", "detail": str(exc)})
        except Exception:
            pass
    finally:
        pool.release()
        try:
            await websocket.close()
        except Exception:
            pass


async def _feed(
    websocket: WebSocket,
    pipeline: Any,
    block: np.ndarray,
    start_sample: int,
    sample_rate: int,
) -> bool:
    """Process one chunk off the event loop and emit its events."""
    chunk = AudioChunk(
        samples=block,
        start_sample=start_sample,
        sample_rate=sample_rate,
        capture_time=time.perf_counter(),
    )
    events = await asyncio.to_thread(pipeline.process_chunk, chunk)

    endpointed = False
    for event in events:
        if event.type is ASREventType.PARTIAL:
            await websocket.send_json({
                "type": "partial",
                "t": round(event.timestamp, 3),
                "committed": event.committed_text,
                "partial": event.partial_text,
                "text": event.full_hypothesis,
                "newly_committed": [w.text for w in event.newly_committed],
                "confidence": event.confidence,
            })
        elif event.type is ASREventType.SEGMENT:
            # Authoritative for its span, and emitted well before the turn
            # ends. A latency-sensitive client consumes these; one that needs a
            # complete thought waits for "final".
            await websocket.send_json({
                "type": "segment",
                "t": round(event.timestamp, 3),
                "start": round(event.window_start or 0.0, 3),
                "end": round(event.window_end or 0.0, 3),
                "text": event.text,
                "transcript": event.committed_text,
                "decoder": event.decoder,
                "used_lm": event.used_lm,
                "forced": bool(event.metrics.get("forced")),
            })
        elif event.type is ASREventType.FINAL:
            # A turn ended mid-stream (silence), not the connection.
            await websocket.send_json({
                "type": "final",
                "t": round(event.timestamp, 3),
                "text": event.text,
                "transcript": event.committed_text,
                "decoder": event.decoder,
                "used_lm": event.used_lm,
            })
        elif event.type is ASREventType.ENDPOINT:
            await websocket.send_json({
                "type": "endpoint",
                "t": round(event.timestamp, 3),
                "reason": event.metrics.get("reason", ""),
            })
            endpointed = True
    return endpointed


async def _finalize(websocket: WebSocket, pipeline: Any) -> None:
    """Run the expensive decode off the event loop and send the result."""
    if pipeline.is_finalized:
        return
    event = await asyncio.to_thread(pipeline.finalize)

    # The last segment almost always closes inside finalize(); publish it
    # before the final so a client accumulating `segment` events sees the whole
    # utterance in arrival order.
    for pending in getattr(pipeline, "drain_pending", list)():
        if pending.type is ASREventType.SEGMENT:
            await websocket.send_json({
                "type": "segment",
                "t": round(pending.timestamp, 3),
                "start": round(pending.window_start or 0.0, 3),
                "end": round(pending.window_end or 0.0, 3),
                "text": pending.text,
                "transcript": pending.committed_text,
                "decoder": pending.decoder,
                "used_lm": pending.used_lm,
                "forced": bool(pending.metrics.get("forced")),
            })

    await websocket.send_json({
        "type": "final",
        "t": round(event.timestamp, 3),
        # The turn still open at end of stream; empty when the last turn
        # already closed on silence.
        "text": event.text,
        # Always the whole session, so a client has one field to read.
        "transcript": getattr(pipeline, "transcript", None) or event.text,
        "provisional_text": event.provisional_text,
        "decoder": event.decoder,
        "used_lm": event.used_lm,
        "end_of_stream": True,
        "metrics": event.metrics,
    })


def _pcm_to_float32(data: bytes, dtype: str) -> np.ndarray:
    """Decode a PCM frame. int16 is the common wire format; float32 is native."""
    if dtype == "int16":
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    array = np.frombuffer(data, dtype="<f4")
    return np.ascontiguousarray(array, dtype=np.float32)


def main() -> None:
    """Entry point used by the container."""
    import uvicorn

    from streaming_asr.console import configure_logging

    settings = server_settings()
    configure_logging()
    uvicorn.run(
        "streaming_asr.server.app:app",
        host=settings["host"],
        port=settings["port"],
        log_level=settings["log_level"],
        # One worker: the model is held in process memory and a second worker
        # would load a second copy (and a second CUDA context). Scale with
        # ASR_MAX_CONCURRENT_STREAMS, or with more containers behind a balancer.
        workers=1,
    )


if __name__ == "__main__":
    main()
