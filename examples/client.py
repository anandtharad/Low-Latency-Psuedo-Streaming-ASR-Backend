"""Example clients for the streaming ASR service.

    python examples/client.py batch  --audio a.wav
    python examples/client.py stream --audio a.wav          # simulated live
    python examples/client.py mic                           # real microphone
    python examples/client.py health

``stream`` paces audio at wall-clock speed, so the partial timings printed here
are what a real caller would experience.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

# Deliberately does NOT import streaming_asr. A client only moves bytes and
# JSON; pulling in the ASR package would drag in torch and onnxruntime -- a
# multi-second, multi-gigabyte cost for something that decodes no audio, and
# enough to fail outright on a memory-constrained machine. Copy this file as
# the starting point for your own client; its only real dependency is
# `websockets` (plus `soundfile` for the file-streaming demo).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_URL = "http://localhost:8000"


def _ws_url(url: str) -> str:
    return url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/transcribe"


def _to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


# ---- REST ----------------------------------------------------------------


def cmd_health(args: argparse.Namespace) -> int:
    import httpx

    response = httpx.get(f"{args.url}/health", timeout=10)
    print(json.dumps(response.json(), indent=2))
    return 0 if response.status_code == 200 else 1


def cmd_batch(args: argparse.Namespace) -> int:
    import httpx

    with open(args.audio, "rb") as fh:
        response = httpx.post(
            f"{args.url}/transcribe",
            files={"file": (Path(args.audio).name, fh, "audio/wav")},
            timeout=300,
        )
    response.raise_for_status()
    body = response.json()

    print(f"text     : {body['text']!r}")
    print(f"decoder  : {body['decoder']} (lm={body['used_lm']})")
    print(f"duration : {body['duration']}s")
    print(f"RTF      : {body['metrics'].get('rtf')}")
    return 0


# ---- WebSocket -----------------------------------------------------------


async def _stream(url: str, blocks, chunk_period: float, real_time: bool) -> int:
    import websockets

    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        print(f"connected: {ready['sample_rate']} Hz, {ready['chunk_ms']} ms chunks\n")
        await ws.send(json.dumps({"type": "config", "format": "int16"}))

        done = asyncio.Event()

        async def receive() -> None:
            """Print events as they arrive rather than after sending finishes."""
            last = ""
            async for raw in ws:
                event = json.loads(raw)
                kind = event.get("type")
                if kind == "partial":
                    # Provisional and revisable -- overwrite one line rather
                    # than logging every revision.
                    line = event["partial"]
                    if line != last:
                        print(f"\r[{event['t']:6.2f}s] ... {line[:90]:<90}",
                              end="", flush=True)
                        last = line
                elif kind == "segment":
                    mark = " (forced)" if event.get("forced") else ""
                    print(f"\r[{event['t']:6.2f}s] SEGMENT{mark}: {event['text']!r}"
                          f"{' ' * 30}")
                    last = ""
                elif kind == "final" and not event.get("end_of_stream"):
                    print(f"[{event['t']:6.2f}s] TURN FINAL: {event['text']!r}\n")
                elif kind == "endpoint":
                    print(f"\n[endpoint] {event['reason']}")
                elif kind == "final":
                    print("\n" + "=" * 64)
                    print(f"TRANSCRIPT ({event['decoder']}, lm={event['used_lm']}):")
                    print(f"  {event['transcript']!r}")
                    print("=" * 64)
                    metrics = event.get("metrics", {})
                    print(f"  first_partial_latency: "
                          f"{metrics.get('first_partial_latency')}s")
                    print(f"  RTF                  : {metrics.get('rtf')}")
                    done.set()
                    return
                elif kind == "error":
                    print(f"ERROR: {event['detail']}", file=sys.stderr)
                    done.set()
                    return

        receiver = asyncio.create_task(receive())
        started = time.perf_counter()
        for index, block in enumerate(blocks):
            if real_time:
                target = started + (index + 1) * chunk_period
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            await ws.send(_to_pcm16(block))

        await ws.send(json.dumps({"type": "end"}))
        await asyncio.wait_for(done.wait(), timeout=120)
        receiver.cancel()
    return 0


def _read_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Read a file as mono float32. The server does the heavy lifting.

    Only supports files already at ``sample_rate``; a real client would either
    resample or just POST the file to /transcribe and let the server decode it.
    """
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate != sample_rate:
        raise SystemExit(
            f"{path} is {rate} Hz; this demo streams {sample_rate} Hz only. "
            f"Use 'batch' instead -- the server resamples for you."
        )
    return data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]


def cmd_stream(args: argparse.Namespace) -> int:
    audio = _read_audio(args.audio)
    step = int(0.16 * 16000)
    blocks = [audio[i:i + step] for i in range(0, len(audio), step)]
    print(f"streaming {Path(args.audio).name}: {len(audio) / 16000:.2f}s "
          f"in {len(blocks)} blocks\n")
    return asyncio.run(_ws_url(args.url) and _stream(
        _ws_url(args.url), blocks, 0.16, real_time=not args.fast
    ))


def cmd_mic(args: argparse.Namespace) -> int:
    """Capture live audio and stream it to the service."""
    import queue
    import signal
    import threading
    import time as _time

    import sounddevice as sd

    step = int(0.16 * 16000)
    buffer: queue.Queue[np.ndarray] = queue.Queue()
    stop = threading.Event()

    def callback(indata, frames, time_info, status):
        if not stop.is_set():
            buffer.put(indata[:, 0].copy())

    # A SIGINT handler that just sets a flag. Raising KeyboardInterrupt into an
    # asyncio loop that is parked on a worker thread does not reliably unwind
    # on Windows, so the loop has to poll a flag instead.
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    async def run() -> int:
        import websockets

        async with websockets.connect(_ws_url(args.url), max_size=None) as ws:
            json.loads(await ws.recv())
            await ws.send(json.dumps({"type": "config", "format": "int16"}))
            print("Listening. Press Ctrl+C to stop.\n")

            finished = asyncio.Event()

            async def receive() -> None:
                last = ""
                async for raw in ws:
                    event = json.loads(raw)
                    kind = event.get("type")
                    if kind == "partial":
                        # Revisable: overwrite one line, padded so a shorter
                        # partial cannot leave the tail of a longer one behind.
                        line = event["partial"]
                        if line != last:
                            print(f"\r  ... {line[:100]:<100}", end="", flush=True)
                            last = line
                    elif kind == "segment":
                        mark = " (forced)" if event.get("forced") else ""
                        print(f"\r  SEGMENT{mark}: {event['text']!r}{' ' * 40}")
                        last = ""
                    elif kind == "final" and not event.get("end_of_stream"):
                        # A turn ended on silence -- the session continues.
                        print(f"  TURN: {event['text']!r}\n")
                        last = ""
                    elif kind == "final":
                        print(f"\n\nTRANSCRIPT: {event['transcript']!r}")
                        finished.set()
                        return
                    elif kind == "error":
                        print(f"\nERROR: {event['detail']}", file=sys.stderr)
                        finished.set()
                        return

            receiver = asyncio.create_task(receive())
            started = _time.perf_counter()

            with sd.InputStream(samplerate=16000, blocksize=step, channels=1,
                                dtype="float32", callback=callback):
                while not stop.is_set() and not receiver.done():
                    if args.seconds and _time.perf_counter() - started >= args.seconds:
                        break
                    try:
                        # Short timeout so the stop flag is checked regularly.
                        block = await asyncio.to_thread(buffer.get, True, 0.2)
                    except queue.Empty:
                        continue
                    await ws.send(_to_pcm16(block))

            stop.set()
            if not receiver.done():
                await ws.send(json.dumps({"type": "end"}))
                try:
                    await asyncio.wait_for(finished.wait(), timeout=60)
                except asyncio.TimeoutError:
                    print("\ntimed out waiting for the final transcript",
                          file=sys.stderr)
            receiver.cancel()
        return 0

    return asyncio.run(run())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health").set_defaults(func=cmd_health)

    batch = sub.add_parser("batch", help="one-shot file transcription")
    batch.add_argument("--audio", required=True)
    batch.set_defaults(func=cmd_batch)

    stream = sub.add_parser("stream", help="stream a file over the WebSocket")
    stream.add_argument("--audio", required=True)
    stream.add_argument("--fast", action="store_true",
                        help="send as fast as possible instead of at wall-clock speed")
    stream.set_defaults(func=cmd_stream)

    mic = sub.add_parser("mic", help="stream from the microphone")
    mic.add_argument("--seconds", type=float,
                     help="stop after this long instead of waiting for Ctrl+C")
    mic.set_defaults(func=cmd_mic)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
