"""Send one recording through every transport and compare the transcripts.

Three of the four paths must agree **exactly** -- they carry identical samples
into an identical pipeline, so any difference is a timing-dependent bug:

    POST /transcribe          the whole file at once
    WS, streamed flat out     chunked, as fast as the socket accepts
    WS, streamed at 1x        chunked, paced like a live caller

The fourth cannot agree exactly, and expecting it to would be a mistake:

    microphone loopback       played through the speakers and re-captured

That path resamples (usually 48 kHz -> 16 kHz), may apply AGC and echo
cancellation, and picks up the room. It verifies that the *capture path* works
end to end, not that the audio survived unchanged. Judge it by whether the
transcript is recognisably the same, not by string equality.

Usage::

    python tools/verify_parity.py --audio sample.wav
    python tools/verify_parity.py --audio sample.wav --mic-device 11
    python tools/verify_parity.py --audio sample.wav --list-devices
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SAMPLE_RATE = 16000
CHUNK = 0.16


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


def read_audio(path: str) -> np.ndarray:
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd

        divisor = gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // divisor, rate // divisor)
    return np.ascontiguousarray(mono, dtype=np.float32)


def to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def ws_url(url: str) -> str:
    return url.replace("http://", "ws://").replace("https://", "wss://") + "/ws/transcribe"


# ---------------------------------------------------------------------------
# transports
# ---------------------------------------------------------------------------


def via_post(url: str, path: str) -> tuple[str, float]:
    import httpx

    started = time.perf_counter()
    with open(path, "rb") as fh:
        response = httpx.post(f"{url}/transcribe",
                              files={"file": (Path(path).name, fh, "audio/wav")},
                              timeout=600)
    response.raise_for_status()
    return response.json()["text"].strip(), time.perf_counter() - started


async def _stream(url: str, blocks, real_time: bool) -> tuple[str, float]:
    import websockets

    started = time.perf_counter()
    async with websockets.connect(ws_url(url), max_size=None) as ws:
        json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "config", "format": "int16"}))

        transcript = {"text": ""}
        done = asyncio.Event()

        async def receive() -> None:
            async for raw in ws:
                event = json.loads(raw)
                if event.get("type") == "final" and event.get("end_of_stream"):
                    transcript["text"] = (event.get("transcript") or "").strip()
                    done.set()
                    return
                if event.get("type") == "error":
                    transcript["text"] = f"<error: {event['detail']}>"
                    done.set()
                    return

        receiver = asyncio.create_task(receive())
        clock = time.perf_counter()
        for index, block in enumerate(blocks):
            if real_time:
                delay = clock + (index + 1) * CHUNK - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            await ws.send(to_pcm16(block))

        await ws.send(json.dumps({"type": "end"}))
        await asyncio.wait_for(done.wait(), timeout=300)
        receiver.cancel()
    return transcript["text"], time.perf_counter() - started


def via_websocket(url: str, audio: np.ndarray, real_time: bool) -> tuple[str, float]:
    step = int(CHUNK * SAMPLE_RATE)
    blocks = [audio[i:i + step] for i in range(0, len(audio), step)]
    return asyncio.run(_stream(url, blocks, real_time))


def via_microphone(url: str, audio: np.ndarray, device: int | str) -> tuple[str, float]:
    """Play the file out and capture it back through a real input device.

    Exercises the capture path with deterministic content. Use a loopback
    device ("Stereo Mix") for a clean signal, or leave speakers and microphone
    open for a genuinely acoustic test.
    """
    import queue
    import threading

    import sounddevice as sd

    captured: queue.Queue[np.ndarray] = queue.Queue()
    finished = threading.Event()

    def on_capture(indata, frames, time_info, status):
        captured.put(indata[:, 0].copy())

    async def run() -> tuple[str, float]:
        import websockets

        started = time.perf_counter()
        async with websockets.connect(ws_url(url), max_size=None) as ws:
            json.loads(await ws.recv())
            await ws.send(json.dumps({"type": "config", "format": "int16"}))

            transcript = {"text": ""}
            done = asyncio.Event()

            async def receive() -> None:
                async for raw in ws:
                    event = json.loads(raw)
                    if event.get("type") == "final" and event.get("end_of_stream"):
                        transcript["text"] = (event.get("transcript") or "").strip()
                        done.set()
                        return

            receiver = asyncio.create_task(receive())
            step = int(CHUNK * SAMPLE_RATE)

            with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=step, channels=1,
                                dtype="float32", device=device, callback=on_capture):
                sd.play(audio, SAMPLE_RATE)
                # Keep capturing past the end of playback so the tail of the
                # recording is not clipped by the segmenter's silence window.
                deadline = time.perf_counter() + len(audio) / SAMPLE_RATE + 2.5
                while time.perf_counter() < deadline:
                    try:
                        block = await asyncio.to_thread(captured.get, True, 0.2)
                    except queue.Empty:
                        continue
                    await ws.send(to_pcm16(block))
                sd.stop()

            await ws.send(json.dumps({"type": "end"}))
            await asyncio.wait_for(done.wait(), timeout=300)
            receiver.cancel()
        return transcript["text"], time.perf_counter() - started

    return asyncio.run(run())


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audio", help="recording to send through every transport")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--mic-device", default=None,
                        help="input device index or name for the loopback test; "
                             "omit to skip it")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print("Input devices (look for 'Stereo Mix' for a clean loopback):")
        for index, info in enumerate(sd.query_devices()):
            if info.get("max_input_channels", 0) > 0:
                print(f"  [{index}] {info['name']}")
        return 0

    if not args.audio:
        parser.error("--audio is required (or use --list-devices)")

    audio = read_audio(args.audio)
    print(f"{Path(args.audio).name}: {len(audio) / SAMPLE_RATE:.2f}s\n")

    results: dict[str, str] = {}
    print(f"{'transport':<26}{'wall':>8}  transcript")
    print("-" * 100)

    for label, runner in (
        ("POST /transcribe", lambda: via_post(args.url, args.audio)),
        ("WS, flat out", lambda: via_websocket(args.url, audio, real_time=False)),
        ("WS, real time (1x)", lambda: via_websocket(args.url, audio, real_time=True)),
    ):
        text, elapsed = runner()
        results[label] = text
        print(f"{label:<26}{elapsed:>7.1f}s  {text[:70]}")

    mic_text = None
    if args.mic_device is not None:
        device = int(args.mic_device) if str(args.mic_device).isdigit() else args.mic_device
        print(f"\nplaying through the speakers and capturing from device {device!r}...")
        mic_text, elapsed = via_microphone(args.url, audio, device)
        print(f"{'microphone loopback':<26}{elapsed:>7.1f}s  {mic_text[:70]}")

    # ---- verdict --------------------------------------------------------

    print("\n" + "=" * 100)
    file_paths = list(results.values())
    identical = len(set(file_paths)) == 1

    if identical:
        print("PASS  the three file transports agree exactly")
    else:
        print("FAIL  the file transports disagree -- identical samples should give "
              "an identical transcript")
        for label, text in results.items():
            print(f"  {label:<26}{text!r}")
        reference = file_paths[0]
        for label, text in list(results.items())[1:]:
            print(f"  WER vs first: {label:<20}{word_error_rate(reference, text):.3f}")

    if mic_text is not None:
        wer = word_error_rate(file_paths[0], mic_text)
        print(f"\nmicrophone loopback WER vs file: {wer:.3f}")
        print(f"  file: {file_paths[0]!r}")
        print(f"  mic : {mic_text!r}")
        if wer == 0.0:
            print("  identical -- a clean loopback with no resampling loss")
        elif wer < 0.2:
            print("  close enough: the capture path works. The difference is the "
                  "device (resampling, AGC) and the room, not the pipeline.")
        else:
            print("  DIVERGENT. Check the input level and whether the device "
                  "resamples; if it is near 1.0 the wrong device is selected.")

    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
