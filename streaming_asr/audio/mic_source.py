"""Live microphone capture.

This is the first *real* audio source: WAV simulation controls the clock, a
microphone does not. That difference drives the whole design here.

Audio arrives on a driver callback thread at exactly real time, while the ASR
loop consumes on the main thread at whatever rate inference allows. A bounded
queue sits between them, and the interesting question is what happens when the
consumer is slower than real time.

At the reference operating point every sample is reprocessed 25 times. If the
streaming RTF exceeds 1.0 the queue grows without bound and latency climbs for
the rest of the session -- the transcript falls further behind the speaker with
every chunk, which is far worse than dropping audio. So the queue is bounded and
overruns are counted and reported rather than silently absorbed. An overrun
count above zero means the configuration is not viable live, whatever the
offline benchmark said.

Requires ``sounddevice`` (preferred) or ``pyaudio``::

    pip install sounddevice
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any, Iterator, Optional

import numpy as np

from streaming_asr.audio.base import AudioChunk, AudioSource

logger = logging.getLogger(__name__)


class MicrophoneUnavailable(RuntimeError):
    """No usable capture backend or device."""


def _import_sounddevice() -> Any:
    try:
        import sounddevice as sd

        return sd
    except Exception as exc:  # pragma: no cover - depends on host audio stack
        raise MicrophoneUnavailable(
            "sounddevice is required for microphone capture. Install it with "
            "'pip install sounddevice'. On Linux you may also need libportaudio2."
        ) from exc


def list_input_devices() -> list[dict[str, Any]]:
    """Enumerate capture devices, for ``--list-devices``."""
    sd = _import_sounddevice()
    devices = []
    for index, info in enumerate(sd.query_devices()):
        if info.get("max_input_channels", 0) > 0:
            devices.append({
                "index": index,
                "name": info.get("name", "?"),
                "channels": info["max_input_channels"],
                "default_samplerate": info.get("default_samplerate"),
            })
    return devices


def default_input_device() -> Optional[int]:
    sd = _import_sounddevice()
    try:
        device = sd.default.device
        return device[0] if isinstance(device, (list, tuple)) else device
    except Exception:  # pragma: no cover
        return None


class MicrophoneSource(AudioSource):
    """Captures live audio and emits fixed-size chunks.

    The ASR engine cannot tell this apart from :class:`WavFileSource` -- that is
    the point of the :class:`AudioSource` interface.

    Args:
        sample_rate: Requested capture rate. 16 kHz matches the model; if the
            device refuses it, capture runs at its native rate and is resampled.
        chunk_samples: Samples per emitted chunk, from ``config.chunk_samples``.
        device: Device index or name. ``None`` uses the system default.
        channels: Capture channels; multi-channel input is downmixed to mono.
        max_queued_chunks: Bound on the backlog. Once full, the oldest chunks
            are dropped and ``overruns`` is incremented.
        max_duration: Optional hard stop, in seconds.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_samples: int = 2560,
        device: Optional[int | str] = None,
        channels: int = 1,
        max_queued_chunks: int = 32,
        max_duration: Optional[float] = None,
    ) -> None:
        self._sd = _import_sounddevice()
        self._sample_rate = int(sample_rate)
        self._chunk_samples = int(chunk_samples)
        self._device = device
        self._channels = int(channels)
        self._max_duration = max_duration

        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(
            maxsize=max_queued_chunks
        )
        self._stream: Any = None
        self._stopped = False
        self._start_sample = 0
        self.overruns = 0

        self._capture_rate = self._resolve_capture_rate()
        self._resampler = None
        if self._capture_rate != self._sample_rate:
            logger.warning(
                "Device does not support %d Hz; capturing at %d Hz and resampling.",
                self._sample_rate, self._capture_rate,
            )
            self._resampler = self._build_resampler()

        # Capture in device-rate blocks that map to one output chunk.
        self._capture_block = int(
            round(self._chunk_samples * self._capture_rate / self._sample_rate)
        )

    # ---- AudioSource interface ------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def chunk_samples(self) -> int:
        return self._chunk_samples

    # ---- setup -----------------------------------------------------------

    def _resolve_capture_rate(self) -> int:
        try:
            self._sd.check_input_settings(
                device=self._device, channels=self._channels,
                samplerate=self._sample_rate, dtype="float32",
            )
            return self._sample_rate
        except Exception:
            info = self._sd.query_devices(self._device, "input")
            return int(info["default_samplerate"])

    def _build_resampler(self) -> Any:
        import torch
        import torchaudio

        return torchaudio.transforms.Resample(
            orig_freq=self._capture_rate, new_freq=self._sample_rate
        ).eval()

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Driver thread. Must not block or do heavy work."""
        if status:
            # Device-level xrun, distinct from our own queue overrun.
            logger.debug("Audio input status: %s", status)

        block = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 \
            else indata.reshape(-1)
        try:
            self._queue.put_nowait(np.array(block, dtype=np.float32, copy=True))
        except queue.Full:
            # Drop the oldest so the newest audio still gets through: falling
            # permanently behind the speaker is worse than a gap.
            self.overruns += 1
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(np.array(block, dtype=np.float32, copy=True))
            except queue.Empty:  # pragma: no cover - race with the consumer
                pass

    # ---- streaming -------------------------------------------------------

    def stream(self) -> Iterator[AudioChunk]:
        """Yield chunks until stopped, interrupted, or ``max_duration`` elapses."""
        self._stopped = False
        self._start_sample = 0
        self.overruns = 0

        self._stream = self._sd.InputStream(
            samplerate=self._capture_rate,
            blocksize=self._capture_block,
            device=self._device,
            channels=self._channels,
            dtype="float32",
            callback=self._callback,
        )

        started = time.perf_counter()
        with self._stream:
            logger.info(
                "Microphone open: %d Hz, %.0f ms chunks%s",
                self._capture_rate, 1000 * self._chunk_samples / self._sample_rate,
                " (resampling)" if self._resampler is not None else "",
            )
            while not self._stopped:
                if self._max_duration is not None and \
                        time.perf_counter() - started >= self._max_duration:
                    break
                try:
                    block = self._queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if block is None:
                    break

                samples = self._to_target_rate(block)
                # capture_time is set here, on arrival, so latency metrics
                # measure the real path from microphone to transcript.
                chunk = AudioChunk(
                    samples=samples,
                    start_sample=self._start_sample,
                    sample_rate=self._sample_rate,
                    is_last=False,
                    capture_time=time.perf_counter(),
                )
                self._start_sample += len(samples)
                yield chunk

        if self.overruns:
            logger.warning(
                "%d input overrun(s): the pipeline could not keep up with real time. "
                "Increase --chunk-ms, reduce --context-sec, or move to GPU.",
                self.overruns,
            )

    def _to_target_rate(self, block: np.ndarray) -> np.ndarray:
        if self._resampler is None:
            return block
        import torch

        with torch.inference_mode():
            tensor = torch.from_numpy(block).unsqueeze(0)
            return self._resampler(tensor).squeeze(0).numpy().astype(np.float32)

    def stop(self) -> None:
        """Ask the capture loop to finish at the next chunk boundary."""
        self._stopped = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def close(self) -> None:
        self.stop()
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:  # pragma: no cover
                pass
            self._stream = None
