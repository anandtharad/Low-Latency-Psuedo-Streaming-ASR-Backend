"""Audio decoding and resampling without torch.

``streaming_asr.audio.decode`` resamples with ``torchaudio.transforms.Resample``,
which is the last thing outside the frontend pulling torch into the runtime.
SciPy's polyphase resampler is equivalent for this purpose and already a
transitive dependency.

Container/codec handling is unchanged -- soundfile for WAV/FLAC/OGG/MP3, ffmpeg
for m4a and WebM/Opus -- so the accepted formats are the same as the main
package.
"""

from __future__ import annotations

import functools
import io
import logging
import shutil
import subprocess
from math import gcd
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

AudioInput = Union[bytes, str, Path]
MAX_DECODE_SECONDS = 3600.0


class UnsupportedAudioError(ValueError):
    """The bytes are not audio, are corrupt, or use an undecodable codec."""


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


def resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resampling.

    ``resample_poly`` applies an anti-aliasing FIR and works on integer
    up/down ratios, which is what the gcd reduction below produces. For the
    rates that occur in practice (48k/44.1k/8k to 16k) the ratios are small and
    the filter is cheap.
    """
    if source_rate == target_rate:
        return np.ascontiguousarray(samples, dtype=np.float32)

    from scipy.signal import resample_poly

    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    logger.debug("resampling %d -> %d Hz (up=%d down=%d)",
                 source_rate, target_rate, up, down)
    return np.ascontiguousarray(
        resample_poly(samples, up, down).astype(np.float32)
    )


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]


def _decode_soundfile(source: AudioInput, target_rate: int) -> np.ndarray:
    import soundfile as sf

    handle = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    data, rate = sf.read(handle, dtype="float32", always_2d=True)
    return resample(_to_mono(data), rate, target_rate)


def _decode_ffmpeg(source: AudioInput, target_rate: int) -> np.ndarray:
    binary = ffmpeg_path()
    if binary is None:
        raise UnsupportedAudioError("ffmpeg is not available")

    is_bytes = isinstance(source, bytes)
    command = [
        binary, "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0" if is_bytes else str(source),
        "-f", "f32le", "-ac", "1", "-ar", str(target_rate), "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command, input=source if is_bytes else None,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=300, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise UnsupportedAudioError("ffmpeg timed out decoding the audio") from exc

    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", "replace").strip()[:200]
        raise UnsupportedAudioError(f"ffmpeg could not decode the audio: {detail}")
    return np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32)


def decode_audio(source: AudioInput, target_sample_rate: int = 16000) -> np.ndarray:
    """Decode audio bytes or a file to mono float32 at the target rate."""
    if isinstance(source, bytes) and not source:
        raise UnsupportedAudioError("the uploaded file is empty")
    if not isinstance(source, bytes) and not Path(source).exists():
        raise FileNotFoundError(f"audio file not found: {source}")

    try:
        samples = _decode_soundfile(source, target_sample_rate)
    except Exception as exc:
        if ffmpeg_path():
            logger.info("libsndfile could not decode this input; trying ffmpeg")
            samples = _decode_ffmpeg(source, target_sample_rate)
        else:
            raise UnsupportedAudioError(
                f"could not decode the audio ({exc}). Install ffmpeg to accept "
                f"m4a/AAC, WebM/Opus and other container formats."
            ) from exc

    if samples.size == 0:
        raise UnsupportedAudioError("the audio decoded to zero samples")
    duration = samples.size / target_sample_rate
    if duration > MAX_DECODE_SECONDS:
        raise UnsupportedAudioError(
            f"decoded audio is {duration / 60:.1f} minutes, above the limit"
        )
    return np.ascontiguousarray(samples, dtype=np.float32)
