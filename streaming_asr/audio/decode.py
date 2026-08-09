"""Container/codec decoding for audio arriving from anywhere.

Format is detected from the **content**, never the filename. A `.wav`
extension on an MP3 decodes fine, and so does the reverse -- clients mislabel
uploads constantly and there is no reason to trust them.

Two tiers:

1. **libsndfile** (via ``soundfile``): WAV, FLAC, OGG/Vorbis, AIFF, CAF, W64,
   RF64 and -- since libsndfile 1.1 -- MP3. Covers most of what a backend
   receives, with no external process.
2. **ffmpeg**, if the binary is on PATH: everything else. This tier is not
   optional in practice. A browser recording with ``MediaRecorder`` produces
   **WebM/Opus** by default and iOS produces **m4a/AAC**; libsndfile decodes
   neither, so a service without ffmpeg rejects the two most likely sources of
   real user audio.

When both tiers fail the error names the formats that *are* available, so the
caller learns what to send instead of just being told no.
"""

from __future__ import annotations

import functools
import io
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

AudioInput = Union[bytes, str, Path]

#: Decoded audio longer than this is almost certainly a mistake or an attack.
MAX_DECODE_SECONDS = 3600.0


class UnsupportedAudioError(ValueError):
    """The bytes are not audio, are corrupt, or use an undecodable codec."""


@functools.lru_cache(maxsize=1)
def libsndfile_formats() -> tuple[str, ...]:
    try:
        import soundfile as sf

        return tuple(sorted(sf.available_formats().keys()))
    except Exception:  # pragma: no cover - soundfile is a hard dependency
        return ()


@functools.lru_cache(maxsize=1)
def ffmpeg_path() -> str | None:
    """Locate ffmpeg once; the answer cannot change during a process."""
    return shutil.which("ffmpeg")


def ffmpeg_available() -> bool:
    return ffmpeg_path() is not None


def describe_support() -> dict[str, object]:
    """What this deployment can actually decode -- surfaced via ``/info``."""
    return {
        "libsndfile_formats": list(libsndfile_formats()),
        "ffmpeg_available": ffmpeg_available(),
        "notes": (
            "Format is sniffed from content, not the file extension. "
            + ("ffmpeg present: m4a/AAC, WebM/Opus and other container formats "
               "are accepted."
               if ffmpeg_available() else
               "ffmpeg NOT found: m4a/AAC and WebM/Opus uploads will be "
               "rejected. Install ffmpeg to accept browser and iOS recordings.")
        ),
    }


# ---------------------------------------------------------------------------


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return data
    return data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    import torch
    import torchaudio

    logger.debug("Resampling %d Hz -> %d Hz", source_rate, target_rate)
    resampler = torchaudio.transforms.Resample(orig_freq=source_rate, new_freq=target_rate)
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
        return resampler(tensor).squeeze(0).numpy()


def _decode_soundfile(source: AudioInput, target_rate: int) -> np.ndarray:
    import soundfile as sf

    handle = io.BytesIO(source) if isinstance(source, bytes) else str(source)
    data, rate = sf.read(handle, dtype="float32", always_2d=True)
    return _resample(_to_mono(data), rate, target_rate)


def _decode_ffmpeg(source: AudioInput, target_rate: int) -> np.ndarray:
    """Shell out to ffmpeg, asking for exactly the layout we need.

    ffmpeg does the downmix and resample itself, so nothing further is needed
    and no intermediate file is written.
    """
    binary = ffmpeg_path()
    if binary is None:  # pragma: no cover - guarded by the caller
        raise UnsupportedAudioError("ffmpeg is not available")

    is_bytes = isinstance(source, bytes)
    command = [
        binary, "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0" if is_bytes else str(source),
        "-f", "f32le",          # raw little-endian float32 on stdout
        "-ac", "1",             # mono
        "-ar", str(target_rate),
        "pipe:1",
    ]

    try:
        completed = subprocess.run(
            command,
            input=source if is_bytes else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise UnsupportedAudioError("ffmpeg timed out decoding the audio") from exc

    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", "replace").strip()[:200]
        raise UnsupportedAudioError(f"ffmpeg could not decode the audio: {detail}")

    return np.frombuffer(completed.stdout, dtype="<f4").astype(np.float32)


def decode(source: AudioInput, target_sample_rate: int = 16000) -> np.ndarray:
    """Decode audio bytes or a file path to mono float32 at the target rate.

    Args:
        source: Encoded audio bytes, or a path to a file.
        target_sample_rate: Rate to resample to. The model's rate.

    Returns:
        1-D float32 array in [-1, 1].

    Raises:
        UnsupportedAudioError: not audio, corrupt, or an undecodable codec.
            The message names what this deployment can decode.
    """
    if isinstance(source, bytes) and not source:
        raise UnsupportedAudioError("the uploaded file is empty")
    if not isinstance(source, bytes) and not Path(source).exists():
        raise FileNotFoundError(f"audio file not found: {source}")

    first_error: Exception | None = None
    try:
        samples = _decode_soundfile(source, target_sample_rate)
    except Exception as exc:
        first_error = exc
        if ffmpeg_available():
            logger.info("libsndfile could not decode this input; trying ffmpeg")
            samples = _decode_ffmpeg(source, target_sample_rate)
        else:
            raise UnsupportedAudioError(
                f"could not decode the audio ({exc}). "
                f"Supported without ffmpeg: {', '.join(libsndfile_formats())}. "
                f"Install ffmpeg to also accept m4a/AAC, WebM/Opus and other "
                f"container formats."
            ) from exc

    if samples.size == 0:
        raise UnsupportedAudioError("the audio decoded to zero samples")

    duration = samples.size / target_sample_rate
    if duration > MAX_DECODE_SECONDS:
        raise UnsupportedAudioError(
            f"decoded audio is {duration / 60:.1f} minutes, above the "
            f"{MAX_DECODE_SECONDS / 60:.0f} minute limit"
        )

    if first_error is not None:
        logger.info("decoded via ffmpeg fallback (%.2fs)", duration)

    return np.ascontiguousarray(samples, dtype=np.float32)
