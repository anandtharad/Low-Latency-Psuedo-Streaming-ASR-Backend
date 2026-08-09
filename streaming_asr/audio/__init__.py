"""Audio input abstractions.

``MicrophoneSource`` is imported lazily: it needs ``sounddevice``, which pulls
in a native PortAudio library. Importing the package must not fail on a
headless box that only ever reads WAV files.
"""

from streaming_asr.audio.base import AudioChunk, AudioSource
from streaming_asr.audio.wav_source import WavFileSource, InMemorySource

__all__ = [
    "AudioChunk",
    "AudioSource",
    "WavFileSource",
    "InMemorySource",
    "MicrophoneSource",
    "list_input_devices",
]


def __getattr__(name: str):
    if name in ("MicrophoneSource", "list_input_devices", "MicrophoneUnavailable"):
        from streaming_asr.audio import mic_source

        return getattr(mic_source, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
