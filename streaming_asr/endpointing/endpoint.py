"""Pluggable speech-end detection.

The ASR engine must not be coupled to any particular VAD. It asks a detector
"is the utterance over?" after each chunk and does not care how the answer is
produced. Swapping the built-in energy gate for Silero, WebRTC VAD or a
server-side turn-taking model means implementing :class:`EndpointDetector`.

The built-in energy detector is deliberately unsophisticated. It is a
placeholder that makes the finalisation path exercisable end-to-end, not a
production VAD -- an RMS gate will misfire on background noise and on quiet
speech. Explicit endpointing is the default for that reason.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EndpointDecision:
    """Result of examining one chunk."""

    is_endpoint: bool
    reason: str = ""
    silence_duration: float = 0.0


class EndpointDetector(ABC):
    """Decides when an utterance has ended."""

    name: str = "base"

    @abstractmethod
    def update(self, chunk: np.ndarray, audio_time: float) -> EndpointDecision:
        """Examine one chunk of audio and report whether speech has ended."""

    def reset(self) -> None:
        """Prepare for a new utterance."""
        return None


class NullEndpointDetector(EndpointDetector):
    """Never endpoints. The stream ends only when the audio source does."""

    name = "none"

    def update(self, chunk: np.ndarray, audio_time: float) -> EndpointDecision:
        return EndpointDecision(is_endpoint=False)


class ExplicitEndpointDetector(EndpointDetector):
    """Endpoints only when the caller says so, via ``asr.end_of_speech()``.

    The default. In a push-to-talk or turn-based application the client already
    knows when the user stopped speaking, and that signal is far more reliable
    than anything inferred from the waveform.
    """

    name = "explicit"

    def __init__(self) -> None:
        self._triggered = False

    def trigger(self) -> None:
        self._triggered = True

    def reset(self) -> None:
        self._triggered = False

    def update(self, chunk: np.ndarray, audio_time: float) -> EndpointDecision:
        if self._triggered:
            return EndpointDecision(is_endpoint=True, reason="explicit")
        return EndpointDecision(is_endpoint=False)


class EnergyVADEndpointDetector(EndpointDetector):
    """RMS-energy silence detector with a hangover.

    Args:
        silence_duration: Continuous silence required to declare the endpoint.
        energy_threshold: RMS below this counts as silence.
        min_speech_duration: Suppress endpointing until this much speech has
            been observed, so a pause before the speaker starts is not mistaken
            for the end of an utterance.
        sample_rate: Used to convert chunk sizes to durations.
    """

    name = "energy"

    def __init__(
        self,
        silence_duration: float = 0.8,
        energy_threshold: float = 0.005,
        min_speech_duration: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        self.silence_duration = silence_duration
        self.energy_threshold = energy_threshold
        self.min_speech_duration = min_speech_duration
        self.sample_rate = sample_rate
        self._silence_accum = 0.0
        self._speech_accum = 0.0

    def reset(self) -> None:
        self._silence_accum = 0.0
        self._speech_accum = 0.0

    def update(self, chunk: np.ndarray, audio_time: float) -> EndpointDecision:
        duration = len(chunk) / self.sample_rate
        rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))) if chunk.size else 0.0

        if rms >= self.energy_threshold:
            self._speech_accum += duration
            self._silence_accum = 0.0
            return EndpointDecision(is_endpoint=False, silence_duration=0.0)

        self._silence_accum += duration
        if (
            self._speech_accum >= self.min_speech_duration
            and self._silence_accum >= self.silence_duration
        ):
            return EndpointDecision(
                is_endpoint=True,
                reason=f"silence for {self._silence_accum:.2f}s",
                silence_duration=self._silence_accum,
            )
        return EndpointDecision(is_endpoint=False, silence_duration=self._silence_accum)


class CompositeEndpointDetector(EndpointDetector):
    """Fires when any wrapped detector fires.

    Lets an explicit client signal coexist with a VAD safety net.
    """

    name = "composite"

    def __init__(self, *detectors: EndpointDetector) -> None:
        self.detectors = list(detectors)

    def reset(self) -> None:
        for d in self.detectors:
            d.reset()

    def update(self, chunk: np.ndarray, audio_time: float) -> EndpointDecision:
        for detector in self.detectors:
            decision = detector.update(chunk, audio_time)
            if decision.is_endpoint:
                return EndpointDecision(
                    is_endpoint=True,
                    reason=f"{detector.name}: {decision.reason}",
                    silence_duration=decision.silence_duration,
                )
        return EndpointDecision(is_endpoint=False)


def build_endpoint_detector(config: "EndpointConfig", sample_rate: int) -> EndpointDetector:
    """Construct the configured detector.

    The explicit detector is always included, so ``end_of_speech()`` works
    regardless of which automatic detector is active.
    """
    explicit = ExplicitEndpointDetector()
    kind = config.detector.lower()

    if kind == "explicit":
        return explicit
    if kind == "none":
        return CompositeEndpointDetector(explicit, NullEndpointDetector())
    if kind == "energy":
        return CompositeEndpointDetector(
            explicit,
            EnergyVADEndpointDetector(
                silence_duration=config.silence_duration,
                energy_threshold=config.energy_threshold,
                min_speech_duration=config.min_speech_duration,
                sample_rate=sample_rate,
            ),
        )
    raise ValueError(f"Unknown endpoint detector: {config.detector!r}")
