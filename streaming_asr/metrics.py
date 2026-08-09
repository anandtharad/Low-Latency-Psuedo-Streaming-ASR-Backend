"""Latency and throughput instrumentation.

This prototype exists to optimise a conversational experience, so the numbers
that matter are latencies, not just WER. Preprocessing, inference and decoding
are timed separately -- knowing the RTF without knowing which stage owns it
gives no lever to pull.

Two distinct notions of latency are tracked and should not be conflated:

*Algorithmic latency* is how far behind the audio the transcript is, measured
in stream time. It is a property of the window geometry and the stability
policy, and it does not improve with a faster GPU.

*Compute latency* is wall-clock time spent processing a chunk. It is what a
faster GPU or a larger step size improves.

A configuration can have excellent RTF and still feel sluggish, if
``stability_window`` is holding text back.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

#: Samples retained for percentile estimation. At the reference operating point
#: a chunk arrives every 160 ms, so this is ~11 minutes of history per metric.
DEFAULT_SAMPLE_WINDOW = 4096


class BoundedSamples:
    """Timing samples with exact aggregates and a bounded memory footprint.

    A plain list grew by one float per chunk per metric, forever. That is fine
    for a file, but a long-lived WebSocket stream that never endpoints would
    accumulate ~22k entries per hour per metric and never release them -- and
    ``snapshot()`` sorts them, so emitting metrics degraded as the call went on.

    Count, sum, mean and max are tracked incrementally and stay **exact** for
    the whole session. Only the percentiles are estimated, from a sliding
    window of the most recent samples -- which is the more useful reading
    anyway: recent latency describes what callers are experiencing now, whereas
    a percentile over a multi-hour session is dominated by ancient history.
    """

    __slots__ = ("_window", "count", "total", "max")

    def __init__(self, maxlen: int = DEFAULT_SAMPLE_WINDOW) -> None:
        self._window: deque[float] = deque(maxlen=maxlen)
        self.count = 0
        self.total = 0.0
        self.max = 0.0

    def add(self, value: float) -> None:
        self._window.append(value)
        self.count += 1
        self.total += value
        if value > self.max:
            self.max = value

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def summary(self) -> dict[str, float]:
        if not self.count:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
        ordered = sorted(self._window)
        return {
            "count": self.count,          # exact, not just the window
            "mean": self.mean,            # exact
            "p50": ordered[len(ordered) // 2],
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "max": self.max,              # exact
        }

    def __len__(self) -> int:
        return self.count


@dataclass
class MetricsCollector:
    """Accumulates timing data over one streaming session."""

    # --- wall-clock anchors ---
    session_start: float = field(default_factory=time.perf_counter)
    first_audio_wall: Optional[float] = None
    first_partial_wall: Optional[float] = None
    endpoint_wall: Optional[float] = None
    final_wall: Optional[float] = None

    # --- per-chunk timings (seconds) ---
    preprocess_times: BoundedSamples = field(default_factory=BoundedSamples)
    inference_times: BoundedSamples = field(default_factory=BoundedSamples)
    greedy_decode_times: BoundedSamples = field(default_factory=BoundedSamples)
    tracker_times: BoundedSamples = field(default_factory=BoundedSamples)
    update_latencies: BoundedSamples = field(default_factory=BoundedSamples)

    # --- algorithmic latencies (stream time) ---
    #: audio_time - word.end_time at the moment each word was committed.
    stable_token_latencies: BoundedSamples = field(default_factory=BoundedSamples)

    #: Wall time to decode one closed segment: preprocessing, one forward pass
    #: over the span, and the final decoder.
    #:
    #: This is the segmented pipeline's real finalisation cost, and
    #: ``finalization_latency`` does not capture it. That metric measures
    #: endpoint -> final, but a segment closes and decodes *during* the stream,
    #: so it reads ~0 and looks free. What a caller actually waits for after
    #: speaking is ``segment_silence + segment_decode``.
    segment_decode_times: BoundedSamples = field(default_factory=BoundedSamples)
    #: Silence required to close a segment, echoed from the config so the
    #: response latency below is computable from the metrics alone.
    segment_silence: float = 0.0

    # --- counters ---
    model_calls: int = 0
    chunks_processed: int = 0
    audio_duration: float = 0.0
    finalize_time: float = 0.0
    final_decode_time: float = 0.0
    final_inference_time: float = 0.0

    # --- config echo, so a result row is self-describing ---
    config_summary: dict[str, Any] = field(default_factory=dict)

    # --- placement, so GPU figures are only reported when GPU work happened ---
    on_gpu: bool = False
    gpu_device_id: int = 0

    #: Whether audio arrived at wall-clock speed. When False the RTF is a
    #: throughput figure, not a live-latency one: feeding audio as fast as the
    #: GPU accepts it keeps the device clocked up, whereas a real caller leaves
    #: it idle between chunks and it downclocks. Measured 1.5x difference on
    #: the same audio, model and pipeline -- enough to make an unpaced RTF
    #: misleading for capacity planning.
    real_time_paced: bool = False

    # ---- recording ------------------------------------------------------

    def mark_first_audio(self) -> None:
        if self.first_audio_wall is None:
            self.first_audio_wall = time.perf_counter()

    def mark_first_partial(self) -> None:
        if self.first_partial_wall is None:
            self.first_partial_wall = time.perf_counter()

    def mark_endpoint(self) -> None:
        self.endpoint_wall = time.perf_counter()

    def mark_final(self) -> None:
        self.final_wall = time.perf_counter()

    def record_window(
        self,
        preprocess_time: float,
        inference_time: float,
        decode_time: float,
        tracker_time: float,
        chunk_capture_time: Optional[float] = None,
    ) -> None:
        self.preprocess_times.add(preprocess_time)
        self.inference_times.add(inference_time)
        self.greedy_decode_times.add(decode_time)
        self.tracker_times.add(tracker_time)
        self.model_calls += 1
        self.chunks_processed += 1
        if chunk_capture_time is not None:
            self.update_latencies.add(time.perf_counter() - chunk_capture_time)

    def record_commit(self, audio_time: float, word_end_time: float) -> None:
        self.stable_token_latencies.add(max(0.0, audio_time - word_end_time))

    def record_segment(self, seconds: float) -> None:
        """Time taken to decode one closed segment."""
        self.segment_decode_times.add(seconds)
        self.finalize_time += seconds

    @property
    def response_latency(self) -> Optional[float]:
        """What a speaker waits between falling silent and the text existing.

        ``segment_silence`` has to elapse before the segment is even recognised
        as closed, then the span is decoded. Both are real waiting, and neither
        appears in ``finalization_latency``.
        """
        if not self.segment_decode_times.count:
            return None
        return self.segment_silence + self.segment_decode_times.mean

    # ---- derived --------------------------------------------------------

    @property
    def first_partial_latency(self) -> Optional[float]:
        """Wall-clock seconds from first audio to first partial transcript."""
        if self.first_audio_wall is None or self.first_partial_wall is None:
            return None
        return self.first_partial_wall - self.first_audio_wall

    @property
    def finalization_latency(self) -> Optional[float]:
        """Wall-clock seconds from endpoint to final transcript."""
        if self.endpoint_wall is None or self.final_wall is None:
            return None
        return self.final_wall - self.endpoint_wall

    @property
    def total_compute_time(self) -> float:
        # Uses the exact running totals, not the retained sample window, so
        # RTF stays correct however long the session runs.
        return (
            self.preprocess_times.total
            + self.inference_times.total
            + self.greedy_decode_times.total
            + self.tracker_times.total
            + self.finalize_time
        )

    @property
    def rtf(self) -> float:
        """Real-time factor: compute seconds per audio second. Lower is better."""
        if self.audio_duration <= 0:
            return 0.0
        return self.total_compute_time / self.audio_duration

    @property
    def streaming_rtf(self) -> float:
        """RTF of the streaming path only, excluding finalisation."""
        if self.audio_duration <= 0:
            return 0.0
        return (self.total_compute_time - self.finalize_time) / self.audio_duration

    @property
    def average_inference_time(self) -> float:
        return self.inference_times.mean

    def gpu_stats(self, device_id: int = 0, on_gpu: bool = True) -> dict[str, Any]:
        """Best-effort GPU utilisation and memory.

        Both sources are reported when available, because they measure
        different things and neither alone is sufficient:

        * NVML gives *device-wide* memory and utilisation, which is the only
          way to see ONNX Runtime's allocations -- ORT has its own CUDA arena
          that torch knows nothing about.
        * ``torch.cuda.max_memory_allocated`` covers only the mel frontend.
          Reported on its own it reads as 0 MB on a busy GPU, which is
          actively misleading.

        Args:
            on_gpu: When False, skip entirely. A CUDA-capable torch build on a
                machine running the CPU pipeline would otherwise report GPU
                figures for work that never touched the GPU.
        """
        if not on_gpu:
            return {}

        stats: dict[str, Any] = {}
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats["gpu_utilization_pct"] = float(util.gpu)
            stats["gpu_memory_used_mb"] = round(mem.used / (1024 ** 2), 1)
            stats["gpu_memory_total_mb"] = round(mem.total / (1024 ** 2), 1)
            pynvml.nvmlShutdown()
        except Exception:
            stats["gpu_nvml"] = "unavailable (pip install pynvml for utilisation)"

        try:
            import torch

            if torch.cuda.is_available():
                stats["frontend_gpu_memory_mb"] = round(
                    torch.cuda.max_memory_allocated(device_id) / (1024 ** 2), 1
                )
        except Exception:
            pass
        return stats

    # ---- reporting ------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """A flat, JSON-serialisable view for benchmark rows and events."""
        data: dict[str, Any] = {
            "model_calls": self.model_calls,
            "chunks_processed": self.chunks_processed,
            "audio_duration": round(self.audio_duration, 3),
            "rtf": round(self.rtf, 4),
            "streaming_rtf": round(self.streaming_rtf, 4),
            "total_compute_time": round(self.total_compute_time, 4),
            "preprocess": self.preprocess_times.summary(),
            "inference": self.inference_times.summary(),
            "greedy_decode": self.greedy_decode_times.summary(),
            "tracker": self.tracker_times.summary(),
            "update_latency": self.update_latencies.summary(),
            "stable_token_latency": self.stable_token_latencies.summary(),
            "segment_decode": self.segment_decode_times.summary(),
            "finalize_time": round(self.finalize_time, 4),
            "final_decode_time": round(self.final_decode_time, 4),
            "final_inference_time": round(self.final_inference_time, 4),
        }
        if self.first_partial_latency is not None:
            data["first_partial_latency"] = round(self.first_partial_latency, 4)
        if self.finalization_latency is not None:
            data["finalization_latency"] = round(self.finalization_latency, 4)
        if self.response_latency is not None:
            data["response_latency"] = round(self.response_latency, 4)
        data.update(self.gpu_stats(self.gpu_device_id, self.on_gpu))
        data.update(self.config_summary)
        return data

    def render(self) -> str:
        s = self.snapshot()
        lines = [
            f"  first_partial_latency : {s.get('first_partial_latency', float('nan')):.4f} s",
        ]

        if self.segment_decode_times.count:
            # Segmented pipeline: report what the speaker actually waits for.
            decode = self.segment_decode_times
            lines += [
                f"  segment_decode        : mean {decode.mean * 1000:.0f} ms, "
                f"max {decode.max * 1000:.0f} ms  ({decode.count} segments)",
                f"  RESPONSE LATENCY      : {(self.response_latency or 0) * 1000:.0f} ms "
                f"= {self.segment_silence * 1000:.0f} ms silence "
                f"+ {decode.mean * 1000:.0f} ms decode",
            ]
        else:
            lines.append(
                f"  finalization_latency  : "
                f"{s.get('finalization_latency', float('nan')):.4f} s"
            )

        lines += [
            f"  average_inference_time: {self.average_inference_time * 1000:.2f} ms",
            f"  avg_preprocess_time   : {self.preprocess_times.mean * 1000:.2f} ms",
            f"  avg_greedy_decode_time: {self.greedy_decode_times.mean * 1000:.2f} ms",
            f"  avg_update_latency    : {self.update_latencies.mean * 1000:.2f} ms",
            f"  stable_token_latency  : {self.stable_token_latencies.mean:.3f} s (audio time)",
            f"  RTF                   : {self.rtf:.3f}  (streaming only: {self.streaming_rtf:.3f})"
            + ("" if self.real_time_paced else
               "  [!] audio fed as fast as possible -- optimistic; use "
               "--mode real-time for a capacity figure"),
            f"  model_calls           : {self.model_calls}",
            f"  audio_duration        : {self.audio_duration:.2f} s",
        ]
        gpu = self.gpu_stats(self.gpu_device_id, self.on_gpu)
        if gpu:
            lines.append(f"  gpu                   : {gpu}")
        return "\n".join(lines)
