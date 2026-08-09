"""Optional hardware sampling during a load run.

Pluggable and entirely optional. A benchmark on a CPU-only CI box must still
produce its latency and throughput numbers, so a missing GPU, a missing
``pynvml`` and a missing ``nvidia-smi`` are all reported as *unavailable* and
the run continues. Nothing here may raise into the load generator.

The GPU reading is the one that changes interpretation. This service shares one
ONNX Runtime session across every stream, so concurrency does not add
parallelism -- it adds contention. Utilisation pinned near 100% while latency
grows says the device is the limit; utilisation flat while latency grows says
something else is (the admission cap, the event loop, or the host).

Note that these are *device-wide* readings. On a shared machine they include
whatever else is running, which is usually what you want when sizing capacity
and is misleading if you forget it.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class ResourceMonitor(Protocol):
    """A source of periodic numeric readings."""

    name: str

    def start(self) -> None: ...

    def sample(self) -> dict[str, float]:
        """Current readings. Must never raise."""

    def stop(self) -> None: ...

    @property
    def unavailable_reason(self) -> Optional[str]:
        """Why this monitor produced nothing, or ``None`` if it worked."""


class HostMonitor:
    """CPU and system memory, via psutil."""

    name = "host"

    def __init__(self) -> None:
        self._psutil: Any = None
        self._reason: Optional[str] = None
        try:
            import psutil

            self._psutil = psutil
        except ImportError:
            self._reason = "psutil not installed (pip install psutil)"

    def start(self) -> None:
        if self._psutil is not None:
            # First call establishes the baseline and always returns 0.0.
            self._psutil.cpu_percent(interval=None)

    def sample(self) -> dict[str, float]:
        if self._psutil is None:
            return {}
        try:
            memory = self._psutil.virtual_memory()
            return {
                "cpu_percent": float(self._psutil.cpu_percent(interval=None)),
                "ram_used_mb": round(memory.used / (1024 ** 2), 1),
                "ram_percent": float(memory.percent),
            }
        except Exception as exc:  # noqa: BLE001
            self._reason = f"psutil failed: {exc}"
            return {}

    def stop(self) -> None:
        return None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._reason


class NvmlMonitor:
    """GPU utilisation and memory through NVML, the accurate source."""

    name = "gpu"

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._handle: Any = None
        self._nvml: Any = None
        self._reason: Optional[str] = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:  # noqa: BLE001 - no GPU is a normal outcome
            self._reason = f"NVML unavailable: {exc}"

    def start(self) -> None:
        return None

    def sample(self) -> dict[str, float]:
        if self._handle is None:
            return {}
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {
                "gpu_percent": float(util.gpu),
                "gpu_memory_percent": float(util.memory),
                "gpu_memory_used_mb": round(memory.used / (1024 ** 2), 1),
            }
        except Exception as exc:  # noqa: BLE001
            self._reason = f"NVML read failed: {exc}"
            self._handle = None
            return {}

    def stop(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._reason


class NvidiaSmiMonitor:
    """GPU readings by shelling out. Fallback when NVML is not importable.

    A subprocess per sample is expensive (tens of milliseconds), so this samples
    less often than NVML would and is a fallback, not a peer.
    """

    name = "gpu"

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self._binary = shutil.which("nvidia-smi")
        self._reason = None if self._binary else "nvidia-smi not on PATH"

    def start(self) -> None:
        return None

    def sample(self) -> dict[str, float]:
        if self._binary is None:
            return {}
        try:
            completed = subprocess.run(
                [self._binary, f"--id={self.device_index}",
                 "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            util, memory = completed.stdout.strip().split(",")
            return {"gpu_percent": float(util), "gpu_memory_used_mb": float(memory)}
        except Exception as exc:  # noqa: BLE001
            self._reason = f"nvidia-smi failed: {exc}"
            self._binary = None
            return {}

    def stop(self) -> None:
        return None

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._reason


def gpu_monitor(device_index: int = 0) -> ResourceMonitor:
    """Best available GPU source, or one that politely reports nothing."""
    nvml = NvmlMonitor(device_index)
    if nvml.unavailable_reason is None:
        return nvml
    smi = NvidiaSmiMonitor(device_index)
    if smi.unavailable_reason is None:
        logger.info("NVML unavailable (%s); falling back to nvidia-smi",
                    nvml.unavailable_reason)
        return smi
    return nvml  # keep NVML's reason: it is the more informative message


class ResourceSampler:
    """Samples a set of monitors on a background thread for the run's duration.

    Runs in a thread rather than on the event loop deliberately: NVML and
    ``nvidia-smi`` both block, and blocking the loop that is pacing the audio
    would corrupt the very timings the run exists to measure.
    """

    def __init__(
        self,
        monitors: Optional[list[ResourceMonitor]] = None,
        interval: float = 1.0,
        device_index: int = 0,
    ) -> None:
        self.monitors = monitors if monitors is not None else [
            HostMonitor(), gpu_monitor(device_index)
        ]
        self.interval = interval
        self._samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "ResourceSampler":
        for monitor in self.monitors:
            try:
                monitor.start()
            except Exception:  # noqa: BLE001
                pass
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="load-resource-sampler")
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        for monitor in self.monitors:
            try:
                monitor.stop()
            except Exception:  # noqa: BLE001
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            reading: dict[str, float] = {}
            for monitor in self.monitors:
                try:
                    reading.update(monitor.sample())
                except Exception:  # noqa: BLE001 - a monitor must never break a run
                    pass
            if reading:
                self._samples.append(reading)
            self._stop.wait(self.interval)

    def summary(self) -> dict[str, Any]:
        """Mean and peak per metric, plus why anything is missing."""
        unavailable = {
            monitor.name: monitor.unavailable_reason
            for monitor in self.monitors if monitor.unavailable_reason
        }
        if not self._samples:
            return {
                "samples": 0,
                "unavailable": unavailable or {"all": "no readings collected"},
            }

        keys = sorted({key for sample in self._samples for key in sample})
        result: dict[str, Any] = {"samples": len(self._samples),
                                  "interval_s": self.interval}
        for key in keys:
            values = [s[key] for s in self._samples if key in s]
            result[f"{key}_mean"] = round(sum(values) / len(values), 1)
            result[f"{key}_max"] = round(max(values), 1)
        if unavailable:
            result["unavailable"] = unavailable
        return result


def render_unavailable(summary: dict[str, Any]) -> list[str]:
    """Human-readable notes about what could not be measured."""
    lines = []
    for name, reason in (summary.get("unavailable") or {}).items():
        label = "GPU metrics unavailable" if name == "gpu" else f"{name} metrics unavailable"
        lines.append(f"{label}: {reason}")
    return lines


def wait_for_first_sample(sampler: ResourceSampler, timeout: float = 3.0) -> None:
    """Block briefly so a short run still has at least one reading."""
    deadline = time.perf_counter() + timeout
    while not sampler._samples and time.perf_counter() < deadline:
        time.sleep(0.05)
