"""Benchmark metadata, so a number can be re-derived six months later.

A latency figure without the machine, the model and the decoder that produced
it is not a measurement, it is an anecdote. Everything recorded here is
captured automatically at run time rather than typed into a document, because
the parts that get typed are the parts that go stale -- in particular *which
execution provider actually loaded*, which on this project has already differed
from what was requested more than once.

Nothing here is required for a run to complete. Anything that cannot be
determined is recorded as ``null`` with the reason, never guessed.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: Bytes hashed from each end of a model file. Full digests of a half-gigabyte
#: checkpoint cost seconds per run for no extra confidence in practice -- this
#: is a fingerprint to detect "someone swapped the model", not a signature.
FINGERPRINT_BYTES = 4 * 1024 * 1024


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _version(module: str) -> Optional[str]:
    try:
        import importlib

        return getattr(importlib.import_module(module), "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return None


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """Size plus a head/tail digest. Enough to tell two checkpoints apart."""
    file = Path(path)
    if not file.exists():
        return {"path": str(path), "available": False,
                "note": "not readable from the machine running the benchmark"}
    size = file.stat().st_size
    digest = hashlib.sha256()
    with open(file, "rb") as handle:
        digest.update(handle.read(FINGERPRINT_BYTES))
        if size > 2 * FINGERPRINT_BYTES:
            handle.seek(-FINGERPRINT_BYTES, 2)
            digest.update(handle.read(FINGERPRINT_BYTES))
    return {
        "path": str(file),
        "available": True,
        "size_bytes": size,
        "modified": datetime.fromtimestamp(file.stat().st_mtime).isoformat(timespec="seconds"),
        "fingerprint_sha256_head_tail": digest.hexdigest()[:32],
    }


def gpu_facts() -> dict[str, Any]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return {"available": False, "reason": "nvidia-smi not on PATH"}
    try:
        completed = subprocess.run(
            [binary, "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        devices = [
            dict(zip(("name", "memory_total", "driver_version"),
                     (part.strip() for part in line.split(","))))
            for line in completed.stdout.strip().splitlines() if line.strip()
        ]
        return {"available": bool(devices), "devices": devices,
                "cuda_version": _cuda_version(binary)}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": f"nvidia-smi failed: {exc}"}


def _cuda_version(binary: str) -> Optional[str]:
    try:
        out = subprocess.run([binary], capture_output=True, text=True,
                             timeout=15, check=True).stdout
        for line in out.splitlines():
            if "CUDA Version" in line:
                return line.split("CUDA Version:")[1].strip().split()[0]
    except Exception:  # noqa: BLE001
        pass
    return None


def host_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
    }
    try:
        import psutil

        facts["cpu_cores_physical"] = psutil.cpu_count(logical=False)
        facts["cpu_cores_logical"] = psutil.cpu_count(logical=True)
        facts["ram_total_mb"] = round(psutil.virtual_memory().total / (1024 ** 2))
    except ImportError:
        import os

        facts["cpu_cores_logical"] = os.cpu_count()
        facts["note"] = "psutil not installed; core/RAM detail limited"
    return facts


def collect(
    *,
    server_url: str,
    server_facts: dict[str, Any],
    audio_fixture: str,
    audio_duration: float,
    chunk_duration: float,
    sample_rate: int,
    mode: str,
    concurrency: list[int],
    pool_size: Optional[int],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """One metadata block, embedded in every results file.

    ``server_facts`` is whatever ``/health`` and ``/info`` returned. The values
    that matter most come from there rather than from local configuration:
    the providers that *actually* loaded, the decoder that is *actually*
    attached, and the segmentation thresholds the latency anchors depend on.
    """
    health = server_facts.get("health", {}) or {}
    config = server_facts.get("config", {}) or {}
    model_path = health.get("model_path", "")

    metadata: dict[str, Any] = {
        "timestamp": timestamp(),
        "server_url": server_url,
        "model": {
            # "family" is what keeps this schema usable for RNNT later: nothing
            # downstream should have to infer the model type from a field name.
            "family": _model_family(server_facts),
            "path": model_path,
            "fingerprint": file_fingerprint(model_path) if model_path else None,
            "vocab_size": health.get("vocab_size"),
            "subsampling_factor": health.get("subsampling_factor"),
            "stateless_graph": health.get("stateless_graph"),
        },
        "runtime": {
            "runtime": health.get("runtime") or config.get("runtime"),
            "requested_device": config.get("device"),
            # The distinction this project keeps getting bitten by: what loaded,
            # not what was asked for.
            "active_providers": health.get("providers"),
            "frontend_device": health.get("frontend_device"),
            "zero_copy": health.get("zero_copy"),
            "onnxruntime": _version("onnxruntime"),
            "numpy": _version("numpy"),
            "torch": _version("torch"),
            "websockets": _version("websockets"),
            "python": sys.version.split()[0],
        },
        "decoder": {
            "backend": health.get("decoder_backend"),
            "used_lm": health.get("used_lm"),
            "final_beam_decode": config.get("final_beam_decode"),
            "greedy_decode": config.get("greedy_decode"),
        },
        "hardware": {"host": host_facts(), "gpu": gpu_facts()},
        "workload": {
            "audio_fixture": audio_fixture,
            "audio_duration_s": round(audio_duration, 3),
            "chunk_duration_ms": round(1000 * chunk_duration, 1),
            "sample_rate": sample_rate,
            "mode": mode,
            "concurrency_levels": concurrency,
            "pool_size": pool_size or health.get("max_concurrent_streams"),
        },
        "segmentation": server_facts.get("segmentation", {}),
    }
    if server_facts.get("error"):
        metadata["server_introspection_error"] = server_facts["error"]
    if extra:
        metadata.update(extra)
    return metadata


def _model_family(server_facts: dict[str, Any]) -> str:
    """Best-effort model family, for a schema that must outlive CTC.

    Derived from the graph rather than assumed: a stateful graph with cache
    inputs is a streaming/transducer model, a stateless one with a single
    logits output is what this project runs today. Reported as ``unknown``
    rather than guessed when the graph is not visible.
    """
    graph = server_facts.get("graph")
    if not graph:
        return "unknown"
    if graph.get("stateless") is False:
        return "stateful"
    return "ctc" if server_facts.get("health", {}).get("subsampling_factor") else "unknown"


def render(metadata: dict[str, Any]) -> str:
    """The header printed above every results table."""
    model = metadata.get("model", {})
    runtime = metadata.get("runtime", {})
    decoder = metadata.get("decoder", {})
    gpu = metadata.get("hardware", {}).get("gpu", {})
    workload = metadata.get("workload", {})

    devices = gpu.get("devices") or []
    gpu_line = (f"{devices[0].get('name')} ({devices[0].get('memory_total')}, "
                f"driver {devices[0].get('driver_version')}, "
                f"CUDA {gpu.get('cuda_version')})") if devices \
        else f"none ({gpu.get('reason', 'unavailable')})"

    providers = runtime.get("active_providers") or []
    lines = [
        f"  model       : {Path(str(model.get('path') or '?')).name}  "
        f"(family={model.get('family')}, vocab={model.get('vocab_size')})",
        f"  runtime     : {runtime.get('runtime')} / "
        f"{', '.join(providers) if providers else 'unknown'}",
        f"  decoder     : {decoder.get('backend')} (lm={decoder.get('used_lm')})",
        f"  gpu         : {gpu_line}",
        f"  host        : {metadata.get('hardware', {}).get('host', {}).get('platform')}",
        f"  audio       : {workload.get('audio_fixture')} "
        f"({workload.get('audio_duration_s')}s), chunk "
        f"{workload.get('chunk_duration_ms')} ms",
    ]
    if "CUDAExecutionProvider" not in providers and devices:
        lines.append(
            "  [!] a GPU is present but the service is not using CUDA -- "
            "these are CPU numbers")
    return "\n".join(lines)
