"""Execution-provider setup that does not depend on torch.

Two decisions that were previously made by accident, and are made deliberately
here so that either runtime can use them.

Both take the same shape: ONNX Runtime's default is correct for one situation
and quietly wrong for another, and nothing in the default path says which one
you are in.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

_cuda_libraries_state: Optional[str] = None

#: Loaded in this order so that each library's own dependencies are already
#: resident: cuDNN 9 splits into sub-libraries that depend on cuBLAS, and
#: everything depends on the CUDA runtime. Globs, because the version suffix
#: moves between CUDA releases.
_CUDA_PRELOAD_ORDER = (
    "cudart64_*.dll",
    "cublasLt64_*.dll",
    "cublas64_*.dll",
    "cufft64_*.dll",
    "curand64_*.dll",
    "cudnn_graph64_*.dll",
    "cudnn_engines_precompiled64_*.dll",
    "cudnn_engines_runtime_compiled64_*.dll",
    "cudnn_heuristic64_*.dll",
    "cudnn_ops64_*.dll",
    "cudnn_adv64_*.dll",
    "cudnn_cnn64_*.dll",
    "cudnn64_*.dll",
)


def ensure_cuda_libraries() -> Optional[str]:
    """Make the CUDA/cuDNN DLLs discoverable, without importing torch.

    ``onnxruntime-gpu`` does not ship the CUDA runtime; it loads ``cublas`` and
    ``cudnn`` from the OS search path. A pip-installed CUDA build of torch does
    ship them, in ``torch/lib``, and registers that directory as a side effect
    of ``import torch``.

    Relying on that side effect is how this project ended up with a service
    whose GPU access depended on an unused import (``PROJECT_REPORT.md`` §5.4):
    torch cost 350 MB of RSS, and deleting the dead import would have silently
    dropped the service to CPU.

    ``importlib.util.find_spec`` locates the same directory **without executing
    the module**, so the libraries can be loaded on their own. Returns the path
    used, or None if there was nothing to do.

    Idempotent, and a no-op off Windows -- Linux resolves these through
    ``LD_LIBRARY_PATH`` and the ``nvidia-*`` wheels, which need no help.
    """
    global _cuda_libraries_state
    if _cuda_libraries_state is not None:
        return _cuda_libraries_state or None
    _cuda_libraries_state = ""

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return None

    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None

    lib = Path(spec.origin).parent / "lib"
    # Check for the libraries themselves rather than the directory: a CPU-only
    # torch build has torch/lib but none of what ORT is looking for, and
    # registering it would be a no-op that reads like a success in the log.
    if not any(lib.glob("cudnn64_*.dll")):
        return None

    # Registering the directory is necessary but *not* sufficient. ONNX Runtime
    # resolves these as ordinary dependencies of its own provider DLL, and that
    # goes through the standard loader search order, which ignores directories
    # added this way. Measured: with only add_dll_directory the session still
    # reports CPUExecutionProvider.
    #
    # What makes it work is loading them into the process, which is what
    # ``import torch`` was doing incidentally. Once a module is resident the
    # loader satisfies later requests for the same name from memory.
    try:
        os.add_dll_directory(str(lib))
    except OSError:
        return None

    import ctypes

    loaded = 0
    for pattern in _CUDA_PRELOAD_ORDER:
        for path in sorted(lib.glob(pattern)):
            try:
                ctypes.WinDLL(str(path))
                loaded += 1
            except OSError:
                # Version mismatches and optional components are expected; ORT
                # will report what it could not find if it matters.
                continue

    if not loaded:
        return None

    _cuda_libraries_state = str(lib)
    logger.debug(
        "Preloaded %d CUDA libraries from %s (torch not imported)", loaded, lib
    )
    return _cuda_libraries_state


def resolve_intra_op_threads(
    configured: int,
    providers: Sequence[str],
    max_concurrent_streams: int = 1,
) -> tuple[int, str]:
    """Pick an intra-op thread count. Returns ``(threads, reason)``.

    ONNX Runtime's default is every core, per inference. That is right for one
    stream and actively harmful for several: N concurrent inferences each claim
    the whole machine and spend their time fighting over it. Measured on 4
    physical cores, real checkpoint, response p50 at 4 concurrent streams:
    **6133 ms** at the default against **3412 ms** pinned to 2. At one stream
    the default wins by 250 ms, which is why this is a function and not a
    constant.

    The rules, in order:

    * an explicit positive setting always wins -- this only ever fills in a
      value nobody chose;
    * on CUDA the setting governs CPU-side ops only and the default is fine;
    * serving one stream at a time, the default is fastest;
    * otherwise divide the machine between the streams that may want it.

    Logical cores rather than physical: on the machine this was measured on
    (4 physical / 8 logical, 4 streams) that yields 2, which is the value that
    measured best. Dividing physical cores would have given 1.
    """
    if configured > 0:
        return configured, "explicitly configured"

    if any("CUDA" in p or "Tensorrt" in p or "ROCM" in p for p in providers):
        return 0, "GPU provider, ORT default"

    if max_concurrent_streams <= 1:
        return 0, "single stream, ORT default (all cores)"

    cores = os.cpu_count() or 1
    threads = max(1, cores // max_concurrent_streams)
    return threads, f"{cores} logical cores / {max_concurrent_streams} streams"
