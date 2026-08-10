"""Execution-provider setup: thread counts and CUDA library discovery.

Both of these were previously decided by accident. The tests exist to keep
them decided on purpose -- particularly the import guards, which protect a
property that nothing else in the code makes obvious and that a single
convenience import would silently undo.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from streaming_asr_lite.execution import (
    ensure_cuda_libraries,
    resolve_intra_op_threads,
)

ROOT = Path(__file__).resolve().parents[1]

CPU = ["CPUExecutionProvider"]
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ---- thread resolution ---------------------------------------------------


def test_an_explicit_thread_count_is_never_second_guessed():
    """The resolver fills in a value nobody chose; it does not override one."""
    for providers in (CPU, CUDA):
        for streams in (1, 8):
            threads, reason = resolve_intra_op_threads(6, providers, streams)
            assert threads == 6
            assert "configured" in reason


def test_a_gpu_session_keeps_the_ort_default():
    """intra_op governs CPU-side ops only when the model runs on CUDA."""
    threads, reason = resolve_intra_op_threads(0, CUDA, 8)
    assert threads == 0
    assert "GPU" in reason


def test_one_stream_at_a_time_gets_the_whole_machine():
    """Measured: the ORT default beats any pinned value at one caller."""
    threads, reason = resolve_intra_op_threads(0, CPU, 1)
    assert threads == 0
    assert "single stream" in reason


def test_concurrent_cpu_streams_divide_the_machine():
    """The case ORT's default gets wrong: 4 streams each claiming every core."""
    cores = os.cpu_count() or 1
    threads, reason = resolve_intra_op_threads(0, CPU, 4)
    assert threads == max(1, cores // 4)
    assert str(cores) in reason


def test_more_streams_than_cores_still_leaves_a_usable_thread():
    """Never 0 by accident: 0 means 'all cores', the opposite of the intent."""
    threads, _ = resolve_intra_op_threads(0, CPU, (os.cpu_count() or 1) * 8)
    assert threads == 1


def test_the_thread_count_never_increases_with_concurrency():
    counts = [resolve_intra_op_threads(0, CPU, n)[0] for n in range(2, 17)]
    assert counts == sorted(counts, reverse=True), counts


# ---- CUDA library discovery ----------------------------------------------


def test_ensure_cuda_libraries_is_idempotent():
    """Called from every engine constructor; must not reload on each session."""
    first = ensure_cuda_libraries()
    assert ensure_cuda_libraries() == first


@pytest.mark.skipif(os.name != "nt", reason="DLL search path is Windows-only")
def test_ensure_cuda_libraries_reports_a_real_directory():
    result = ensure_cuda_libraries()
    if result is None:
        pytest.skip("no CUDA-enabled torch on this machine")
    assert Path(result).is_dir()


# ---- import guards -------------------------------------------------------
#
# These are the regression tests for PROJECT_REPORT.md 5.4: model_pool carried an
# unused `from streaming_asr.pipeline import StreamingASRPipeline` that pulled
# torch into every server process regardless of ASR_RUNTIME, and -- because
# importing torch also registers its CUDA DLLs on Windows -- was the only
# reason the service found a GPU at all. Both halves are now explicit.


def _imports_torch(module: str) -> bool:
    """Import `module` in a clean interpreter and report whether torch came."""
    code = (
        f"import sys; sys.path.insert(0, r'{ROOT}');"
        f" import {module};"
        " print('TORCH' if 'torch' in sys.modules else 'NO_TORCH')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return "NO_TORCH" not in result.stdout


def test_the_server_does_not_import_torch():
    """The lite runtime's footprint is only real if the service gets it too.

    Measured before the fix: 425 MB RSS and a torch import on every server
    process, including `ASR_RUNTIME=lite`, from one unused import.
    """
    assert not _imports_torch("streaming_asr.server.model_pool")


def test_the_server_app_does_not_import_torch():
    assert not _imports_torch("streaming_asr.server.app")


def test_the_cli_does_not_import_torch():
    """Same defect, same cause -- there it was an annotation-only import."""
    assert not _imports_torch("streaming_asr.cli")


def test_the_execution_helper_does_not_import_torch():
    """It locates torch/lib via find_spec, which must not execute the module."""
    code = (
        f"import sys; sys.path.insert(0, r'{ROOT}');"
        " from streaming_asr_lite.execution import ensure_cuda_libraries;"
        " ensure_cuda_libraries();"
        " print('TORCH' if 'torch' in sys.modules else 'NO_TORCH')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "NO_TORCH" in result.stdout, result.stdout
