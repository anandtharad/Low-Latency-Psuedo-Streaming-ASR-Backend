"""Device placement: frontend and ONNX session must be decided together."""

from __future__ import annotations

import numpy as np
import pytest

from streaming_asr.device import (
    RuntimePlacement,
    cuda_available_onnxruntime,
    cuda_available_torch,
    resolve_device,
)

CUDA_READY = cuda_available_torch() and cuda_available_onnxruntime()
requires_cuda = pytest.mark.skipif(not CUDA_READY, reason="no CUDA runtime available")


def test_cpu_means_cpu_for_the_session_too():
    """device='cpu' must not leave the session on CUDA.

    Deriving providers from global CUDA availability rather than from the
    requested device produced a CPU frontend feeding a CUDA session -- a host
    copy of the features on every window, which is the split placement this
    module exists to prevent.
    """
    placement = resolve_device("cpu")
    assert placement.torch_device == "cpu"
    assert placement.providers == ["CPUExecutionProvider"]
    assert not placement.on_cuda
    assert not placement.zero_copy_possible


def test_auto_never_splits_the_placement_towards_a_gpu_frontend():
    """A GPU frontend feeding a CPU session would copy features every window."""
    placement = resolve_device("auto")
    if placement.torch_device.startswith("cuda"):
        # Frontend on GPU is only chosen when the session is on GPU too.
        assert any("CUDA" in p or "Tensorrt" in p for p in placement.providers)


def test_explicit_cuda_raises_when_unavailable():
    """Silently downgrading to CPU turns a 10x slowdown into an invisible one."""
    if CUDA_READY:
        pytest.skip("CUDA is available here, so the failure path cannot be exercised")
    with pytest.raises(RuntimeError, match="onnxruntime-gpu|CUDA device"):
        resolve_device("cuda")


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unknown device"):
        resolve_device("tpu")


def test_explicit_providers_override_device_derivation():
    placement = resolve_device("cpu", providers=["CPUExecutionProvider"])
    assert placement.providers == ["CPUExecutionProvider"]


def test_device_ordinal_is_parsed():
    if not CUDA_READY:
        pytest.skip("needs CUDA to resolve cuda:N")
    assert resolve_device("cuda:0").device_id == 0


def test_zero_copy_flag_requires_both_sides_on_cuda():
    cpu_both = RuntimePlacement("cpu", ["CPUExecutionProvider"])
    split = RuntimePlacement("cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"])
    both = RuntimePlacement("cuda:0", ["CUDAExecutionProvider", "CPUExecutionProvider"])

    assert not cpu_both.zero_copy_possible
    assert not split.zero_copy_possible          # frontend on CPU -> host copy
    assert both.zero_copy_possible


# ---- CUDA-only behaviour --------------------------------------------------


@requires_cuda
def test_auto_selects_cuda_when_both_runtimes_support_it():
    placement = resolve_device("auto")
    assert placement.on_cuda
    assert placement.zero_copy_possible


@requires_cuda
def test_frontend_produces_cuda_tensors():
    import torch

    from streaming_asr.config import PreprocessingConfig
    from streaming_asr.preprocessing.filterbank import Preprocessor

    pre = Preprocessor(PreprocessingConfig(), device="cuda:0")
    features, lengths = pre(np.zeros((1, 16000), dtype=np.float32), n_samples=16000)

    assert features.is_cuda
    assert isinstance(int(lengths[0]), int)


@requires_cuda
def test_gpu_and_cpu_produce_equivalent_features():
    """Placement must not meaningfully change the numbers.

    cuFFT and the CPU FFT are not bit-identical in float32, so an exact match
    is the wrong assertion. What matters is that the difference is far below
    anything acoustically meaningful: these are per-feature normalised features
    with unit variance, so a 1e-3 deviation is ~0.1% of a standard deviation.
    Measured worst case here is ~1.3e-4.
    """
    from streaming_asr.config import PreprocessingConfig
    from streaming_asr.preprocessing.filterbank import Preprocessor

    audio = (0.2 * np.sin(
        2 * np.pi * 440 * np.arange(16000, dtype=np.float32) / 16000
    )).astype(np.float32)

    cpu, _ = Preprocessor(PreprocessingConfig(), device="cpu")(
        audio.reshape(1, -1), n_samples=len(audio)
    )
    gpu, _ = Preprocessor(PreprocessingConfig(), device="cuda:0")(
        audio.reshape(1, -1), n_samples=len(audio)
    )

    max_deviation = float((cpu - gpu.cpu()).abs().max())
    assert max_deviation < 1e-3, f"features diverge by {max_deviation:.2e}"
