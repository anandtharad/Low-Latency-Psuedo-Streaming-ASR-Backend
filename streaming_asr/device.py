"""Device selection for the preprocessing frontend and the ONNX session.

These two must be decided *together*. The frontend produces the tensor the
model consumes, so a split placement forces a round trip:

    frontend on GPU + ORT on CPU  ->  GPU -> host copy every window
    frontend on CPU + ORT on GPU  ->  host -> GPU copy every window

At the reference operating point that happens 6.25 times per second of audio,
on top of a strategy that already reprocesses every sample 25 times. So
``resolve_device`` only selects CUDA when *both* torch and ONNX Runtime can use
it; if either cannot, both stay on CPU and the handoff stays local.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimePlacement:
    """Where the frontend and the model will actually run."""

    torch_device: str            # "cpu" or "cuda:N"
    providers: list[str]         # ONNX Runtime execution providers, in order
    device_id: int = 0
    reason: str = ""

    @property
    def on_cuda(self) -> bool:
        return self.torch_device.startswith("cuda")

    @property
    def zero_copy_possible(self) -> bool:
        """True when features can go frontend -> model without touching host."""
        return self.on_cuda and self.providers and "CUDA" in self.providers[0]

    def describe(self) -> str:
        return (
            f"frontend={self.torch_device}, providers={self.providers}"
            + (f" ({self.reason})" if self.reason else "")
        )


def cuda_available_torch() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_available_onnxruntime() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def resolve_device(
    device: str = "auto",
    providers: str | list[str] = "auto",
    device_id: int = 0,
) -> RuntimePlacement:
    """Decide where the frontend and the model run.

    Args:
        device: ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"cuda:N"``. ``"auto"``
            picks CUDA only when torch *and* ONNX Runtime both support it.
        providers: ``"auto"`` derives providers from the resolved device;
            anything else is passed through and overrides that choice.
        device_id: GPU ordinal when ``device`` does not name one.

    Raises:
        RuntimeError: if CUDA was requested explicitly but is unavailable.
            Silently downgrading would turn a performance problem into an
            invisible one.
    """
    requested = (device or "auto").lower().strip()

    if requested.startswith("cuda"):
        if ":" in requested:
            device_id = int(requested.split(":", 1)[1])
        torch_ok, ort_ok = cuda_available_torch(), cuda_available_onnxruntime()
        if not torch_ok and not ort_ok:
            raise RuntimeError(
                "device='cuda' requested but neither torch nor ONNX Runtime sees a "
                "CUDA device. Install onnxruntime-gpu (not onnxruntime) and a CUDA "
                "build of torch, or pass device='cpu'."
            )
        if not ort_ok:
            raise RuntimeError(
                "device='cuda' requested but ONNX Runtime has no CUDAExecutionProvider. "
                "You most likely have the CPU-only 'onnxruntime' package installed; "
                "replace it with 'onnxruntime-gpu'."
            )
        if not torch_ok:
            # ORT can use the GPU but the frontend cannot. Still worth it: the
            # model is far more expensive than the mel frontend.
            logger.warning(
                "ONNX Runtime will use CUDA but torch has no CUDA device; the mel "
                "frontend stays on CPU and features are copied to the GPU each window."
            )
            return _placement("cpu", providers, device_id, "ORT-only CUDA",
                              use_cuda_providers=True)
        return _placement(f"cuda:{device_id}", providers, device_id, "explicit",
                          use_cuda_providers=True)

    if requested == "cpu":
        # CPU means CPU for both halves, even where a GPU is present.
        return _placement("cpu", providers, device_id, "explicit",
                          use_cuda_providers=False)

    if requested != "auto":
        raise ValueError(f"Unknown device {device!r}; expected auto, cpu, cuda or cuda:N")

    # auto
    if cuda_available_torch() and cuda_available_onnxruntime():
        return _placement(f"cuda:{device_id}", providers, device_id,
                          "auto-detected CUDA", use_cuda_providers=True)
    if cuda_available_onnxruntime():
        return _placement("cpu", providers, device_id,
                          "auto: ORT CUDA, CPU frontend", use_cuda_providers=True)

    logger.info(
        "No CUDA runtime detected; running on CPU. Rolling-window inference "
        "reprocesses each sample %s, so expect a much worse RTF than on GPU.",
        "many times over",
    )
    return _placement("cpu", providers, device_id, "auto: no CUDA")


def _placement(
    torch_device: str,
    providers: str | list[str],
    device_id: int,
    reason: str,
    use_cuda_providers: bool = False,
) -> RuntimePlacement:
    """Build a placement.

    ``use_cuda_providers`` is passed explicitly rather than inferred from
    global CUDA availability: ``device="cpu"`` must mean CPU for the session
    too, even on a machine where ONNX Runtime *could* use the GPU. Inferring it
    made ``--device cpu`` produce a CPU frontend feeding a CUDA session -- the
    split placement this module exists to prevent.
    """
    if providers != "auto":
        resolved = [providers] if isinstance(providers, str) else list(providers)
    elif use_cuda_providers and cuda_available_onnxruntime():
        resolved = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        resolved = ["CPUExecutionProvider"]
    return RuntimePlacement(
        torch_device=torch_device, providers=resolved, device_id=device_id, reason=reason
    )


def gpu_name(device_id: int = 0) -> Optional[str]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(device_id)
    except Exception:
        pass
    return None
