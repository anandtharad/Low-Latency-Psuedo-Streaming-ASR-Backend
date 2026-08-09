"""Build the right engine and pipeline for the configured runtime.

Single place where the lite/torch choice is resolved, so the CLI, the server
and any future entry point agree on it -- including the parts that are easy to
get subtly wrong: which engine class to use, and avoiding torch imports on the
lite path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from streaming_asr.config import StreamingASRConfig

logger = logging.getLogger(__name__)

DEFAULT_FRONTEND = "fixtures/frontend.onnx"


def resolve_frontend_path(config: StreamingASRConfig) -> Path:
    """Locate the frontend export, with an actionable error if it is missing."""
    path = Path(config.frontend_path or DEFAULT_FRONTEND)
    if not path.exists():
        raise FileNotFoundError(
            f"runtime='lite' needs the exported mel frontend, not found at: {path}\n"
            f"Build it once (this step needs torch; nothing afterwards does):\n"
            f"  python -m streaming_asr_lite.export_frontend --out {path}\n"
            f"Or pass --runtime torch / ASR_RUNTIME=torch to use the torchaudio "
            f"frontend instead."
        )
    return path


def build_engine(config: StreamingASRConfig, providers: Any = None) -> Any:
    """Engine for the configured runtime.

    The lite path must not use :class:`ONNXASREngine`: its constructor imports
    torch so ONNX Runtime can find CUDA libraries from ``torch/lib`` on Windows,
    which would restore the dependency this runtime exists to remove.
    """
    if config.runtime == "lite":
        from streaming_asr_lite.engine import LiteONNXEngine

        return LiteONNXEngine(
            model_path=config.onnx_model_path,
            providers=providers if providers is not None else config.providers,
            intra_op_threads=config.intra_op_threads,
        )

    from streaming_asr.inference.onnx_engine import ONNXASREngine

    return ONNXASREngine(
        model_path=config.onnx_model_path,
        providers=providers if providers is not None else config.providers,
        intra_op_threads=config.intra_op_threads,
    )


def build_pipeline(
    config: StreamingASRConfig,
    engine: Optional[Any] = None,
    final_decoder: Optional[Any] = None,
) -> Any:
    """Pipeline for the configured runtime and pipeline kind."""
    if config.runtime not in ("lite", "torch"):
        raise ValueError(f"unknown runtime {config.runtime!r}; expected lite or torch")

    if config.runtime == "lite":
        if config.pipeline != "segmented":
            raise ValueError(
                f"runtime='lite' implements the segmented pipeline only "
                f"(requested {config.pipeline!r}). Use --runtime torch for the "
                f"windowed pipeline."
            )
        from streaming_asr_lite.pipeline import LiteSegmentedPipeline

        return LiteSegmentedPipeline(
            config,
            frontend_path=resolve_frontend_path(config),
            engine=engine,
            final_decoder=final_decoder,
        )

    if config.pipeline == "segmented":
        from streaming_asr.segmented import SegmentedASRPipeline

        return SegmentedASRPipeline(config, engine=engine, final_decoder=final_decoder)

    from streaming_asr.pipeline import StreamingASRPipeline

    return StreamingASRPipeline(config, engine=engine, final_decoder=final_decoder)


def describe_runtime(config: StreamingASRConfig, engine: Any) -> str:
    """One-line summary for startup logs and /health."""
    if config.runtime == "lite":
        detail = "ONNX mel frontend, no torch"
    else:
        detail = "torchaudio mel frontend"
    return (
        f"runtime={config.runtime} ({detail}), pipeline={config.pipeline}, "
        f"providers={', '.join(engine.active_providers)}"
    )
