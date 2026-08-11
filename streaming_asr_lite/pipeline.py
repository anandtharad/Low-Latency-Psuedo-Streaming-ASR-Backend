"""Pause-segmented pipeline, torch-free.

The segmentation loop lives in :mod:`streaming_asr.segmented` and is shared.
This module supplies only the two things that actually differ in this runtime:
an ONNX mel frontend in place of the torchaudio one, and an engine that has no
``run_torch``.

It used to carry a full copy of that loop -- roughly 400 duplicated lines --
because ``segmented.py`` imported the torchaudio preprocessor at module scope
and so could not be imported at all without torch. Making the preprocessor
injectable, and that one import lazy, is the whole of what the split was
buying. A fix now lands in one place instead of needing to be mirrored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from streaming_asr.config import StreamingASRConfig
from streaming_asr.decoding.beam_ctc_lm import FinalDecoder
from streaming_asr.segmented import SegmentedASRPipeline
from streaming_asr_lite.engine import LiteONNXEngine
from streaming_asr_lite.frontend import OnnxMelFrontend


class LiteSegmentedPipeline(SegmentedASRPipeline):
    """Pause-segmented ASR with no torch dependency.

    Args:
        config: Shared configuration; ``config.segmentation`` holds thresholds.
        frontend_path: The exported ``frontend.onnx``.
        engine / final_decoder: Pre-built and shared, so a server loads once.
    """

    pipeline_name = "lite-segmented"

    def __init__(
        self,
        config: StreamingASRConfig,
        frontend_path: str | Path,
        engine: Optional[LiteONNXEngine] = None,
        final_decoder: Optional[FinalDecoder] = None,
    ) -> None:
        if engine is None:
            if not config.onnx_model_path:
                raise ValueError("config.onnx_model_path is required")
            engine = LiteONNXEngine(
                model_path=config.onnx_model_path,
                providers=config.providers,
                intra_op_threads=config.intra_op_threads,
            )

        # The frontend is built here rather than by the base class because it
        # needs the engine's *resolved* providers -- sharing them is what keeps
        # features in device memory between the two sessions on CUDA.
        super().__init__(
            config,
            engine=engine,
            final_decoder=final_decoder,
            preprocessor=OnnxMelFrontend(
                frontend_path,
                providers=engine.active_providers,
                hop_duration=config.preprocessing.hop_duration,
            ),
        )
