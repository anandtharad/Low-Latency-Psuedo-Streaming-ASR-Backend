"""Low-latency streaming ASR around an offline IndicConformer-CTC ONNX model.

This package implements *sliding-window / rolling-buffer pseudo-streaming*
inference. The underlying model is a full-context (offline) Conformer-CTC
model: it has no encoder cache and no stateful inference. We repeatedly run it
over overlapping windows of a rolling audio buffer, decode greedily for low
latency, stabilise the resulting hypotheses over time, and run the expensive
CTC beam search + KenLM decoder once at the speech endpoint.

See ``docs/ARCHITECTURE.md`` for the design rationale.
"""

from streaming_asr.config import StreamingASRConfig
from streaming_asr.events import ASREvent, ASREventType

__all__ = [
    "StreamingASRConfig",
    "ASREvent",
    "ASREventType",
    "StreamingASRPipeline",
    "SegmentedASRPipeline",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Import the pipelines lazily.

    They pull in torch, which costs seconds and a large amount of memory. A
    process that only needs the config or the event types -- a client, a test,
    a script that builds a request -- should not pay for that, and on a
    memory-constrained machine eagerly loading torch here is enough to fail
    outright with a paging-file error.
    """
    if name == "StreamingASRPipeline":
        from streaming_asr.pipeline import StreamingASRPipeline

        return StreamingASRPipeline
    if name == "SegmentedASRPipeline":
        from streaming_asr.segmented import SegmentedASRPipeline

        return SegmentedASRPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
