"""A torch-free runtime for the streaming ASR pipeline.

Parallel to :mod:`streaming_asr`, which remains the working reference and the
fallback. Nothing here modifies it.

The difference is one dependency. ``streaming_asr`` computes mel features with
torchaudio; this package runs an ONNX export of the same frontend through the
ONNX Runtime session that already hosts the encoder. Measured on this machine,
that removes ~330 MB of resident memory and ~2 s of import time -- torch is
~85% of the deployed footprint and is used for nothing else.

Build the frontend export first (this step needs torch, once, offline)::

    python -m streaming_asr_lite.export_frontend --out fixtures/frontend.onnx

Everything after that runs on onnxruntime + numpy + soundfile.
"""

__all__ = ["OnnxMelFrontend", "decode_audio", "resample"]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "OnnxMelFrontend":
        from streaming_asr_lite.frontend import OnnxMelFrontend

        return OnnxMelFrontend
    if name in ("decode_audio", "resample"):
        from streaming_asr_lite import audio

        return getattr(audio, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
