"""The torch-free runtime.

Two things must hold, and the second is the whole point:

1. It produces the **same transcript** as the torch pipeline. A frontend that
   is subtly different degrades accuracy with nothing raised, so equivalence
   has to be asserted, not assumed.
2. It **never imports torch**. Easy to break by accident -- one convenience
   import anywhere in the chain silently restores a 330 MB dependency.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
MODEL = FIXTURES / "synthetic_model.onnx"
AUDIO = FIXTURES / "synthetic.wav"
VOCAB = FIXTURES / "vocabulary.txt"
FRONTEND = FIXTURES / "frontend.onnx"

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and AUDIO.exists() and VOCAB.exists() and FRONTEND.exists()),
    reason="fixtures not built; run tools/build_synthetic_fixture.py and "
           "python -m streaming_asr_lite.export_frontend",
)


# ---- the frontend matches torch ------------------------------------------


def test_onnx_frontend_matches_the_torch_frontend():
    """The reason this whole package can exist."""
    torch = pytest.importorskip("torch")

    from streaming_asr.config import PreprocessingConfig
    from streaming_asr.preprocessing.filterbank import Preprocessor
    from streaming_asr_lite.frontend import OnnxMelFrontend

    reference = Preprocessor(PreprocessingConfig())
    lite = OnnxMelFrontend(FRONTEND)

    rng = np.random.default_rng(0)
    for seconds in (0.5, 2.0, 8.83):
        audio = (0.3 * rng.standard_normal(int(seconds * 16000))).astype(np.float32)

        expected, _ = reference(audio.reshape(1, -1), n_samples=len(audio))
        actual, lengths = lite(audio.reshape(1, -1))

        assert actual.shape == tuple(expected.shape)
        assert int(lengths[0]) == actual.shape[2]
        # Features are unit-variance, so this is a fraction of a std deviation.
        deviation = float(np.abs(expected.numpy() - actual).max())
        assert deviation < 1e-3, f"{seconds}s: deviation {deviation:.2e}"


def test_frontend_reports_a_helpful_error_when_not_exported(tmp_path):
    from streaming_asr_lite.frontend import OnnxMelFrontend

    with pytest.raises(FileNotFoundError, match="export_frontend"):
        OnnxMelFrontend(tmp_path / "missing.onnx")


# ---- resampling ----------------------------------------------------------


def test_resample_preserves_duration():
    from streaming_asr_lite.audio import resample

    audio = np.sin(2 * np.pi * 440 * np.arange(8000, dtype=np.float32) / 8000)
    out = resample(audio.astype(np.float32), 8000, 16000)

    assert out.dtype == np.float32
    assert abs(len(out) / 16000 - 1.0) < 0.01      # 1 second, still 1 second


def test_resample_is_a_noop_at_the_same_rate():
    from streaming_asr_lite.audio import resample

    audio = np.ones(100, dtype=np.float32)
    np.testing.assert_array_equal(resample(audio, 16000, 16000), audio)


def test_decode_rejects_non_audio():
    from streaming_asr_lite.audio import UnsupportedAudioError, decode_audio

    with pytest.raises(UnsupportedAudioError):
        decode_audio(b"not audio at all" * 20)


# ---- the pipelines agree -------------------------------------------------


def _config():
    from streaming_asr.config import (
        BeamDecoderConfig, SegmentationConfig, StreamingASRConfig, load_vocabulary,
    )

    vocab = load_vocabulary(VOCAB)
    return StreamingASRConfig(
        onnx_model_path=str(MODEL), vocabulary=vocab, blank_id=len(vocab) - 1,
        final_beam_decode=False, providers="CPUExecutionProvider",
        beam=BeamDecoderConfig(backend="pure_python", beam_size=5),
        segmentation=SegmentationConfig(energy_threshold=0.01),
    )


def test_lite_pipeline_transcribes_identically_to_the_torch_pipeline():
    pytest.importorskip("torch")

    from streaming_asr.audio.wav_source import InMemorySource, load_wav
    from streaming_asr.inference.onnx_engine import ONNXASREngine
    from streaming_asr.segmented import SegmentedASRPipeline
    from streaming_asr_lite.engine import LiteONNXEngine
    from streaming_asr_lite.pipeline import LiteSegmentedPipeline

    audio = load_wav(AUDIO, 16000)
    # Repeat with a pause so more than one segment is exercised.
    audio = np.concatenate([audio, np.zeros(16000, dtype=np.float32), audio])
    config = _config()

    # Separate engines on purpose: the torch pipeline calls run_torch(), which
    # the lite engine deliberately does not implement.
    reference = SegmentedASRPipeline(
        config, engine=ONNXASREngine(str(MODEL), providers="CPUExecutionProvider")
    )
    list(reference.stream(InMemorySource(audio, 16000, config.chunk_samples)))

    lite = LiteSegmentedPipeline(
        config, frontend_path=FRONTEND,
        engine=LiteONNXEngine(str(MODEL), providers="CPUExecutionProvider"),
    )
    list(lite.stream(InMemorySource(audio, 16000, config.chunk_samples)))

    assert lite.transcript == reference.transcript
    assert lite.transcript.strip()


def test_lite_pipeline_emits_the_same_event_shape():
    from streaming_asr.audio.wav_source import InMemorySource, load_wav
    from streaming_asr.events import ASREventType
    from streaming_asr_lite.engine import LiteONNXEngine
    from streaming_asr_lite.pipeline import LiteSegmentedPipeline

    config = _config()
    engine = LiteONNXEngine(str(MODEL), providers="CPUExecutionProvider")
    lite = LiteSegmentedPipeline(config, frontend_path=FRONTEND, engine=engine)

    events = list(lite.stream(
        InMemorySource(load_wav(AUDIO, 16000), 16000, config.chunk_samples)
    ))
    kinds = [e.type for e in events]

    assert ASREventType.PARTIAL in kinds
    assert ASREventType.SEGMENT in kinds          # incl. the one closed in finalize()
    assert kinds[-1] is ASREventType.FINAL

    joined = " ".join(e.text for e in events
                      if e.type is ASREventType.SEGMENT and e.text)
    assert joined == lite.transcript


# ---- the point of the exercise -------------------------------------------


def test_the_lite_runtime_never_imports_torch():
    """Guards the entire reason this package exists.

    A single convenience import anywhere in the chain silently restores a
    330 MB, ~2 s dependency. Run in a subprocess because torch is already
    resident in the test process.
    """
    code = f"""
import sys
sys.path.insert(0, r"{ROOT}")

from streaming_asr_lite.pipeline import LiteSegmentedPipeline
from streaming_asr_lite.frontend import OnnxMelFrontend
from streaming_asr_lite.audio import decode_audio
from streaming_asr_lite.engine import LiteONNXEngine
from streaming_asr.config import StreamingASRConfig, load_vocabulary

vocab = load_vocabulary(r"{VOCAB}")
config = StreamingASRConfig(
    onnx_model_path=r"{MODEL}", vocabulary=vocab, blank_id=len(vocab) - 1,
    final_beam_decode=False, providers="CPUExecutionProvider",
)
engine = LiteONNXEngine(r"{MODEL}", providers="CPUExecutionProvider")
pipeline = LiteSegmentedPipeline(config, frontend_path=r"{FRONTEND}", engine=engine)

audio = decode_audio(r"{AUDIO}")
from streaming_asr.audio.wav_source import InMemorySource
events = list(pipeline.stream(InMemorySource(audio, 16000, config.chunk_samples)))

assert pipeline.transcript.strip(), "no transcript produced"
print("TORCH" if "torch" in sys.modules else "NO_TORCH")
"""
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-1500:]
    assert "NO_TORCH" in result.stdout, (
        "torch was imported by the lite runtime:\n" + result.stdout
    )
