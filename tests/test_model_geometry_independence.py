"""The pipeline must not be coupled to one model's geometry.

Every other test runs against either a stub engine or the synthetic fixture,
and the fixture deliberately mirrors the real IndicConformer: 80 mels, 4x
subsampling, 129 output units. That is a blind spot -- code that hard-codes any
of those numbers would pass the entire suite while being wrong for any other
export.

So this builds a model with deliberately different geometry (64 mels, 2x
subsampling, 40 tokens) and checks the pipeline derives what it needs instead of
assuming it. In particular the CTC frame duration -- which sets the resolution
of every token timestamp, and therefore the behaviour of the whole
time-based stabilisation layer -- must follow from the graph, not from a
constant.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from streaming_asr.audio.wav_source import InMemorySource  # noqa: E402
from streaming_asr.config import (  # noqa: E402
    PreprocessingConfig,
    StabilityConfig,
    StreamingASRConfig,
)
from streaming_asr.events import ASREventType  # noqa: E402
from streaming_asr.inference.onnx_engine import ONNXASREngine  # noqa: E402
from streaming_asr.pipeline import StreamingASRPipeline  # noqa: E402

N_MELS = 64          # not 80
SUBSAMPLE = 2        # not 4
VOCAB_SIZE = 40      # not 129


class _OddGeometryModel(nn.Module):
    """Same ONNX interface as the NeMo export, different dimensions."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(N_MELS, 96, kernel_size=5, stride=SUBSAMPLE, padding=2)
        self.out = nn.Linear(96, VOCAB_SIZE)

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv(audio_signal)).transpose(1, 2)
        logits = self.out(x)
        out_len = torch.div(length, SUBSAMPLE, rounding_mode="floor").clamp(min=1)
        frames = torch.arange(logits.shape[1], device=logits.device)
        valid = frames.unsqueeze(0) < out_len.unsqueeze(1)
        blank = torch.zeros_like(logits)
        blank[..., -1] = 30.0
        return F.log_softmax(torch.where(valid.unsqueeze(-1), logits, blank), dim=-1)


@pytest.fixture(scope="module")
def odd_model_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("odd") / "odd_model.onnx"
    torch.manual_seed(0)
    model = _OddGeometryModel().eval()
    torch.onnx.export(
        model,
        (torch.randn(1, N_MELS, 300), torch.tensor([300])),
        str(path),
        input_names=["audio_signal", "length"],
        output_names=["logprobs"],
        dynamic_axes={
            "audio_signal": {0: "b", 2: "t"},
            "length": {0: "b"},
            "logprobs": {0: "b", 1: "t"},
        },
        opset_version=17,
    )
    return str(path)


@pytest.fixture(scope="module")
def odd_engine(odd_model_path: str) -> ONNXASREngine:
    return ONNXASREngine(odd_model_path, providers="auto")


def _config(model_path: str) -> StreamingASRConfig:
    vocab = [f"▁w{i}" for i in range(VOCAB_SIZE - 1)] + ["__"]
    return StreamingASRConfig(
        chunk_duration=0.16,
        context_duration=1.84,
        onnx_model_path=model_path,
        vocabulary=vocab,
        blank_id=VOCAB_SIZE - 1,
        final_beam_decode=False,
        preprocessing=PreprocessingConfig(features=N_MELS),
        stability=StabilityConfig(stability_window=0.4, min_stable_updates=2),
    )


def test_subsampling_factor_is_detected_not_assumed(odd_engine):
    odd_engine.run(np.zeros((1, N_MELS, 300), dtype=np.float32), 300)
    assert odd_engine.subsampling_factor == SUBSAMPLE


def test_ctc_frame_duration_follows_the_graph(odd_engine):
    """Timestamp resolution must be derived; it drives the whole tracker."""
    odd_engine.run(np.zeros((1, N_MELS, 300), dtype=np.float32), 300)
    assert odd_engine.ctc_frame_duration(hop_duration=0.01) == pytest.approx(0.02)


def test_vocab_size_is_read_from_the_graph(odd_engine):
    assert odd_engine.graph_report.vocab_size == VOCAB_SIZE


def test_full_pipeline_runs_on_a_foreign_geometry(odd_model_path, odd_engine):
    config = _config(odd_model_path)
    pipeline = StreamingASRPipeline(config, engine=odd_engine)

    t = np.arange(16000 * 3, dtype=np.float32) / 16000
    audio = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    events = list(pipeline.stream(InMemorySource(audio, 16000, config.chunk_samples)))

    assert any(e.type is ASREventType.FINAL for e in events)
    assert pipeline.metrics.model_calls > 0


def test_non_80_mel_frontend_feeds_the_model(odd_model_path, odd_engine):
    """A shape mismatch would raise inside ONNX Runtime, not pass silently."""
    config = _config(odd_model_path)
    pipeline = StreamingASRPipeline(config, engine=odd_engine)

    features, lengths = pipeline.preprocessor(
        np.zeros((1, config.buffer_samples), dtype=np.float32),
        n_samples=config.buffer_samples,
    )
    assert features.shape[1] == N_MELS
    # run_torch, not run(np.asarray(...)): the frontend may be on CUDA, and
    # np.asarray cannot convert a device tensor.
    result = odd_engine.run_torch(features, int(lengths[0]))
    assert result.logits.shape[2] == VOCAB_SIZE
