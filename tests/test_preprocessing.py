"""Mel frontend fidelity.

The frontend must match the one used at training/export. A mismatch degrades
accuracy silently, with no error raised anywhere, so these tests pin the shape,
the determinism and the normalisation behaviour rather than trusting them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from streaming_asr.config import PreprocessingConfig
from streaming_asr.preprocessing.filterbank import (
    Preprocessor,
    create_pre_processor,
    make_seq_mask_like,
)

SAMPLE_RATE = 16000


@pytest.fixture
def audio() -> np.ndarray:
    t = np.arange(SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_feature_shape_matches_reference_formula(audio):
    """T = floor(n_samples / hop) + 1, as ``_compute_output_lengths`` defines."""
    pre = Preprocessor(PreprocessingConfig())
    features, lengths = pre(audio.reshape(1, -1), n_samples=len(audio))

    assert features.shape[0] == 1
    assert features.shape[1] == 80                    # mel bins
    assert int(lengths[0]) == len(audio) // 160 + 1
    assert features.shape[2] == int(lengths[0])


def test_reference_buffer_geometry_gives_401_frames():
    """The 4.0 s reference window yields 401 feature frames."""
    pre = Preprocessor(PreprocessingConfig())
    buffer = np.zeros((1, 64000), dtype=np.float32)
    features, lengths = pre(buffer, n_samples=64000)
    assert int(lengths[0]) == 401
    assert features.shape[2] == 401


def test_eval_mode_is_deterministic(audio):
    """Critical: no dithering at inference.

    The reference never calls ``.eval()``, and ``_apply_dithering`` is gated on
    ``self.training``. Left in train mode, the same audio yields different
    features on each of its ~25 overlapping passes -- injecting hypothesis
    instability that has nothing to do with the acoustics.
    """
    pre = Preprocessor(PreprocessingConfig())
    assert pre.module.training is False

    first, _ = pre(audio.reshape(1, -1), n_samples=len(audio))
    second, _ = pre(audio.reshape(1, -1), n_samples=len(audio))
    torch.testing.assert_close(first, second)


def test_train_mode_would_dither(audio):
    """Demonstrates that the eval() call is load-bearing, not cosmetic."""
    module = create_pre_processor()
    module.train()
    tensor = torch.from_numpy(audio).unsqueeze(0)
    length = torch.tensor([len(audio)])

    with torch.inference_mode():
        first, _ = module.forward(tensor, length=length)
        second, _ = module.forward(tensor, length=length)

    assert not torch.allclose(first, second), "expected dithering in train mode"


def test_per_feature_normalisation(audio):
    """Each mel bin should be roughly zero-mean, unit-variance across time."""
    pre = Preprocessor(PreprocessingConfig())
    features, _ = pre(audio.reshape(1, -1), n_samples=len(audio))

    means = features[0].mean(dim=1)
    stds = features[0].std(dim=1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-3)
    assert torch.allclose(stds, torch.ones_like(stds), atol=0.05)


def test_normalisation_can_be_disabled():
    """Without per-feature normalisation the mel bins keep their native scale."""
    audio = np.random.default_rng(0).standard_normal(8000).astype(np.float32)

    raw, _ = Preprocessor(PreprocessingConfig(normalize=None))(
        audio.reshape(1, -1), n_samples=len(audio)
    )
    normalised, _ = Preprocessor(PreprocessingConfig())(
        audio.reshape(1, -1), n_samples=len(audio)
    )

    # Normalisation forces every bin to unit variance; raw log-mel does not.
    assert torch.allclose(normalised[0].std(dim=1), torch.ones(80), atol=0.05)
    assert not torch.allclose(raw[0].std(dim=1), torch.ones(80), atol=0.05)
    # Bin means also span a wide range without normalisation.
    assert float(raw[0].mean(dim=1).max() - raw[0].mean(dim=1).min()) > 3.0


def test_hop_duration_is_ten_milliseconds():
    pre = Preprocessor(PreprocessingConfig())
    assert pre.hop_duration == pytest.approx(0.01)


def test_accepts_numpy_and_torch_and_1d_and_2d(audio):
    pre = Preprocessor(PreprocessingConfig())
    from_numpy_2d, _ = pre(audio.reshape(1, -1), n_samples=len(audio))
    from_numpy_1d, _ = pre(audio, n_samples=len(audio))
    from_torch, _ = pre(torch.from_numpy(audio), n_samples=len(audio))

    torch.testing.assert_close(from_numpy_2d, from_numpy_1d)
    torch.testing.assert_close(from_numpy_2d, from_torch)


def test_length_tensor_is_cached():
    """The length tensor is reused rather than allocated per window."""
    pre = Preprocessor(PreprocessingConfig())
    first = pre._length_tensor(64000)
    second = pre._length_tensor(64000)
    assert first is second


def test_zero_padding_changes_normalisation_statistics():
    """Documents a deliberate reference-fidelity caveat.

    The pipeline declares the whole rolling buffer valid, warm-up zero padding
    included, exactly as the reference does. The padding therefore contributes
    to the per-feature statistics, so an early window is normalised differently
    from the same audio in a full window.
    """
    pre = Preprocessor(PreprocessingConfig())
    t = np.arange(16000, dtype=np.float32) / SAMPLE_RATE
    speech = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    padded = np.zeros(64000, dtype=np.float32)
    padded[-16000:] = speech

    full, _ = pre(padded.reshape(1, -1), n_samples=64000)
    valid_only, _ = pre(speech.reshape(1, -1), n_samples=16000)

    tail = full[0, :, -100:]
    head = valid_only[0, :, -100:]
    assert not torch.allclose(tail, head, atol=0.1)


def test_make_seq_mask_like():
    like = torch.zeros(2, 4, 5)
    lengths = torch.tensor([3, 5])

    valid = make_seq_mask_like(lengths, like, time_dim=-1, valid_ones=True)
    assert valid.shape == (2, 1, 5)
    assert valid[0, 0].tolist() == [True, True, True, False, False]
    assert valid[1, 0].tolist() == [True] * 5

    inverted = make_seq_mask_like(lengths, like, time_dim=-1, valid_ones=False)
    assert inverted[0, 0].tolist() == [False, False, False, True, True]


def test_rejects_conflicting_window_specification():
    with pytest.raises(ValueError, match="Only one should be specified"):
        create_pre_processor(window_size=0.025, n_window_size=400)
