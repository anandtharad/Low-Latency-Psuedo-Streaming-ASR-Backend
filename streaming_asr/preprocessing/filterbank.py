"""TorchAudio mel filterbank frontend, preserved from the reference pipeline.

This is a faithful port of the reference ``FilterbankFeaturesTA`` /
``create_pre_processor``. The frontend must match the one used when the model
was trained and exported -- a mismatch here degrades accuracy silently, with no
error anywhere. Do not substitute another mel implementation.

Three corrections relative to the reference notebook, all called out in the
design brief:

1. ``import torch.nn as nn`` -- the reference cell subclasses ``nn.Module``
   without importing it and only works because an earlier cell leaked the name.
2. The module is forced into ``eval()`` mode by :class:`Preprocessor`.
3. Dithering is therefore disabled at inference. ``_apply_dithering`` is gated
   on ``self.training``, so a module left in train mode would add fresh
   Gaussian noise to every window -- meaning the *same* audio would produce
   different features on each of the 25 overlapping passes, injecting
   hypothesis instability that has nothing to do with the acoustics.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio


@torch.jit.script_if_tracing
def make_seq_mask_like(
    lengths: torch.Tensor,
    like: torch.Tensor,
    time_dim: int = -1,
    valid_ones: bool = True,
) -> torch.Tensor:
    """Build a broadcastable padding mask shaped like ``like``."""
    mask = (
        torch.arange(like.shape[time_dim], device=like.device)
        .repeat(lengths.shape[0], 1)
        .lt(lengths.view(-1, 1))
    )
    for _ in range(like.dim() - mask.dim()):
        mask = mask.unsqueeze(1)
    if time_dim != -1 and time_dim != mask.dim() - 1:
        mask = mask.transpose(-1, time_dim)
    if not valid_ones:
        mask = ~mask
    return mask


class FilterbankFeaturesTA(nn.Module):
    """Log-mel filterbank features via ``torchaudio.transforms.MelSpectrogram``.

    Pipeline: dither (train only) -> pre-emphasis -> STFT -> mel -> log ->
    per-feature normalisation -> optional pad.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_window_size: int = 320,
        n_window_stride: int = 160,
        normalize: Optional[str] = "per_feature",
        nfilt: int = 64,
        n_fft: Optional[int] = None,
        preemph: float = 0.97,
        lowfreq: float = 0,
        highfreq: Optional[float] = None,
        log: bool = True,
        log_zero_guard_type: str = "add",
        log_zero_guard_value: Union[float, str] = 2 ** -24,
        dither: float = 1e-5,
        window: str = "hann",
        pad_to: int = 0,
        pad_value: float = 0.0,
        mel_norm: str = "slaney",
        # Deprecated arguments; kept for config compatibility with NeMo.
        use_grads: bool = False,
        max_duration: float = 16.7,
        frame_splicing: int = 1,
        exact_pad: bool = False,
        nb_augmentation_prob: float = 0.0,
        nb_max_freq: int = 4000,
        mag_power: float = 2.0,
        rng: Optional[random.Random] = None,
        stft_exact_pad: bool = False,
        stft_conv: bool = False,
    ) -> None:
        super().__init__()

        supported_log_zero_guard_strings = {"eps", "tiny"}
        if isinstance(log_zero_guard_value, str) and log_zero_guard_value not in supported_log_zero_guard_strings:
            raise ValueError(
                f"Log zero guard value must either be a float or a member of "
                f"{supported_log_zero_guard_strings}"
            )

        self.torch_windows = {"hann": torch.hann_window}
        if window not in self.torch_windows:
            raise ValueError(
                f"Got window value '{window}' but expected a member of {self.torch_windows.keys()}"
            )

        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self._sample_rate = sample_rate
        self._normalize_strategy = normalize
        self._use_log = log
        self._preemphasis_value = preemph
        self.log_zero_guard_type = log_zero_guard_type
        self.log_zero_guard_value: Union[str, float] = log_zero_guard_value
        self.dither = dither
        self.pad_to = pad_to
        self.pad_value = pad_value
        self.n_fft = n_fft
        self.nfilt = nfilt

        self._mel_spec_extractor: torchaudio.transforms.MelSpectrogram = (
            torchaudio.transforms.MelSpectrogram(
                sample_rate=self._sample_rate,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=nfilt,
                window_fn=self.torch_windows[window],
                mel_scale="slaney",
                norm=mel_norm,
                n_fft=n_fft,
                f_max=highfreq,
                f_min=lowfreq,
                wkwargs={"periodic": False},
            )
        )

    @property
    def filter_banks(self) -> torch.Tensor:
        """Matches the analogous NeMo class."""
        return self._mel_spec_extractor.mel_scale.fb

    def _resolve_log_zero_guard_value(self, dtype: torch.dtype) -> float:
        if isinstance(self.log_zero_guard_value, float):
            return self.log_zero_guard_value
        return getattr(torch.finfo(dtype), self.log_zero_guard_value)

    def _apply_dithering(self, signals: torch.Tensor) -> torch.Tensor:
        if self.training and self.dither > 0.0:
            noise = torch.randn_like(signals) * self.dither
            signals = signals + noise
        return signals

    def _apply_preemphasis(self, signals: torch.Tensor) -> torch.Tensor:
        if self._preemphasis_value is not None:
            padded = torch.nn.functional.pad(signals, (1, 0))
            signals = signals - self._preemphasis_value * padded[:, :-1]
        return signals

    def _compute_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()

    def _apply_pad_to(self, features: torch.Tensor) -> torch.Tensor:
        # Only applied during training; inference needs a dynamic shape so the
        # exported model sees exactly the frames we computed.
        if not self.training or self.pad_to == 0 or features.shape[-1] % self.pad_to == 0:
            return features
        pad_length = self.pad_to - (features.shape[-1] % self.pad_to)
        return torch.nn.functional.pad(features, pad=(0, pad_length), value=self.pad_value)

    def _apply_log(self, features: torch.Tensor) -> torch.Tensor:
        if self._use_log:
            zero_guard = self._resolve_log_zero_guard_value(features.dtype)
            if self.log_zero_guard_type == "add":
                features = features + zero_guard
            elif self.log_zero_guard_type == "clamp":
                features = features.clamp(min=zero_guard)
            else:
                raise ValueError(f"Unsupported log zero guard type: '{self.log_zero_guard_type}'")
            features = features.log()
        return features

    def _extract_spectrograms(self, signals: torch.Tensor) -> torch.Tensor:
        # Complex FFT must run in single precision. ``torch.amp.autocast`` is
        # the non-deprecated spelling of the reference's
        # ``torch.cuda.amp.autocast``; numerics are identical.
        with torch.amp.autocast("cuda", enabled=False):
            features = self._mel_spec_extractor(waveform=signals)
        return features

    def _apply_normalization(
        self, features: torch.Tensor, lengths: torch.Tensor, eps: float = 1e-5
    ) -> torch.Tensor:
        # Always masked-fill, even when not normalising, for consistency.
        mask: torch.Tensor = make_seq_mask_like(
            lengths=lengths, like=features, time_dim=-1, valid_ones=False
        )
        features = features.masked_fill(mask, 0.0)
        if self._normalize_strategy is None:
            return features

        guard_value = self._resolve_log_zero_guard_value(features.dtype)
        if self._normalize_strategy in ("per_feature", "all_features"):
            # 'all_features' reduces over each sample; 'per_feature' per channel.
            reduce_dim: int | list[int] = 2
            if self._normalize_strategy == "all_features":
                reduce_dim = [1, 2]
            means = features.sum(dim=reduce_dim, keepdim=True).div(lengths.view(-1, 1, 1))
            stds = (
                features.sub(means)
                .masked_fill(mask, 0.0)
                .pow(2.0)
                .sum(dim=reduce_dim, keepdim=True)
                .div(lengths.view(-1, 1, 1) - 1)  # assume biased estimator
                .clamp(min=guard_value)           # avoid sqrt(0)
                .sqrt()
            )
            features = (features - means) / (stds + eps)
        else:
            raise ValueError(f"Unsupported norm type: '{self._normalize_strategy}'")
        return features.masked_fill(mask, 0.0)

    def forward(
        self, input_signal: torch.Tensor, length: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_lengths = self._compute_output_lengths(input_lengths=length)
        signals = self._apply_dithering(signals=input_signal)
        signals = self._apply_preemphasis(signals=signals)
        features = self._extract_spectrograms(signals=signals)
        features = self._apply_log(features=features)
        features = self._apply_normalization(features=features, lengths=feature_lengths)
        features = self._apply_pad_to(features=features)
        return features, feature_lengths


def create_pre_processor(
    sample_rate: int = 16000,
    window_size: float = 0.025,
    window_stride: float = 0.01,
    n_window_size: Optional[int] = None,
    n_window_stride: Optional[int] = None,
    window: str = "hann",
    normalize: Optional[str] = "per_feature",
    n_fft: int = 512,
    preemph: float = 0.97,
    features: int = 80,
    lowfreq: float = 0,
    highfreq: Optional[float] = None,
    log: bool = True,
    log_zero_guard_type: str = "add",
    log_zero_guard_value: float = 2 ** -24,
    dither: float = 0.00001,
    pad_to: int = 0,
    frame_splicing: int = 1,
    exact_pad: bool = False,
    pad_value: float = 0.0,
    mag_power: float = 2.0,
    rng: Optional[random.Random] = None,
    nb_augmentation_prob: float = 0.0,
    nb_max_freq: int = 4000,
    use_torchaudio: bool = False,
    mel_norm: str = "slaney",
    stft_exact_pad: bool = False,
    stft_conv: bool = False,
) -> FilterbankFeaturesTA:
    """Build the frontend from duration-based parameters, as NeMo does."""
    if window_size and n_window_size:
        raise ValueError("Received both window_size and n_window_size. Only one should be specified.")
    if window_stride and n_window_stride:
        raise ValueError("Received both window_stride and n_window_stride. Only one should be specified.")
    if window_size:
        n_window_size = int(window_size * sample_rate)
    if window_stride:
        n_window_stride = int(window_stride * sample_rate)

    return FilterbankFeaturesTA(
        sample_rate=sample_rate,
        n_window_size=n_window_size,
        n_window_stride=n_window_stride,
        window=window,
        normalize=normalize,
        n_fft=n_fft,
        preemph=preemph,
        nfilt=features,
        lowfreq=lowfreq,
        highfreq=highfreq,
        log=log,
        log_zero_guard_type=log_zero_guard_type,
        log_zero_guard_value=log_zero_guard_value,
        dither=dither,
        pad_to=pad_to,
        frame_splicing=frame_splicing,
        exact_pad=exact_pad,
        pad_value=pad_value,
        mag_power=mag_power,
        rng=rng,
        nb_augmentation_prob=nb_augmentation_prob,
        nb_max_freq=nb_max_freq,
        mel_norm=mel_norm,
        stft_exact_pad=stft_exact_pad,
        stft_conv=stft_conv,
    )


class Preprocessor:
    """Reusable, inference-mode wrapper around :class:`FilterbankFeaturesTA`.

    Built once and reused for every window. Holds a pinned length tensor to
    avoid re-allocating a one-element tensor 6.25 times per second of audio.
    """

    def __init__(self, config: "PreprocessingConfigLike", device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        self.module = create_pre_processor(
            sample_rate=config.sample_rate,
            window_size=config.window_size,
            window_stride=config.window_stride,
            window=config.window,
            normalize=config.normalize,
            n_fft=config.n_fft,
            preemph=config.preemph,
            features=config.features,
            lowfreq=config.lowfreq,
            highfreq=config.highfreq,
            log=config.log,
            log_zero_guard_type=config.log_zero_guard_type,
            log_zero_guard_value=config.log_zero_guard_value,
            dither=config.dither,
            pad_to=config.pad_to,
            pad_value=config.pad_value,
            mel_norm=config.mel_norm,
        )
        # Critical: eval() disables dithering. See module docstring.
        self.module.eval()
        self.module.to(self.device)

        self._length_cache: dict[int, torch.Tensor] = {}

    @property
    def hop_duration(self) -> float:
        """Seconds per feature frame."""
        return self.module.hop_length / self.config.sample_rate

    def _length_tensor(self, n_samples: int) -> torch.Tensor:
        cached = self._length_cache.get(n_samples)
        if cached is None:
            cached = torch.tensor([n_samples], dtype=torch.long, device=self.device)
            self._length_cache[n_samples] = cached
        return cached

    @torch.inference_mode()
    def __call__(self, waveform: np.ndarray | torch.Tensor, n_samples: Optional[int] = None):
        """Compute features for one window.

        Args:
            waveform: Shape ``(1, N)`` or ``(N,)``, float32.
            n_samples: Valid length. Defaults to the full width of ``waveform``,
                which is what the reference does -- it always declares the whole
                rolling buffer as valid, including warm-up zero padding.

        Returns:
            ``(features, feature_lengths)`` as torch tensors of shape
            ``(1, n_mels, T)`` and ``(1,)``.
        """
        if isinstance(waveform, np.ndarray):
            tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
        else:
            tensor = waveform
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32, non_blocking=True)

        length = self._length_tensor(int(n_samples if n_samples is not None else tensor.shape[1]))
        return self.module.forward(tensor, length=length)


# Structural type hint alias; avoids a circular import of config.py.
PreprocessingConfigLike = "streaming_asr.config.PreprocessingConfig"
