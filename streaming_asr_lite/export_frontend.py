"""Export the mel frontend to ONNX so the runtime does not need torch.

torch accounts for ~330 MB of resident memory and ~2 s of startup in the
deployed stack, and is used for nothing but this frontend and resampling.
Moving it into ONNX Runtime -- which already hosts the encoder -- removes that
entirely, and makes the frontend language-independent as a side effect.

Why the frontend is reimplemented rather than exported directly
--------------------------------------------------------------
``torch.onnx.export`` cannot export ``torch.stft``::

    SymbolicValueError: STFT does not currently support complex types

So the STFT is rebuilt from operators that do export: a **convolution whose
kernels are the windowed DFT basis**. Framing, windowing and the transform
collapse into one ``conv1d``, which ONNX handles natively.

    real[k, t] = sum_n  x[t*hop + n] * w[n] * cos(2*pi*k*n / N)
    imag[k, t] = sum_n  x[t*hop + n] * w[n] * -sin(2*pi*k*n / N)
    power      = real^2 + imag^2

Everything downstream of the spectrogram (mel projection, log, per-feature
normalisation) is already plain tensor arithmetic and exports unchanged.

**The window and mel filterbank are taken from a live torchaudio module rather
than recomputed.** Reproducing ``mel_scale="slaney"`` with ``norm="slaney"`` and
a ``periodic=False`` Hann window by hand is exactly the kind of near-miss that
degrades accuracy with no error raised anywhere. Copying the tensors removes
that risk by construction; the export then only has to reproduce the *dataflow*,
which :func:`verify` checks numerically.

Run once, offline. torch is needed here and nowhere else::

    python -m streaming_asr_lite.export_frontend --out fixtures/frontend.onnx
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def build_exportable_frontend(config) -> "torch.nn.Module":
    """Wrap the reference frontend's constants in export-safe operations."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from streaming_asr.preprocessing.filterbank import create_pre_processor

    reference = create_pre_processor(
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
    reference.eval()

    mel_spec = reference._mel_spec_extractor
    n_fft = int(mel_spec.spectrogram.n_fft)
    hop = int(mel_spec.spectrogram.hop_length)
    win_length = int(mel_spec.spectrogram.win_length)

    # Lifted from the live module, not recomputed. See the module docstring.
    window = mel_spec.spectrogram.window.detach().clone()          # (win_length,)
    mel_fb = mel_spec.mel_scale.fb.detach().clone()                # (n_freqs, n_mels)

    # torch.stft centres a shorter window inside the FFT frame.
    padded_window = torch.zeros(n_fft, dtype=window.dtype)
    offset = (n_fft - win_length) // 2
    padded_window[offset:offset + win_length] = window

    n_freqs = n_fft // 2 + 1
    n = torch.arange(n_fft, dtype=torch.float64)
    k = torch.arange(n_freqs, dtype=torch.float64)
    angle = 2.0 * math.pi * k.unsqueeze(1) * n.unsqueeze(0) / n_fft

    w = padded_window.to(torch.float64).unsqueeze(0)
    cos_kernel = (torch.cos(angle) * w).unsqueeze(1).float()       # (F, 1, N)
    sin_kernel = (-torch.sin(angle) * w).unsqueeze(1).float()

    class ExportableFrontend(nn.Module):
        """Log-mel frontend using only ONNX-exportable operators."""

        def __init__(self) -> None:
            super().__init__()
            self.hop = hop
            self.pad = n_fft // 2
            self.preemph = config.preemph
            self.log_guard = float(config.log_zero_guard_value)
            self.eps = 1e-5
            self.register_buffer("cos_kernel", cos_kernel)
            self.register_buffer("sin_kernel", sin_kernel)
            self.register_buffer("mel_fb", mel_fb.float())

        def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            # (B, T) -> (B, n_mels, frames)
            x = waveform
            if self.preemph is not None:
                padded = F.pad(x, (1, 0))
                x = x - self.preemph * padded[:, :-1]

            # center=True in torch.stft is reflect padding of n_fft // 2.
            x = F.pad(x.unsqueeze(1), (self.pad, self.pad), mode="reflect")

            real = F.conv1d(x, self.cos_kernel, stride=self.hop)
            imag = F.conv1d(x, self.sin_kernel, stride=self.hop)
            power = real * real + imag * imag                       # (B, F, T)

            # torchaudio applies the filterbank as (T, F) @ (F, mels).
            mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)
            mel = torch.log(mel + self.log_guard)

            # per_feature normalisation over the whole (fully valid) utterance.
            frames = mel.shape[-1]
            means = mel.mean(dim=2, keepdim=True)
            # The reference divides by (length - 1): a deliberately biased
            # estimator. Matching it matters -- at 100 frames the difference is
            # 0.5%, which is small but systematic across every feature.
            var = (mel - means).pow(2).sum(dim=2, keepdim=True) / (frames - 1)
            stds = var.clamp(min=self.log_guard).sqrt()
            return (mel - means) / (stds + self.eps)

    return ExportableFrontend().eval(), reference


def verify(exportable, reference, sample_rate: int, seed: int = 0) -> float:
    """Compare the export-safe frontend against the reference implementation.

    A frontend mismatch degrades recognition accuracy and raises nothing, so
    this is the only thing standing between a working export and a silently
    worse one.
    """
    import torch

    rng = np.random.default_rng(seed)
    worst = 0.0

    print(f"{'input':>12}{'max abs diff':>16}{'rel to std':>14}")
    print("-" * 42)
    for seconds in (0.5, 1.0, 2.0, 4.0, 8.83):
        n = int(seconds * sample_rate)
        audio = torch.from_numpy(
            (0.3 * rng.standard_normal(n)).astype(np.float32)
        ).unsqueeze(0)

        with torch.inference_mode():
            expected, _ = reference(audio, torch.tensor([n], dtype=torch.long))
            actual = exportable(audio)

        if expected.shape != actual.shape:
            raise SystemExit(
                f"shape mismatch at {seconds}s: reference {tuple(expected.shape)} "
                f"vs export {tuple(actual.shape)}"
            )
        diff = float((expected - actual).abs().max())
        worst = max(worst, diff)
        # Features are unit-variance after normalisation, so absolute error is
        # directly interpretable as a fraction of a standard deviation.
        print(f"{seconds:>10.2f}s{diff:>16.2e}{diff:>14.2e}")

    return worst


def export(module, path: Path, n_mels: int) -> None:
    import torch

    dummy = torch.randn(1, 16000)
    torch.onnx.export(
        module, (dummy,), str(path),
        input_names=["waveform"],
        output_names=["features"],
        dynamic_axes={"waveform": {0: "batch", 1: "samples"},
                      "features": {0: "batch", 2: "frames"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"\nexported -> {path} ({path.stat().st_size / 1024:.0f} KB)")


def check_onnx(path: Path, reference, sample_rate: int) -> float:
    """Run the exported graph through ONNX Runtime and re-compare."""
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    worst = 0.0

    print(f"\n{'input':>12}{'ORT vs torch':>16}")
    print("-" * 28)
    for seconds in (0.7, 3.3, 8.83):
        n = int(seconds * sample_rate)
        audio = (0.3 * rng.standard_normal(n)).astype(np.float32)[None, :]

        with torch.inference_mode():
            expected, _ = reference(
                torch.from_numpy(audio), torch.tensor([n], dtype=torch.long)
            )
        actual = session.run(None, {"waveform": audio})[0]
        diff = float(np.abs(expected.numpy() - actual).max())
        worst = max(worst, diff)
        print(f"{seconds:>10.2f}s{diff:>16.2e}")
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="fixtures/frontend.onnx")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="max acceptable absolute deviation, in units of "
                             "feature standard deviations")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    from streaming_asr.config import PreprocessingConfig

    config = PreprocessingConfig()
    print(f"frontend: {config.features} mels, {config.n_fft}-point FFT, "
          f"{config.n_window_size}/{config.n_window_stride} window/hop\n")

    exportable, reference = build_exportable_frontend(config)

    print("verifying the export-safe implementation against the reference")
    worst = verify(exportable, reference, config.sample_rate)
    if worst > args.tolerance:
        print(f"\nFAILED: worst deviation {worst:.2e} exceeds {args.tolerance:.0e}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    export(exportable, out, config.features)

    worst_ort = check_onnx(out, reference, config.sample_rate)
    if worst_ort > args.tolerance:
        print(f"\nFAILED: ONNX Runtime deviates by {worst_ort:.2e}")
        return 1

    print(f"\nOK: worst deviation {max(worst, worst_ort):.2e} "
          f"(tolerance {args.tolerance:.0e})")
    print("The runtime no longer needs torch. See streaming_asr_lite/frontend.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
