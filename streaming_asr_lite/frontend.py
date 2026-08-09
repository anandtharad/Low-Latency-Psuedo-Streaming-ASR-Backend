"""Mel frontend running on ONNX Runtime. No torch.

Drop-in for :class:`streaming_asr.preprocessing.filterbank.Preprocessor`, with
the same call signature, so the pipeline code below is unchanged apart from
which class it constructs.

Verified against the torch implementation at export time to within 2.5e-04 on
unit-variance features -- 0.025% of a standard deviation. See
``export_frontend.py``; regenerate and re-verify if the frontend configuration
ever changes, because a mismatch here degrades accuracy without raising
anything.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class OnnxMelFrontend:
    """Log-mel features via an exported ONNX graph.

    Args:
        model_path: The ``frontend.onnx`` produced by ``export_frontend.py``.
        providers: ONNX Runtime providers. Sharing the encoder's providers is
            usually right -- on CUDA it keeps features in device memory.
        hop_duration: Seconds per feature frame, used by callers to convert
            frame indices to time. Must match what the frontend was exported
            with.
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: Optional[Sequence[str]] = None,
        hop_duration: float = 0.01,
    ) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"frontend not found: {path}\n"
                f"Build it once with:\n"
                f"  python -m streaming_asr_lite.export_frontend --out {path}"
            )

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path), sess_options=options,
            providers=list(providers) if providers else ["CPUExecutionProvider"],
        )
        self._input = self.session.get_inputs()[0].name
        self._hop_duration = hop_duration
        self.n_mels = self._probe_n_mels()

        logger.info(
            "ONNX mel frontend: %d mels, providers=%s",
            self.n_mels, self.session.get_providers(),
        )

    def _probe_n_mels(self) -> int:
        shape = self.session.get_outputs()[0].shape
        if isinstance(shape[1], int):
            return shape[1]
        # Dynamic: one tiny run settles it.
        probe = np.zeros((1, 4000), dtype=np.float32)
        return int(self.session.run(None, {self._input: probe})[0].shape[1])

    @property
    def hop_duration(self) -> float:
        return self._hop_duration

    def __call__(
        self, waveform: np.ndarray, n_samples: Optional[int] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute features for one utterance.

        Args:
            waveform: ``(1, N)`` or ``(N,)`` float32.
            n_samples: Accepted for signature compatibility with the torch
                preprocessor. The exported graph normalises over the whole
                input, so anything shorter than the array is not honoured --
                slice the waveform instead.

        Returns:
            ``(features, feature_lengths)`` with features ``(1, n_mels, T)``,
            matching the torch preprocessor's return shape.
        """
        audio = np.ascontiguousarray(waveform, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        if n_samples is not None and n_samples != audio.shape[1]:
            audio = audio[:, :n_samples]

        features = self.session.run(None, {self._input: audio})[0]
        lengths = np.array([features.shape[2]], dtype=np.int64)
        return features, lengths
