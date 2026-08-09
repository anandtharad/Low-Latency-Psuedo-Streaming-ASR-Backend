"""Final CTC beam search + KenLM decoding.

This is the authoritative decoder. It runs **once**, at the speech endpoint --
never per chunk. At the reference operating point a chunk arrives every 160 ms
while a beam search over a 4 s window costs far more than that, so running it
per update would make the pipeline non-real-time for no benefit: the streaming
transcript is provisional and gets replaced by this result anyway.

Three backends are provided because the primary one is not installable
everywhere:

``flashlight``
    ``torchaudio.models.decoder.ctc_decoder`` + KenLM. Reproduces the reference
    configuration exactly. Requires ``flashlight-text``, which has no Windows
    wheel -- so this backend is Linux/macOS only in practice.
``pyctcdecode``
    ``pyctcdecode`` + ``kenlm``. Same LM file, different beam implementation.
``pure_python``
    Prefix beam search in NumPy with **no language model**. Always available.
    Use it to exercise the pipeline where flashlight cannot be installed; it is
    a fallback for plumbing, not an accuracy substitute for the KenLM path.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from streaming_asr.types import tokens_to_text

logger = logging.getLogger(__name__)

_NEG_INF = -float("inf")


@dataclass
class BeamDecodeResult:
    """Outcome of a final decode."""

    text: str
    tokens: list[str] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    score: float = 0.0
    timesteps: list[int] = field(default_factory=list)
    decode_time: float = 0.0
    backend: str = "unknown"
    #: True when the decode used a language model.
    used_lm: bool = False


class FinalDecoder(ABC):
    """Interface for the authoritative end-of-utterance decoder."""

    name: str = "base"
    used_lm: bool = False

    @abstractmethod
    def decode(self, logits: np.ndarray) -> BeamDecodeResult:
        """Decode ``(B, T, V)`` log-probabilities into a final transcript."""


class FlashlightBeamLMDecoder(FinalDecoder):
    """torchaudio + flashlight + KenLM. Mirrors the reference configuration."""

    name = "flashlight"
    used_lm = True

    def __init__(
        self,
        vocabulary: Sequence[str],
        lexicon_path: Optional[str],
        lm_path: Optional[str],
        beam_size: int = 50,
        beam_size_token: int = 50,
        beam_threshold: float = 20.0,
        lm_weight: float = 2.0,
        word_score: float = 0.0,
        nbest: int = 1,
    ) -> None:
        import torch  # noqa: F401  (needed for the tensor handoff below)
        from torchaudio.models.decoder import ctc_decoder

        self.vocabulary = list(vocabulary)
        self.nbest = nbest
        self.used_lm = lm_path is not None

        # blank_token and sil_token are both the appended blank, exactly as in
        # the reference. The model has no separate silence unit.
        blank = self.vocabulary[-1]
        self._decoder = ctc_decoder(
            lexicon=lexicon_path,
            nbest=nbest,
            tokens=self.vocabulary,
            blank_token=blank,
            sil_token=blank,
            beam_size_token=beam_size_token,
            lm=lm_path,
            beam_threshold=beam_threshold,
            beam_size=beam_size,
            lm_weight=lm_weight,
            word_score=word_score,
        )
        logger.info(
            "Flashlight beam decoder ready (beam=%d, lm_weight=%.2f, lm=%s, lexicon=%s)",
            beam_size, lm_weight, lm_path, lexicon_path,
        )

    def decode(self, logits: np.ndarray) -> BeamDecodeResult:
        import torch

        start = time.perf_counter()
        # flashlight requires a contiguous CPU float32 tensor.
        tensor = torch.from_numpy(np.ascontiguousarray(logits, dtype=np.float32))
        hypotheses = self._decoder(tensor)
        elapsed = time.perf_counter() - start

        if not hypotheses or not hypotheses[0]:
            return BeamDecodeResult(text="", decode_time=elapsed, backend=self.name,
                                    used_lm=self.used_lm)

        best = hypotheses[0][0]
        token_ids = [int(t) for t in best.tokens]
        tokens = list(self._decoder.idxs_to_tokens(best.tokens))
        words = list(best.words)
        # With a lexicon flashlight returns real words; without one it returns
        # no words and the transcript must be rebuilt from the tokens.
        text = " ".join(words).strip() if words else tokens_to_text(tokens)

        return BeamDecodeResult(
            text=text,
            tokens=tokens,
            token_ids=token_ids,
            words=words,
            score=float(best.score),
            timesteps=[int(t) for t in best.timesteps],
            decode_time=elapsed,
            backend=self.name,
            used_lm=self.used_lm,
        )


class PyCTCDecodeBeamLMDecoder(FinalDecoder):
    """pyctcdecode + KenLM. Alternative to flashlight with the same LM file."""

    name = "pyctcdecode"
    used_lm = True

    def __init__(
        self,
        vocabulary: Sequence[str],
        lm_path: Optional[str],
        beam_size: int = 50,
        lm_weight: float = 2.0,
        word_score: float = 0.0,
    ) -> None:
        from pyctcdecode import build_ctcdecoder

        # pyctcdecode expects the blank as "" and infers it from the label set;
        # the reference passes the model vocabulary without the appended blank.
        self.vocabulary = list(vocabulary)
        self.beam_size = beam_size
        self.used_lm = lm_path is not None
        self._decoder = build_ctcdecoder(
            self.vocabulary[:-1],
            kenlm_model_path=lm_path,
            alpha=lm_weight,
            beta=word_score,
        )

    def decode(self, logits: np.ndarray) -> BeamDecodeResult:
        start = time.perf_counter()
        # pyctcdecode wants probabilities or logits, not log-probs; softmax to
        # be safe regardless of what the export emits.
        probs = _softmax(logits[0])
        text = self._decoder.decode(probs, beam_width=self.beam_size)
        return BeamDecodeResult(
            text=text.strip(),
            words=text.split(),
            decode_time=time.perf_counter() - start,
            backend=self.name,
            used_lm=self.used_lm,
        )


class PurePythonBeamDecoder(FinalDecoder):
    """CTC prefix beam search in NumPy. No language model.

    Present so the pipeline is exercisable where flashlight cannot be built.
    It will not match the KenLM path on accuracy and is not intended to.
    """

    name = "pure_python"
    used_lm = False

    def __init__(
        self,
        vocabulary: Sequence[str],
        blank_id: Optional[int] = None,
        beam_size: int = 50,
        beam_size_token: int = 50,
    ) -> None:
        self.vocabulary = list(vocabulary)
        self.blank_id = blank_id if blank_id is not None else len(self.vocabulary) - 1
        self.beam_size = beam_size
        self.beam_size_token = beam_size_token

    def decode(self, logits: np.ndarray) -> BeamDecodeResult:
        start = time.perf_counter()
        log_probs = _ensure_log_probs(logits[0])
        n_frames, vocab_size = log_probs.shape
        top_k = min(self.beam_size_token, vocab_size)

        # prefix -> (log P(prefix ending in blank), log P(prefix ending in non-blank))
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, _NEG_INF)}

        for t in range(n_frames):
            frame = log_probs[t]
            candidates = np.argpartition(-frame, top_k - 1)[:top_k] if top_k < vocab_size \
                else np.arange(vocab_size)
            nxt: dict[tuple[int, ...], list[float]] = defaultdict(
                lambda: [_NEG_INF, _NEG_INF]
            )

            for prefix, (p_blank, p_nonblank) in beams.items():
                p_total = np.logaddexp(p_blank, p_nonblank)
                last = prefix[-1] if prefix else -1

                for c in candidates:
                    c = int(c)
                    p = float(frame[c])

                    if c == self.blank_id:
                        entry = nxt[prefix]
                        entry[0] = np.logaddexp(entry[0], p_total + p)
                    elif c == last:
                        # Repeat without an intervening blank collapses into the
                        # existing token...
                        entry = nxt[prefix]
                        entry[1] = np.logaddexp(entry[1], p_nonblank + p)
                        # ...whereas a repeat *after* a blank is a new token.
                        extended = nxt[prefix + (c,)]
                        extended[1] = np.logaddexp(extended[1], p_blank + p)
                    else:
                        extended = nxt[prefix + (c,)]
                        extended[1] = np.logaddexp(extended[1], p_total + p)

            ranked = sorted(
                nxt.items(),
                key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[: self.beam_size]
            beams = {prefix: (probs[0], probs[1]) for prefix, probs in ranked}

        best_prefix, (pb, pnb) = max(
            beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1])
        )
        tokens = [self.vocabulary[i] for i in best_prefix]
        text = tokens_to_text(tokens)

        return BeamDecodeResult(
            text=text,
            tokens=tokens,
            token_ids=list(best_prefix),
            words=text.split(),
            score=float(np.logaddexp(pb, pnb)),
            decode_time=time.perf_counter() - start,
            backend=self.name,
            used_lm=False,
        )


# ---- helpers -------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _ensure_log_probs(x: np.ndarray) -> np.ndarray:
    """Normalise to log-probabilities whether or not the export already did."""
    probe = np.exp(x[: min(8, len(x))].astype(np.float64)).sum(axis=-1)
    if np.allclose(probe, 1.0, atol=0.05):
        return x.astype(np.float64)
    shifted = x - x.max(axis=-1, keepdims=True)
    return (shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))).astype(np.float64)


def flashlight_available() -> bool:
    try:
        from torchaudio.models.decoder import ctc_decoder  # noqa: F401
    except Exception:
        return False
    # torchaudio imports the symbol lazily; constructing is the real test, but
    # that needs assets. Probe the underlying package instead.
    try:
        import flashlight.lib.text  # noqa: F401
        return True
    except Exception:
        return False


def pyctcdecode_available() -> bool:
    try:
        import pyctcdecode  # noqa: F401
        import kenlm  # noqa: F401
        return True
    except Exception:
        return False


def build_final_decoder(config: "StreamingASRConfig") -> FinalDecoder:
    """Construct the configured final decoder, resolving ``backend="auto"``.

    ``auto`` prefers the reference-faithful flashlight+KenLM path, then
    pyctcdecode, then the LM-free fallback. The chosen backend is logged
    loudly, because silently degrading to a decoder with no language model
    would make final-transcript quality look far worse than it is.
    """
    vocab = config.ensure_blank_in_vocabulary()
    beam = config.beam
    backend = beam.backend

    if backend == "auto":
        if flashlight_available() and config.lm_path:
            backend = "flashlight"
        elif pyctcdecode_available() and config.lm_path:
            backend = "pyctcdecode"
        else:
            reason = "no lm_path configured" if not config.lm_path else \
                "neither flashlight-text nor pyctcdecode+kenlm is importable"
            logger.warning(
                "Falling back to the LM-free pure-Python beam decoder (%s). "
                "Final-transcript quality will be materially worse than the "
                "reference KenLM configuration.", reason,
            )
            backend = "pure_python"

    if backend == "flashlight":
        return FlashlightBeamLMDecoder(
            vocabulary=vocab,
            lexicon_path=config.lexicon_path,
            lm_path=config.lm_path,
            beam_size=beam.beam_size,
            beam_size_token=beam.beam_size_token,
            beam_threshold=beam.beam_threshold,
            lm_weight=beam.lm_weight,
            word_score=beam.word_score,
            nbest=beam.nbest,
        )
    if backend == "pyctcdecode":
        return PyCTCDecodeBeamLMDecoder(
            vocabulary=vocab,
            lm_path=config.lm_path,
            beam_size=beam.beam_size,
            lm_weight=beam.lm_weight,
            word_score=beam.word_score,
        )
    if backend == "pure_python":
        return PurePythonBeamDecoder(
            vocabulary=vocab,
            blank_id=config.resolved_blank_id,
            beam_size=beam.beam_size,
            beam_size_token=beam.beam_size_token,
        )
    raise ValueError(f"Unknown beam decoder backend: {backend!r}")
