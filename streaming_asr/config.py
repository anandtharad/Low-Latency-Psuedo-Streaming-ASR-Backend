"""Central configuration for the streaming ASR pipeline.

Every tunable lives here. Nothing in the pipeline hard-codes a path, a window
size or a decoder hyper-parameter -- the reference notebook did, and that made
the 4s/160ms operating point impossible to question. Treat the defaults below
as *the reference operating point*, not as known-optimal values.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


# The vocabulary shipped with the reference Conformer-CTC-BPE checkpoint
# (SentencePiece, 128 tokens). The reference appends "__" as the CTC blank,
# giving 129 output units. Kept here so the package is runnable without the
# .nemo checkpoint, but prefer loading the real vocabulary from a file.
REFERENCE_VOCABULARY: tuple[str, ...] = (
    '<unk>', 'e', '▁', 's', 'a', 't', 'i', 'd', '▁a', 'n', '▁the', 'l', 'y',
    'u', 'o', 'm', 'p', '▁to', '▁s', 'h', '▁p', 'r', 'er', 'k', 'c', 're',
    '▁m', 'f', 'g', '▁in', '▁i', '▁of', 'ing', 'ar', '▁f', '▁w', '▁b', 'w',
    'an', 'ed', 'in', '▁t', 'or', '▁and', '▁d', '▁on', 'b', '▁c', 'en', 'le',
    've', 'ch', 'st', '▁he', '▁is', 'll', '▁be', '▁you', 'al', 'on', 'ro',
    '▁for', 'es', '▁co', '▁it', 'ur', 'at', '▁e', '▁g', '▁re', '▁ha', 'th',
    'us', 'ra', '▁we', "▁'", '▁so', 'ent', 'ri', 'ce', '▁that', '▁at', 'it',
    'ir', '▁o', '▁ma', '▁th', 'ic', '▁sh', 'ver', '▁do', 'j', '▁mo', 'ly',
    '▁st', '▁was', '▁ho', 'tion', 'ng', 'v', 'ow', 'ight', 'ter', 'x', '▁lo',
    'vi', '▁not', '▁go', '▁with', '▁have', 'ck', '▁can', '▁what', '▁want',
    '▁no', '▁this', '▁will', '▁li', '▁his', '▁but', '▁two', '▁from', 'z',
    '▁book', '▁table', '▁out', 'q', "'",
)

#: Blank symbol appended to the vocabulary by the reference implementation.
DEFAULT_BLANK_TOKEN = "__"


@dataclass
class PreprocessingConfig:
    """Mel filterbank frontend parameters.

    These MUST match the values used when the model was trained and exported.
    The defaults reproduce the reference ``create_pre_processor()`` exactly.
    Changing any of them silently degrades accuracy without raising an error,
    so they are grouped separately to make that risk visible.
    """

    sample_rate: int = 16000
    window_size: float = 0.025          # 25 ms analysis window
    window_stride: float = 0.01         # 10 ms hop
    window: str = "hann"
    normalize: Optional[str] = "per_feature"
    n_fft: int = 512
    preemph: float = 0.97
    features: int = 80                  # number of mel bins
    lowfreq: float = 0.0
    highfreq: Optional[float] = None
    log: bool = True
    log_zero_guard_type: str = "add"
    log_zero_guard_value: float = 2 ** -24
    dither: float = 1e-5
    pad_to: int = 0
    pad_value: float = 0.0
    mel_norm: str = "slaney"

    @property
    def n_window_size(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def n_window_stride(self) -> int:
        return int(self.window_stride * self.sample_rate)

    @property
    def hop_duration(self) -> float:
        """Seconds of audio per feature frame (10 ms by default)."""
        return self.n_window_stride / self.sample_rate


@dataclass
class BeamDecoderConfig:
    """CTC beam search + KenLM parameters, mirroring the reference decoder."""

    beam_size: int = 50
    beam_size_token: int = 50
    beam_threshold: float = 20.0
    lm_weight: float = 2.0
    word_score: float = 0.0
    nbest: int = 1
    #: "flashlight" (torchaudio + KenLM, matches the reference exactly),
    #: "pyctcdecode" (kenlm via pyctcdecode) or "pure_python" (no LM, always
    #: available -- for environments where flashlight cannot be installed).
    backend: str = "auto"


@dataclass
class StabilityConfig:
    """Controls how aggressively partial hypotheses are promoted to committed."""

    #: A token is only eligible for commitment once the newest audio is at
    #: least this far ahead of the token's end time. This buys right-context,
    #: which a full-context model needs before its output settles.
    stability_window: float = 0.60

    #: Number of consecutive window hypotheses that must agree on a token
    #: before it is committed.
    min_stable_updates: int = 2

    #: Aligner used to compare successive hypotheses:
    #: "time" (default, exploits the known window offset), "prefix", "levenshtein", "dtw".
    aligner: str = "time"

    #: Two tokens are considered the same observation if their start times are
    #: within this many seconds. Should be a small multiple of the CTC frame
    #: period (40 ms for a 4x-subsampled Conformer).
    time_tolerance: float = 0.12

    def validate_against(self, context_duration: float) -> None:
        """A token must be committed before it scrolls out of the buffer.

        The reference notebook demonstrates the failure mode this guards
        against: once the utterance is longer than the buffer, early words fall
        off the left edge and vanish from the transcript entirely
        ("india versus pakistan world cup final" -> "pakistan world cup final").
        """
        commit_latency = self.stability_window
        if commit_latency >= context_duration:
            raise ValueError(
                f"stability_window ({self.stability_window}s) must be smaller than "
                f"context_duration ({context_duration}s), otherwise tokens leave the "
                f"rolling buffer before they can be committed and are lost forever."
            )
        if commit_latency > 0.5 * context_duration:
            warnings.warn(
                f"stability_window ({self.stability_window}s) exceeds half of "
                f"context_duration ({context_duration}s). Tokens have little margin "
                f"before scrolling out of the buffer; consider more context.",
                stacklevel=2,
            )


@dataclass
class SegmentationConfig:
    """Thresholds for the pause-segmented pipeline.

    Two silence thresholds, because there are two different decisions and
    conflating them is what forces word-level commitment:

    * ``segment_silence`` -- is this a safe place to cut the audio? Short; the
      caller sees a ``segment`` event with authoritative text for that span.
    * ``turn_silence`` -- has the speaker finished? Longer; the caller sees a
      ``final`` event joining the turn's segments.

    Cutting at a pause means no partial word ever sits on a boundary, which is
    what removes the need to decide whether a word was already emitted.
    """

    #: Silence that closes a segment and publishes its text.
    segment_silence: float = 0.5
    #: Silence that ends the speaker's turn.
    turn_silence: float = 1.5
    #: Hard cap on a segment with no pause in it. Keep at or below the
    #: checkpoint's training ``max_duration`` (11 s for the reference model):
    #: beyond that the single-pass decode goes out of distribution and degrades
    #: sharply.
    max_segment_duration: float = 10.0
    #: Speech shorter than this is a blip, not a segment.
    min_segment_speech: float = 0.2
    #: RMS above which a chunk counts as speech.
    energy_threshold: float = 0.005
    #: Audio retained across a cut so the next segment does not start clipped.
    speech_pad: float = 0.2

    #: Emit at most one partial per this many seconds. The pipeline raises this
    #: on its own when decoding is slower than the interval, because a partial
    #: re-decodes the whole open segment and therefore costs more as the segment
    #: grows: on a live microphone the loop otherwise falls behind, the input
    #: queue overruns, and segments arrive seconds late. Self-regulating -- when
    #: decoding is fast, partials arrive every chunk.
    min_partial_interval: float = 0.0

    def __post_init__(self) -> None:
        if self.turn_silence < self.segment_silence:
            raise ValueError(
                f"turn_silence ({self.turn_silence}s) must be at least "
                f"segment_silence ({self.segment_silence}s); a turn cannot end "
                f"before the segment inside it closes."
            )


@dataclass
class EndpointConfig:
    """Speech-end detection. Explicit by default; VAD is opt-in and pluggable."""

    #: "explicit" (caller invokes end_of_speech), "energy" (built-in RMS VAD),
    #: or "none" (never auto-endpoints).
    detector: str = "explicit"
    #: Silence required before declaring the end of an utterance.
    silence_duration: float = 0.8
    #: RMS threshold below which a chunk counts as silence (energy detector).
    energy_threshold: float = 0.005
    #: Ignore endpointing until at least this much speech has been seen.
    min_speech_duration: float = 0.5


@dataclass
class StreamingASRConfig:
    """Top-level configuration object.

    The reference operating point is ``chunk_duration=0.16`` /
    ``context_duration=3.84``, i.e. a 4.0 s window advanced 160 ms at a time.
    That is what the existing prototype does; section 34 of the design brief is
    explicit that it is not assumed optimal. ``benchmark.py`` sweeps it.
    """

    # ---- audio geometry -------------------------------------------------
    sample_rate: int = 16000
    chunk_duration: float = 0.16
    context_duration: float = 3.84

    # ---- model / assets -------------------------------------------------
    onnx_model_path: Optional[str] = None
    lexicon_path: Optional[str] = None
    lm_path: Optional[str] = None
    vocabulary: Sequence[str] = field(default_factory=lambda: list(REFERENCE_VOCABULARY))
    blank_token: str = DEFAULT_BLANK_TOKEN
    #: Index of the CTC blank. ``None`` means "last entry of the vocabulary",
    #: which is what the reference does after appending ``blank_token``.
    blank_id: Optional[int] = None

    # ---- pipeline -------------------------------------------------------
    #: "segmented" cuts at pauses and decodes each span whole (default);
    #: "windowed" is the rolling-buffer pipeline with word-level commitment,
    #: kept for comparison. See docs/ARCHITECTURE.md for why segmented won.
    pipeline: str = "segmented"

    # ---- runtime --------------------------------------------------------
    #: "lite" runs the mel frontend as an ONNX graph and needs no torch:
    #: measured 68 MB / 0.28 s startup against 425 MB / 2.52 s, with identical
    #: transcripts. "torch" is the original torchaudio frontend, kept as a
    #: fallback and as the reference the lite frontend is verified against.
    runtime: str = "lite"
    #: The frontend export used by ``runtime="lite"``. Build it once with
    #: ``python -m streaming_asr_lite.export_frontend``. Only valid for the
    #: preprocessing configuration it was exported from -- re-export if the mel
    #: count, FFT size or window geometry ever change.
    frontend_path: Optional[str] = None

    # ---- decoding toggles ----------------------------------------------
    greedy_decode: bool = True
    final_beam_decode: bool = True

    # ---- history --------------------------------------------------------
    #: Seconds of audio retained for the final full-utterance decode. Bounds
    #: memory on very long utterances.
    max_history: float = 120.0

    # ---- finalisation of long utterances --------------------------------
    #: The reference checkpoint was trained on utterances of 0.5-11 s
    #: (``max_duration: 11`` in its train config). Feeding a 60 s recording to
    #: the encoder in one pass is out of distribution and may exceed the
    #: Conformer's positional-encoding extent. Beyond this threshold the final
    #: decode runs over overlapping segments whose logits are stitched.
    #:
    #: Keep this at or below the checkpoint's training ``max_duration``.
    #: Measured on real speech (``tools/real_audio_wer.py``): single-pass greedy
    #: WER rises with turn length -- 0.053 at 6 s, 0.061 at 66 s, 0.076 at 124 s
    #: -- while the segmented path stays flat at ~0.05. Past ~150 s on a 4 GB GPU
    #: it stops working entirely: CUDA OOM, then a hard broadcast failure in the
    #: model's relative-position buffer, then a native abort that kills the
    #: process. Self-attention is quadratic in sequence length.
    final_segment_duration: float = 10.0
    #: Overlap between final-decode segments. Half is discarded on each side of
    #: a seam so that no token is decoded without context on both sides.
    final_segment_overlap: float = 2.0
    #: How far a segment boundary may be moved to land in a local energy
    #: minimum, so a cut falls in a pause rather than mid-word.
    #:
    #: Defaults to 0 (disabled) because it did not help when measured -- but the
    #: only measurement was on the synthetic fixture, which has uniform 0.12 s
    #: inter-word gaps and no real pauses, so snapping had nothing to find. That
    #: result cannot support a conclusion either way and its numbers are not
    #: quoted. Re-measure on real audio before enabling or dismissing it.
    final_segment_snap: float = 0.0

    # ---- warm-up --------------------------------------------------------
    #: Before the rolling buffer has filled, its left portion is zero padding.
    #: The reference feeds the whole padded buffer and declares it all valid.
    #: That hands the model up to ``context_duration`` of digital silence --
    #: something it never saw in training (min_duration 0.5 s) -- and the
    #: observed result is hallucinated tokens in the padding region, plus
    #: normalisation statistics computed over silence. When False (the
    #: default) only the real audio is fed until the buffer is warm. Set True
    #: for bit-exact parity with the reference implementation.
    pad_warmup_window: bool = False

    # ---- execution ------------------------------------------------------
    #: Where the mel frontend and the model run: "auto", "cpu", "cuda" or
    #: "cuda:N". "auto" selects CUDA only when torch *and* ONNX Runtime can
    #: both use it -- a split placement would copy features across the PCIe
    #: bus on every window. Requesting "cuda" explicitly raises if it is
    #: unavailable, rather than quietly running 10x slower on CPU.
    device: str = "auto"
    #: "auto" derives providers from ``device``; an explicit list overrides it.
    providers: str = "auto"
    #: ONNX Runtime intra-op thread count. 0 = library default.
    intra_op_threads: int = 0

    # ---- sub-configs ----------------------------------------------------
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    beam: BeamDecoderConfig = field(default_factory=BeamDecoderConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)

    # ---- diagnostics ----------------------------------------------------
    #: Emit a metrics event alongside every partial.
    emit_metrics: bool = False
    #: Retain per-window frame posteriors. Useful for DTW experiments and
    #: alignment debugging, but memory-hungry on long audio.
    retain_frame_posteriors: bool = False

    def __post_init__(self) -> None:
        if self.preprocessing.sample_rate != self.sample_rate:
            logger.warning(
                "Overriding preprocessing.sample_rate (%d) with top-level sample_rate (%d)",
                self.preprocessing.sample_rate, self.sample_rate,
            )
            self.preprocessing.sample_rate = self.sample_rate
        if self.chunk_duration <= 0:
            raise ValueError("chunk_duration must be positive")
        if self.context_duration < 0:
            raise ValueError("context_duration must be non-negative")
        self.stability.validate_against(self.context_duration)

    # ---- derived geometry ----------------------------------------------

    @property
    def buffer_duration(self) -> float:
        """Total window length fed to the model each step."""
        return self.context_duration + self.chunk_duration

    @property
    def chunk_samples(self) -> int:
        return int(round(self.chunk_duration * self.sample_rate))

    @property
    def context_samples(self) -> int:
        return int(round(self.context_duration * self.sample_rate))

    @property
    def buffer_samples(self) -> int:
        return self.chunk_samples + self.context_samples

    @property
    def window_redundancy(self) -> float:
        """How many times each audio sample is re-processed by the model.

        4.0 / 0.16 = 25 at the reference operating point -- i.e. 96% of every
        model call is recomputation of audio already seen. This is the headline
        cost of pseudo-streaming a full-context model.
        """
        return self.buffer_duration / self.chunk_duration

    @property
    def resolved_blank_id(self) -> int:
        if self.blank_id is not None:
            return self.blank_id
        return len(self.vocabulary) - 1

    # ---- vocabulary helpers ---------------------------------------------

    def ensure_blank_in_vocabulary(self) -> list[str]:
        """Return the vocabulary with the blank symbol appended if missing.

        The reference does ``vocabulary.append("__")`` before building the
        decoder; the exported model has ``len(vocab) + 1`` output units.
        """
        vocab = list(self.vocabulary)
        if self.blank_id is None and (not vocab or vocab[-1] != self.blank_token):
            vocab.append(self.blank_token)
        return vocab

    # ---- serialisation ---------------------------------------------------

    def to_dict(self, summarize_vocabulary: bool = False) -> dict[str, Any]:
        """Serialise the config.

        The default output round-trips through :meth:`from_dict`.
        ``summarize_vocabulary`` replaces the 129-entry token list with a count,
        which is what you want for a human-readable dump but would break a
        reload.
        """
        d = asdict(self)
        if summarize_vocabulary:
            d["vocabulary"] = f"<{len(self.vocabulary)} tokens>"
        d["derived"] = {
            "buffer_duration": self.buffer_duration,
            "buffer_samples": self.buffer_samples,
            "chunk_samples": self.chunk_samples,
            "window_redundancy": self.window_redundancy,
        }
        return d

    def describe(self) -> str:
        return json.dumps(self.to_dict(summarize_vocabulary=True), indent=2, default=str)

    @classmethod
    def from_json(cls, path: str | Path) -> "StreamingASRConfig":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StreamingASRConfig":
        raw = dict(raw)
        raw.pop("derived", None)
        sub = {
            "preprocessing": PreprocessingConfig,
            "beam": BeamDecoderConfig,
            "stability": StabilityConfig,
            "endpoint": EndpointConfig,
            "segmentation": SegmentationConfig,
        }
        for key, klass in sub.items():
            if isinstance(raw.get(key), dict):
                raw[key] = klass(**raw[key])
        return cls(**raw)


def load_vocabulary(path: str | Path) -> list[str]:
    """Load a vocabulary from a newline-delimited or JSON file.

    Newline format preserves whitespace-bearing SentencePiece tokens exactly,
    so only the trailing newline is stripped.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return list(json.loads(text))
    return [line.rstrip("\n") for line in text.splitlines()]
