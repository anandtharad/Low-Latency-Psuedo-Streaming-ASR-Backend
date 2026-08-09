"""Build a stand-in Conformer-CTC ONNX model for local verification.

The real IndicConformer checkpoint, the KenLM binary and the lexicon live on
the ASR server (``/data/asr/...``). Without them the streaming pipeline cannot
be run, let alone measured, on a development machine -- and "it imports
cleanly" is not evidence that a streaming stabiliser works.

So this script builds a fixture that stands in for the real model:

* synthetic speech-like audio for a known transcript, with known word timings;
* a small CTC acoustic model with the **same ONNX interface** as the NeMo
  export (``audio_signal`` / ``length`` in, log-probs out; 4x subsampling;
  129 output units), trained to overfit that one utterance;
* trained on random *crops* of the audio, so it behaves sensibly when fed the
  4-second rolling windows the pipeline actually sends it.

The encoder is a **bidirectional** GRU, chosen deliberately. A causal model
would produce stable partial hypotheses and make the stabiliser look better
than it is. A bidirectional one revises its reading of earlier audio when the
window shifts and new right-context arrives -- which is exactly the failure
mode described in section 15 of the brief, and the reason the hypothesis
tracker exists at all.

This is a test fixture. It is not a speech model and its transcripts mean
nothing beyond the one sentence it memorised.

Usage::

    python tools/build_synthetic_fixture.py --out fixtures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming_asr.config import REFERENCE_VOCABULARY, PreprocessingConfig  # noqa: E402
from streaming_asr.console import configure_stdout  # noqa: E402
from streaming_asr.preprocessing.filterbank import Preprocessor  # noqa: E402

configure_stdout()

DEFAULT_TRANSCRIPT = "i have been having chest pain for three days after eating"
SAMPLE_RATE = 16000
#: Each BPE token gets its own acoustic segment; a word's duration is the sum
#: over its tokens, so "i" is short and "having" is long -- as in real speech.
#: At a 40 ms CTC frame this gives 6 frames per token, enough for the blank
#: separators CTC needs between repeated units.
TOKEN_DURATION = 0.24
GAP_DURATION = 0.12
LEAD_SILENCE = 0.30
TAIL_SILENCE = 0.50


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

def tokenize(text: str, vocabulary: list[str]) -> list[int]:
    """Greedy longest-match SentencePiece-style tokenisation.

    Not a real SentencePiece implementation -- it only needs to produce a valid
    token sequence over the reference vocabulary for the fixture transcript.
    """
    lookup = {tok: i for i, tok in enumerate(vocabulary)}
    ids: list[int] = []
    for word in text.split():
        piece = "▁" + word
        cursor = 0
        while cursor < len(piece):
            for end in range(len(piece), cursor, -1):
                candidate = piece[cursor:end]
                if candidate in lookup:
                    ids.append(lookup[candidate])
                    cursor = end
                    break
            else:
                raise ValueError(
                    f"Cannot tokenise {word!r}: no vocabulary entry matches at "
                    f"{piece[cursor:]!r}. Pick a transcript the reference "
                    f"vocabulary covers."
                )
    return ids


# --------------------------------------------------------------------------
# Synthetic audio
# --------------------------------------------------------------------------

def _render_token(token_id: int, seed: int, duration: float, rng) -> np.ndarray:
    """Render one BPE token as a distinct formant pattern.

    Keyed by token id, so a given token always sounds the same wherever it
    appears -- the same property real sub-word units have, and the thing that
    makes the mapping learnable at all.
    """
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE

    token_rng = np.random.default_rng(seed * 1000 + token_id)
    formants = np.sort(token_rng.uniform([250, 900, 2000], [850, 1900, 3600]))
    am_rate = float(token_rng.uniform(3.0, 20.0))

    signal = np.zeros(n, dtype=np.float32)
    for k, freq in enumerate(formants):
        # A slight sweep gives the token an internal trajectory, so it cannot
        # be identified from a single static frame.
        sweep = freq * (1.0 + 0.06 * np.sin(2 * np.pi * t / max(duration, 1e-6)))
        signal += (0.7 ** k) * np.sin(2 * np.pi * sweep * t).astype(np.float32)

    signal *= (0.65 + 0.35 * np.sin(2 * np.pi * am_rate * t)).astype(np.float32)
    envelope = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / max(n - 1, 1)))
    signal *= envelope.astype(np.float32)
    signal += 0.005 * rng.standard_normal(n).astype(np.float32)
    signal /= (np.abs(signal).max() + 1e-6)
    return (signal * 0.6).astype(np.float32)


def synthesize_audio(
    word_token_ids: list[list[int]], seed: int = 0
) -> tuple[np.ndarray, list[tuple[float, float]], list[tuple[int, float, float]]]:
    """Render the utterance one BPE token at a time.

    Rendering per *word* leaves multi-token words such as "having"
    (``▁ha`` + ``vi`` + ``ng``) as a single uniform sound, giving CTC no
    acoustic cue for where one token ends and the next begins. The model then
    under-emits, collapsing "having" to "hang". Giving every token its own
    sub-segment removes that ambiguity.

    Returns the waveform, per-word ``(start, end)`` times, and per-token
    ``(token_id, start, end)`` spans. Knowing the true timings is what makes it
    possible to build crop-level CTC targets, to supervise the alignment
    directly, and to check the pipeline's token timestamps against ground
    truth.
    """
    rng = np.random.default_rng(seed)
    pieces = [np.zeros(int(LEAD_SILENCE * SAMPLE_RATE), dtype=np.float32)]
    spans: list[tuple[float, float]] = []
    token_spans: list[tuple[int, float, float]] = []
    cursor = LEAD_SILENCE

    for token_ids in word_token_ids:
        word_start = cursor
        for token_id in token_ids:
            pieces.append(_render_token(token_id, seed, TOKEN_DURATION, rng))
            token_spans.append((token_id, cursor, cursor + TOKEN_DURATION))
            cursor += TOKEN_DURATION
        spans.append((word_start, cursor))

        gap = int(GAP_DURATION * SAMPLE_RATE)
        pieces.append(0.001 * rng.standard_normal(gap).astype(np.float32))
        cursor += GAP_DURATION

    pieces.append(np.zeros(int(TAIL_SILENCE * SAMPLE_RATE), dtype=np.float32))
    return np.concatenate(pieces).astype(np.float32), spans, token_spans


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

class SyntheticCTCModel(nn.Module):
    """A tiny CTC acoustic model with the NeMo export's interface.

    Two strided convolutions give the 4x subsampling a Conformer has (10 ms
    frames in, 40 ms frames out), and a bidirectional GRU supplies the
    full-context behaviour that makes hypotheses revise as the window slides.
    """

    def __init__(self, n_mels: int = 80, hidden: int = 192, vocab_size: int = 129) -> None:
        super().__init__()
        self.subsample = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.encoder = nn.GRU(
            hidden, hidden // 2, num_layers=2,
            bidirectional=True, batch_first=True,
        )
        self.classifier = nn.Linear(hidden, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, audio_signal: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
        # audio_signal: (B, n_mels, T) -> (B, T/4, V) log-probabilities
        x = self.subsample(audio_signal)
        x = x.transpose(1, 2)
        x, _ = self.encoder(x)
        logits = self.classifier(x)

        # Force padded frames to blank, mirroring how the real model uses the
        # length input. Keeps 'length' live in the exported graph instead of
        # being pruned as unused.
        out_len = torch.div(length, 4, rounding_mode="floor").clamp(min=1)
        frames = torch.arange(logits.shape[1], device=logits.device)
        valid = frames.unsqueeze(0) < out_len.unsqueeze(1)
        blank_bias = torch.zeros_like(logits)
        blank_bias[..., -1] = 30.0
        logits = torch.where(valid.unsqueeze(-1), logits, blank_bias)

        return F.log_softmax(logits, dim=-1)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

CTC_FRAME_DURATION = 0.04       # 10 ms hop x 4x subsampling


def frame_targets_for_crop(
    crop_start: float,
    n_out_frames: int,
    token_spans: list[tuple[int, float, float]],
    blank_id: int,
) -> torch.Tensor:
    """Build a frame-level label sequence for an auxiliary alignment loss.

    CTC loss is alignment-free: any monotonic placement of the tokens scores
    identically, so nothing in it requires a spike to fire near the sound that
    caused it. Left to itself on a small fixture the model discovers it can
    dump every token into the first few frames of the window -- which makes a
    word's absolute timestamp advance by one chunk per window and never settle,
    breaking time-based stabilisation.

    Real well-trained Conformer-CTC models do align to the acoustics. Since the
    true token spans are known by construction here, the alignment is simply
    supervised, so the fixture exhibits the same property.

    Only the middle of each token's span is labelled; the edges stay blank.
    That guarantees a blank separator between adjacent tokens, so CTC collapse
    cannot merge two neighbours into one.
    """
    targets = torch.full((n_out_frames,), blank_id, dtype=torch.long)
    for token_id, start, end in token_spans:
        # Trim to the central 60% of the span.
        margin = 0.2 * (end - start)
        lo = int(np.ceil((start + margin - crop_start) / CTC_FRAME_DURATION))
        hi = int(np.floor((end - margin - crop_start) / CTC_FRAME_DURATION))
        lo, hi = max(lo, 0), min(hi, n_out_frames - 1)
        if lo <= hi:
            targets[lo:hi + 1] = token_id
    return targets


def build_crop_batch(
    audio: np.ndarray,
    spans: list[tuple[float, float]],
    word_token_ids: list[list[int]],
    preprocessor: Preprocessor,
    rng: np.random.Generator,
    batch_size: int,
    window_options: tuple[float, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """Sample random crops and their CTC targets.

    A crop's target is the words *fully contained* in it. That mirrors what the
    pipeline does at run time: the model sees a window and is expected to
    transcribe what is inside it, not to guess at words clipped by the edges.
    """
    features_list, targets, target_lengths, input_lengths = [], [], [], []
    crop_starts: list[float] = []
    duration = len(audio) / SAMPLE_RATE

    for _ in range(batch_size):
        # Resample until the crop contains at least one complete word. An
        # empty CTC target divides the mean-reduced loss by zero, which
        # zero_infinity silently drops -- wasting the slot and destabilising
        # the gradient.
        for _attempt in range(20):
            window = min(float(rng.choice(window_options)), duration)
            start = max(0.0, float(rng.uniform(-0.5, max(0.0, duration - window))))
            end = min(duration, start + window)
            target: list[int] = []
            for (word_start, word_end), ids in zip(spans, word_token_ids):
                if word_start >= start - 1e-6 and word_end <= end + 1e-6:
                    target.extend(ids)
            if target:
                break
        else:
            start, end = 0.0, duration
            target = [t for ids in word_token_ids for t in ids]

        crop = audio[int(start * SAMPLE_RATE):int(end * SAMPLE_RATE)]

        with torch.inference_mode():
            feats, feat_len = preprocessor(crop.reshape(1, -1), n_samples=len(crop))

        features_list.append(feats[0])
        input_lengths.append(int(feat_len[0]))
        targets.append(torch.tensor(target, dtype=torch.long))
        target_lengths.append(len(target))
        crop_starts.append(start)

    max_frames = max(f.shape[-1] for f in features_list)
    padded = torch.zeros(len(features_list), features_list[0].shape[0], max_frames)
    for i, feats in enumerate(features_list):
        padded[i, :, : feats.shape[-1]] = feats

    flat_targets = torch.cat(targets) if any(target_lengths) else torch.zeros(0, dtype=torch.long)
    return (
        padded,
        torch.tensor(input_lengths, dtype=torch.long),
        flat_targets,
        torch.tensor(target_lengths, dtype=torch.long),
        crop_starts,
    )


def train(
    model: SyntheticCTCModel,
    audio: np.ndarray,
    spans: list[tuple[float, float]],
    word_token_ids: list[list[int]],
    token_spans: list[tuple[int, float, float]],
    preprocessor: Preprocessor,
    steps: int,
    batch_size: int,
    blank_id: int,
    seed: int,
    alignment_weight: float = 1.0,
) -> None:
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    window_options = (1.0, 2.0, 3.0, 4.0, 5.0, len(audio) / SAMPLE_RATE)

    model.train()
    for step in range(steps):
        features, input_lengths, targets, target_lengths, crop_starts = build_crop_batch(
            audio, spans, word_token_ids, preprocessor, rng, batch_size, window_options
        )
        log_probs = model(features, input_lengths)
        out_lengths = torch.div(input_lengths, 4, rounding_mode="floor").clamp(min=1)

        # ctc_loss wants (T, B, V).
        ctc = F.ctc_loss(
            log_probs.transpose(0, 1), targets, out_lengths, target_lengths,
            blank=blank_id, zero_infinity=True,
        )

        # Auxiliary frame-level supervision, so spikes fire near the acoustics
        # rather than wherever CTC finds convenient. See frame_targets_for_crop.
        frame_loss = torch.zeros((), dtype=log_probs.dtype)
        for b, crop_start in enumerate(crop_starts):
            valid = int(out_lengths[b])
            frame_target = frame_targets_for_crop(
                crop_start, valid, token_spans, blank_id
            )
            frame_loss = frame_loss + F.nll_loss(log_probs[b, :valid], frame_target)
        frame_loss = frame_loss / len(crop_starts)

        loss = ctc + alignment_weight * frame_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if step % 25 == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}  ctc={ctc.item():.4f} "
                  f"align={frame_loss.item():.4f}", flush=True)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

@torch.inference_mode()
def validate(
    model: SyntheticCTCModel,
    audio: np.ndarray,
    preprocessor: Preprocessor,
    vocabulary: list[str],
    blank_id: int,
    expected: str,
) -> bool:
    """Greedy-decode the full utterance and the 4 s windows the pipeline uses.

    A fixture that only works on the full utterance is useless here -- the
    streaming path never sees more than one window at a time.
    """
    from streaming_asr.decoding.greedy_ctc import ctc_collapse
    from streaming_asr.types import tokens_to_text

    model.eval()

    def decode(segment: np.ndarray) -> str:
        feats, lengths = preprocessor(segment.reshape(1, -1), n_samples=len(segment))
        log_probs = model(feats, lengths)
        spans = ctc_collapse(log_probs[0].argmax(dim=-1).numpy(), blank_id)
        return tokens_to_text([vocabulary[t] for t, _, _ in spans])

    full = decode(audio)
    print(f"  full-utterance greedy: {full!r}")
    print(f"  expected             : {expected!r}")

    window = int(4.0 * SAMPLE_RATE)
    for start in (0, len(audio) // 3, max(0, len(audio) - window)):
        piece = audio[start:start + window]
        if len(piece) > SAMPLE_RATE:
            print(f"  window @{start / SAMPLE_RATE:5.2f}s: {decode(piece)!r}")

    matched = full.strip() == expected.strip()
    print("  CONVERGED" if matched else "  NOT CONVERGED (train longer or raise --steps)")
    return matched


def export_onnx(model: SyntheticCTCModel, path: Path, n_mels: int) -> None:
    """Export with the same input/output names and dynamic axes as NeMo."""
    model.eval()
    dummy_features = torch.randn(1, n_mels, 400)
    dummy_length = torch.tensor([400], dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_features, dummy_length),
        str(path),
        input_names=["audio_signal", "length"],
        output_names=["logprobs"],
        dynamic_axes={
            "audio_signal": {0: "batch", 2: "time"},
            "length": {0: "batch"},
            "logprobs": {0: "batch", 1: "time_out"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  exported -> {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="fixtures", help="output directory")
    parser.add_argument("--transcript", default=DEFAULT_TRANSCRIPT)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--alignment-weight", type=float, default=1.0,
                        help="weight of the auxiliary frame-alignment loss; 0 disables it")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = list(REFERENCE_VOCABULARY) + ["__"]
    blank_id = len(vocabulary) - 1
    words = args.transcript.split()

    print(f"Transcript: {args.transcript!r} ({len(words)} words)")
    word_token_ids = [tokenize(w, vocabulary) for w in words]
    flat = [t for ids in word_token_ids for t in ids]
    print(f"  tokens: {[vocabulary[i] for i in flat]}")

    print("Synthesising audio...")
    audio, spans, token_spans = synthesize_audio(word_token_ids, seed=args.seed)
    duration = len(audio) / SAMPLE_RATE
    print(f"  {duration:.2f}s, {len(words)} words / {len(flat)} tokens "
          f"at {TOKEN_DURATION}s per token")

    import soundfile as sf

    wav_path = out_dir / "synthetic.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    print(f"  wrote {wav_path}")

    preprocessor = Preprocessor(PreprocessingConfig())
    model = SyntheticCTCModel(n_mels=80, vocab_size=len(vocabulary))

    print(f"Training ({args.steps} steps)...")
    train(
        model, audio, spans, word_token_ids, token_spans, preprocessor,
        steps=args.steps, batch_size=args.batch_size, blank_id=blank_id, seed=args.seed,
        alignment_weight=args.alignment_weight,
    )

    print("Validating...")
    validate(model, audio, preprocessor, vocabulary, blank_id, args.transcript)

    print("Exporting ONNX...")
    export_onnx(model, out_dir / "synthetic_model.onnx", n_mels=80)

    (out_dir / "vocabulary.txt").write_text("\n".join(vocabulary), encoding="utf-8")
    (out_dir / "transcript.txt").write_text(args.transcript, encoding="utf-8")
    (out_dir / "word_spans.json").write_text(
        json.dumps([{"word": w, "start": s, "end": e}
                    for w, (s, e) in zip(words, spans)], indent=2),
        encoding="utf-8",
    )
    print(f"Wrote fixture assets to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
