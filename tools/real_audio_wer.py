"""Word error rate on real speech, from the Common Voice clips in this repo.

Every WER figure quoted in the documentation comes from here. The synthetic
fixture built by ``build_synthetic_fixture.py`` is a *code-path* exercise -- it
proves the plumbing works without a checkpoint -- and its error rates describe a
toy model reading toy audio. They must never be quoted as accuracy results.

What this measures
------------------
Two questions, both about the choice this project actually made:

**Short clips.** Does the pause-segmented streaming path cost anything on
utterances short enough that a single ``model.transcribe()`` call is already
in-distribution? It should not: with one pause-bounded segment per clip the two
paths do the same work.

**Long turns.** What happens when the turn is longer than the checkpoint was
trained on (``max_duration: 11`` s for this one)? Common Voice ships single
sentences, so a long turn is built by concatenating consecutive clips from one
speaker with a silent gap between them -- which is what a person dictating
several sentences sounds like to the segmenter. The reference is the
concatenation of the clips' own sentences, so it is exact, not approximated.

Both paths decode the *same audio* with the *same model*; the only difference is
whether the encoder sees it in one pass or one pause-bounded span at a time.

Text normalisation
------------------
Common Voice references are cased and punctuated; a CTC checkpoint emits neither.
Comparing them raw would report a large error rate that is entirely formatting.
Both sides are therefore lowercased, hyphens become spaces, everything except
``a-z0-9'`` and spaces is dropped, and whitespace is collapsed. This is applied
identically to reference and hypothesis, so it cannot flatter either path.

Usage::

    python tools/real_audio_wer.py --model model.onnx --vocabulary vocab.txt \\
        --clips common_voice_en/en_train_28 --manifest common_voice_en/train.tsv \\
        --clip-count 50 --turn-count 8 --json-out results/real_audio_wer.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming_asr.audio.base import AudioChunk  # noqa: E402
from streaming_asr.config import (  # noqa: E402
    SegmentationConfig,
    StreamingASRConfig,
    load_vocabulary,
)
from streaming_asr.events import ASREventType  # noqa: E402
from streaming_asr_lite.audio import decode_audio  # noqa: E402
from streaming_asr_lite.factory import build_engine, build_pipeline  # noqa: E402

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_KEEP = re.compile(r"[^a-z0-9' ]+")
_SPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Applied to reference and hypothesis alike. Hyphens become spaces rather
    than disappearing, so ``Wisconsin-Milwaukee`` scores as two words against a
    model that can only ever emit two.
    """
    text = text.lower().replace("-", " ").replace("—", " ").replace("–", " ")
    text = text.replace("’", "'")
    return _SPACE.sub(" ", _KEEP.sub(" ", text)).strip()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, start=1):
        current = [i]
        for j, h in enumerate(hypothesis, start=1):
            current.append(min(previous[j] + 1,
                               current[j - 1] + 1,
                               previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


@dataclass
class ErrorCount:
    """Accumulates errors and reference length, so WER can be pooled.

    Averaging per-utterance WER weights a three-word clip the same as a
    thirty-word one. Pooling errors over total reference words is the standard
    definition and the one reported here.
    """

    errors: int = 0
    words: int = 0
    utterances: int = 0

    def add(self, reference: str, hypothesis: str) -> float:
        ref, hyp = reference.split(), hypothesis.split()
        distance = edit_distance(ref, hyp)
        self.errors += distance
        self.words += len(ref)
        self.utterances += 1
        return distance / len(ref) if ref else (0.0 if not hyp else 1.0)

    @property
    def wer(self) -> float:
        return self.errors / self.words if self.words else 0.0


# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


@dataclass
class Clip:
    path: Path
    sentence: str
    client_id: str
    duration: float = 0.0


def read_manifest(manifest: Path, clips_dir: Path) -> list[Clip]:
    """Join the TSV against the clips actually present on disk.

    ``train.tsv`` describes the whole Common Voice English train split (over a
    million rows); this repository holds one shard of it. Streaming the file and
    keeping only rows whose audio exists avoids loading 375 MB of metadata for
    the ~19k clips that matter.
    """
    available = {p.name: p for p in clips_dir.iterdir() if p.is_file()}
    if not available:
        raise FileNotFoundError(f"no clips found in {clips_dir}")

    found: list[Clip] = []
    with open(manifest, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            path = available.get(row.get("path", ""))
            if path is None:
                continue
            sentence = normalize(row.get("sentence", ""))
            if sentence:
                found.append(Clip(path=path, sentence=sentence,
                                  client_id=row.get("client_id", "")))
            if len(found) == len(available):
                break

    logger.info("matched %d of %d clips in %s against the manifest",
                len(found), len(available), clips_dir.name)
    if not found:
        raise ValueError(
            f"no rows in {manifest} reference any file in {clips_dir}. "
            f"Check that the manifest covers this shard."
        )
    return found


def load_clip(clip: Clip) -> np.ndarray:
    audio = decode_audio(clip.path, SAMPLE_RATE)
    clip.duration = len(audio) / SAMPLE_RATE
    return audio


# ---------------------------------------------------------------------------
# decoding paths
# ---------------------------------------------------------------------------


class Decoders:
    """The two ways of turning audio into text that are being compared."""

    def __init__(self, config: StreamingASRConfig) -> None:
        self.config = config
        if config.runtime == "torch":
            from streaming_asr.device import resolve_device

            self.engine = build_engine(config, providers=resolve_device(
                config.device, config.providers).providers)
        else:
            self.engine = build_engine(config)
        # One pipeline is kept purely for its frontend, engine and greedy
        # decoder; the single-pass path needs those but no segmentation state.
        self._reference_pipeline = build_pipeline(config, engine=self.engine)
        self._reference_pipeline.warmup(iterations=1)

    def single_pass(self, audio: np.ndarray) -> tuple[str, float]:
        """One forward pass over the whole recording -- what ``transcribe()`` does.

        Deliberately routed through the pipeline's own ``_decode``, so this and
        the segmented path run *identical* code and differ only in how much
        audio reaches the encoder at once. Reimplementing the forward pass here
        would leave room for the comparison to be measuring something else.
        """
        pipeline = self._reference_pipeline
        started = time.perf_counter()
        logits, _, _ = pipeline._decode(audio)
        text = pipeline.greedy.decode_text(logits)
        return normalize(text), time.perf_counter() - started

    def segmented(
        self, audio: np.ndarray, segmentation: Optional[SegmentationConfig] = None
    ) -> tuple[str, float, list[dict[str, Any]]]:
        """Feed the audio through the streaming pipeline chunk by chunk."""
        config = self.config
        if segmentation is not None:
            config = _with_segmentation(self.config, segmentation)
        pipeline = build_pipeline(config, engine=self.engine)

        started = time.perf_counter()
        segments: list[dict[str, Any]] = []
        for event in pipeline.stream(_chunks(audio, config.chunk_samples)):
            if event.type is ASREventType.SEGMENT:
                segments.append({
                    "start": round(event.window_start or 0.0, 2),
                    "end": round(event.window_end or 0.0, 2),
                    "text": event.text,
                    "forced": bool(event.metrics.get("forced")),
                })
        elapsed = time.perf_counter() - started
        return normalize(pipeline.transcript), elapsed, segments


def _with_segmentation(
    config: StreamingASRConfig, segmentation: SegmentationConfig
) -> StreamingASRConfig:
    raw = config.to_dict()
    raw["segmentation"] = asdict(segmentation)
    return StreamingASRConfig.from_dict(raw)


def _chunks(audio: np.ndarray, chunk_samples: int) -> Iterator[AudioChunk]:
    """Chunk iterator with the trailing partial chunk zero-padded, not dropped."""
    for start in range(0, len(audio), chunk_samples):
        block = audio[start:start + chunk_samples]
        if block.size < chunk_samples:
            padded = np.zeros(chunk_samples, dtype=np.float32)
            padded[: block.size] = block
            block = padded
        yield AudioChunk(samples=block, start_sample=start, sample_rate=SAMPLE_RATE)


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------


@dataclass
class ShortClipResult:
    clips: int = 0
    audio_seconds: float = 0.0
    single_pass: ErrorCount = field(default_factory=ErrorCount)
    segmented: ErrorCount = field(default_factory=ErrorCount)
    worst: list[dict[str, Any]] = field(default_factory=list)


def run_short_clips(decoders: Decoders, clips: list[Clip]) -> ShortClipResult:
    """Per-clip WER. Both paths should agree: one clip is one segment."""
    result = ShortClipResult()
    scored: list[tuple[float, dict[str, Any]]] = []

    for index, clip in enumerate(clips, start=1):
        audio = load_clip(clip)
        single, _ = decoders.single_pass(audio)
        streamed, _, _ = decoders.segmented(audio)

        single_wer = result.single_pass.add(clip.sentence, single)
        stream_wer = result.segmented.add(clip.sentence, streamed)
        result.clips += 1
        result.audio_seconds += clip.duration
        scored.append((stream_wer, {
            "clip": clip.path.name,
            "reference": clip.sentence,
            "single_pass": single,
            "segmented": streamed,
            "wer_single_pass": round(single_wer, 3),
            "wer_segmented": round(stream_wer, 3),
        }))
        if index % 10 == 0:
            print(f"  {index}/{len(clips)} clips  "
                  f"single-pass WER {result.single_pass.wer:.3f}  "
                  f"segmented WER {result.segmented.wer:.3f}")

    scored.sort(key=lambda item: -item[0])
    result.worst = [row for _, row in scored[:5]]
    return result


@dataclass
class Turn:
    audio: np.ndarray
    reference: str
    clips: int
    duration: float


def build_turns(
    clips: list[Clip], target_seconds: float, gap: float, count: int
) -> list[Turn]:
    """Concatenate consecutive clips into turns of roughly ``target_seconds``.

    Clips are grouped by speaker where possible, so a turn sounds like one
    person talking. The gap is digital silence long enough for the segmenter to
    cut on (it must exceed ``segment_silence``), which is what makes this a fair
    stand-in for continuous dictation rather than an artificially easy case.
    """
    silence = np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32)
    by_speaker: dict[str, list[Clip]] = {}
    for clip in clips:
        by_speaker.setdefault(clip.client_id, []).append(clip)
    ordered = [clip for group in by_speaker.values() for clip in group]

    turns: list[Turn] = []
    parts: list[np.ndarray] = []
    sentences: list[str] = []
    total = 0.0

    for clip in ordered:
        audio = load_clip(clip)
        if parts:
            parts.append(silence)
            total += gap
        parts.append(audio)
        sentences.append(clip.sentence)
        total += clip.duration

        if total >= target_seconds:
            turns.append(Turn(audio=np.concatenate(parts),
                              reference=" ".join(sentences),
                              clips=len(sentences), duration=total))
            parts, sentences, total = [], [], 0.0
            if len(turns) >= count:
                break
    return turns


def run_long_turns(
    decoders: Decoders,
    turns: list[Turn],
    segment_caps: Sequence[float],
    checkpoint: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Single-pass versus pause-segmented on turns longer than training saw."""
    single = ErrorCount()
    per_cap = {cap: ErrorCount() for cap in segment_caps}
    forced_cuts = {cap: 0 for cap in segment_caps}
    detail: list[dict[str, Any]] = []

    base = decoders.config.segmentation
    for index, turn in enumerate(turns, start=1):
        row: dict[str, Any] = {
            "turn": index, "clips": turn.clips,
            "duration": round(turn.duration, 1),
        }
        try:
            text, _ = decoders.single_pass(turn.audio)
        except Exception as exc:  # OOM on a small GPU is a real outcome, not a crash
            logger.error("single-pass decode of turn %d (%.0fs) failed: %s",
                         index, turn.duration, exc)
            row["single_pass_error"] = str(exc)
            text = ""
        row["wer_single_pass"] = round(single.add(turn.reference, text), 3)

        for cap in segment_caps:
            settings = SegmentationConfig(
                segment_silence=base.segment_silence,
                turn_silence=base.turn_silence,
                max_segment_duration=cap,
                min_segment_speech=base.min_segment_speech,
                energy_threshold=base.energy_threshold,
                speech_pad=base.speech_pad,
            )
            streamed, _, segments = decoders.segmented(turn.audio, settings)
            row[f"wer_segmented_{cap:g}s"] = round(
                per_cap[cap].add(turn.reference, streamed), 3
            )
            forced_cuts[cap] += sum(1 for s in segments if s["forced"])

        detail.append(row)
        print(f"  turn {index}/{len(turns)} ({turn.duration:.0f}s, {turn.clips} clips): "
              f"single-pass {single.wer:.3f} vs segmented "
              + ", ".join(f"{cap:g}s={per_cap[cap].wer:.3f}" for cap in segment_caps))
        # Per turn, not per block: a long decode can abort the whole process
        # from native code, and a block-level checkpoint loses everything since
        # the last one. This has already cost two runs.
        if checkpoint is not None:
            checkpoint(_summarise(turns, single, per_cap, forced_cuts,
                                  segment_caps, detail))

    return _summarise(turns, single, per_cap, forced_cuts, segment_caps, detail)


def _summarise(
    turns: list[Turn],
    single: ErrorCount,
    per_cap: dict[float, ErrorCount],
    forced_cuts: dict[float, int],
    segment_caps: Sequence[float],
    detail: list[dict[str, Any]],
) -> dict[str, Any]:
    done = turns[: len(detail)]
    return {
        # Counts what was actually scored, so a checkpoint written mid-block is
        # honest about how much of the block it covers.
        "turns": len(done),
        "mean_turn_seconds": round(
            sum(t.duration for t in done) / max(1, len(done)), 1
        ),
        "single_pass": {"wer": round(single.wer, 4), **asdict(single)},
        "segmented": {
            f"{cap:g}s": {
                "wer": round(per_cap[cap].wer, 4),
                "forced_cuts": forced_cuts[cap],
                **asdict(per_cap[cap]),
            }
            for cap in segment_caps
        },
        "per_turn": detail,
    }


# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--frontend", default="fixtures/frontend.onnx")
    parser.add_argument("--clips", default="common_voice_en/en_train_28")
    parser.add_argument("--manifest", default="common_voice_en/train.tsv")
    parser.add_argument("--clip-count", type=int, default=50,
                        help="clips for the short-utterance comparison")
    parser.add_argument("--turn-count", type=int, default=8,
                        help="concatenated long turns; 0 skips that experiment")
    parser.add_argument("--turn-seconds", nargs="+", type=float, default=[60.0],
                        help="one experiment per length, so the degradation "
                             "curve against turn length is visible rather than "
                             "a single point")
    parser.add_argument("--gap", type=float, default=0.7,
                        help="silence inserted between clips in a turn; must "
                             "exceed ASR segment_silence for a cut to happen")
    parser.add_argument("--segment-caps", nargs="+", type=float,
                        default=[6.0, 10.0, 20.0, 40.0],
                        help="max_segment_duration values to compare")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--runtime", default="lite", choices=["lite", "torch"],
                        help="'torch' also preloads torch's bundled CUDA "
                             "libraries, which on Windows is what lets ONNX "
                             "Runtime find them at all")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--json-out", default="results/real_audio_wer.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    vocabulary = load_vocabulary(args.vocabulary)
    config = StreamingASRConfig(
        onnx_model_path=args.model,
        vocabulary=vocabulary,
        blank_id=len(vocabulary) - 1,
        frontend_path=args.frontend,
        runtime=args.runtime,
        pipeline="segmented",
        device=args.device,
        # Greedy only. An LM would improve both paths and obscure the thing
        # being measured, which is what the *encoder* does with long input.
        final_beam_decode=False,
    )

    print("Real-audio WER  (Common Voice English, this repository's shard)")
    print("=" * 78)
    clips = read_manifest(Path(args.manifest), Path(args.clips))
    random.Random(args.seed).shuffle(clips)
    print(f"manifest        : {len(clips)} clips matched")

    decoders = Decoders(config)
    print(f"model           : {Path(args.model).name}")
    print(f"providers       : {', '.join(decoders.engine.active_providers)}")
    print(f"decoder         : greedy CTC (no language model)")
    print()

    payload: dict[str, Any] = {
        "model": str(args.model),
        "providers": decoders.engine.active_providers,
        "decoder": "greedy",
        "corpus": {
            "manifest": str(args.manifest),
            "clips_dir": str(args.clips),
            "matched_clips": len(clips),
            "seed": args.seed,
        },
        "normalisation": "lowercase, hyphens to spaces, keep [a-z0-9' ], collapse whitespace",
    }

    selected = clips[: args.clip_count]
    if selected:
        print(f"--- short clips ({len(selected)}) " + "-" * 44)
        short = run_short_clips(decoders, selected)
        print(f"\n  audio            : {short.audio_seconds:.0f}s over {short.clips} "
              f"clips (mean {short.audio_seconds / max(1, short.clips):.1f}s)")
        print(f"  single-pass WER  : {short.single_pass.wer:.4f}")
        print(f"  segmented WER    : {short.segmented.wer:.4f}")
        payload["short_clips"] = {
            "clips": short.clips,
            "audio_seconds": round(short.audio_seconds, 1),
            "single_pass": {"wer": round(short.single_pass.wer, 4),
                            **asdict(short.single_pass)},
            "segmented": {"wer": round(short.segmented.wer, 4),
                          **asdict(short.segmented)},
            "worst_five": short.worst,
        }

    out = Path(args.json_out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        """Persist after every block.

        Writing once at the end lost a 35-minute run: a single-pass decode of a
        4-minute recording aborted inside ``onnxruntime_providers_cuda.dll``,
        which is a native ``terminate()`` and cannot be caught by ``except``.
        Everything measured before it went with it.
        """
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    save()

    payload["long_turns"] = []
    for target in (args.turn_seconds if args.turn_count else []):
        print(f"\n--- long turns (~{target:.0f}s, {args.gap:.1f}s gaps) " + "-" * 30)
        pool = clips[args.clip_count:] or clips
        turns = build_turns(pool, target, args.gap, args.turn_count)
        if not turns:
            print("  not enough audio to build a turn at that length")
            continue

        block: dict[str, Any] = {"target_seconds": target, "gap_seconds": args.gap}
        payload["long_turns"].append(block)

        def checkpoint(partial: dict[str, Any], _block=block) -> None:
            _block.update(partial)
            save()

        block.update(run_long_turns(decoders, turns, args.segment_caps, checkpoint))
        save()
        print(f"\n  {'segment cap':<16}{'WER':>8}{'forced cuts':>14}")
        print("  " + "-" * 38)
        print(f"  {'single pass':<16}{long_result['single_pass']['wer']:>8.4f}"
              f"{'n/a':>14}")
        for cap in args.segment_caps:
            entry = long_result["segmented"][f"{cap:g}s"]
            print(f"  {f'segmented {cap:g}s':<16}{entry['wer']:>8.4f}"
                  f"{entry['forced_cuts']:>14}")

    if len(payload["long_turns"]) > 1:
        print("\n  turn length vs WER" + " " * 8 + "single pass   best segmented")
        print("  " + "-" * 52)
        for entry in payload["long_turns"]:
            best = min(entry["segmented"].values(), key=lambda e: e["wer"])
            print(f"  {entry['mean_turn_seconds']:>6.0f}s"
                  f"{'':<21}{entry['single_pass']['wer']:>8.4f}"
                  f"{best['wer']:>15.4f}")

    save()
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
