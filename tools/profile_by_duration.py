"""Per-duration profile: where the time goes, as a function of how long someone talks.

One recording per duration bin, smallest to largest, each run through the real
pipeline with the real checkpoint. Answers four questions that the aggregate
load numbers cannot:

* **How much of the wall clock is actually decoding?** Reported against speech
  rather than audio. A recording is mostly silence at the edges of every
  utterance, and RTF-against-audio quietly credits the model for time it spent
  waiting. ``rtf_speech`` is the honest compute figure; ``rtf_audio`` is the one
  that matters for capacity.
* **Does cost grow with turn length?** Segmented decoding says it should not --
  every forward pass is the same size no matter how long the speaker goes on.
* **What does the decoder add?** Greedy, beam without an LM, and beam with one.
  Note *where* that cost lands: the segmented pipeline has no separate final
  pass. Each segment is decoded authoritatively as it closes, so choosing a beam
  moves the cost into **every segment decode** rather than into one step at the
  end, and ``final_decode_time`` is structurally zero. The figure that matters
  is the per-segment decode time, and the difference between variants is the
  decoder's real price -- which a caller pays on every pause, not once per turn.
* **Where is the cliff?** Past roughly 150 s a single pass stops working
  (``PROJECT_REPORT.md`` 2.3). The segmented path should be flat through it.

Bins come from real clips where the corpus has them. Common Voice tops out
around 10 s, so anything longer is *constructed*: consecutive clips from one
speaker joined by ``--gap`` seconds of silence, which is what a person dictating
several sentences sounds like to a segmenter. Every row says which it is.

Usage::

    python tools/profile_by_duration.py \\
        --model model.onnx --vocabulary vocab.txt \\
        --clips common_voice_en/en_train_28 \\
        --manifest common_voice_en/train.tsv \\
        --json-out results/profile.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streaming_asr.config import StreamingASRConfig, load_vocabulary
from streaming_asr.events import ASREventType
from streaming_asr_lite.factory import build_engine, build_pipeline
from tools.real_audio_wer import (
    SAMPLE_RATE,
    Clip,
    _chunks,
    load_clip,
    normalize,
    read_manifest,
)

logger = logging.getLogger("profile")

#: Upper edges, seconds. Below ~10 s these are satisfied by real single clips;
#: above it they are constructed. The top of the range sits just under the
#: single-pass failure wall so the run completes.
DEFAULT_BINS = (5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0)


# ---------------------------------------------------------------------------
# building one recording per bin
# ---------------------------------------------------------------------------


@dataclass
class Item:
    """One recording to profile."""

    label: str
    audio: np.ndarray
    reference: str
    source: str          # "clip" or "constructed"
    clips: int

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def build_items(
    clips: list[Clip],
    bins: Sequence[float],
    gap: float,
    seed: int = 0,
) -> list[Item]:
    """One recording per bin, real where possible and constructed where not."""
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(clips)))
    loaded: dict[int, np.ndarray] = {}

    def audio_for(index: int) -> np.ndarray:
        if index not in loaded:
            loaded[index] = load_clip(clips[index])
        return loaded[index]

    items: list[Item] = []
    lower = 0.0
    used: set[int] = set()

    for upper in bins:
        # A real clip that lands in this bin, if the corpus has one.
        picked: Optional[int] = None
        for index in order:
            if index in used:
                continue
            duration = len(audio_for(index)) / SAMPLE_RATE
            if lower < duration <= upper:
                picked = index
                break

        if picked is not None:
            used.add(picked)
            items.append(Item(
                label=f"{upper:g}s",
                audio=audio_for(picked),
                reference=clips[picked].sentence,
                source="clip",
                clips=1,
            ))
            lower = upper
            continue

        # Nothing that long in the corpus: build it. One speaker throughout, so
        # the encoder is not asked to switch voices mid-utterance.
        target = (lower + upper) / 2 if lower else upper
        parts: list[np.ndarray] = []
        words: list[str] = []
        silence = np.zeros(int(gap * SAMPLE_RATE), dtype=np.float32)
        speaker: Optional[str] = None
        total = 0.0
        for index in order:
            if index in used:
                continue
            clip = clips[index]
            if speaker is None:
                speaker = clip.client_id
            elif clip.client_id != speaker:
                continue
            chunk = audio_for(index)
            if parts:
                parts.append(silence)
                total += gap
            parts.append(chunk)
            words.append(clip.sentence)
            used.add(index)
            total += len(chunk) / SAMPLE_RATE
            if total >= target:
                break

        if not parts:
            logger.warning("no clips left to build the %g s bin", upper)
            break
        items.append(Item(
            label=f"{upper:g}s",
            audio=np.concatenate(parts),
            reference=normalize(" ".join(words)),
            source="constructed",
            clips=len(words),
        ))
        lower = upper

    return items


# ---------------------------------------------------------------------------
# decoder variants
# ---------------------------------------------------------------------------


@dataclass
class Variant:
    """One finalisation setting to measure."""

    name: str
    final_beam_decode: bool
    beam_backend: str = "auto"
    lm_path: str = ""
    lexicon_path: str = ""


def plan_variants(args: argparse.Namespace) -> list[Variant]:
    variants = [Variant("greedy", final_beam_decode=False)]
    if "beam" in args.variants:
        variants.append(Variant("beam", True, beam_backend=args.beam_backend))
    if "beam_lm" in args.variants:
        if args.lm and args.lexicon:
            variants.append(Variant(
                "beam_lm", True, beam_backend="flashlight",
                lm_path=args.lm, lexicon_path=args.lexicon,
            ))
        else:
            logger.warning(
                "beam_lm requested but --lm/--lexicon not given; skipping. "
                "Without them 'beam' already measures the LM-free beam."
            )
    return variants


def config_for(base: StreamingASRConfig, variant: Variant) -> StreamingASRConfig:
    raw = base.to_dict()
    raw["final_beam_decode"] = variant.final_beam_decode
    beam = dict(raw.get("beam") or {})
    if variant.beam_backend:
        beam["backend"] = variant.beam_backend
    if variant.lm_path:
        beam["lm_path"] = variant.lm_path
    if variant.lexicon_path:
        beam["lexicon_path"] = variant.lexicon_path
    raw["beam"] = beam
    return StreamingASRConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def profile(
    item: Item,
    config: StreamingASRConfig,
    engine: Any,
    repeat: int,
) -> dict[str, Any]:
    """Run one recording through the pipeline and report where the time went."""
    runs: list[dict[str, Any]] = []

    for _ in range(max(1, repeat)):
        pipeline = build_pipeline(config, engine=engine)
        spans: list[tuple[float, float]] = []
        decoder_names: list[str] = []
        forced = 0

        started = time.perf_counter()
        for event in pipeline.stream(_chunks(item.audio, config.chunk_samples)):
            if event.type is ASREventType.SEGMENT:
                spans.append((event.window_start or 0.0, event.window_end or 0.0))
                forced += bool(event.metrics.get("forced"))
                if event.decoder:
                    decoder_names.append(event.decoder)
        wall = time.perf_counter() - started

        metrics = pipeline.metrics
        snapshot = metrics.snapshot()
        speech = sum(end - start for start, end in spans)
        audio_seconds = item.duration

        inference = metrics.inference_times
        segment_decode = metrics.segment_decode_times
        # p95 is not offered: BoundedSamples reports p50/p90 over a sliding
        # window, and inventing a p95 from a handful of segments would be
        # precision this sample size does not have.
        decode_stats = segment_decode.summary()

        runs.append({
            "wall_seconds": round(wall, 4),
            "speech_seconds": round(speech, 3),
            "silence_seconds": round(max(0.0, audio_seconds - speech), 3),
            "speech_fraction": round(speech / audio_seconds, 4) if audio_seconds else 0.0,
            "segments": len(spans),
            "forced_cuts": forced,
            "model_calls": metrics.model_calls,

            # Compute, isolated from pacing and from silence.
            "inference_seconds": round(inference.total, 4),
            "preprocess_seconds": round(metrics.preprocess_times.total, 4),
            "compute_seconds": round(metrics.total_compute_time, 4),

            # The two RTFs. rtf_speech is what the model costs per second of
            # speech; rtf_audio is what it costs per second of wall-clock a
            # caller occupies, which is the capacity figure.
            "rtf_audio": round(metrics.total_compute_time / audio_seconds, 4)
            if audio_seconds else 0.0,
            "rtf_speech": round(metrics.total_compute_time / speech, 4) if speech else 0.0,
            "inference_rtf_speech": round(inference.total / speech, 4) if speech else 0.0,

            # What a speaker waits for after a pause: silence wait + decode.
            "segment_decode_mean_ms": round(decode_stats["mean"] * 1000, 1)
            if segment_decode.count else None,
            "segment_decode_p50_ms": round(decode_stats["p50"] * 1000, 1)
            if segment_decode.count else None,
            "segment_decode_p90_ms": round(decode_stats["p90"] * 1000, 1)
            if segment_decode.count else None,
            "response_p50_ms": round(
                (metrics.segment_silence + decode_stats["p50"]) * 1000, 1
            ) if segment_decode.count else None,

            # Finalisation. Zero for the segmented pipeline by construction --
            # kept so the JSON is comparable if this is ever run against the
            # windowed pipeline, which does have a real final pass.
            "final_decode_seconds": round(metrics.final_decode_time, 4),
            "final_inference_seconds": round(metrics.final_inference_time, 4),
            "first_partial_ms": round(snapshot.get("first_partial_latency", 0.0) * 1000, 1),
            "decoder": sorted(set(decoder_names))[0] if decoder_names else "",
            "transcript": normalize(pipeline.transcript),
        })

    return _median_run(runs)


def _median_run(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Median across repeats for the numbers; first run for everything else."""
    if len(runs) == 1:
        return runs[0]
    merged = dict(runs[0])
    for key, value in runs[0].items():
        if isinstance(value, (int, float)) and value is not None:
            values = [r[key] for r in runs if isinstance(r.get(key), (int, float))]
            if values:
                merged[key] = round(statistics.median(values), 4)
    merged["repeats"] = len(runs)
    return merged


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--clips", default="common_voice_en/en_train_28")
    parser.add_argument("--manifest", default="common_voice_en/train.tsv")
    parser.add_argument("--audio", nargs="*", default=[],
                        help="profile these files instead of sampling the corpus")
    parser.add_argument("--bins", default=",".join(f"{b:g}" for b in DEFAULT_BINS),
                        help="upper edges in seconds, comma separated")
    parser.add_argument("--gap", type=float, default=0.7,
                        help="silence between clips when a bin must be constructed")
    parser.add_argument("--variants", default="greedy,beam,beam_lm",
                        help="which finalisation paths to measure")
    parser.add_argument("--beam-backend", default="auto",
                        choices=("auto", "flashlight", "pyctcdecode", "pure_python"))
    parser.add_argument("--lm", default="", help="KenLM binary, enables beam_lm")
    parser.add_argument("--lexicon", default="", help="lexicon for flashlight")
    parser.add_argument("--repeat", type=int, default=1,
                        help="runs per cell; the median is reported")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--providers", default="auto")
    parser.add_argument("--runtime", default="lite", choices=("lite", "torch"))
    parser.add_argument("--frontend", default="fixtures/frontend.onnx")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    vocabulary = load_vocabulary(args.vocabulary)
    base = StreamingASRConfig(
        onnx_model_path=args.model,
        vocabulary=vocabulary,
        blank_id=len(vocabulary) - 1,
        device=args.device,
        providers=args.providers,
        runtime=args.runtime,
        frontend_path=args.frontend,
        pipeline="segmented",
        final_beam_decode=False,
    )

    if args.runtime == "torch":
        from streaming_asr.device import resolve_device

        engine = build_engine(base, providers=resolve_device(
            args.device, args.providers).providers)
    else:
        engine = build_engine(base)

    bins = [float(b) for b in args.bins.split(",") if b.strip()]
    if args.audio:
        from streaming_asr_lite.audio import decode_audio

        items = []
        for path in args.audio:
            audio = decode_audio(Path(path), SAMPLE_RATE)
            items.append(Item(Path(path).stem, audio, "", "file", 1))
    else:
        clips = read_manifest(Path(args.manifest), Path(args.clips))
        items = build_items(clips, bins, args.gap, args.seed)

    args.variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    variants = plan_variants(args)

    print(f"model    : {args.model}")
    print(f"runtime  : {args.runtime} / providers={engine.active_providers}")
    print(f"variants : {', '.join(v.name for v in variants)}")
    print(f"items    : {len(items)} ({sum(i.source == 'clip' for i in items)} real clips, "
          f"{sum(i.source == 'constructed' for i in items)} constructed)")
    print()

    payload: dict[str, Any] = {
        "model": args.model,
        "runtime": args.runtime,
        "providers": list(engine.active_providers),
        "gap_seconds": args.gap,
        "variants": [v.name for v in variants],
        "items": [],
    }

    # Warm up the session and then every variant. The first forward pass on a
    # fresh session pays for kernel selection and arena growth, and each
    # decoder backend has its own first-call cost -- both would otherwise land
    # entirely on the smallest bin and make it look like the most expensive.
    # Measured before this existed: the 5 s beam cell read 3231 ms against
    # 1864 ms for the 10 s cell, which is backwards.
    warm = build_pipeline(base, engine=engine)
    warm.warmup(iterations=2)
    if items:
        sample = items[0].audio[: 3 * SAMPLE_RATE]
        for variant in variants:
            pipeline = build_pipeline(config_for(base, variant), engine=engine)
            for _ in pipeline.stream(_chunks(sample, base.chunk_samples)):
                pass

    for item in items:
        row: dict[str, Any] = {
            "bin": item.label,
            "audio_seconds": round(item.duration, 2),
            "source": item.source,
            "clips": item.clips,
            "variants": {},
        }
        for variant in variants:
            config = config_for(base, variant)
            try:
                row["variants"][variant.name] = profile(
                    item, config, engine, args.repeat
                )
            except Exception as exc:
                logger.warning("%s / %s failed: %s", item.label, variant.name, exc)
                row["variants"][variant.name] = {"error": str(exc)}
        payload["items"].append(row)
        _print_row(row, variants)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")

    _print_summary(payload, variants)
    return 0


def _print_row(row: dict[str, Any], variants: list[Variant]) -> None:
    greedy = row["variants"].get("greedy", {})
    if "error" in greedy:
        print(f"{row['bin']:>6}  FAILED: {greedy['error'][:60]}")
        return
    print(
        f"{row['bin']:>6}  {row['audio_seconds']:6.1f}s audio  "
        f"{greedy['speech_seconds']:6.1f}s speech ({greedy['speech_fraction']*100:.0f}%)  "
        f"{greedy['segments']:3d} seg  "
        f"infer {greedy['inference_seconds']:6.2f}s  "
        f"rtf_audio {greedy['rtf_audio']:.3f}  rtf_speech {greedy['rtf_speech']:.3f}  "
        f"[{row['source'][:5]}]"
    )


def _print_summary(payload: dict[str, Any], variants: list[Variant]) -> None:
    print("\n" + "=" * 78)
    print("Compute, isolated from silence")
    print("=" * 78)
    print(f"{'bin':>6} {'audio':>7} {'speech':>7} {'sil':>6} {'seg':>4} "
          f"{'infer_s':>8} {'rtf_a':>6} {'rtf_s':>6}")
    for row in payload["items"]:
        g = row["variants"].get("greedy", {})
        if "error" in g:
            continue
        print(f"{row['bin']:>6} {row['audio_seconds']:7.1f} {g['speech_seconds']:7.1f} "
              f"{g['silence_seconds']:6.1f} {g['segments']:4d} "
              f"{g['inference_seconds']:8.2f} {g['rtf_audio']:6.3f} {g['rtf_speech']:6.3f}")
    print("\nrtf_a = compute / audio duration   (capacity: what a caller occupies)")
    print("rtf_s = compute / speech duration  (what the model actually costs)")

    print("\n" + "=" * 78)
    print("Decoder cost, per segment decode")
    print("=" * 78)
    print("The segmented pipeline decodes each segment authoritatively as it")
    print("closes, so a beam is paid on every pause -- not once per turn.")
    print()
    header = f"{'bin':>6} {'seg':>4}"
    for v in variants:
        header += f" {v.name:>14}"
    if len(variants) > 1:
        header += f" {'vs greedy':>12}"
    print(header)
    for row in payload["items"]:
        g = row["variants"].get("greedy", {})
        if "error" in g or g.get("segment_decode_mean_ms") is None:
            continue
        line = f"{row['bin']:>6} {g['segments']:4d}"
        for v in variants:
            data = row["variants"].get(v.name, {})
            value = data.get("segment_decode_mean_ms")
            line += f" {value:11.0f} ms" if value is not None else f" {'-':>14}"
        if len(variants) > 1:
            last = row["variants"].get(variants[-1].name, {})
            value = last.get("segment_decode_mean_ms")
            base = g.get("segment_decode_mean_ms")
            if value is not None and base is not None:
                line += f" {value - base:+9.0f} ms"
        print(line)

    if not any(v.name == "beam_lm" for v in variants):
        print("\nNOTE: beam_lm was not measured. Without --lm/--lexicon the 'beam'")
        print("column is an LM-free beam, which is a different thing: the LM is")
        print("the expensive half. See PROJECT_REPORT.md 7 on the untested path.")


if __name__ == "__main__":
    raise SystemExit(main())
