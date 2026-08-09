"""Measure whether the model's CTC timestamps are temporally faithful.

**Run this against the real IndicConformer before trusting the default
configuration.** It answers one question, and the answer decides which aligner
is safe to use.

The time-aware stabilisation strategy rests on an assumption: when the same
spoken word is decoded from two overlapping windows, it lands at the same
*absolute* stream time in both. That holds when a model's CTC spikes fire near
the acoustic evidence, which well-trained Conformer-CTC models do -- but CTC
loss is alignment-free and does not enforce it. An undertrained or unusually
peaky model can instead emit its spikes at a fixed offset from the *window
start*, in which case a word's apparent timestamp advances by one chunk every
window and never settles.

The distinction is measurable. For each pair of adjacent windows, take the
words decoded from both and compare their absolute start times:

    drift ~= 0                 -> timestamps track the audio.
                                  `aligner="time"` is appropriate.
    drift ~= chunk_duration    -> timestamps track the window.
                                  Time-based matching will never commit;
                                  switch to `aligner="levenshtein"`.

Usage::

    python tools/check_alignment_fidelity.py \\
        --audio sample.wav --model model.onnx --vocabulary vocab.txt
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming_asr.audio.wav_source import WavFileSource  # noqa: E402
from streaming_asr.config import StreamingASRConfig, load_vocabulary  # noqa: E402
from streaming_asr.console import configure_logging, configure_stdout  # noqa: E402
from streaming_asr.events import ASREventType  # noqa: E402
from streaming_asr.hypothesis.aligner import (  # noqa: E402
    HypothesisAligner,
    LevenshteinAligner,
)
from streaming_asr.pipeline import StreamingASRPipeline  # noqa: E402
from streaming_asr.types import TimedWord  # noqa: E402

configure_stdout()


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def drift_samples(
    previous: Sequence[TimedWord],
    current: Sequence[TimedWord],
    aligner: HypothesisAligner,
) -> list[float]:
    """Timestamp drift for words present in both hypotheses.

    Words are paired by **sequence alignment**, not by spelling. Keying a
    lookup on the word text -- the obvious approach, and the one this tool
    originally used -- silently collapses repeated words: a sentence containing
    "it" twice keeps only one entry, so the next window can end up subtracting
    one occurrence from the *other* and recording their separation (often
    seconds) as drift. A handful of those wreck the spread and any tolerance
    derived from it.

    The aligner is deliberately **text-only** (Levenshtein). Using the
    time-aware aligner would be circular: it decides matches partly by
    timestamp, and timestamps are the thing being measured. Monotonic
    alignment is what disambiguates repeated words -- the first "it" can only
    pair with the first "it".
    """
    if not previous or not current:
        return []
    alignment = aligner.align(previous, current)
    return [
        current[c].start_time - previous[p].start_time
        for p, c in alignment.matched_pairs
    ]


def summarise(drifts: Sequence[float], windows: int, unmatched: int, total: int) -> dict:
    if not drifts:
        return {"windows": windows, "samples": 0}

    magnitudes = [abs(d) for d in drifts]
    return {
        "windows": windows,
        "samples": len(drifts),
        "unmatched_words": unmatched,
        "total_words": total,
        "mean_drift": statistics.fmean(drifts),
        "median_drift": statistics.median(drifts),
        # Robust spread: the median absolute deviation is not dragged around by
        # a few bad pairings the way a standard deviation is.
        "mad_drift": statistics.median([abs(d - statistics.median(drifts)) for d in drifts]),
        "p90_abs": _percentile(magnitudes, 90),
        "p95_abs": _percentile(magnitudes, 95),
        "max_abs": max(magnitudes),
    }


def measure(pipeline: StreamingASRPipeline, source: WavFileSource) -> dict:
    """Run the stream and collect per-word timestamp drift between windows."""
    aligner = LevenshteinAligner()
    drifts: list[float] = []
    previous: list[TimedWord] = []
    windows = 0
    unmatched = 0
    total = 0

    for event in pipeline.stream(source):
        if event.type is not ASREventType.PARTIAL:
            continue
        hypothesis = pipeline.last_hypothesis
        if hypothesis is None:
            continue
        windows += 1

        current = list(hypothesis.words)
        samples = drift_samples(previous, current, aligner)
        drifts.extend(samples)
        if previous and current:
            total += len(current)
            unmatched += len(current) - len(samples)
        previous = current

    return summarise(drifts, windows, unmatched, total)


def suggest_tolerance(stats: dict, frame_duration: float, chunk: float) -> tuple[float, str]:
    """Pick a ``time_tolerance`` that covers observed drift without over-reaching.

    Two competing constraints:

    * it must exceed the drift a word actually shows between windows, or the
      aligner stops recognising a word as itself;
    * it must stay well below the gap between two *distinct* utterances of the
      same word, or the aligner starts conflating them -- which is precisely
      the ambiguity time-awareness exists to resolve.

    So: cover the 95th percentile of observed drift plus one frame of slack,
    then clamp into a sane band. The old formula (three standard deviations)
    inherited every outlier the text-keyed pairing produced and suggested
    values larger than two chunks.
    """
    needed = stats["p95_abs"] + frame_duration
    # Never below two frames; never past 0.25s, where distinct repetitions of a
    # word start falling inside the window.
    lower, upper = max(0.08, 2 * frame_duration), 0.25
    tolerance = min(upper, max(lower, needed))

    if needed > upper:
        note = (f"drift exceeds what a safe tolerance can cover ({needed:.2f}s "
                f"needed); prefer aligner='levenshtein'")
    elif abs(tolerance - 0.12) < 0.005:
        note = "matches the default"
    else:
        note = "differs from the 0.12s default"
    return round(tolerance, 2), note


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocabulary")
    parser.add_argument("--chunk-ms", type=float, default=160.0)
    parser.add_argument("--context-sec", type=float, default=3.84)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    configure_logging(quiet=not args.verbose, verbose=args.verbose)

    kwargs: dict = {
        "chunk_duration": args.chunk_ms / 1000.0,
        "context_duration": args.context_sec,
        "onnx_model_path": args.model,
        "final_beam_decode": False,
    }
    if args.vocabulary:
        vocabulary = load_vocabulary(args.vocabulary)
        kwargs["vocabulary"] = vocabulary
        kwargs["blank_id"] = len(vocabulary) - 1
    config = StreamingASRConfig(**kwargs)

    pipeline = StreamingASRPipeline(config)
    # Warm up on a full-size window first. The encoder's subsampling factor is
    # measured from the inputs it sees, and a short warm-up window measures it
    # wrong -- which would scale every timestamp, and therefore every drift
    # figure this tool reports.
    pipeline.warmup(iterations=1)

    source = WavFileSource(args.audio, config.sample_rate, config.chunk_samples)
    stats = measure(pipeline, source)

    chunk = config.chunk_duration
    frame = pipeline.engine.ctc_frame_duration(pipeline.preprocessor.hop_duration)

    print(f"\nCTC alignment fidelity ({Path(args.audio).name})")
    print("=" * 62)
    print(f"  windows decoded          : {stats['windows']}")
    print(f"  aligned word pairs       : {stats.get('samples', 0)}")

    if not stats.get("samples"):
        print("\n  No word was decoded in two consecutive windows -- the model "
              "produced too little output to judge. Try longer audio.")
        return 1

    print(f"  words new each window    : {stats['unmatched_words']} of {stats['total_words']}")
    print(f"  chunk (step) size        : {chunk * 1000:.0f} ms "
          f"= {chunk / frame:.0f} CTC frames of {frame * 1000:.0f} ms")
    print("\n  drift (current window - previous), per word:")
    print(f"    median                 : {stats['median_drift'] * 1000:+.1f} ms")
    print(f"    mean                   : {stats['mean_drift'] * 1000:+.1f} ms")
    print(f"    MAD (robust spread)    : {stats['mad_drift'] * 1000:.1f} ms")
    print(f"    p90 |drift|            : {stats['p90_abs'] * 1000:.1f} ms")
    print(f"    p95 |drift|            : {stats['p95_abs'] * 1000:.1f} ms")
    print(f"    max |drift|            : {stats['max_abs'] * 1000:.1f} ms")
    print()

    # The median, not the mean: it is the statistic a few bad pairings cannot move.
    drift = abs(stats["median_drift"])
    fraction = drift / chunk if chunk else 0.0
    tolerance, note = suggest_tolerance(stats, frame, chunk)

    if drift < 0.3 * chunk:
        print(f"  VERDICT: timestamps track the AUDIO "
              f"(median drift is {fraction:.0%} of a chunk).")
        print("           aligner='time' (the default) is appropriate.")
        print(f"           Suggested time_tolerance: {tolerance:.2f}s ({note})")
        return 0
    if drift > 0.7 * chunk:
        print(f"  VERDICT: timestamps track the WINDOW, not the audio "
              f"(median drift is {fraction:.0%} of a chunk).")
        print("           A word's apparent time advances by roughly one chunk per")
        print("           window, so it never settles and time-based matching")
        print("           will never commit.")
        print("           Use aligner='levenshtein' and treat timestamps as")
        print("           unreliable for anything except ordering.")
        return 2
    print(f"  VERDICT: inconclusive -- median drift is {fraction:.0%} of a chunk, "
          f"between the two regimes.")
    print(f"           Try time_tolerance={tolerance:.2f}s, and compare against")
    print("           aligner='levenshtein' on real audio before deciding.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
