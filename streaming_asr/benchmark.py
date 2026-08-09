"""Benchmark runner: sweep the window geometry and measure the tradeoffs.

The reference operating point (4.0 s window / 160 ms step) is what the existing
prototype happens to do, not a measured optimum. It re-processes every sample
25 times. This sweeps context and chunk size so the question can be settled
with numbers.

Two things are measured that are easy to conflate:

*Perceived latency* is dominated by ``chunk_duration`` (how often a partial can
appear at all) and by ``stability_window`` (how long a word is withheld). It is
independent of GPU speed.

*Compute* is dominated by ``window_redundancy = buffer / chunk``. Halving the
window halves the per-call cost; doubling the step halves the number of calls.
Both reduce load, but only the second increases latency.

Section 23 asks specifically how trustworthy greedy decoding is as an
intermediate signal for this model. :func:`compare_greedy_vs_beam` answers it
by decoding the same logits both ways.

Usage::

    python -m streaming_asr.benchmark \\
        --audio short.wav long.wav --model model.onnx \\
        --contexts 1 2 3 4 5 --chunks 160 320 500
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from streaming_asr.audio.wav_source import InMemorySource, load_wav
from streaming_asr.config import (
    BeamDecoderConfig,
    SegmentationConfig,
    StabilityConfig,
    StreamingASRConfig,
    load_vocabulary,
)
from streaming_asr.console import configure_logging, configure_stdout
from streaming_asr.device import resolve_device
from streaming_asr.events import ASREventType
from streaming_asr.inference.onnx_engine import ONNXASREngine
from streaming_asr.pipeline import StreamingASRPipeline

logger = logging.getLogger(__name__)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = list(reference), list(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


@dataclass
class BenchmarkRow:
    """One (audio, context, chunk) measurement."""

    audio: str
    audio_duration: float
    context_sec: float
    chunk_ms: float
    window_sec: float
    window_redundancy: float
    model_calls: int
    rtf: float
    streaming_rtf: float
    first_partial_latency: Optional[float]
    avg_update_latency: float
    stable_token_latency: float
    finalization_latency: Optional[float]
    avg_inference_ms: float
    avg_preprocess_ms: float
    final_text: str = ""
    streaming_text: str = ""
    wer_final: Optional[float] = None
    wer_streaming: Optional[float] = None
    gpu: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def run_one(
    audio: np.ndarray,
    audio_name: str,
    sample_rate: int,
    context_sec: float,
    chunk_ms: float,
    engine: ONNXASREngine,
    base_config: StreamingASRConfig,
    reference: Optional[str] = None,
    real_time: bool = False,
    pipeline_kind: str = "segmented",
) -> BenchmarkRow:
    """Run the full pipeline once at one operating point.

    ``context_sec`` means different things per pipeline: the rolling window for
    ``windowed``, the segment length cap for ``segmented``. Both are "how much
    audio does one model call see", which is what the sweep is varying.
    """
    config = StreamingASRConfig(
        sample_rate=sample_rate,
        chunk_duration=chunk_ms / 1000.0,
        context_duration=context_sec if pipeline_kind == "windowed" else 3.84,
        onnx_model_path=base_config.onnx_model_path,
        lexicon_path=base_config.lexicon_path,
        lm_path=base_config.lm_path,
        vocabulary=base_config.vocabulary,
        blank_id=base_config.blank_id,
        pipeline=pipeline_kind,
        greedy_decode=True,
        final_beam_decode=base_config.final_beam_decode,
        device=base_config.device,
        providers=base_config.providers,
        beam=base_config.beam,
        stability=StabilityConfig(
            # Keep the stability policy proportional to the step size, so the
            # sweep compares window geometries rather than accidentally
            # comparing commit policies.
            stability_window=min(
                base_config.stability.stability_window, 0.45 * context_sec
            ),
            min_stable_updates=base_config.stability.min_stable_updates,
            aligner=base_config.stability.aligner,
            time_tolerance=max(base_config.stability.time_tolerance, chunk_ms / 1000.0),
        ),
        endpoint=base_config.endpoint,
        segmentation=SegmentationConfig(
            segment_silence=base_config.segmentation.segment_silence,
            turn_silence=base_config.segmentation.turn_silence,
            max_segment_duration=context_sec,
            energy_threshold=base_config.segmentation.energy_threshold,
        ),
    )

    if pipeline_kind == "segmented":
        from streaming_asr.segmented import SegmentedASRPipeline

        pipeline = SegmentedASRPipeline(config, engine=engine)
    else:
        pipeline = StreamingASRPipeline(config, engine=engine)
    pipeline.warmup(iterations=1)
    pipeline.metrics.real_time_paced = real_time

    source = InMemorySource(
        samples=audio, sample_rate=sample_rate,
        chunk_samples=config.chunk_samples, real_time=real_time,
    )

    final = None
    for event in pipeline.stream(source):
        if event.type is ASREventType.FINAL:
            final = event

    metrics = pipeline.metrics
    snapshot = metrics.snapshot()
    row = BenchmarkRow(
        audio=audio_name,
        audio_duration=len(audio) / sample_rate,
        context_sec=context_sec,
        chunk_ms=chunk_ms,
        window_sec=config.buffer_duration,
        window_redundancy=config.window_redundancy,
        model_calls=metrics.model_calls,
        rtf=metrics.rtf,
        streaming_rtf=metrics.streaming_rtf,
        first_partial_latency=metrics.first_partial_latency,
        avg_update_latency=snapshot["update_latency"]["mean"],
        stable_token_latency=snapshot["stable_token_latency"]["mean"],
        finalization_latency=metrics.finalization_latency,
        avg_inference_ms=snapshot["inference"]["mean"] * 1000,
        avg_preprocess_ms=snapshot["preprocess"]["mean"] * 1000,
        final_text=getattr(pipeline, "transcript", None) or (final.text if final else ""),
        streaming_text=final.provisional_text if final else "",
        gpu=metrics.gpu_stats(),
    )
    if reference:
        row.wer_final = word_error_rate(reference, row.final_text)
        row.wer_streaming = word_error_rate(reference, row.streaming_text)
    return row


def compare_greedy_vs_beam(
    audio: np.ndarray,
    sample_rate: int,
    engine: ONNXASREngine,
    config: StreamingASRConfig,
    reference: Optional[str] = None,
) -> dict[str, Any]:
    """Decode one utterance both ways from identical logits.

    This is the measurement section 23 calls for. Greedy partials are only
    useful as an incremental semantic signal if they are close enough to the
    authoritative transcript; how close is a property of this specific model
    and must be measured rather than assumed.
    """
    pipeline = StreamingASRPipeline(config, engine=engine)

    features, lengths = pipeline.preprocessor(audio.reshape(1, -1), n_samples=len(audio))
    result = engine.run_torch(features, int(lengths[0]))

    greedy_start = time.perf_counter()
    greedy_text = pipeline.greedy.decode_text(result.logits)
    greedy_time = time.perf_counter() - greedy_start

    beam_result = pipeline.final_decoder.decode(result.logits)

    comparison: dict[str, Any] = {
        "greedy_text": greedy_text,
        "beam_text": beam_result.text,
        "beam_backend": beam_result.backend,
        "beam_used_lm": beam_result.used_lm,
        "greedy_decode_time": round(greedy_time, 5),
        "beam_decode_time": round(beam_result.decode_time, 5),
        "beam_slowdown_x": round(beam_result.decode_time / max(greedy_time, 1e-9), 1),
        "wer_greedy_vs_beam": round(word_error_rate(beam_result.text, greedy_text), 4),
        "cer_greedy_vs_beam": round(character_error_rate(beam_result.text, greedy_text), 4),
    }
    if reference:
        comparison["wer_greedy_vs_reference"] = round(word_error_rate(reference, greedy_text), 4)
        comparison["wer_beam_vs_reference"] = round(word_error_rate(reference, beam_result.text), 4)
    return comparison


def render_table(rows: Sequence[BenchmarkRow]) -> str:
    header = (
        f"{'audio':<14}{'ctx':>6}{'chunk':>7}{'win':>6}{'redun':>7}{'calls':>7}"
        f"{'RTF':>7}{'infer':>8}{'1st_par':>9}{'stable':>8}{'final':>8}{'WER':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{r.audio[:13]:<14}{r.context_sec:>6.2f}{r.chunk_ms:>7.0f}{r.window_sec:>6.2f}"
            f"{r.window_redundancy:>7.1f}{r.model_calls:>7d}{r.rtf:>7.3f}"
            f"{r.avg_inference_ms:>7.1f}m"
            f"{(r.first_partial_latency or 0):>9.3f}"
            f"{r.stable_token_latency:>8.2f}"
            f"{(r.finalization_latency or 0):>8.2f}"
            f"{(r.wer_final if r.wer_final is not None else float('nan')):>7.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--audio", nargs="+", required=True,
                        help="one or more WAV files; use both ~10s and ~60s recordings")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocabulary")
    parser.add_argument("--lexicon")
    parser.add_argument("--lm")
    parser.add_argument("--reference", nargs="*", default=[],
                        help="reference transcript per audio file, in order")
    parser.add_argument("--pipeline", default="segmented",
                        choices=["segmented", "windowed"])
    parser.add_argument("--mode", default="simulated-streaming",
                        choices=["simulated-streaming", "real-time"],
                        help="'real-time' paces audio at wall-clock speed. Slower to "
                             "run, but the only RTF worth sizing capacity from: fed "
                             "flat out the GPU stays clocked up and the figure comes "
                             "out ~1.5x optimistic")
    parser.add_argument("--contexts", nargs="+", type=float, default=[1.0, 2.0, 3.0, 4.0, 5.0],
                        help="rolling window (windowed) or segment cap (segmented), seconds")
    parser.add_argument("--chunks", nargs="+", type=float, default=[160.0, 320.0, 500.0])
    parser.add_argument("--beam-backend", default="auto",
                        choices=["auto", "flashlight", "pyctcdecode", "pure_python"])
    parser.add_argument("--no-final-beam", dest="final_beam", action="store_false", default=True,
                        help="skip finalisation; measures the streaming path alone")
    parser.add_argument("--compare-decoders", action="store_true",
                        help="also run the greedy-vs-beam+LM comparison")
    parser.add_argument("--json-out", default="benchmark_results.json")
    parser.add_argument("--device", default="auto",
                        help="'auto', 'cpu', 'cuda' or 'cuda:N'")
    parser.add_argument("--providers", default="auto")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    configure_logging(verbose=args.verbose, quiet=not args.verbose)

    vocabulary = load_vocabulary(args.vocabulary) if args.vocabulary else None
    base_kwargs: dict = {
        "onnx_model_path": args.model,
        "lexicon_path": args.lexicon,
        "lm_path": args.lm,
        "final_beam_decode": args.final_beam,
        "device": args.device,
        "providers": args.providers,
        "beam": BeamDecoderConfig(backend=args.beam_backend),
    }
    if vocabulary is not None:
        base_kwargs["vocabulary"] = vocabulary
        base_kwargs["blank_id"] = len(vocabulary) - 1
    base_config = StreamingASRConfig(**base_kwargs)

    # One engine for the whole sweep. Reloading the ONNX session per cell would
    # dominate the timings and measure nothing useful.
    placement = resolve_device(args.device, args.providers)
    engine = ONNXASREngine(
        args.model, providers=placement.providers, device_id=placement.device_id
    )
    print(engine.graph_report.render())
    print(f"runtime: frontend={placement.torch_device}, "
          f"session={', '.join(engine.active_providers)}, "
          f"zero_copy={placement.zero_copy_possible}")
    print()

    references = list(args.reference) + [None] * (len(args.audio) - len(args.reference))
    rows: list[BenchmarkRow] = []
    comparisons: dict[str, Any] = {}
    real_time = args.mode == "real-time"

    if real_time:
        # Real-time pacing means the sweep takes at least as long as the audio,
        # once per cell. Say so up front rather than letting it look hung.
        total_audio = sum(
            len(load_wav(p, base_config.sample_rate)) / base_config.sample_rate
            for p in args.audio
        )
        cells = len(args.contexts) * len(args.chunks)
        print(f"real-time pacing: {cells} configurations x {total_audio:.0f}s of audio "
              f"= at least {cells * total_audio / 60:.0f} minutes.\n")

    for audio_path, reference in zip(args.audio, references):
        if reference and Path(reference).exists():
            reference = Path(reference).read_text(encoding="utf-8").strip()
        audio = load_wav(audio_path, target_sample_rate=base_config.sample_rate)
        name = Path(audio_path).stem
        print(f"=== {name} ({len(audio) / base_config.sample_rate:.1f}s) ===")

        if args.compare_decoders:
            comparisons[name] = compare_greedy_vs_beam(
                audio, base_config.sample_rate, engine, base_config, reference
            )
            print(json.dumps(comparisons[name], indent=2))
            print()

        for context in args.contexts:
            for chunk in args.chunks:
                try:
                    row = run_one(
                        audio=audio, audio_name=name,
                        sample_rate=base_config.sample_rate,
                        context_sec=context, chunk_ms=chunk,
                        engine=engine, base_config=base_config, reference=reference,
                        real_time=real_time, pipeline_kind=args.pipeline,
                    )
                except Exception as exc:  # keep the sweep going
                    logger.error("context=%.2f chunk=%.0f failed: %s", context, chunk, exc)
                    continue
                rows.append(row)
                print(f"  ctx={context:.2f}s chunk={chunk:.0f}ms -> "
                      f"RTF={row.rtf:.3f} calls={row.model_calls} "
                      f"1st_partial={(row.first_partial_latency or 0):.3f}s "
                      f"WER={row.wer_final if row.wer_final is not None else float('nan'):.3f}")
        print()

    print(render_table(rows))

    payload = {
        "rows": [r.to_dict() for r in rows],
        "decoder_comparison": comparisons,
        "graph_report": {
            "stateless": engine.graph_report.is_stateless,
            "inputs": [str(s) for s in engine.graph_report.inputs],
            "outputs": [str(s) for s in engine.graph_report.outputs],
            "providers": engine.graph_report.providers,
            "subsampling_factor": engine.subsampling_factor,
        },
    }
    Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
