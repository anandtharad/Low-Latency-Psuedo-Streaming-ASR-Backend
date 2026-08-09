"""Command-line entry point for the streaming ASR prototype.

Example::

    python -m streaming_asr.cli \\
        --audio patient.wav \\
        --model /data/asr/ams/Conformer-CTC-BPE-v1_95_400-averaged.onnx \\
        --lexicon /data/asr/lms/12.0/merged_lm.lexicon \\
        --lm /data/asr/lms/12.0/merged_lm.bin \\
        --chunk-ms 160 --context-sec 3.84
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from streaming_asr.audio.wav_source import WavFileSource
from streaming_asr.config import (
    BeamDecoderConfig,
    EndpointConfig,
    SegmentationConfig,
    StabilityConfig,
    StreamingASRConfig,
    load_vocabulary,
)
from streaming_asr.console import configure_logging, configure_stdout
from streaming_asr.events import ASREvent, ASREventType
from streaming_asr.pipeline import StreamingASRPipeline
from streaming_asr.trace import HypothesisTracer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streaming_asr",
        description="Rolling-buffer pseudo-streaming ASR over an offline Conformer-CTC ONNX model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--audio", help="path to a WAV file")
    io_group.add_argument("--mic", nargs="?", const="default", metavar="DEVICE",
                          help="capture from a microphone instead of a file; "
                               "optionally a device index or name")
    io_group.add_argument("--list-devices", action="store_true",
                          help="list capture devices and exit")
    io_group.add_argument("--max-seconds", type=float,
                          help="stop capturing after this long (mic only)")
    io_group.add_argument("--model", required=True, help="path to the ONNX model")
    io_group.add_argument("--vocabulary", help="vocabulary file (one token per line)")
    io_group.add_argument("--lexicon", help="KenLM lexicon path")
    io_group.add_argument("--lm", help="KenLM binary path")
    io_group.add_argument("--json-out", help="write the transcript and metrics as JSON")

    geom = parser.add_argument_group("window geometry")
    geom.add_argument("--chunk-ms", type=float, default=160.0,
                      help="step size: how much new audio per model call")
    geom.add_argument("--context-sec", type=float, default=3.84,
                      help="retained history; window = context + chunk")

    dec = parser.add_argument_group("decoding")
    dec.add_argument("--greedy", dest="greedy", action="store_true", default=True,
                     help="greedy-decode every window for partials")
    dec.add_argument("--no-greedy", dest="greedy", action="store_false")
    dec.add_argument("--final-beam", dest="final_beam", action="store_true", default=True,
                     help="run beam+KenLM at the endpoint")
    dec.add_argument("--no-final-beam", dest="final_beam", action="store_false")
    dec.add_argument("--beam-size", type=int, default=50)
    dec.add_argument("--beam-size-token", type=int, default=50)
    dec.add_argument("--beam-threshold", type=float, default=20.0)
    dec.add_argument("--lm-weight", type=float, default=2.0)
    dec.add_argument("--word-score", type=float, default=0.0)
    dec.add_argument("--beam-backend", default="auto",
                     choices=["auto", "flashlight", "pyctcdecode", "pure_python"])

    stab = parser.add_argument_group("stabilisation")
    stab.add_argument("--stability-updates", type=int, default=2,
                      help="consecutive agreeing windows required to commit a word")
    stab.add_argument("--stability-window", type=float, default=0.6,
                      help="right-context (seconds) required before committing")
    stab.add_argument("--aligner", default="time",
                      choices=["time", "prefix", "levenshtein", "dtw"])
    stab.add_argument("--enable-dtw", action="store_true",
                      help="shorthand for --aligner dtw")
    stab.add_argument("--time-tolerance", type=float, default=0.12)

    ep = parser.add_argument_group("endpointing")
    ep.add_argument("--endpoint", default="explicit", choices=["explicit", "energy", "none"])
    ep.add_argument("--silence-sec", type=float, default=0.8)
    ep.add_argument("--energy-threshold", type=float, default=0.005)

    run = parser.add_argument_group("execution")
    run.add_argument("--runtime", default="lite", choices=["lite", "torch"],
                     help="'lite' runs the mel frontend as ONNX and needs no torch "
                          "(68 MB / 0.28s startup vs 425 MB / 2.52s); 'torch' is the "
                          "original torchaudio frontend, kept as a fallback")
    run.add_argument("--frontend", default=None,
                     help="path to the exported frontend.onnx (default "
                          "fixtures/frontend.onnx); only used by --runtime lite")
    run.add_argument("--pipeline", default="segmented",
                     choices=["segmented", "windowed"],
                     help="'segmented' cuts at pauses and decodes each span whole "
                          "(default); 'windowed' is the rolling-buffer pipeline with "
                          "word-level commitment")
    run.add_argument("--segment-silence", type=float, default=0.5,
                     help="silence that closes a segment and publishes its text")
    run.add_argument("--turn-silence", type=float, default=1.5,
                     help="silence that ends the turn and emits the final")
    run.add_argument("--max-segment-sec", type=float, default=10.0,
                     help="hard cap on a segment with no pause in it")
    run.add_argument("--mode", default="simulated-streaming",
                     choices=["simulated-streaming", "real-time"],
                     help="real-time paces chunks at wall-clock speed")
    run.add_argument("--device", default="auto",
                     help="'auto', 'cpu', 'cuda' or 'cuda:N'. Places the mel frontend "
                          "and the ONNX session together; 'cuda' errors out if "
                          "unavailable rather than silently using the CPU")
    run.add_argument("--providers", default="auto",
                     help="'auto', or an explicit ONNX Runtime provider name "
                          "(overrides what --device would choose)")
    run.add_argument("--reference", help="reference transcript, to report WER")
    run.add_argument("--inspect-model", action="store_true",
                     help="print the ONNX graph report and exit")
    run.add_argument("--compare-greedy", action="store_true",
                     help="also run an offline greedy decode and compare with beam+LM")
    run.add_argument("--show-windows", action="store_true",
                     help="log window/new-audio spans with every partial")
    run.add_argument("--trace", metavar="FILE",
                     help="write a per-window hypothesis trace (.jsonl for machine-readable)")
    run.add_argument("--trace-tokens", action="store_true",
                     help="include token-level timings in the trace")
    run.add_argument("--quiet", action="store_true")
    run.add_argument("--verbose", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> StreamingASRConfig:
    vocabulary = load_vocabulary(args.vocabulary) if args.vocabulary else None
    aligner = "dtw" if args.enable_dtw else args.aligner

    kwargs: dict = {
        "chunk_duration": args.chunk_ms / 1000.0,
        "context_duration": args.context_sec,
        "onnx_model_path": args.model,
        "lexicon_path": args.lexicon,
        "lm_path": args.lm,
        "greedy_decode": args.greedy,
        "final_beam_decode": args.final_beam,
        "runtime": args.runtime,
        "frontend_path": args.frontend,
        "pipeline": args.pipeline,
        "device": args.device,
        "providers": args.providers,
        "beam": BeamDecoderConfig(
            beam_size=args.beam_size,
            beam_size_token=args.beam_size_token,
            beam_threshold=args.beam_threshold,
            lm_weight=args.lm_weight,
            word_score=args.word_score,
            backend=args.beam_backend,
        ),
        "stability": StabilityConfig(
            stability_window=args.stability_window,
            min_stable_updates=args.stability_updates,
            aligner=aligner,
            time_tolerance=args.time_tolerance,
        ),
        "endpoint": EndpointConfig(
            detector=args.endpoint,
            silence_duration=args.silence_sec,
            energy_threshold=args.energy_threshold,
        ),
        "segmentation": SegmentationConfig(
            segment_silence=args.segment_silence,
            turn_silence=args.turn_silence,
            max_segment_duration=args.max_segment_sec,
            energy_threshold=args.energy_threshold,
        ),
    }
    if vocabulary is not None:
        kwargs["vocabulary"] = vocabulary
        # A vocabulary file is expected to already include the blank symbol.
        kwargs["blank_id"] = len(vocabulary) - 1
    return StreamingASRConfig(**kwargs)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER via Levenshtein distance over words."""
    ref, hyp = reference.split(), hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (r != h),
            ))
        previous = current
    return previous[-1] / len(ref)


def _render_runtime(pipeline) -> str:
    """Report where things actually ran, not where they were asked to run."""
    lines = [
        "",
        "Runtime placement",
        "-" * 60,
        f"onnx session : {', '.join(pipeline.engine.active_providers)}",
    ]

    placement = getattr(pipeline, "placement", None)
    if placement is None:
        # Lite runtime: the frontend is an ONNX graph on the same providers as
        # the encoder, so there is no separate torch device to report.
        lines.append("mel frontend : ONNX (same providers as the encoder)")
        return "\n".join(lines)

    from streaming_asr.device import gpu_name

    lines.insert(3, f"mel frontend : {placement.torch_device}")
    name = gpu_name(placement.device_id)
    if name:
        lines.append(f"gpu          : {name}")
    if placement.zero_copy_possible:
        lines.append("transfers    : zero-copy (features stay in device memory)")
    elif pipeline.engine.on_cuda:
        lines.append("transfers    : host->device per window (frontend is on CPU)")
    else:
        lines.append("transfers    : n/a (CPU)")
    return "\n".join(lines)


def _render_partial(event: ASREvent, show_windows: bool) -> str:
    lines = []
    if show_windows:
        lines.append(
            f"          window=[{event.window_start:.2f}, {event.window_end:.2f}] "
            f"new_audio=[{event.new_audio_start:.2f}, {event.new_audio_end:.2f}]"
        )
    if event.newly_committed:
        promoted = " ".join(w.text for w in event.newly_committed)
        lines.append(f"[{event.timestamp:6.2f}s] +COMMIT: {promoted!r}")
    lines.append(
        f"[{event.timestamp:6.2f}s] committed: {event.committed_text!r}\n"
        f"           partial: {event.partial_text!r}"
    )
    return "\n".join(lines)


def _open_source(args: argparse.Namespace, config: StreamingASRConfig):
    """Build the audio source. The pipeline cannot tell mic from file."""
    if args.mic:
        from streaming_asr.audio.mic_source import MicrophoneSource

        device: int | str | None = None
        if args.mic != "default":
            device = int(args.mic) if args.mic.isdigit() else args.mic
        source = MicrophoneSource(
            sample_rate=config.sample_rate,
            chunk_samples=config.chunk_samples,
            device=device,
            max_duration=args.max_seconds,
        )
        label = f"microphone ({args.mic})"
        return source, label

    source = WavFileSource(
        path=args.audio,
        sample_rate=config.sample_rate,
        chunk_samples=config.chunk_samples,
        real_time=(args.mode == "real-time"),
    )
    return source, f"{Path(args.audio).name} ({source.duration:.2f}s)"


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = build_parser().parse_args(argv)
    configure_logging(verbose=args.verbose, quiet=args.quiet)

    if args.list_devices:
        from streaming_asr.audio.mic_source import default_input_device, list_input_devices

        default = default_input_device()
        print("Input devices:")
        for info in list_input_devices():
            marker = " (default)" if info["index"] == default else ""
            print(f"  [{info['index']}] {info['name']}{marker} "
                  f"- {info['channels']}ch @ {info['default_samplerate']:.0f} Hz")
        return 0

    # --inspect-model is a setup step; requiring audio to look at a graph
    # would be gratuitous.
    if not args.inspect_model:
        if not args.audio and not args.mic:
            print("Provide --audio FILE or --mic. See --list-devices.", file=sys.stderr)
            return 2
        if args.audio and args.mic:
            print("--audio and --mic are mutually exclusive.", file=sys.stderr)
            return 2

    config = config_from_args(args)

    if args.inspect_model:
        # Report the graph even when the config does not match it -- that
        # mismatch is exactly what the user is here to diagnose.
        from streaming_asr.inference.onnx_engine import ONNXASREngine

        engine = ONNXASREngine(args.model, providers=args.providers)
        print(engine.graph_report.render())
        vocab_size = engine.graph_report.vocab_size
        configured = len(config.ensure_blank_in_vocabulary())
        print(f"\nvocabulary: {configured} entries configured, "
              f"{vocab_size} output units in the model", end="")
        print(" -- MATCH" if vocab_size == configured else
              "\n  MISMATCH: extract the checkpoint's own vocabulary with "
              "tools/extract_vocabulary.py")
        return 0

    from streaming_asr_lite.factory import build_pipeline, describe_runtime

    pipeline = build_pipeline(config)

    print(pipeline.engine.graph_report.render())
    print(f"\n{describe_runtime(config, pipeline.engine)}")
    print(_render_runtime(pipeline))
    print()

    source, label = _open_source(args, config)
    print(
        f"Streaming {label} at {config.chunk_duration * 1000:.0f}ms steps over a "
        f"{config.buffer_duration:.2f}s window "
        f"({config.window_redundancy:.0f}x recompute)\n"
    )
    pipeline.warmup()
    # A microphone is inherently real time; a file only is when paced.
    pipeline.metrics.real_time_paced = bool(args.mic) or args.mode == "real-time"

    if args.mic:
        print("Listening. Press Ctrl+C to stop and produce the final transcript.\n")
    print("Starting streaming ASR...\n")
    final_event: ASREvent | None = None
    previous_line = ""
    tracer = HypothesisTracer(args.trace, include_tokens=args.trace_tokens) if args.trace else None
    if tracer:
        tracer.__enter__()

    def handle(event: ASREvent) -> None:
        nonlocal final_event, previous_line
        if event.type is ASREventType.PARTIAL:
            if tracer:
                tracer.record(event, pipeline.last_hypothesis)
            line = _render_partial(event, args.show_windows)
            # Suppress consecutive identical states; at 160 ms steps most
            # windows change nothing and the log is unreadable otherwise.
            key = f"{event.committed_text}|{event.partial_text}"
            if key != previous_line:
                print(line)
                previous_line = key
        elif event.type is ASREventType.SEGMENT:
            marker = " (forced cut)" if event.metrics.get("forced") else ""
            print(f"\n[{event.timestamp:6.2f}s] SEGMENT{marker}: {event.text!r}")
            print(f"           transcript: {event.committed_text!r}\n")
            previous_line = ""
        elif event.type is ASREventType.ENDPOINT:
            print(f"\n[END OF SPEECH] ({event.metrics.get('reason', '')})")
            print("Running final decoding...\n")
        elif event.type is ASREventType.FINAL:
            # At end of stream the open turn may be empty because the last one
            # already closed on silence; printing it reads as a lost transcript.
            if event.text:
                print(f"[{event.timestamp:6.2f}s] TURN FINAL: {event.text!r}\n")
            final_event = event

    try:
        for event in pipeline.stream(source):
            handle(event)
    except KeyboardInterrupt:
        # A live stream has no natural end, so Ctrl+C is the normal way to
        # finish. Abandoning the generator would skip finalisation entirely and
        # throw away the utterance, so finalise explicitly here.
        print("\n\n[interrupted] finalising...\n")
        source.close()
        pipeline.end_of_speech()
        if not pipeline.is_finalized:
            handle(pipeline.finalize())
    finally:
        source.close()

    overruns = getattr(source, "overruns", 0)
    if overruns:
        print(f"\n!! {overruns} input overrun(s): the pipeline fell behind real time.")
        print("   This configuration is not viable live. Increase --chunk-ms, "
              "reduce --context-sec, or run on GPU.")

    if final_event is None:
        print("No final transcript produced.", file=sys.stderr)
        return 1

    # In segmented mode a FINAL marks the end of a *turn*, so the session
    # transcript is the accumulation of every segment, not the last event.
    session_text = getattr(pipeline, "transcript", None) or final_event.text

    if final_event.decoder == "streaming":
        source_label = "no final decode; streaming transcript as-is"
    elif final_event.used_lm:
        source_label = f"beam+KenLM via {final_event.decoder}"
    else:
        source_label = f"beam, no LM via {final_event.decoder}"

    print("=" * 68)
    print(f"STREAMING (provisional): {final_event.provisional_text!r}")
    print(f"FINAL ({source_label}):\n  {session_text!r}")
    print("=" * 68)

    results: dict = {
        "final": session_text,
        "provisional": final_event.provisional_text,
        "used_lm": final_event.used_lm,
        "decoder": final_event.decoder,
        "metrics": final_event.metrics,
    }

    if args.compare_greedy:
        offline = _offline_greedy(pipeline)
        results["offline_greedy"] = offline
        print(f"\nOffline greedy (whole utterance): {offline!r}")
        print(f"  token edit distance vs final: "
              f"{word_error_rate(final_event.text, offline):.3f} (word level)")

    if args.reference:
        reference = Path(args.reference).read_text(encoding="utf-8").strip() \
            if Path(args.reference).exists() else args.reference
        results["reference"] = reference
        results["wer_final"] = word_error_rate(reference, session_text)
        results["wer_streaming"] = word_error_rate(reference, final_event.provisional_text)
        print(f"\nReference: {reference!r}")
        print(f"  WER final     : {results['wer_final']:.3f}")
        if final_event.provisional_text:
            print(f"  WER streaming : {results['wer_streaming']:.3f}")

    if tracer:
        summary = tracer.summary()
        tracer.close()
        results["trace_summary"] = summary
        print(f"\nHypothesis instability (from {args.trace}):")
        for key, value in summary.items():
            print(f"  {key:38s}: {value}")

    print("\nMetrics:")
    print(pipeline.metrics.render())

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json_out}")
    return 0


def _offline_greedy(pipeline: StreamingASRPipeline) -> str:
    """Greedy-decode the whole utterance in one pass, for comparison."""
    import numpy as np

    audio = pipeline.retained_audio()
    features, lengths = pipeline.preprocessor(audio.reshape(1, -1), n_samples=len(audio))
    result = pipeline.engine.run_torch(features, int(lengths[0]))
    return pipeline.greedy.decode_text(result.logits)


if __name__ == "__main__":
    raise SystemExit(main())
