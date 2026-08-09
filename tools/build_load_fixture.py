"""Build a realistic multi-turn recording for load testing.

The Common Voice clips in this repository are single sentences of about six
seconds. Streamed at wall-clock speed that is one segment and a handful of
partials -- too short to say anything about a service under sustained load, and
too few samples for a p95 worth quoting.

This concatenates real clips from one speaker with silent gaps, producing audio
that segments the way a person dictating several sentences does. It is real
speech throughout; only the pauses are manufactured, and the gap length is
printed so the result can be read with that in mind.

Usage::

    python tools/build_load_fixture.py --seconds 60 --out fixtures/load_sample.wav
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.real_audio_wer import SAMPLE_RATE, build_turns, read_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clips", default="common_voice_en/en_train_28")
    parser.add_argument("--manifest", default="common_voice_en/train.tsv")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--gap", type=float, default=0.7,
                        help="silence between clips; must exceed the service's "
                             "segment_silence for segments to close on a pause")
    parser.add_argument("--out", default="fixtures/load_sample.wav")
    parser.add_argument("--transcript", default=None,
                        help="also write the reference text alongside the audio")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)

    import soundfile as sf

    clips = read_manifest(Path(args.manifest), Path(args.clips))
    random.Random(args.seed).shuffle(clips)
    turns = build_turns(clips, args.seconds, args.gap, count=1)
    if not turns:
        print("not enough audio to reach that length", file=sys.stderr)
        return 1

    turn = turns[0]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, np.clip(turn.audio, -1.0, 1.0), SAMPLE_RATE, subtype="PCM_16")

    if args.transcript:
        Path(args.transcript).write_text(turn.reference + "\n", encoding="utf-8")

    print(f"{out}: {turn.duration:.1f}s from {turn.clips} clips, "
          f"{args.gap:.1f}s gaps, {len(turn.reference.split())} reference words")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
