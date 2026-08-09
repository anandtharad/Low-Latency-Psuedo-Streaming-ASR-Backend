"""Extract the CTC vocabulary from a .nemo checkpoint.

**Run this before anything else with a real model.** The vocabulary is the one
piece of configuration that fails silently: an index-to-token table that does
not match the checkpoint produces confident, fluent-looking, completely wrong
text. Nothing raises.

The brief states the tokenizer was reinitialized/retrained for the monolingual
healthcare model, so the reference vocabulary hard-coded in ``config.py`` --
lifted from the notebook's older English checkpoint -- will almost certainly be
wrong for it.

Two extraction paths, tried in order:

1. **NeMo**, if installed. Authoritative: reads ``model.decoder.vocabulary``,
   exactly as the reference notebook does.
2. **Tar**, otherwise. A ``.nemo`` file is an (optionally gzipped) tar archive
   containing ``model_config.yaml`` and the SentencePiece model/vocab. This
   path needs no NeMo, no torch and no GPU, so it runs on the serving box.

Both then append the CTC blank, matching ``vocabulary.append("__")`` in the
reference.

Usage::

    python tools/extract_vocabulary.py --nemo model.nemo --out vocabulary.txt
    python tools/extract_vocabulary.py --nemo model.nemo --out vocab.txt --verify model.onnx
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streaming_asr.console import configure_stdout  # noqa: E402

configure_stdout()

DEFAULT_BLANK = "__"


def from_nemo_api(path: Path) -> Optional[list[str]]:
    """Authoritative path: ask NeMo itself."""
    try:
        import nemo.collections.asr as nemo_asr
    except Exception:
        return None

    print("  using the NeMo API (authoritative)")
    model = nemo_asr.models.EncDecCTCModelBPE.restore_from(
        str(path), map_location="cpu"
    )
    return list(model.decoder.vocabulary)


def from_tar(path: Path) -> Optional[list[str]]:
    """Fallback: read the tokenizer straight out of the .nemo archive."""
    print("  NeMo not installed; reading the .nemo archive directly")

    with tarfile.open(path, "r:*") as archive:
        names = archive.getnames()

        # Preferred: the SentencePiece model, which gives the exact ordering.
        spm_name = next(
            (n for n in names if n.endswith(".model") and "tokenizer" in n.lower()),
            None,
        ) or next((n for n in names if n.endswith(".model")), None)

        if spm_name is not None:
            try:
                import sentencepiece as spm

                with tempfile.TemporaryDirectory() as tmp:
                    archive.extract(spm_name, path=tmp)
                    processor = spm.SentencePieceProcessor()
                    processor.load(str(Path(tmp) / spm_name))
                    print(f"  read SentencePiece model: {spm_name}")
                    return [processor.id_to_piece(i)
                            for i in range(processor.get_piece_size())]
            except ImportError:
                print("  sentencepiece not installed; falling back to the .vocab file")

        # Secondary: the plain vocab listing, "<piece>\t<score>" per line.
        vocab_name = next((n for n in names if n.endswith(".vocab")), None)
        if vocab_name is not None:
            print(f"  read vocab file: {vocab_name}")
            member = archive.extractfile(vocab_name)
            assert member is not None
            lines = member.read().decode("utf-8").splitlines()
            return [line.split("\t")[0] for line in lines if line]

        print(f"  no tokenizer found in the archive. Contents: {names[:20]}")
        return None


def verify_against_onnx(vocabulary: list[str], onnx_path: Path) -> bool:
    """The decisive check: does the table match the model's output width?

    A mismatch here is the difference between a working system and fluent
    nonsense, so it is worth failing loudly at setup time.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed; skipping verification)")
        return True

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    output = session.get_outputs()[0]
    width = output.shape[-1]

    print(f"\nVerifying against {onnx_path.name}")
    print(f"  model output units : {width}")
    print(f"  vocabulary entries : {len(vocabulary)}")

    if not isinstance(width, int):
        print("  output width is dynamic; cannot verify statically.")
        return True
    if width == len(vocabulary):
        print("  MATCH")
        return True

    print(f"  MISMATCH: off by {width - len(vocabulary)}")
    if width == len(vocabulary) + 1:
        print("  The model has one extra unit -- the CTC blank was not appended.")
    elif width == len(vocabulary) - 1:
        print("  The blank was appended twice; drop one.")
    else:
        print("  This vocabulary does not belong to this model. Using it would "
              "produce confident, completely wrong transcripts.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nemo", required=True, help="path to the .nemo checkpoint")
    parser.add_argument("--out", required=True, help="output vocabulary file")
    parser.add_argument("--verify", help="ONNX model to check the size against")
    parser.add_argument("--blank", default=DEFAULT_BLANK,
                        help="CTC blank symbol to append")
    parser.add_argument("--no-blank", action="store_true",
                        help="do not append a blank (if the checkpoint already has one)")
    args = parser.parse_args()

    nemo_path = Path(args.nemo)
    if not nemo_path.exists():
        print(f"not found: {nemo_path}", file=sys.stderr)
        return 1

    print(f"Extracting vocabulary from {nemo_path.name}")
    vocabulary = from_nemo_api(nemo_path)
    if vocabulary is None:
        vocabulary = from_tar(nemo_path)
    if vocabulary is None:
        print("\nCould not extract a vocabulary. Install NeMo, or "
              "'pip install sentencepiece' and retry.", file=sys.stderr)
        return 1

    print(f"  {len(vocabulary)} tokens before the blank")
    print(f"  first 12: {vocabulary[:12]}")

    if not args.no_blank:
        if vocabulary[-1] == args.blank:
            print(f"  blank {args.blank!r} already present; not appending")
        else:
            vocabulary.append(args.blank)
            print(f"  appended blank {args.blank!r} -> {len(vocabulary)} units")

    ok = True
    if args.verify:
        ok = verify_against_onnx(vocabulary, Path(args.verify))

    out = Path(args.out)
    # Newline-delimited: preserves SentencePiece pieces containing whitespace,
    # which a CSV or naive split would mangle.
    out.write_text("\n".join(vocabulary), encoding="utf-8")
    print(f"\nWrote {len(vocabulary)} tokens to {out}")
    print(f"Use it with:  --vocabulary {out}")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
