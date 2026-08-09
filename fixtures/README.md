# Fixtures

The binaries in this directory are **not committed** — they are megabytes of
generated artefacts, and every one of them is reproducible from the scripts in
`tools/`. Only the small text files that describe them are tracked.

Rebuild everything from a clean checkout:

```bash
# Synthetic model + audio + vocabulary. Needed by most of the test suite;
# tests that require it skip cleanly when it is absent.
python tools/build_synthetic_fixture.py --out fixtures

# The ONNX mel frontend used by the torch-free runtime. This step needs torch;
# nothing afterwards does.
python -m streaming_asr_lite.export_frontend --out fixtures/frontend.onnx

# A multi-turn recording for load testing, built from real Common Voice clips.
# Needs common_voice_en/ (not committed either — see the note below).
python tools/build_load_fixture.py --seconds 45 \
    --out fixtures/load_sample.wav --transcript fixtures/load_sample.txt
```

| file | produced by | used by |
|---|---|---|
| `synthetic_model.onnx`, `synthetic.wav`, `vocabulary.txt`, `transcript.txt`, `word_spans.json` | `tools/build_synthetic_fixture.py` | most of `tests/` |
| `synthetic_long.wav`, `synthetic_pauses.wav` + transcripts | the same script | long-form and segmentation tests |
| `frontend.onnx` | `streaming_asr_lite.export_frontend` | the `lite` runtime, the server's default |
| `load_sample.wav`, `load_sample.txt` | `tools/build_load_fixture.py` | `tests/load/` benchmarks |

## The synthetic model is not a quality reference

It exists so the pipeline runs on a machine with no checkpoint, and it does
catch real bugs. It cannot support an accuracy claim: it is a toy model reading
toy audio, and error rates measured against it have been wrong by more than 2×
in both directions. Accuracy comes from `tools/real_audio_wer.py` on real
speech.

## The corpora are not committed either

`common_voice_en/` (Common Voice English, ~1 GB here) and the
`stt_en_conformer_ctc_large` checkpoint are redistributable only under their own
licences. Point the tools at your own copies — every script takes `--clips`,
`--manifest`, `--model` and `--vocabulary`.
