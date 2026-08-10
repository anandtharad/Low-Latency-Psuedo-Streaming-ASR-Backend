# Streaming ASR over an offline Conformer-CTC ONNX model

Low-latency streaming ASR built **around** an offline, full-context
Conformer-CTC model — no retraining, no re-export, no model surgery.

The model has no encoder cache and no streaming state, so this is
*pause-segmented pseudo-streaming*: speech is cut at natural pauses and each
span is decoded whole. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for
why that beat the rolling-window design it replaced.

> The NVIDIA model card that shipped with the checkpoint is preserved at
> [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

---

## Run as a service

```bash
cp .env.example .env          # point ASR_MODEL_PATH / ASR_VOCAB_PATH at your files
docker compose up asr                        # CPU
docker compose --profile gpu up asr-gpu      # GPU (needs nvidia-container-toolkit)
```

Or directly, without Docker:

```bash
set ASR_MODEL_PATH=C:\anand\ASR Backend\stt_en_cconformer_ctc_large-averaged.onnx
set ASR_VOCAB_PATH=C:\anand\ASR Backend\vocab.txt
set ASR_DEVICE=cuda
set ASR_BEAM_BACKEND=pure_python
python -m streaming_asr.server.app
```

The model loads once at startup and stays resident. Check what actually loaded:

```bash
curl localhost:8000/health
```

```json
{"ready": true, "frontend_device": "cuda:0", "zero_copy": true,
 "decoder_backend": "pure_python", "used_lm": false, "stateless_graph": true}
```

It reports **reality, not intent** — a service that silently fell back to CPU,
or to a decoder with no language model, is still "up" but will miss its budget.
That is the line to alert on.

| Endpoint | Purpose |
|---|---|
| `GET /health` | readiness + what actually loaded |
| `GET /info` | config, ONNX graph report, accepted audio formats |
| `POST /transcribe` | upload a file → transcript + per-segment breakdown |
| `WS /ws/transcribe` | live streaming |

---

## Calling it from an app

### One-shot file

```bash
curl -F file=@patient.wav localhost:8000/transcribe
```

```jsonc
{
  "text": "i have been having chest pain for three days",
  "segments": [
    {"start": 0.28, "end": 8.64, "text": "i have been having chest pain", "forced": false},
    {"start": 10.04, "end": 18.24, "text": "for three days", "forced": false}
  ],
  "duration": 31.12, "decoder": "flashlight", "used_lm": true
}
```

Format is sniffed from **content, not the extension** — WAV, FLAC, OGG, MP3
natively; m4a/AAC and WebM/Opus when ffmpeg is present (the Docker images
include it). Browser `MediaRecorder` produces WebM/Opus and iOS produces m4a,
so without ffmpeg those are refused with a **415** naming what is accepted.
Resampling and stereo downmix are automatic.

### Live streaming

Send binary PCM frames (`int16` or `float32`, mono, at `sample_rate`). Frames
need **not** be chunk-aligned — the server re-blocks them. Control messages:
`{"type":"end"}` to close, `{"type":"reset"}` to start over.

**Build your app on `segment`.** Four event types:

| event | meaning | use for |
|---|---|---|
| `partial` | open segment, re-decoded every chunk | live captions only — **revisable, never accumulate** |
| `segment` | a pause-bounded span, decoded whole | the authoritative unit. Append these. |
| `final` | a turn's segments joined (~1.5 s silence) | a complete thought — feed anything semantic from here |
| `final` with `end_of_stream` | connection closed | `transcript` holds the whole session |

```jsonc
{"type":"segment", "start":10.04, "end":18.24, "text":"i have chest pain",
 "transcript":"…everything so far…", "forced":false}
{"type":"final", "text":"<this turn>", "transcript":"<whole session>"}
```

`transcript` appears on every `segment` and `final`, so a client that misses an
event has one field to resync from.

Outstanding work, known defects and measured performance figures live in
[`docs/TODO.md`](docs/TODO.md).

### A torch-free runtime

[`streaming_asr_lite/`](streaming_asr_lite/) runs the same pipeline with the mel
frontend exported to ONNX, so the runtime needs only onnxruntime + numpy +
soundfile. Measured on identical audio, producing identical transcripts:

| runtime | RSS | startup | inference |
|---|---|---|---|
| `streaming_asr` (torch) | 425 MB | 2.52 s | 0.28 s |
| `streaming_asr_lite` | **68 MB** | **0.28 s** | 0.27 s |

Build the frontend once (this step needs torch; nothing afterwards does):

```bash
python -m streaming_asr_lite.export_frontend --out fixtures/frontend.onnx
```

`streaming_asr/` is unmodified and remains the reference and fallback.

The service gets this too: `tests/test_execution.py` imports the server, the app
and the CLI in clean interpreters and fails if torch appears in any of them. On
Windows the CUDA libraries are still taken from `torch/lib` when that is the only
copy on the machine — located with `find_spec` and loaded with `ctypes`, without
importing torch. See [`PROJECT_REPORT.md` §5.4](docs/PROJECT_REPORT.md).

Two design notes for the wider system (**not implemented**):

- [`docs/PROGRESSIVE_CONSUMPTION.md`](docs/PROGRESSIVE_CONSUMPTION.md) — using
  all three tiers together: speculative downstream work off `partial`, with
  anything user-visible gated on `segment`.
- [`docs/CONVERSATIONAL_LOOP.md`](docs/CONVERSATIONAL_LOOP.md) — the audio path:
  ring buffer, echo cancellation, barge-in, turn boundaries, and where NMT
  belongs. Contains one latent bug against the server as it stands.

Working clients in [`examples/client.py`](examples/client.py):

```bash
python examples/client.py health
python examples/client.py batch  --audio patient.wav
python examples/client.py stream --audio patient.wav   # paced at wall-clock speed
python examples/client.py mic                          # live microphone
```

---

## Tuning

Three knobs decide behaviour; the defaults are starting points, not answers.

| Variable | Default | Effect |
|---|---|---|
| `ASR_SEGMENT_SILENCE` | 0.5 s | Silence that closes a segment. Too low splits phrases at hesitations (`"i have"` / `"chest pain"`); too high delays every `segment`. |
| `ASR_TURN_SILENCE` | 1.5 s | Silence that ends a turn. Cutting someone off mid-thought is worse than half a second of latency — err high. |
| `ASR_ENERGY_THRESHOLD` | 0.005 | RMS above which a chunk is speech. **If this sits below your room's noise floor, silence never registers and nothing ever segments.** |
| `ASR_MAX_SEGMENT_SEC` | 10 s | Cap for speech with no pause. Keep at or below the checkpoint's training `max_duration` (11 s) — beyond that the decode goes out of distribution and degrades sharply. |

`forced: true` on a segment means it hit the cap without finding a pause. A few
are fine; many means `ASR_ENERGY_THRESHOLD` is wrong for your audio.

Capacity is **capped, not queued**: past `ASR_MAX_CONCURRENT_STREAMS` the
service returns 503 rather than pushing every existing caller past real time.
One worker per container — a second uvicorn worker loads a second copy of the
model and a second CUDA context.

**Set the cap from a measurement, not a guess.** Measured on a GTX 1650 (4 GB)
with the real checkpoint, a 16-caller burst:

| cap | outcome |
|---|---|
| 32 (guessed) | CUDA out of memory, **the service died**, 0 of 16 served |
| 4 (measured) | 4 served at 1.1 s p95, 12 refused cleanly, 0 errors |

The cap is the only thing standing between a busy minute and an outage.
`python -m tests.load.run_load_sweep` produces the curve it should come from —
see [`tests/load/README.md`](tests/load/README.md).

---

## CLI

```bash
python -m streaming_asr.cli --audio patient.wav \
    --model model.onnx --vocabulary vocab.txt --device cuda

python -m streaming_asr.cli --mic --model model.onnx --vocabulary vocab.txt
```

Setup and diagnostics:

```bash
python tools/extract_vocabulary.py --nemo model.nemo --out vocab.txt --verify model.onnx
python -m streaming_asr.cli --model model.onnx --vocabulary vocab.txt --inspect-model
python tools/check_alignment_fidelity.py --audio x.wav --model model.onnx --vocabulary vocab.txt
python -m streaming_asr.benchmark --audio short.wav long.wav --model model.onnx --compare-decoders
```

`extract_vocabulary.py` is not optional with a new checkpoint: a vocabulary that
does not match produces fluent, confident, completely wrong text, and nothing
raises. The pipeline refuses to start if the size disagrees with the model.

---

## Layout

```
streaming_asr/
├── segmented.py           pause-segmented pipeline (default)
├── pipeline.py            rolling-window pipeline (kept for comparison)
├── config.py              every tunable
├── audio/                 sources + container/codec decoding
├── preprocessing/         mel frontend, preserved from the reference
├── inference/             ONNX Runtime engine + graph introspection
├── decoding/              greedy_ctc.py, beam_ctc_lm.py
├── hypothesis/            aligners + tracker (windowed mode only)
├── server/                model_pool.py, settings.py, app.py
├── benchmark.py, cli.py
tools/                     extract_vocabulary, check_alignment_fidelity,
                           real_audio_wer, verify_parity, fixtures
tests/                     unit suite
tests/load/                concurrency + load framework (its own README)
```

```bash
python -m pytest tests -q                    # everything
python -m pytest tests -q -m "not slow"      # skip the tests that start a service
```

Scope is ASR only — the downstream question-generation system is deliberately
not implemented.

---

## Measuring it

Two questions, two tools. Neither reports a number the other produced.

### Accuracy — `tools/real_audio_wer.py`

WER on the Common Voice clips in `common_voice_en/`, joined against
`train.tsv`. Compares the pause-segmented streaming path against a single
`transcribe()`-style pass over the same audio, on short clips and on long turns
built by concatenating clips from one speaker.

```bash
python tools/real_audio_wer.py \
    --model stt_en_cconformer_ctc_large-averaged.onnx --vocabulary vocab.txt \
    --runtime torch --device cuda --clip-count 100 --turn-count 6
```

The synthetic fixture under `fixtures/` is a *code-path* exercise for machines
with no checkpoint. Its error rates describe a toy model reading toy audio and
are not quoted anywhere as accuracy.

### Behaviour under load — `tests/load/`

Multiple simultaneous WebSocket users against a running service, with
real-time and maximum-throughput modes kept separate.

```bash
python -m tests.load.load_test      --audio sample.wav --concurrency 8
python -m tests.load.run_load_sweep --audio sample.wav --levels 1,2,4,8,16,32
```

Full metric definitions, saturation thresholds, pool-size sweeps and the
recommended procedure are in [`tests/load/README.md`](tests/load/README.md).
The one thing worth repeating here: **RTF below 1 is necessary for real-time
work but not sufficient for good conversational latency** — what a speaker
waits for is `segment_silence + decode`, and the first term does not care how
fast the GPU is.
