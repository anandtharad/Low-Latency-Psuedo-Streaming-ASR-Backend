# Measurement scripts

## `benchmark_all.sh`

Runs the whole measurement matrix — per-duration profile and concurrency sweep,
on every device the machine has — and writes it to `results/<timestamp>/`.

```bash
MODEL=/path/model.onnx VOCAB=/path/vocab.txt ./scripts/benchmark_all.sh
```

Everything else has a default; override from the environment.

| variable | default | what it controls |
|---|---|---|
| `MODEL`, `VOCAB` | repo-root names | the checkpoint. Not in the repository |
| `CLIPS`, `MANIFEST` | `common_voice_en/...` | corpus for the duration profile. Skipped if absent |
| `LOAD_AUDIO` | `fixtures/load_sample.wav` | sweep fixture; built from the corpus if missing |
| `RUNTIME` | `lite` | `lite` or `torch` |
| `BINS` | `5,…,120` | duration bins, seconds |
| `GPU_LEVELS` / `CPU_LEVELS` | `1,…,16` / `1,…,8` | concurrency levels |
| `MAX_STREAMS` | `8` | admission cap — **also sets the CPU thread count** |
| `LM`, `LEXICON` | unset | enables the `beam_lm` variant |
| `DEVICES` | `auto` | `cpu`, `gpu`, or `"cpu gpu"` |

20–40 min on a GPU box, longer CPU-only. Non-interactive; safe under `nohup`.

### What it measures, and why each piece exists

**1. Duration profile** (`tools/profile_by_duration.py`) — one recording per
duration bin, single stream, measuring compute in isolation from pacing:

- `rtf_speech` — compute per second of *speech*. The model's true cost.
- `rtf_audio` — compute per second of *audio*, silence included. The capacity
  figure, and always the lower of the two.
- per-segment decode time, greedy vs beam vs beam+LM.

Bins up to ~10 s are real Common Voice clips. The corpus has nothing longer, so
larger bins are **constructed**: consecutive clips from one speaker joined by
0.7 s pauses. Every row says which it is.

**2. Concurrency sweep** (`tests/load/`) — N simultaneous WebSocket callers
against the live service, paced at wall-clock speed.

### Reading the output

Three traps, all of which have caught this project:

- **Size on response latency, not RTF.** RTF stays under 1 well past the point
  where callers wait 20 s. See `PROJECT_REPORT.md` §6.2.
- **Believe the run header about providers.** It reports what ONNX Runtime
  actually loaded, which is not always what was requested. A GPU present but
  unused is called out explicitly.
- **`MAX_STREAMS` is not cosmetic on CPU.** It sets the derived intra-op thread
  count. Sweeping at a cap of 8 while only ever running 2 streams under-threads
  every one of them.

### The `beam_lm` variant

Skipped unless both `LM` and `LEXICON` are set. Without them the `beam` column
is an **LM-free beam**, which is a different measurement — the LM is the
expensive half, and the flashlight + KenLM path has never executed anywhere in
this project (no Windows wheel; see `docs/TODO.md` §1.4). Tomorrow's Linux box
is the first chance to run it. Treat the first numbers it produces as suspect
until something confirms the transcript actually improved.

Also worth knowing before reading a beam column: the default `pure_python`
backend is a **reference implementation**, not a production decoder. Measured on
a GTX 1650 it adds ~1.3 s to *every segment decode* against greedy's ~60 ms.
That is not the cost of beam search, it is the cost of beam search in Python.
