# Outstanding work

Accumulated from building and measuring this pipeline. Ordered by leverage
within each section. Figures marked **measured** come from runs in this repo on
a GTX 1650; anything else is an estimate and is labelled as such.

---

## 1. Known defects

### 1.1 Timeline discontinuity when the client pauses transmission — **live bug**

The pipeline derives `audio_time` from cumulative samples received, not wall
clock. A client that stops sending during TTS playback and resumes afterwards
produces **no gap** from the server's point of view: `trailing_silence` never
accumulates, the open segment never closes, and pre-TTS and post-TTS speech
merge into one segment.

Not yet triggered because nothing pauses transmission today. Barge-in will
trigger it immediately. Fix: explicit turn boundaries (`{"type":"end"}` then
`{"type":"reset"}`), documented in
[`CONVERSATIONAL_LOOP.md` §4](CONVERSATIONAL_LOOP.md).

### 1.2 A large `ASR_MAX_SEGMENT_SEC` would walk into a process-killing wall

**Not currently reachable.** Every service path — `POST /transcribe` included —
runs the segmented pipeline, so no request hands the encoder more than
`max_segment_duration` (10 s). What follows is the wall that sits behind that
guard, and the reason the guard must not be widened casually.

Measured on a 4 GB GPU, decoding one recording in a single forward pass:

| length | result |
|---|---|
| 150 s | works, 18.1 s (8.6× slower than 120 s) |
| 180 s | CUDA OOM — one attention `MatMul` requests 1.30 GB |
| 210 s | `Add_2: right operand cannot broadcast on dim 3, {1,8,5251,5251}` |
| 245 s | garbage transcript, then a **native abort that killed the process** |

Self-attention is quadratic in length, so this is not a tuning problem. The
210 s error is the model's own relative-position buffer, which a bigger GPU does
not move at all. The 245 s abort comes from inside
`onnxruntime_providers_cuda.dll` and is **not catchable in Python**.

`ASR_MAX_SEGMENT_SEC` is the one knob that reaches this. It is documented as
"keep at or below the checkpoint's training `max_duration` (11 s)" for accuracy
reasons, but nothing *enforces* it, and a value in the hundreds would take the
process down rather than merely degrade quality. Cheap fix: refuse or warn at
startup above some multiple of the training duration.

Related and quieter: the first OOM permanently disables zero-copy `io_binding`
for the rest of the process, so one oversized decode leaves the service slower
for every request after it.

### 1.3 The server loads torch whatever `ASR_RUNTIME` says — **live defect**

`streaming_asr/server/model_pool.py` line 43:

```python
from streaming_asr.pipeline import StreamingASRPipeline
```

Unused since `new_pipeline()` started delegating to
`streaming_asr_lite.factory.build_pipeline`. It still executes, pulling in the
torch preprocessor. `streaming_asr/cli.py` does the equivalent.

Two consequences, in opposite directions:

* **The lite runtime's footprint benefit is not realised by the service.** The
  68 MB / 0.28 s figures are for a process importing only
  `streaming_asr_lite.*`, which is what `tests/test_lite.py` verifies. The
  server is the 425 MB number regardless of `ASR_RUNTIME`.
* **…but on Windows that accident is what gives the server CUDA at all.**
  Importing torch first puts the CUDA/cuDNN DLLs where ONNX Runtime can find
  them. Measured on this machine: lite runtime standalone →
  `CPUExecutionProvider`; lite runtime under the server → `CUDAExecutionProvider`.

So deleting the dead import is correct *and* would cost this machine its GPU.
Do both together: remove the import **and** install the CUDA runtime properly
(or deploy the container, which already has it). Add a test asserting the server
process does not import torch under `ASR_RUNTIME=lite` — it will fail today,
which is the point.

Found by the benchmark metadata in `tests/load/`, which records active providers
per run; the discrepancy showed up as the same configuration reporting different
providers from two entry points.

### 1.4 Backlog chunks pollute latency metrics

`AudioChunk.capture_time` feeds `update_latency`. A flushed ring-buffer backlog
arrives faster than real time, so the first turn of every session will report an
inflated latency that never happened. Flag backlog chunks and exclude them.

### 1.5 The flashlight + KenLM path has never executed

No Windows wheel for `flashlight-text`, so the reference decoder is written but
completely untested at runtime. Everything to date has run on the LM-free
`pure_python` backend. **Exercise it on Linux before trusting any final-decode
quality number.**

### 1.6 Merge-across-boundary duplicates (windowed pipeline only)

If the model merges a committed word with the next one (`"is"` + `"closed"` →
`"isclosed"` after `"is"` is committed), a short duplicate results. Accepted
deliberately: the stricter rule dropped real words whenever drift exceeded the
inter-word gap, and losing speech is worse than duplicating it. Only affects
`--pipeline windowed`, which is no longer the default.

---

## 2. Measurements that decide design

Cheap, and each one settles a question currently being guessed at.

### 2.1 Fetch latency in the conversational loop

**Decides whether the speculative layer gets built at all.** At 50 ms per
question fetch the complexity is unjustifiable; at 800 ms it is transformative.
Measurable today without writing any of the speculation machinery. See
[`PROGRESSIVE_CONSUMPTION.md` §7](PROGRESSIVE_CONSUMPTION.md).

### 2.2 Multilingual retrieval instead of MT-in-the-loop

Embed ~50 real patient utterances with LaBSE or multilingual-E5 and check
whether the correct question ranks top-3 against the bank **without
translating**. About a day of work. If it works, NMT leaves the latency-critical
path entirely and the partial-translation hazard disappears.
[`CONVERSATIONAL_LOOP.md` §7](CONVERSATIONAL_LOOP.md).

### 2.3 Greedy vs beam+LM on the real model

Section 23 of the original brief, still unanswered — we never had an LM. Run
`benchmark.py --compare-decoders` once KenLM is available. If the gap is small,
`--no-final-beam` takes finalisation latency to zero.

### 2.4 Alignment fidelity on the healthcare checkpoint

`tools/check_alignment_fidelity.py`. Decides whether `aligner="time"` is safe.
Only relevant to the windowed pipeline, but run it if that path is ever revived.

### 2.5 Concurrency and capacity — **framework built, one machine measured**

`tests/load/` runs N simultaneous WebSocket callers against the live service and
reports latency percentiles, RTF, success rate and resource use per concurrency
level. See [`tests/load/README.md`](../tests/load/README.md).

**Measured**, GTX 1650 (4 GB), real checkpoint, greedy, 46 s recording, paced at
wall-clock speed:

| streams | RTF p95 | response p95 | GPU % |
|---|---|---|---|
| 1 | 0.427 | 0.75 s | 27 |
| 4 | 0.728 | 1.38 s | 64 |
| 8 | 0.905 | 4.68 s | 91 |
| 12 | 1.318 | 22.5 s | 90 |
| 16 | — | service died (CUDA OOM) | 93 |

**RTF stays under 1 up to 12 streams while response latency is already 22 s.**
Size on `segment_response_latency`, never on RTF. Usable capacity here is 4.

Still outstanding:

* Sweep on the **deployment target**. The shape should transfer, the numbers
  will not.
* Sweep `--pool-sizes` to settle whether raising `ASR_MAX_CONCURRENT_STREAMS`
  buys throughput or only spreads the same throughput over more unhappy
  callers. It is an admission cap over one shared session, not a worker pool.
* Re-run after §3.1 (batching) — that change is expected to move this curve more
  than anything else on the list.

### 2.5.1 The service dies instead of shedding load — **defect**

At 16 concurrent streams the CUDA allocator failed inside a device-to-host copy
and the **process went down**, taking every in-flight caller with it. The
admission cap prevents this only if it is set below what the GPU can hold, which
means a correct cap depends on hardware the code cannot see.

Worth considering: a GPU-memory-aware admission check, or catching allocator
failures at the pipeline boundary and refusing that stream rather than letting
the exception reach the process. The second is cheap and strictly better than
what happens now.

### 2.6 Tune the segmentation thresholds on real audio

`ASR_SEGMENT_SILENCE` (0.5), `ASR_TURN_SILENCE` (1.5), `ASR_ENERGY_THRESHOLD`
(0.005) are defaults, not answers. The energy threshold is the one most likely
to be wrong: **if it sits below the room's noise floor, silence never registers
and every segment gets force-cut at the cap.** Watch for `"forced": true`.

`ASR_TURN_SILENCE` is a product decision, not a tuning parameter — it is the
conversational feel of the agent. Tune it with clinicians present.

### 2.7 Re-run the benchmark sweep with `--mode real-time`

**Measured:** unpaced RTF 0.165 vs real-time-paced 0.245 on identical audio,
model and pipeline — the GPU downclocks between chunks when fed at wall-clock
speed. Unpaced figures are ~1.5× optimistic and useless for capacity planning.

---

## 3. Performance

### Where the time actually goes — **measured**, real Conformer, 160 ms chunks

| stage | per chunk | implemented in |
|---|---|---|
| ONNX inference | 39.74 ms | C++ (ONNX Runtime) |
| mel frontend | 2.47 ms | C++/CUDA (torchaudio) |
| greedy CTC decode | 0.36 ms | C (NumPy) |
| sum of native work | 42.57 ms | |
| measured end-to-end | 42.98 ms | |
| **Python glue** | **≈0.4 ms ≈ 1%** | |

**The binding constraint is the GPU, not the language.** Real-time RTF 0.66 is
~1.5 concurrent streams per GPU. The GIL saturates at roughly 400 streams
(0.4 ms × 6.25 chunks/s = 2.5 ms/s per stream). The GPU limit arrives ~250×
sooner.

### 3.1 Batch multiple streams into one inference call

**Highest-leverage change available.** A GPU running 8 sequences as one batch
does not take 8× as long; *estimated* 3–4× throughput, as sherpa-onnx does it.
This is a change to `ModelPool`, not to the language.

### 3.2 Export the mel frontend to ONNX and drop torch — **DONE**

Implemented in [`streaming_asr_lite/`](../streaming_asr_lite/), parallel to
`streaming_asr/`, which is untouched and remains the fallback.

**Measured, same audio, same model, identical transcripts:**

| runtime | RSS | startup | inference |
|---|---|---|---|
| `streaming_asr` (torch) | 425 MB | 2.52 s | 0.28 s |
| `streaming_asr_lite` | **68 MB** | **0.28 s** | 0.27 s |

**6.3× less memory, 9× faster startup, no change in inference time.**

`torch.onnx.export` cannot export `torch.stft` ("does not currently support
complex types"), so the STFT is rebuilt as a convolution whose kernels are the
windowed DFT basis. The window and mel filterbank are lifted from a live
torchaudio module rather than recomputed — reproducing `slaney` normalisation
and a `periodic=False` Hann window by hand is exactly the near-miss that
degrades accuracy silently. Verified to **2.5e-04** on unit-variance features.

Outstanding:

* **The segmentation loop is duplicated** between `streaming_asr/segmented.py`
  and `streaming_asr_lite/pipeline.py`, because the former imports torch via its
  preprocessor and could not be reused without modification. A fix applied to
  one must be applied to the other. Resolve by making the preprocessor
  injectable in `segmented.py`, after which the two collapse into one.
* Likewise `LiteONNXEngine` exists only because `ONNXASREngine.__init__` calls
  `_preload_cuda_runtime()`, which imports torch so ONNX Runtime can find CUDA
  DLLs from `torch/lib` on Windows. A torch-free runtime cannot rely on that and
  must get CUDA from the toolkit or the image.
* Numbers above are CPU-side. Re-measure on the CUDA deployment target, where
  torch also carries a CUDA context.
* The server, CLI and Docker images still use the torch path. Switching them
  over is a follow-up.

### 3.3 INT8 quantisation of the ONNX model

*Estimated* 2–3× inference speedup on supported hardware, at some WER cost.
Measure the trade on real data.

### 3.4 Chunk size

**Measured:** 160 → 320 ms took model calls from 52 to 26 on the same audio.
Halves the call count; costs first-partial latency.

### 3.5 Rewriting in C++/Go — **deprioritised**

Recovers ~1% of per-chunk time (see table above). Go genuinely helps with
deployment (single static binary), per-process memory, and connection
concurrency at scale not currently approached.

Two costs if attempted:

* **The frontend must match bit-for-bit.** A mismatch degrades accuracy with no
  error raised. Doing §3.2 first converts this from a hazard into a solved
  problem.
* **321 tests and the diagnostic tooling would need rebuilding** — vocabulary
  extraction, alignment fidelity, the benchmark sweep. They exist because this
  pipeline has several silent-failure modes.

---

## 4. Protocol additions

Both additive; neither breaks existing consumers.

* **`seq`** — monotonic sequence number on every event, so clients can order and
  discard stale async results without relying on arrival time.
* **`segment_id`** on `partial` events, so a client can distinguish "this
  partial replaced the previous one" from "this is a new segment".

---

## 5. Deferred design

* [`PROGRESSIVE_CONSUMPTION.md`](PROGRESSIVE_CONSUMPTION.md) — speculative
  downstream work off `partial`, gated display on `segment`.
* [`CONVERSATIONAL_LOOP.md`](CONVERSATIONAL_LOOP.md) — ring buffer, echo
  cancellation, barge-in, turn boundaries, where NMT belongs.

Neither is implemented.

---

## 6. Settled, for the record

Things already measured so they are not re-litigated:

* **Pause-segmented beat rolling-window.** Word-level commitment produced
  cascading duplicates (`in in in`, `the the the`) on real audio; three repairs
  each fixed one case and broke another. Cutting at pauses removes the decision
  entirely.
* **Segmented is not slower.** Measured on identical audio: RTF 0.101 vs 0.162
  (beam off), 0.135 vs 0.119 (beam on).
* **Snapping final-decode seams to silence did not help.** Measured
  neutral-to-worse on the synthetic fixture and shipped off by default. That
  fixture has uniform inter-word gaps and no real pauses, so snapping had
  nothing to find; **not yet re-measured on real speech**, and the numbers it
  produced are not quoted anywhere.
* **Long single-pass decodes degrade, but less than the synthetic fixture
  suggested.** Measured on real Common Voice audio with the real checkpoint —
  see `docs/PROJECT_REPORT.md` §2.2 and `tools/real_audio_wer.py`. The earlier
  figures in this file came from the synthetic stand-in and overstated the
  effect by more than 2×; they have been removed.
* **A segment cap below the natural phrase length is actively harmful.**
  Measured: at a 6 s cap the segmenter force-cuts mid-phrase and scores *worse*
  than a single pass over the whole turn. The cap is a safety valve for speech
  with no pauses, not a tuning knob to minimise.
