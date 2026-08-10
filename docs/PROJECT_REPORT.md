# Streaming ASR — project report

A low-latency streaming speech-recognition service built around an **offline**
Conformer-CTC model, with no retraining, no re-export and no model surgery.

Every figure below was measured in this repository. Estimates are labelled as
such. Where something was tried and did not work, that is recorded too — several
of the most useful conclusions came from measurements that contradicted the
plan.

**On the numbers.** Every accuracy figure here comes from real speech: the
Common Voice English clips in `common_voice_en/`, scored against their own
`train.tsv` transcripts by [`tools/real_audio_wer.py`](../tools/real_audio_wer.py),
decoded by the real `stt_en_conformer_ctc_large` checkpoint. An earlier draft of
this document quoted error rates from the synthetic fixture in `fixtures/`. That
fixture is a *code-path* exercise — it lets the pipeline run on a machine with no
checkpoint — and its error rates describe a toy model reading toy audio. They
overstated the central result by more than 2×, and they have been removed rather
than adjusted. Behaviour figures (latency, RTF, concurrency) come from the real
service under [`tests/load/`](../tests/load/README.md).

---

## 1. What was asked for

An application-level streaming layer over an existing IndicConformer-CTC model
exported to ONNX, for a conversational healthcare use case. The model has no
encoder cache, no streaming attention and no stateful inference — it is a
full-context offline model, and the graph was verified to confirm it:

```
state: NONE FOUND -> model is stateless/full-context.
```

The brief asked for incremental transcript output during speech, an
authoritative transcript at the end, and enough instrumentation to decide the
operating point empirically rather than by assumption.

Downstream layers — NMT, NLU, question retrieval, TTS — consume this. That
context matters: the value of partial output is that it lets those layers start
working before the speaker has finished.

---

## 2. Why not just Flask + `model.transcribe()`

This is the right question to ask, and for some use cases the answer is "you
should". The distinction is not speed; it is what the two approaches are
*capable of*.

```python
@app.post("/transcribe")
def transcribe():
    return model.transcribe(request.files["audio"])   # the obvious approach
```

### 2.1 It cannot produce output during speech

Nothing exists until the speaker stops, the file is uploaded, and the whole
utterance is decoded. For a conversational agent this is structural, not a
tuning problem: the downstream NLU, retrieval and TTS stages cannot begin until
everything else has finished, so their latencies **add**.

This service emits authoritative text at every pause:

```
[  8.6s] SEGMENT: 'i have been having chest pain'
[ 10.2s] SEGMENT: 'for three days'
```

Retrieval for "chest pain" can run while the patient is still saying "for three
days". That is the whole point, and it is unavailable to a request/response API.

### 2.2 Single-pass decoding of a long turn measurably degrades

The naive approach hands the model the entire recording. The checkpoint's own
training config caps utterances at 11 s (`max_duration: 11`), so a 90 s turn is
out of distribution.

**Measured on real speech.** 6 turns averaging 92 s, each built by concatenating
consecutive Common Voice clips from one speaker with 0.7 s pauses between them —
what a person dictating several sentences sounds like to a segmenter. 810
reference words, real checkpoint, greedy decoding, no LM:

| | WER | relative |
|---|---|---|
| single pass over the whole turn | 0.0630 | — |
| pause-segmented streaming | **0.0519** | **−17.6 %** |

Same audio, same model, same decoder. The only difference is whether the encoder
sees 92 seconds at once or one pause-bounded span at a time.

**The gap opens with turn length**, and below ~20 s it runs the other way.
Same construction, 5 turns per length, 10 s segment cap:

| turn length | ref words | single pass | pause-segmented |
|---|---|---|---|
| 6.4 s (100 real clips, no concatenation) | 984 | 0.0528 | **0.0518** |
| 17.5 s | 136 | **0.0441** | 0.0662 |
| 33.9 s | 246 | 0.0610 | **0.0569** |
| 66.1 s | 474 | 0.0612 | **0.0485** |
| 92.4 s | 810 | 0.0630 | **0.0519** |
| 124 s | 87 clips | 0.0758 | **0.0505** |

Two things to read off this, and the second is the one that would be easy to
hide:

* **Single-pass WER climbs with turn length; segmented WER does not.** The
  segmented path sits at 0.05 ± 0.01 across the whole range, because every
  decode it performs is the same size regardless of how long the speaker talks.
* **At 17.5 s, segmenting is worse** — 0.066 against 0.044. Cutting a short turn
  into two or three spans throws away cross-sentence context the encoder was
  happily using, and there is no length penalty yet to pay for it. The sample is
  the smallest in the table (136 reference words, so this is 9 errors against 6)
  and should not be over-read, but the direction is consistent with the 6.4 s row
  above, where the two paths tie because there is nothing to cut.

So the accuracy case for segmenting is **not** "it is always better". It is
"below ~20 s it costs a little, above ~30 s it wins, and the win grows without
bound while the cost does not". A conversational turn is well past that
crossover; a voice-search query is not.

**And segmenting too aggressively is worse than not segmenting at all.** With
the cap lowered to 6 s, the segmenter ran out of pauses and force-cut mid-phrase
43 times across the same six turns:

| `max_segment_duration` | WER | forced cuts |
|---|---|---|
| 6 s | 0.0642 | 43 |
| 10 s | **0.0519** | 0 |
| 20 s | 0.0519 | 0 |
| 40 s | 0.0519 | 0 |

At 6 s it scores worse than the single pass it was supposed to improve on. The
cap is a safety valve for speech that never pauses, not a knob to minimise —
and above 10 s it stops mattering entirely, because natural pauses close every
segment long before the cap fires.

Cutting **at pauses** is what keeps each decode in distribution.
`model.transcribe()` on a long turn cannot, and cutting on a timer is worse than
either.

### 2.3 Past a certain length, the single pass does not degrade — it fails

The WER table above only covers lengths where a single pass still *works*.
Measured on the same hardware (GTX 1650, 4 GB), decoding one recording in one
forward pass:

| recording | single pass | pause-segmented |
|---|---|---|
| 120 s | 2.1 s, WER 0.076 | fine |
| 150 s | **18.1 s** (8.6× slower), works | fine |
| 180 s | **CUDA OOM** — one attention `MatMul` asks for 1.30 GB | fine |
| 210 s | **model error** — `Add_2: right operand cannot broadcast on dim 3, LeftShape {1,8,5251,5251}` | fine |
| 245 s | WER **1.000**, then a native `terminate()` **killed the process** | WER 0.052 |

Three separate failure modes, none of them graceful:

* Self-attention is **quadratic in sequence length**, so memory grows as the
  square of the turn. At 4 GB the wall arrives just short of three minutes.
* At 210 s the model's own relative-position buffer runs out — a hard
  structural limit, not a memory one. A bigger GPU does not fix it.
* At 245 s the abort came from inside `onnxruntime_providers_cuda.dll`. It is a
  native `terminate()`, **not a Python exception**, so no `try/except` catches
  it. In a Flask worker that is the worker gone, taking every other in-flight
  request with it.

And a quieter one: the first OOM permanently disabled zero-copy transfers for
the rest of the process —

```
CUDA zero-copy io_binding failed (...); falling back to host transfers
for the rest of this session.
```

— so a *single* oversized request leaves the service slower for every request
after it, with nothing but one log line to say why.

The pause-segmented path decoded all of the same recordings, including the 245 s
one, at WER ~0.05, because it never hands the encoder more than
`max_segment_duration` at a time. Its cost is linear in turn length by
construction. This service has no single-pass route at all — `POST /transcribe`
runs the same segmented pipeline the WebSocket does, so a long upload is
decoded span by span rather than in one go.

This is the part of the argument that does not depend on caring about latency at
all. If a user can upload a four-minute recording, `model.transcribe()` on this
class of hardware is not slow — it is a crash.

### 2.4 Measured latency, end to end

| | naive request/response | this service |
|---|---|---|
| first text available | after the speaker stops **and** uploads | ~0.7 s into speech |
| response latency after speech ends | upload + full-utterance decode | **727 ms** (p50, one caller) |
| of which the deliberate silence wait | — | 500 ms |

Measured through the real service over a WebSocket with
[`tests/load/`](../tests/load/README.md), on a 46 s recording, GTX 1650. The
500 ms is `segment_silence` — waiting long enough to be sure the speaker paused.
It is a product decision, not a cost, and it does not shrink with a faster GPU.

The full curve, and what happens when several people call at once, is §6.2.

### 2.5 The operational differences

| | naive Flask | this service |
|---|---|---|
| model residency | reloaded per worker/request unless handled | loaded once, **0.2 s** startup |
| concurrency | whatever the WSGI worker does | capped; past the limit it refuses instead of degrading everyone — **measured**, §6.2 |
| a 4-minute upload | crashes the worker (§2.3) | decoded segment by segment |
| transport | HTTP upload only | WebSocket streaming + HTTP |
| audio formats | whatever you wire up | content-sniffed; WAV/FLAC/OGG/MP3 native, m4a/WebM via ffmpeg |
| bad input | 500 | **415** naming the accepted formats |
| observability | none by default | per-stage timings, RTF, response latency, GPU utilisation |
| footprint | ~425 MB with torch | **68 MB** (torch-free runtime) |

### 2.6 When Flask *is* the right answer

If you transcribe complete recordings offline, latency is not a constraint, and
utterances are short — use the simple thing. Most of this document is machinery
for problems that only exist when a human is waiting.

---

## 3. What was built

```
streaming_asr/            reference implementation (torch frontend)
├── segmented.py          pause-segmented pipeline — the default
├── pipeline.py           rolling-window pipeline — superseded, kept for comparison
├── config.py             every tunable, nothing hard-coded downstream
├── audio/                sources, container/codec decoding, microphone
├── preprocessing/        mel frontend, ported faithfully from the reference
├── inference/            ONNX Runtime engine + graph introspection
├── decoding/             greedy CTC, beam + KenLM (3 backends)
├── hypothesis/           aligners + stabiliser (windowed pipeline only)
├── server/               model pool, env config, FastAPI REST + WebSocket
└── benchmark.py, cli.py

streaming_asr_lite/       torch-free runtime — 68 MB vs 425 MB
├── export_frontend.py    mel frontend → ONNX, with numeric verification
├── frontend.py, engine.py, audio.py, pipeline.py, factory.py

tools/                    vocabulary extraction, alignment fidelity,
                          transport parity, real-audio WER, synthetic fixtures
tests/                    unit + integration suite
tests/load/               concurrency and load framework — simultaneous
                          WebSocket users, latency percentiles, saturation
```

---

## 4. How it works

```
audio → VAD → growing segment buffer
                 │
                 ├─ every chunk (rate-limited): re-decode the open segment  → partial
                 ├─ 0.5 s silence: close, decode the span whole             → segment
                 └─ 1.5 s silence: join the turn's segments                 → final
```

Three tiers of trust, so a consumer picks its own latency/certainty trade:

| event | meaning | use for |
|---|---|---|
| `partial` | open segment, re-decoded, **revisable** | live captions; speculative prefetch |
| `segment` | pause-bounded span, decoded whole | the authoritative unit — accumulate these |
| `final` | a turn's segments joined | a complete thought; anything with consequences |

The design commitment that makes it work: **nothing is ever committed
mid-utterance.** Partials are replaced wholesale, so an error in one cannot
survive into the transcript. Section 5.2 is the story of learning that the hard
way.

---

## 5. What went wrong, and what fixed it

### 5.1 Inherited from the reference notebook

The starting point was a working prototype. Reading it carefully surfaced five
issues, all of which shaped the design:

**The rolling-buffer loop loses text.** Its own output shows it — chunk 31 reads
`india versus pakistan world cup final`, chunk 35 reads `pakistan world cup
final`. Each window's transcript *replaced* the previous one, so words scrolling
out of the 4 s buffer vanished. Nothing accumulated a transcript. This created
the hard deadline the whole stabiliser was built around: a word must be
committed before it leaves the buffer, or it is gone.

**Beam + KenLM ran on every chunk** — the expensive decoder invoked every
160 ms, when the streaming transcript it produced was going to be replaced
anyway.

**`FilterbankFeaturesTA` subclasses `nn.Module` without importing `nn`.** It
worked only because an earlier cell leaked the name.

**The preprocessor was never put in `eval()` mode.** Dithering is gated on
`self.training`, so fresh Gaussian noise was added to every window — the *same
audio* produced different features on each of its 25 overlapping passes,
injecting instability with no acoustic cause. A stabiliser built on top would
have been measuring its own noise.

**`AudioChunkIterator` silently drops the trailing partial chunk**, losing up to
160 ms of speech. This implementation zero-pads it.

### 5.2 The duplicate-commit saga — three failed fixes

The most instructive part of the project. On real audio the rolling-window
pipeline produced:

```
'...climb in in in colorado which traverses the the the highest paved road...'
```

Words committed two and three times. Three successive repairs each fixed the
case in front of them and broke another:

| attempt | fixed | broke |
|---|---|---|
| timestamp cutoff | nothing | words drift past the boundary and re-commit |
| exact suffix/prefix match | the simple case | any re-spelling (`war craft`→`warcraft`) matches nothing, strips nothing — and once one duplicate commits, the tail is corrupted and **never resyncs**, so it cascades |
| alignment without a time gate | re-spelling | a phrase genuinely repeated 30 s later matched by text and was **deleted** |

Each fix passed its targeted tests and failed on the next real recording.

**The turning point was recognising the problem as structural.** Committing
words irreversibly at points chosen by a *timer* means every model quirk at
such a point becomes permanent, and errors compound — which is exactly the
"clean for five seconds, then degrades" signature the logs showed.

The fix was to stop doing it: cut at **pauses**, where the question "is this
word new?" does not arise, and re-decode each span whole. That deleted the
failing machinery rather than patching it a fourth time.

**Process change:** targeted tests were clearly insufficient, so a randomised
simulation was built — 36 seeded runs streaming a sentence through the tracker
while perturbing hypotheses the way a real Conformer does (timestamp drift,
re-segmentation, clipped edges, unsettled trailing words). It immediately caught
10 failures in a fix that had passed every hand-written test.

### 5.3 Concurrency bugs that only appear under load

Moving from a single-threaded CLI to a shared-session server made state shared
without a re-audit. Two consequences:

**A data race corrupting output silently.** The engine reused one array for the
model's `length` input. With one session serving several streams, thread B could
overwrite it before ONNX Runtime read thread A's value — masking A's features to
the wrong frame count. Wrong transcript, no exception, nothing in the logs.

Proven by reintroducing the bug against the regression test: **74.3% of output
elements wrong.**

**A capacity-slot leak causing a hard outage.** `try_acquire()` took a slot, but
pipeline construction and the opening `ready` send sat *outside* the `try` whose
`finally` released it. A client disconnecting immediately — a load-balancer
health probe suffices — leaked a slot permanently. After
`max_concurrent_streams` of them the service 503s every caller until restarted.

Both were found by reading, then pinned by a test. What neither had was a way to
*observe* the service under concurrent load end to end, which is what
[`tests/load/`](../tests/load/README.md) now provides. Its central assertion is
the one that catches this whole bug class: every client sends byte-identical
audio, so every transcript must equal a serially-produced reference. A leak of
per-stream state shows up as text that is wrong only when several people are
talking — no exception, nothing in the log, and invisible to any single-stream
test.

### 5.4 Silent failures — the recurring theme

The most dangerous class in this system: things that produce confident, wrong
output with no error.

**Vocabulary mismatch.** A token table that does not match the checkpoint makes
every argmax index a different vocabulary — fluent, completely wrong text.
Mitigated with `tools/extract_vocabulary.py` and a startup guard that refuses to
run if the size disagrees with the model's output width.

**Subsampling factor latched from a warm-up window.** Convolutional subsampling
pads at the edges, so a short input measures a 4× encoder as 3×. Latching that
scaled *every timestamp by ¾* — invisibly, since the values stayed
self-consistent and the transcript still read correctly. It corrupted the very
diagnostic built to measure timestamp drift. Now estimated from the largest
input seen.

**CUDA that isn't.** `onnxruntime.get_available_providers()` reports what the
build was *compiled* with, not what can load. A version mismatch still lists
`CUDAExecutionProvider`, then silently falls back to CPU. The pipeline checks
`session.get_providers()` — reality — and `/health` reports what actually
loaded. Related: on Windows, torch must be imported *before* onnxruntime or ORT
cannot find the CUDA DLLs at all.

That last point has a consequence nobody intended, and it took the load
framework's metadata capture to notice. On this machine CUDA and cuDNN exist
only inside `torch/lib`, never on the system path, so **whether the service gets
a GPU at all depends on whether something happened to import torch first.**

A program that imports only `streaming_asr_lite.*` — as the torch-free runtime
intends — gets:

```
LoadLibrary failed with error 126 ... onnxruntime_providers_cuda.dll
Failed to create CUDAExecutionProvider. Require cuDNN 9.* and CUDA 12.*
```

…and runs on CPU. The **server**, configured identically with
`ASR_RUNTIME=lite`, gets `CUDAExecutionProvider`. The difference is one line in
`server/model_pool.py`:

```python
from streaming_asr.pipeline import StreamingASRPipeline   # unused since the
                                                          # factory refactor
```

It is dead — `new_pipeline()` has delegated to the factory for some time — but
it drags in the torch preprocessor, and with it torch, and with torch the CUDA
DLLs. So the lite runtime is fast on the server *by accident*, and would silently
lose its GPU the moment someone deleted an unused import. `streaming_asr/cli.py`
pulls torch the same way.

Two things followed, and the second was the more embarrassing:

* **CUDA availability rested on an import side effect.** Nothing lied about it —
  `/health` reports the providers that actually loaded — but "delete an unused
  import, lose the GPU" is not a property anyone chose.
* **The service did not realise the torch-free footprint at all.** The 68 MB /
  0.28 s figures elsewhere in this document are for the runtime *in isolation*,
  which is what `tests/test_lite.py` asserts. The server loaded torch regardless
  of `ASR_RUNTIME`. That claim was made for the package and quietly read as a
  claim about the service.

**Both are now fixed**, and the fix is more useful than the deletion would have
been. The trap was that removing the dead import is *correct* and would have
cost this machine its GPU — so the two had to be separated rather than traded
off.

`importlib.util.find_spec("torch")` locates `torch/lib` **without executing the
module**, so the CUDA libraries can be found without paying for torch.
Registering the directory turned out to be necessary but not sufficient: ONNX
Runtime resolves those DLLs as ordinary dependencies of its own provider DLL,
which goes through the standard loader search order and ignores
`os.add_dll_directory`. Measured — with only the directory registered, the
session still came up `CPUExecutionProvider`. What `import torch` was actually
doing was *loading* them, so the loader satisfies later requests for the same
name from memory. `ctypes.WinDLL` on each library, in dependency order, does the
same thing deliberately.

The result, measured on a running server with `ASR_RUNTIME=lite`:

```
runtime   : lite
providers : ['CUDAExecutionProvider', 'CPUExecutionProvider']
torch runtime DLLs mapped : NONE      (torch_python / torch_cpu / c10)
CUDA DLLs mapped          : cudart64_12, cublas64_12, cudnn64_9, +5 cuDNN 9 parts
```

CUDA, no torch. `streaming_asr/cli.py` had the same defect, where the import was
annotation-only and `TYPE_CHECKING` was enough. Four tests in
[`tests/test_execution.py`](../tests/test_execution.py) import each entry point
in a clean interpreter and fail if torch appears.

One caveat worth stating: this makes torch a convenient *source* of CUDA
libraries rather than a dependency. A deployment with the CUDA toolkit installed
properly — or the container, which has it — never reaches this path. It exists
so that a machine that only has CUDA inside a torch wheel still gets a GPU
without importing torch to do it.

Every results file records the *active* providers for exactly this reason, and
the run header warns when a GPU is present but unused.

**Frontend mismatch.** A mel frontend that differs from the one used at training
degrades accuracy with nothing raised. This is why the ONNX frontend export
lifts the window and filterbank from a live torchaudio module rather than
recomputing them, and is verified numerically (2.5e-04 on unit-variance
features) rather than assumed.

### 5.5 Measurements that contradicted the plan

**Snapping segment seams to silence.** Plausible, standard in long-form ASR,
and measured neutral-to-worse — on the synthetic fixture, which has uniform
inter-word gaps and no real pauses, so snapping had nothing to find. Shipped
**off**. The numbers are not quoted here because that fixture cannot support an
accuracy claim; re-measure on real speech before dismissing it permanently.

**"The gain from segmenting long turns is huge."** The synthetic fixture said
WER 0.545 unsegmented against 0.208 at a 6 s cap, and 0.506 against 0.000 for
streaming versus single-pass. Re-measured on real speech with the real
checkpoint, the accuracy gain is real but much smaller — 0.063 → 0.052 at 92 s —
and at short turns or a small cap segmenting is *worse*. Every one of those
synthetic figures overstated the effect, and two of them pointed the wrong way.
They are removed from this document.

What the synthetic fixture missed entirely was the *actual* argument, which is
not gradual at all: past ~150 s the single pass stops working — OOM, then a hard
model error, then a process-killing native abort (§2.3). A toy model with a
small hidden size never runs out of memory, so the fixture could not have shown
this and its WER column quietly implied the wrong shape of curve.

The lesson is not "synthetic fixtures are useless" — that fixture caught real
bugs and still does. It is that a stand-in model validates *code paths*, never
*accuracy* and never *resource behaviour*, and the three are easy to conflate
when the fixture prints a WER column.

**"Segmented is slower because it re-decodes the whole segment."** Asserted,
then measured false — 0.101 vs 0.162 RTF with beam off. The real cause of the
observed slowdown was **pacing**: unpaced runs keep the GPU clocked up and come
out ~1.5× optimistic. The CLI now says so next to the number.

**"Rewrite the ML layer in C++."** Measured: the ML layer is *already* C++
(ONNX Runtime, torchaudio kernels, NumPy). Python glue is **≈0.4 ms of a 43 ms
chunk — about 1%.** The GPU binds at ~1.5 concurrent streams; the GIL at ~400.
The rewrite would remove a limit ~250× further away than the real one.

---

## 6. Measured results

**Accuracy** — real Conformer-CTC large, Common Voice English, greedy, no LM
([`tools/real_audio_wer.py`](../tools/real_audio_wer.py)):

| workload | single pass | pause-segmented |
|---|---|---|
| 100 clips, mean 6.4 s (984 words) | 0.0528 | **0.0518** |
| 5 turns, mean 17.5 s (136 words) | **0.0441** | 0.0662 |
| 5 turns, mean 66.1 s (474 words) | 0.0612 | **0.0485** |
| 6 turns, mean 92.4 s (810 words) | 0.0630 | **0.0519** |
| 5 turns, mean 124 s (87 clips) | 0.0758 | **0.0505** |
| ≥180 s | **fails** (§2.3) | ~0.05 |

Reference and hypothesis are normalised identically — lowercased, hyphens to
spaces, punctuation dropped — so the figure is recognition error, not
formatting. Segmented WER is flat across the range by construction: every
decode it performs is the same size whatever the turn length.

**Compute breakdown** — real Conformer-CTC large, GTX 1650, 160 ms chunks:

| stage | per chunk | implemented in |
|---|---|---|
| ONNX inference | 39.74 ms | C++ |
| mel frontend | 2.47 ms | C++/CUDA |
| greedy CTC decode | 0.36 ms | C |
| Python glue | **≈0.4 ms (1%)** | Python |

**Latency**, pause-segmented, greedy, one caller:

| | |
|---|---|
| first partial | ~0.7 s |
| segment decode | **46 ms** (max 58 ms), CLI, single stream |
| response latency, through the service | **727 ms** p50 (§6.2) |

The CLI figure and the service figure differ because the service one includes
the socket, the event loop and a longer recording. Both are real; quote the
service one, because that is what a caller gets.

**Footprint**, torch-free runtime, identical transcripts:

| | RSS | startup | inference |
|---|---|---|---|
| torch frontend | 425 MB | 2.52 s | 0.28 s |
| ONNX frontend | **68 MB** | **0.28 s** | 0.27 s |

**Decoder cost** — the pure-Python beam, with no LM:

| | segment decode | response latency |
|---|---|---|
| greedy | 46 ms | 546 ms |
| pure-Python beam | 1924 ms | 2424 ms |

42× the cost for no language model. This is a property of the *fallback
implementation*, not of beam search: native backends (flashlight, pyctcdecode)
are expected in the 10–50 ms range — **estimate, not measured.**

### 6.2 Behaviour under concurrent load

Measured end to end through the WebSocket service with
[`tests/load/`](../tests/load/README.md): N simultaneous callers, each streaming
the same 46 s recording **paced at wall-clock speed**, real checkpoint on a
GTX 1650 (4 GB), greedy decoding, `ASR_MAX_CONCURRENT_STREAMS=32` so admission
control never intervenes.

| streams | success | RTF p50 | RTF p95 | response p50 | response p95 | GPU % |
|---|---|---|---|---|---|---|
| 1 | 100 % | 0.427 | 0.427 | **727 ms** | 749 ms | 27 |
| 2 | 100 % | 0.501 | 0.501 | 756 ms | 835 ms | 38 |
| 4 | 100 % | 0.719 | 0.728 | **1 178 ms** | 1 383 ms | 64 |
| 8 | 100 % | 0.883 | 0.905 | **3 396 ms** | 4 678 ms | 91 |
| 12 | 100 % | 1.188 | 1.318 | 14 151 ms | 22 539 ms | 90 |
| 16 | **0 %** | — | — | — | — | 93 |

*response* is speaker-perceived: from the end of a segment's speech to its text
arriving. It includes the 500 ms silence wait by construction. At one caller the
remaining ~227 ms is not all decode — the segment decode itself measures 46 ms;
the rest is chunk granularity (a segment can only close on a 160 ms boundary),
the socket, and the event loop. Worth knowing before optimising the decoder to
chase it.

Three things fall out of this that were previously guesses:

**RTF below 1 does not mean the service is usable.** At 8 streams RTF p95 is
**0.905** — comfortably real-time by that measure — while a speaker waits
**3.4 seconds** for their words. Sizing on RTF alone would have put capacity at
roughly 12 streams; sizing on what a person experiences puts it at **4**. This is
the single most useful thing the load framework produced, and it is why
`segment_response_latency` is the metric the tool leads with.

**The knee is between 4 and 8.** Response latency roughly triples while RTF moves
by a fifth. GPU utilisation over the same step goes 64 % → 91 %, so the device is
the constraint — which also says the fix is batching (§8.1), not more processes.

**Past the limit it does not degrade, it dies.** At 16 streams the service hit
CUDA out-of-memory inside a device-to-host copy and the process went down,
taking all 16 callers with it — 0 % success. That is what an admission cap set
above what the hardware can hold buys you.

With the cap set correctly, the same 16-caller burst behaves completely
differently:

| `ASR_MAX_CONCURRENT_STREAMS` | 16 simultaneous callers |
|---|---|
| 32 (oversized) | **service crashes**, 0 % success, everyone loses |
| 4 (measured) | 4 served at **1 142 ms p95**, 12 refused cleanly, 0 errors |

Same hardware, same burst. The four admitted callers get the same latency they
would have had alone at that concurrency, and the twelve refused get an
immediate, actionable refusal instead of a dead socket. **"Capped, not queued"
was a design assertion; this is the measurement behind it.** It also shows the
cap is only correct if it comes from a curve like the one above — 32 was a
plausible-looking number and it was catastrophic.

**On the two modes.** The same sweep in maximum-throughput mode returned RTF
0.419 at one stream against 0.427 real-time-paced — a 2 % difference, far
smaller than the ~1.5× recorded earlier in this project from a different
measurement path, and worth re-checking on other hardware before either figure
is relied on. What is *not* small is the latency: throughput mode reported
"response latency" of 9–19 s, because the whole recording is delivered at once
and the number becomes queueing depth rather than responsiveness. The two modes
are reported separately for that reason, and throughput-mode latency is never
quoted as a user-facing figure.

### 6.3 Where the response latency actually goes

The decomposition holds at every concurrency level measured, on both devices:

```
response  =  segment_silence  +  segment processing  +  chunk granularity
             (policy, 500 ms)     (queue + decode)       (+ socket, ~50-140 ms)
```

| | GTX 1650 | CPU, 4 cores |
|---|---|---|
| policy wait | 500 ms | 500 ms |
| segment processing | **87 ms** | **780 ms** |
| chunk granularity + socket | ~140 ms | ~50 ms |
| **one caller, total** | **727 ms** | **1 329 ms** |

Only the middle row is hardware. Two thirds of the single-caller latency on a
GPU is policy and framing, which is worth knowing before anyone optimises the
decoder to chase it.

**It degrades in the right order.** Under load the rate limiter sheds partials
while authoritative output stays complete:

| streams (GPU) | partials/stream | segments/stream |
|---|---|---|
| 1 | 244 | **6** |
| 4 | 152 | **6** |
| 8 | 67 | **6** |

Captions get choppier before the transcript loses anything. That was not
designed in — it falls out of rate-limiting partials on measured cost — but it
is the correct thing to shed, and it means a brief overload costs smoothness
rather than words.

### 6.4 CPU only — measured, and it works

Same checkpoint, same recording, `ASR_PROVIDERS=CPUExecutionProvider`, on 4
physical cores (Tiger Lake laptop):

| streams | RTF p50 | response p50 | response p50, `INTRA_OP_THREADS=2` |
|---|---|---|---|
| 1 | 0.742 | 1 329 ms | 1 577 ms |
| 2 | 0.863 | 1 567 ms | 1 803 ms |
| 4 | 1.109 | **6 133 ms** | **3 412 ms** |

Two findings.

**ONNX Runtime's default threading is self-defeating under concurrency.** With
`intra_op_threads = 0` (the library default, and this project's default) every
inference claims every core, so four streams put roughly sixteen threads on
eight logical cores. Pinning to 2 costs 250 ms at one caller and saves 2.7 s at
four. **Any CPU deployment must set `ASR_INTRA_OP_THREADS` explicitly**; 0 is
the wrong default there, and this is the cheapest tuning win in the project.

**A 4-core laptop CPU is within 2× of a GTX 1650 on RTF** — 0.74 against 0.43.
That says more about the GTX 1650 than about the CPU, but it does mean CPU-only
is a real option rather than a fallback. Usable concurrency is **1–2 streams per
4 physical cores**, roughly 2 cores per stream.

CPU-only is also where the torch-free runtime finally earns its keep: no CUDA,
no torch, 68 MB instead of 425 MB, and since §5.4 that applies to the service
and not only to the package.

### 6.5 What transfers to other hardware — **estimates**

Everything in this subsection is extrapolation and is labelled as such. The
measured parts are §6.2–6.4.

GPU utilisation fits **≈ 15 % baseline + 12 % per stream** (27 / 38 / 64 / 91 %
at 1 / 2 / 4 / 8 streams). Two thresholds follow:

* compute saturation at **N ≈ 7**, where the fit crosses 100 %;
* the latency knee at **N ≈ 4**, at 64 % utilisation.

**Usable concurrency is ~55–60 % of the saturation point, not the saturation
point.** That ratio is the thing worth carrying to other hardware. The absolute
numbers are not.

Memory ran out at nearly the same concurrency here — ~1.7 GB resident plus
~250–400 MB per stream against 4 GB — but that is a coincidence of a small card.
On 24 GB, memory stops binding entirely and compute binds alone.

Scaling the per-stream cost by FP32 throughput. Weight traffic is ~2.6 GB/s per
stream against 128 GB/s available, so this workload is compute-bound rather than
bandwidth-bound, which makes FLOPS the right lever — discounted for the poorer
occupancy a small-batch model gets on a large die:

| | usable streams | basis |
|---|---|---|
| GTX 1650, 4 GB | **4** | **measured** |
| T4, 16 GB | ~10–12 | *est.*, 2.8× FP32 |
| L4 / A10G, 24 GB | ~20–40 | *est.*, 10× FP32, discounted 50–100 % |
| 32-core server CPU | ~8–12 | *est.*, sublinear scaling, lower clocks, needs thread pinning |

Two multipliers compound on top of any of these, and both matter more than the
choice of card: **batching** streams into one inference call (§8.1, *est.*
3–4×) and **INT8** (*est.* 2–3× on x86 with VNNI, more with AMX).

### 6.6 What no hardware fixes

* **The 500 ms policy floor.** Lower `ASR_SEGMENT_SILENCE` if you want a faster
  answer; that trades against false cuts, and no GPU affects it.
* **The ~210 s structural limit** (§2.3). The model's own relative-position
  buffer. An A100 hits it at the same place.
* **The ~1 % Python glue.** Already measured; nothing to reclaim.

The one number to carry to a new machine is not a concurrency figure — it is the
**utilisation-to-latency relationship**. Keep the device under ~65 % and callers
see ~1.2 s; let it reach 90 % and they see 3.4 s *while RTF still reads 0.9 and
looks healthy*. Set `ASR_MAX_CONCURRENT_STREAMS` from that, measured on the real
box.

---

## 7. Current limitations

**The flashlight + KenLM path has never executed.** No Windows wheel, so the
reference decoder is written but has zero runtime coverage. Everything to date
ran on the LM-free fallback. Exercise it on Linux before trusting any final
quality figure.

**Timeline discontinuity — a latent bug.** The pipeline derives time from
cumulative samples, not wall clock. A client that pauses transmission during TTS
and resumes produces no gap from the server's view, so the open segment never
closes and pre/post-TTS speech merge. Not yet triggered because nothing pauses
transmission; barge-in will trigger it immediately.

**The segmentation loop is duplicated** between `segmented.py` and
`streaming_asr_lite/pipeline.py`, because the former imports torch via its
preprocessor. A fix applied to one must be applied to the other.

**Compute grows with segment length.** Re-decoding the open segment on every
partial is the price of never committing mid-utterance. A rate limiter bounds
it, but on long pauseless speech it costs more than a fixed window would.

**Thresholds are defaults, not answers.** `segment_silence`, `turn_silence` and
especially `energy_threshold` need tuning on real audio. If the energy threshold
sits below the room's noise floor, silence never registers and every segment
gets force-cut — and §2.2 shows force-cutting is worse than not segmenting at
all, so this is not a cosmetic misconfiguration.

**The default runtime does not get CUDA on a bare Windows host.** See §5.4.
`--runtime torch` does. Check `/health` before believing any performance number.

**Long turns here are constructed, not recorded.** Common Voice ships single
sentences, so the 92 s turns in §2.2 are consecutive clips from one speaker
joined by 0.7 s of digital silence. The speech is real and the references are
exact, but the pauses are uniform and clean. Real conversational pauses vary in
length and contain room noise, which is precisely what `energy_threshold` has to
cope with — so the segmentation thresholds are *less* stressed here than they
will be in production.

**The capacity figure in §6.2 is for one specific machine.** A 4 GB GTX 1650
sharing a desktop with an IDE. The shape of the curve should transfer; the
numbers will not. Re-run the sweep on the deployment target before sizing
anything — it takes an afternoon and it changes `ASR_MAX_CONCURRENT_STREAMS`,
which §6.2 shows is the difference between a refusal and an outage.

**The 4 GB GPU is the binding constraint throughout.** Both the long-recording
failures (§2.3) and the 16-stream crash (§6.2) are memory. On a 24 GB card both
limits move a long way out, and some of the conclusions above would change in
degree — though not the quadratic-attention scaling or the 210 s structural
limit, which are properties of the model.

**`ASR_MAX_SEGMENT_SEC` is documented but not enforced.** It is the one setting
that can walk the service into §2.3's failure zone, and nothing rejects an
absurd value at startup. Documented in `TODO.md` §1.2 with a cheap fix.

**The service dies rather than shedding load when the cap is set too high.**
§6.2. Admission control is the only defence, and it depends on a number the code
cannot derive from the hardware.

**The load generator is single-process.** At high concurrency its own event loop
can become the constraint. That is measured (`loop_lag_p95_ms`) and warned
about, but not yet worked around by sharding across processes.

---

## 8. What to do next

Ordered by measured or expected leverage. Details in
[`TODO.md`](TODO.md).

0. **Run the load sweep on the deployment target.** Everything below is sized
   against a capacity figure nobody has yet. The framework is built and
   verified; it needs the real hardware and the real configuration for an
   afternoon. Cheapest item on this list and it re-ranks the rest.
1. **Batch multiple streams into one inference call.** The single largest
   throughput change available — a GPU running 8 sequences as one batch does not
   take 8× as long. *Estimated* 3–4×. A change to the model pool, not the
   language. Sweep before and after; the curve is the acceptance test.
2. **Measure the downstream fetch latency.** Decides whether the speculative
   consumption layer ([`PROGRESSIVE_CONSUMPTION.md`](PROGRESSIVE_CONSUMPTION.md))
   is worth building at all. Measurable today.
3. **Prototype multilingual retrieval.** If an Indic utterance can match an
   English question bank directly (LaBSE, multilingual-E5), NMT leaves the
   latency-critical path entirely — and with it the partial-translation
   hallucination hazard. ~1 day of work; decides the shape of the whole loop.
4. **INT8 quantisation.** *Estimated* 2–3× inference speedup at some WER cost.
5. **Ring buffer, AEC and turn boundaries** for barge-in
   ([`CONVERSATIONAL_LOOP.md`](CONVERSATIONAL_LOOP.md)), which also fixes the
   timeline discontinuity.
6. **Pre-render TTS** if the question bank is bounded. Removes TTS from the
   runtime budget entirely — usually a larger saving than anything speculative.

### What these compound to

The current 546 ms response latency is 500 ms policy + 46 ms compute. The
remaining wins are not in ASR: they are in overlapping the *downstream* stages
with speech that has not finished yet. That capability exists only because the
pipeline emits authoritative text mid-utterance — which is precisely what the
request/response design cannot do.

---

## 9. Decision guide

**Use this if** you need text while the speaker is still talking, turns run
longer than ~10 s, you are building a conversational loop, or you need several
concurrent streams with predictable behaviour under load.

**Use Flask + `model.transcribe()` if** you transcribe complete recordings
offline, latency is not a constraint, and utterances are short. It is less code
and less to operate.

**The honest summary:** most of this repository is machinery for problems that
only exist when a human is waiting for an answer. If nobody is waiting, you do
not need it.

The accuracy argument on its own is not enough to justify the switch, and it
would be dishonest to pretend otherwise: on real speech, segmenting a 92 s turn
at pauses buys **17.6 % relative WER** (0.063 → 0.052), and on short clips it
buys nothing at all. Worth having, not worth this much code.

The argument that *is* decisive is what the shape of the output makes possible.
A request/response API produces one string after the speaker stops. This
produces authoritative text at every pause, ~0.5 s behind the speech — so
retrieval for "chest pain" runs while the patient is still saying "for three
days", and the downstream NLU, translation and TTS stages overlap with speech
instead of queueing behind it. No amount of tuning a single `transcribe()` call
gets there, because the limitation is not its speed. It is that nothing exists
until the end.

Take the accuracy as a bonus, and the incrementality as the reason.
