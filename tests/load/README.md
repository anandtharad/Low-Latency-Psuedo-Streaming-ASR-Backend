# Concurrency and load testing

Measures what the **running service** does when several people talk to it at
once. Not a unit test of threading: every client opens its own WebSocket, sends
real audio, and is scored on the events that come back.

```bash
# one level
python -m tests.load.load_test --audio sample.wav --concurrency 8

# the whole curve
python -m tests.load.run_load_sweep --audio sample.wav --levels 1,2,4,8,16,32
```

Both need a service already running:

```bash
set ASR_MODEL_PATH=stt_en_cconformer_ctc_large-averaged.onnx
set ASR_VOCAB_PATH=vocab.txt
set ASR_FINAL_BEAM=false
python -m streaming_asr.server.app
```

---

## What is here

| file | role |
|---|---|
| `websocket_client.py` | one simulated user: connects, streams, times everything |
| `metrics.py` | per-stream records, percentiles, thresholds, JSON/CSV output |
| `load_test.py` | runs N simultaneous users at one concurrency level |
| `run_load_sweep.py` | walks a list of levels (and optionally pool sizes) |
| `monitors.py` | optional CPU / RAM / GPU sampling |
| `environment.py` | benchmark metadata, captured automatically |
| `plots.py` | optional charts (needs `matplotlib`) |
| `server_process.py` | starts/stops the service, for sweeping its startup config |
| `fake_server.py` | scripted protocol-compatible server, **no model** |

### Two things that must not be confused

**Harness correctness** — `test_load_harness.py`, `test_failure_modes.py`.
Fast, deterministic, no model. They run under the normal suite:

```bash
python -m pytest tests/load -q -m "not slow"
```

They prove the *ruler* is accurate — that a server told to take exactly 400 ms
is measured at 400 ms, that clients really do run simultaneously, that a
dropped connection is recorded rather than fatal. **They say nothing about ASR
performance.** Numbers from the fake server are constants.

**Real-service behaviour** — `test_real_service.py`. Starts the actual
`streaming_asr.server.app` with a real ONNX session and asks what only a real
model can answer, principally: *do concurrent streams contaminate each other?*
Every client sends identical audio, so every transcript must be identical to a
serially-produced reference. That failure mode is silent — wrong text, no
exception — and no single-stream test can see it.

```bash
python -m pytest tests/load/test_real_service.py -q            # synthetic fixture
set ASR_LOAD_TEST_MODEL=stt_en_cconformer_ctc_large-averaged.onnx
set ASR_LOAD_TEST_VOCAB=vocab.txt
set ASR_LOAD_TEST_AUDIO=2086-149220-0033.wav
python -m pytest tests/load/test_real_service.py -q            # the real checkpoint
```

Still not a benchmark: four streams on a developer laptop measures nothing
about capacity. `run_load_sweep.py` is the benchmark.

---

## The two modes

| | `--mode realtime` (default) | `--mode throughput` |
|---|---|---|
| pacing | one chunk per chunk-duration of wall clock | as fast as the socket accepts |
| models | a phone call | an offline batch job |
| answers | *how many simultaneous callers stay acceptable?* | *how much audio can this box chew through?* |

They are reported separately and labelled in every results row, because the
difference is large and in a consistent direction. Audio fed flat out keeps the
GPU clocked up; a real caller leaves it idle between chunks and it downclocks.
This project has measured **~1.5× optimistic RTF** from exactly that. A
throughput figure quoted as capacity oversells the service by that much.

---

## Metric definitions

Every latency resolves through one anchor: the wall-clock instant at which the
client had finished transmitting the audio up to a given point in the stream.
The server derives its own `t` from cumulative samples received, so both sides
refer to the same point in the audio and the difference between them is elapsed
time and nothing else.

| metric | measured from | to |
|---|---|---|
| `connection_latency` | `connect()` returning | the `ready` event |
| `first_partial_latency` | the audio in the partial finishing transmission | the partial arriving |
| `segment_latency` | the endpoint condition becoming detectable (`end + segment_silence`) | the `segment` arriving |
| `segment_response_latency` | the end of that segment's speech | the `segment` arriving |
| `final_latency` | `{"type":"end"}` being sent | the `final` with `end_of_stream` |

**`segment_response_latency` is the number to quote to a product owner.** It is
what the speaker experiences and it *includes* the silence the service
deliberately waits out before it will call a segment closed — so it can never
fall below `segment_silence` (0.5 s by default). That floor is a product
decision. Everything above it is the machine.

**`segment_latency` is the number to quote to an engineer.** It removes the
fixed policy wait and leaves queueing plus decode, which is the part that
degrades under load.

### Documented limitations

* `segment_latency` needs `segment_silence`, which is server configuration, not
  part of the event. It is read from `/info` at startup. If that read fails,
  only `segment_response_latency` is reported — the tool does not guess.
* A **forced** segment (`forced: true`, the cap hit with no pause found) has no
  silence in front of it, so it is anchored at `end` instead. Forced segments
  are counted separately in the results.
* Client and server chunk boundaries only coincide when `--chunk-ms` is left
  unset (the client then adopts the server's advertised chunk). Set it to
  something else and every anchor rounds up to the next client frame.
* `turn_final_latency` is anchored like a segment, because the client knows
  `segment_silence` but not `turn_silence`. It is therefore an overstatement of
  turn latency by the difference between the two.

### RTF

`rtf` is compute seconds per audio second, taken from the **server's own**
metrics attached to the final event — a client cannot see inference time.
`wall_rtf` (wall duration over audio duration) is also recorded but is only
meaningful in throughput mode; under real-time pacing it is ~1.0 by
construction.

> **RTF < 1 is necessary for real-time work but not sufficient for good
> conversational latency.** A service can sit at RTF 0.05 and still feel slow,
> because what the speaker waits for is `segment_silence + decode`, and the
> first term does not care how fast the GPU is. Judge the experience on
> `segment_response_latency`; use RTF to judge headroom.

---

## Saturation

No production limits are hard-coded. The sweep reports the whole curve and
names a capacity limit only if you say what acceptable means:

```bash
python -m tests.load.run_load_sweep \
    --audio sample.wav --levels 1,2,4,8,16,32 \
    --max-p95-ms 1500 --max-rtf 1.0 --min-success-rate 0.99
```

| flag | applies to |
|---|---|
| `--max-p95-ms` | p95 of `--latency-metric` (default `segment_response_latencies`) |
| `--max-rtf` | p95 of the server-reported RTF |
| `--min-success-rate` | fraction of streams reaching a final |
| `--max-error-rate` | fraction that failed or timed out (**rejections excluded**) |

With none supplied, every measurement is still reported and no limit is
claimed.

The sweep does **not** stop at the first bad level unless you pass
`--stop-on-failure`. How a service degrades past its limit — refusing politely,
or admitting everyone and making them all late — is worth knowing before it
happens for real.

### Reading the curve

* **Latency climbs, GPU utilisation is pinned near 100%** — the device is the
  limit. More concurrency will not help; batching or a bigger GPU might.
* **Latency climbs, GPU utilisation is flat** — something else is the limit.
  Check the admission cap, then the client's `loop_lag_p95_ms`.
* **Rejections rise, latency stays flat** — admission control is doing its job.
  This is the good failure.
* **Success rate falls with no rejections** — streams are timing out or
  erroring. Read the `errors` list; this is the bad failure.
* **`loop_lag_p95_ms` above a quarter of the chunk duration** — the load
  generator itself may be the bottleneck. The tool prints a warning; treat the
  latencies as an upper bound and run fewer clients per process.

---

## Pool size

`ASR_MAX_CONCURRENT_STREAMS` is fixed when the service starts, so sweeping it
means restarting:

```bash
python -m tests.load.run_load_sweep --audio sample.wav \
    --levels 1,2,4,8 --pool-sizes 1,2,4,8 --spawn-server
```

`--spawn-server` launches the service from the current `ASR_*` environment, so
it is the same service with one knob moved.

Be clear about what that knob is: **an admission cap, not a worker pool.** One
ONNX session serves everyone and `Run()` is re-entrant, so raising the cap does
not add parallelism — it admits more callers onto the same device. Whether that
improves aggregate throughput or merely spreads the same throughput across more
unhappy users is exactly what the sweep shows.

---

## Decoder backends

The live backend is read from `/health` and recorded in every row, so results
from `greedy`, `pure_python` beam, `pyctcdecode` and `flashlight` (+ KenLM) are
distinguishable after the fact. To require one:

```bash
python -m tests.load.load_test --audio sample.wav --require-backend flashlight
```

If the service is not running it, the tool prints

```
SKIPPED: optional backend unavailable: requested 'flashlight', service is running 'greedy'
```

and exits **0**. A missing optional decoder is not a load-test failure — there
is no `flashlight-text` wheel for Windows, so that would be red on every
machine this project has run on.

---

## Hardware monitoring

Optional and pluggable. GPU readings come from NVML if `pynvml` imports,
otherwise `nvidia-smi`, otherwise nothing:

```
GPU metrics unavailable: NVML unavailable: ...
```

The run continues either way. CPU and RAM need `psutil`; without it the host
section degrades the same way. Readings are **device-wide**, so on a shared
machine they include whatever else is running.

---

## Results

```
results/
    load_test_<timestamp>.json    metadata + per-level summaries + every stream
    load_test_<timestamp>.csv     one row per stream
```

The CSV schema is deliberately model-agnostic: `model_family`,
`decoder_backend`, `used_lm`, and generic `partials` / `segments` / `finals`
counts. Nothing in it names CTC, so RNNT and RNNT+RNNLM runs land in the same
columns and can be compared directly. Columns are appended, never renamed.

Charts, if `matplotlib` is installed:

```bash
python -m tests.load.run_load_sweep --audio sample.wav --levels 1,2,4,8 --plots
python -m tests.load.plots results/load_test_20260809_143210.json   # after the fact
```

---

## Recommended procedure

1. **Start the service the way you deploy it** — same model, same providers,
   same decoder. Confirm with `curl localhost:8000/health` that CUDA actually
   loaded; `get_available_providers()` reports what the build was *compiled*
   with, not what can load, and this project has already shipped a silent
   CPU fallback once.
2. **Use representative audio.** Turn length drives everything: a 3 s clip
   produces one segment and never exercises the segmenter. Several real
   recordings via `--audio a.wav b.wav c.wav` beats one clip repeated.
3. **Warm up.** The first stream after startup pays for ORT arena allocation
   and CUDA autotuning. The service warms itself at startup; a discarded
   `--concurrency 1` run costs nothing and removes the doubt.
4. **Sweep in real-time mode first.** That is the capacity number.
5. **Then throughput mode**, if offline batch work matters to you.
6. **Repeat** the levels near the knee with `--repeat 3`. A non-monotonic curve
   means the run was noisy, and the sweep will say so.
7. **Record the environment.** It is in the JSON automatically: model
   fingerprint, active providers, decoder, GPU, driver, CUDA, Python,
   onnxruntime, chunk size, sample rate, fixture, concurrency, pool size.

### What must be recorded with any number you quote

Model path and fingerprint · active execution providers (not the requested
ones) · decoder backend and whether an LM is attached · GPU model, driver and
CUDA version · CPU and RAM · chunk duration and sample rate · the audio fixture
and its duration · concurrency and pool size · **and the mode**.

All of it is in the JSON. Quoting a latency without the mode is the single
easiest way to publish a wrong number.

---

## Limitations

* **One process drives all the clients.** At high concurrency the client's own
  event loop can become the constraint; `loop_lag_p95_ms` is measured and
  warned about, but the tool does not yet shard across processes. If the
  warning fires, run several instances against the same service.
* **Latency includes the network.** Over loopback that is negligible. Over a
  real network it is not, and the tool cannot separate the two.
* **`turn_final_latency` is an overstatement**, as described above.
* **Resource readings are device-wide**, not per-process.
* **Nothing measures accuracy.** WER on real audio is
  `tools/real_audio_wer.py`; this package measures behaviour under load only.
* **A short run has few samples.** p99 over 8 streams is the max with extra
  steps. Use `--repeat` for percentiles you intend to quote.

---

## Why some requested tests are not here

* **A "maximum real-time capacity" benchmark from throughput mode.** Throughput
  mode is a batch measurement. Labelling it as real-time capacity would
  overstate it by the GPU-clocking factor described above, so the two are kept
  separate and named accordingly.
* **Per-request model-pool worker tests.** There is no worker pool to test.
  `ASR_MAX_CONCURRENT_STREAMS` is an admission cap over a single shared
  session; `--pool-sizes` measures what it actually does rather than what its
  name suggests.
* **Assertions on absolute latency in the pytest suite.** A CI machine's
  latency is a property of that machine. The tests assert *behaviour* —
  isolation, correctness under concurrency, failure handling — and the
  benchmark reports numbers without judging them.

* **A separate "server timeout" case.** The service has no idle timeout of its
  own; it holds a WebSocket until the client ends it or the connection breaks.
  A server-side timeout would reach the client as a close, which is exactly the
  `drop` case already covered. Adding a third name for the same observable
  behaviour would imply a distinction the protocol does not make.

### Covered failure modes

| scenario | test |
|---|---|
| server not listening | `test_a_server_that_is_not_listening_is_a_result_not_a_crash` |
| connection refused at capacity | `test_a_refused_connection_is_recorded_as_rejected_not_failed` |
| pool exhaustion | `test_pool_exhaustion_still_yields_a_complete_measurement` |
| dropped mid-stream | `test_a_connection_dropped_mid_stream_is_labelled` |
| malformed event | `test_a_malformed_event_is_a_protocol_error_not_an_unhandled_exception` |
| binary frame where JSON was promised | `test_a_binary_frame_from_the_server_is_a_protocol_error` |
| server error event | `test_a_server_error_event_is_attributed_to_the_server` |
| server stops responding | `test_a_server_that_stops_responding_times_out_rather_than_hanging` |
| final never arrives | `test_a_missing_final_is_a_timeout_with_the_partial_data_kept` |
| stuck but chatty server | `test_the_total_budget_bounds_a_stream_whose_events_never_stop` |
| **one client failing, others continue** | `test_one_failing_client_does_not_take_down_the_others` |
| a client raising despite its guards | `test_a_client_raising_despite_its_guards_is_still_a_row` |
| concurrency above capacity | `test_arbitrary_concurrency_values_are_supported`, plus the two above |
| cross-stream contamination (real model) | `test_concurrent_streams_do_not_contaminate_each_others_transcripts` |
| capacity leak after a burst (real model) | `test_capacity_is_returned_after_a_burst` |
