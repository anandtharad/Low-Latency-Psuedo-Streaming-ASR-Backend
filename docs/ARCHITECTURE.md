# Architecture and design rationale

## 1. What this is, and what it is not

The IndicConformer export is an **offline, full-context Conformer-CTC** model.
It has no encoder cache, no streaming attention and no stateful inference.
`ONNXASREngine.graph_report` verifies this at load time rather than assuming
it, by scanning every graph input and output for cache/state tensors:

```
ONNX graph report
------------------------------------------------------------
providers: CPUExecutionProvider
inputs:
  audio_signal: tensor(float)[batch x 80 x time]
  length: tensor(int64)[batch]
outputs:
  logprobs: tensor(float)[batch x time_out x 129]
state: NONE FOUND -> model is stateless/full-context.
       Rolling-window inference is the correct strategy.
```

If a future export gains cache inputs, that line changes and true incremental
inference becomes worth investigating. Until then:

> This is **sliding-window / rolling-buffer pseudo-streaming inference**.
> Calling it "streaming Conformer inference" would misdescribe both the cost
> and the failure modes.

The cost is explicit in the config: `window_redundancy = buffer / chunk` is
**25** at the reference operating point. Every audio sample is processed 25
times; 96 % of each model call is recomputation.

---

## 2. The problem the tracker exists to solve

### 2.1 The reference implementation loses text

The reference notebook's own output is the clearest statement of the problem:

```
chunk 31: india versus pakistan world cup final
chunk 32: ya versus pakistan world cup final
chunk 33: versus pakistan world cup final
chunk 35: pakistan world cup final
chunk 38: sustan world cup final
```

Each window's transcript **replaces** the previous one. The buffer holds 4
seconds; once the utterance is longer than that, the opening words scroll out
of view and disappear from the output entirely. Nothing accumulates a
transcript across windows.

This creates a hard deadline that shapes the whole stabilisation design:

> **A word must be committed before it scrolls out of the rolling buffer.**
> After that, no future window can see it, and it is unrecoverable.

`StabilityConfig.validate_against` enforces `stability_window < context_duration`
at construction time, and warns when the margin drops below 2x.

### 2.2 Successive hypotheses are not prefixes of each other

Section 13 of the brief warns against naive string comparison. The failure is
sharper than "words sometimes change": once the window slides, consecutive
hypotheses share a *suffix/infix*, not a prefix. A longest-common-prefix
comparison of `india versus pakistan` against `versus pakistan world` finds
**zero** agreement and would never commit anything.

`tests/test_aligner.py::test_prefix_aligner_fails_once_the_window_slides`
pins this behaviour for both aligners.

### 2.3 A full-context model revises its past

Window N covers `[t-3.84, t]`; window N+1 covers `[t-3.68, t+0.16]`. Because
attention is bidirectional, the model's reading of audio at 2 s can change when
new right-context arrives. So:

> A token is **not** final merely because it lies outside the newest 160 ms.

Temporal maturity — not position in the window — is the criterion.

---

## 3. The stabilisation strategy

Every token carries an **absolute stream timestamp**, not just a window-relative
frame index. Frame 50 of window N and frame 50 of window N+1 describe audio
160 ms apart; absolute times are comparable across windows.

That single decision is what makes cross-window merging tractable, and it is
the brief's prescribed escalation order (section 14) taken seriously:

```
known window offset  →  token/frame timing  →  simple sequence alignment
    →  edit-distance alignment  →  DTW only if measurement demands it
```

The default `TimeAwareAligner` sits at step 4 with step 1–2 folded in: it is a
Levenshtein alignment whose substitution cost is gated by timestamp agreement,
so temporally impossible pairs cannot align at all. This is what disambiguates
a repeated word ("pain … pain") that pure text alignment cannot.

DTW is implemented (`dtw_aligner.py`) but **off by default**. It is a poor fit
for word sequences — it forces every element to match something, whereas real
revision inserts and deletes — but a natural fit for the one thing it is
offered for: `align_posteriors()`, comparing the frame-level CTC posterior
trajectories of two overlapping windows.

### The assumption this rests on — verify it on the real model

Time-based matching requires that **the model's CTC spikes fire near the
acoustic evidence**, so that the same spoken word decoded from two overlapping
windows lands at the same absolute time in both.

This is not guaranteed by anything in the training objective. CTC loss is
alignment-free: every monotonic placement of the target tokens scores
identically, so nothing in it requires a spike to coincide with the sound that
caused it. Well-trained Conformer-CTC models do align well in practice — the
easiest solution is to fire when the evidence appears — but it is an empirical
property, not a structural one.

Building the synthetic fixture demonstrated the failure mode concretely. An
undertrained model learned to emit every token in the first few frames of
whatever window it was given. A word's apparent timestamp then advanced by
exactly one chunk per window and never settled, so the time-aware aligner
refused every match, streaks reset every update, and **nothing was ever
committed**.

`tools/check_alignment_fidelity.py` measures this directly:

```bash
python tools/check_alignment_fidelity.py --audio sample.wav --model model.onnx
```

| Measured drift | Meaning | Action |
|---|---|---|
| ≈ 0 | timestamps track the audio | `aligner="time"` (default) |
| ≈ `chunk_duration` | timestamps track the window | `aligner="levenshtein"`; treat timestamps as ordering only |

Run it against the IndicConformer before trusting the defaults. Note that the
maturity criterion itself is drift-immune — `window_end − word.end_time` is
computed inside a single window — so only streak propagation is affected.

### Commitment rules

A word commits only when **both** hold:

1. **Temporal maturity** — `audio_time - word.end_time ≥ stability_window`.
2. **Repeated agreement** — the aligner has matched it in
   `min_stable_updates` consecutive windows.

plus two structural rules:

3. **Prefix-only.** Commitment never skips an unstable word to reach a stable
   later one. A hole in the transcript cannot be filled later without
   retracting text the caller already has.
4. **No truncated words.** A word whose first token lacks the `▁` boundary
   marker began before the window did, so it is a fragment. It is never
   committed from that window.

Agreement counts are carried **through the alignment** (`_propagate_streaks`)
rather than keyed on text or rounded time, which keeps the bookkeeping correct
when words shift position between windows.

---

## 4. Latency: two different quantities

| | Determined by | Improves with |
|---|---|---|
| **Algorithmic latency** | `chunk_duration`, `stability_window` | window geometry only |
| **Compute latency** | model size, window size, provider | faster GPU, bigger step |

A configuration can have an excellent RTF and still feel sluggish, because
`stability_window` is withholding text. `MetricsCollector` reports both:
`stable_token_latency` is measured in *audio* time and is invariant to hardware.

Minimum time-to-commit is approximately:

```
chunk_duration + stability_window + (min_stable_updates - 1) × chunk_duration
```

≈ 0.92 s at defaults (0.16 + 0.60 + 0.16). Lowering `stability_window` is the
most direct lever; the cost is more retracted-looking output, since the model
revises text it has not yet had time to settle on.

---

## 5. Why finalisation re-decodes instead of merging

`finalize()` re-runs the model over the whole retained utterance and decodes
with beam+KenLM. It does **not** stitch the streaming fragments together.

Each streaming hypothesis was produced from a truncated ≤4 s view with no
language model. Concatenating them preserves every error they contain. The
final decode sees the entire utterance *and* has an LM, so it is strictly
better-informed. The provisional text is still returned as
`ASREvent.provisional_text` so the drift between the two can be measured — that
drift is the real quality signal for the streaming layer.

### Long utterances — where "final is authoritative" stops being true

The checkpoint's own training config caps utterances at 11 s
(`max_duration: 11`). A single forward pass over a 60 s recording is therefore
out of distribution. Past `final_segment_duration`, `_final_inference` cuts the
audio into overlapping segments and stitches the logits, discarding half the
overlap at each seam so no frame is taken from a segment edge.

Measured on **real speech** — Common Voice clips concatenated into turns with
0.7 s pauses, real checkpoint, greedy decoding, via
[`tools/real_audio_wer.py`](../tools/real_audio_wer.py):

| turn length | single pass | pause-segmented (10 s cap) |
|---|---|---|
| ~6 s (unmodified clips) | 0.0528 | 0.0518 |
| ~17 s | **0.0441** | 0.0662 |
| ~34 s | 0.0610 | **0.0569** |
| ~66 s | 0.0612 | **0.0485** |
| ~92 s | 0.0630 | **0.0519** |
| ~124 s | 0.0758 | **0.0505** |

Single-pass WER climbs with turn length; segmented WER does not, because every
decode it performs is the same size. Below ~20 s segmenting *costs* a little —
it discards cross-sentence context the encoder was using and there is no length
penalty yet to pay for it.

**Keep `max_segment_duration` at or below the checkpoint's training
`max_duration`** — the default is 10 s for this reason, not 20 s. Going the
other way is worse than useless: at a 6 s cap the segmenter runs out of pauses
and force-cuts mid-phrase, scoring 0.0642 against 0.0630 for the single pass it
was meant to improve on.

The harder limit is not accuracy at all. Past ~150 s on a 4 GB GPU a single pass
stops working: CUDA OOM at 180 s (self-attention is quadratic in length), a hard
`Add_2` broadcast failure at 210 s from the model's own relative-position buffer,
and at 245 s a native abort inside `onnxruntime_providers_cuda.dll` that kills
the process and cannot be caught in Python. The segmented path is unaffected by
construction. See [`PROJECT_REPORT.md` §2.3](PROJECT_REPORT.md).

Both transcripts are always reported — `ASREvent.text` and
`ASREvent.provisional_text` — and for long inputs the choice between them
should be made from measurement on real audio, not by rule.

### One thing that was tried and did not work

Snapping segment boundaries to a local energy minimum, so seams fall in pauses
rather than mid-word. Plausible, and standard in long-form ASR, but on the
synthetic fixture it measured neutral to worse. It is implemented and defaulted
**off** (`final_segment_snap = 0.0`) rather than shipped on an untested
rationale. The numbers are not quoted here: that fixture has uniform 0.12 s
inter-word gaps and no real pauses, so snapping had nothing to find and its
error rates cannot support an accuracy claim either way. **Re-measure on real
speech before enabling or dismissing it.**

---

## 6. Preprocessing fidelity

The mel frontend must match training/export exactly; a mismatch degrades
accuracy silently, with no error anywhere. `filterbank.py` is a faithful port,
with three corrections:

1. `import torch.nn as nn` — the reference cell subclasses `nn.Module` without
   importing it and only works because an earlier cell leaked the name.
2. `Preprocessor` forces `eval()`.
3. Consequently **dithering is disabled at inference**. This matters more than
   it appears: `_apply_dithering` is gated on `self.training`, so a module left
   in train mode adds fresh Gaussian noise to every window. The same audio
   would then yield different features on each of its 25 overlapping passes,
   injecting hypothesis instability that has nothing to do with the acoustics —
   and the stabiliser would be measuring its own noise.

### Warm-up padding

The reference feeds the whole rolling buffer every step and declares all of it
valid — including, for the first ~24 windows, up to 3.84 s of zero padding.
Two things go wrong with that:

* The model is handed seconds of digital silence it never saw in training
  (`min_duration: 0.5`). Observed result on the fixture: **hallucinated tokens
  in the padding region**, at negative stream timestamps.
* Per-feature normalisation statistics are computed over the padding, so early
  windows are normalised differently from steady-state ones.

`pad_warmup_window` defaults to `False`, feeding only real audio until the
buffer fills. Set it `True` for bit-exact reference parity.

---

## 7. Long-lived streams: what is bounded and what is not

Measured on a connection that never endpoints (~150 wpm, 160 ms chunks):

| | Growth | Status |
|---|---|---|
| Metrics sample lists | was ~4.1 MB/hour, forever | **fixed** — 0.13 MB, flat |
| Rolling audio buffer | fixed allocation | bounded by design |
| Retained audio (`_history`) | capped at `max_history` | bounded by config |
| Committed transcript | ~4.4 MB/hour (106 MB/day) | **by design — see below** |

`MetricsCollector` appended one float per chunk per metric with no bound, and
`snapshot()` sorted those lists, so emitting metrics also got slower the longer
a session ran. It now keeps exact running `count`/`sum`/`mean`/`max` plus a
sliding window of the last 4096 samples for percentiles. RTF stays exact
because it uses the running total, not the window; only percentiles are
estimated, and from recent history — which is the more useful reading anyway.

The committed transcript is **not** treated as a leak and is deliberately left
alone. It is the product, and truncating it to save a few MB/hour would
silently discard output the caller asked for. The correct mitigation is
endpointing: `finalize()` closes the utterance and `reset()` clears it. Note
that the WebSocket protocol already streams `newly_committed` to the client on
every partial, so a client that endpoints regularly never accumulates anything
server-side. A client that connects and never sends `{"type":"end"}` will grow
~106 MB/day; cap it with `ASR_MAX_UPLOAD_SEC`-style session limits at the edge,
or enable `ASR_ENDPOINT=energy` so utterances close themselves.

`reset()` is verified by behavioural equivalence rather than by inspection: a
reset pipeline must reproduce a fresh pipeline's entire committed/partial trail
byte for byte (`tests/test_pipeline_reuse.py`). Comparing final transcripts
alone would miss state that changes the stabilisation path without changing the
end result.

## 8. Open questions to settle with measurement

1. **Is greedy good enough as an intermediate semantic signal?**
   `benchmark.py --compare-decoders` decodes identical logits both ways and
   reports WER/CER between them. Do not assume; measure.
2. **Is 4 s / 160 ms the right operating point?** The sweep exists to answer
   this. Expect `2 s / 320 ms` to cut compute ~4x; the question is what it
   costs in WER.
3. **Does the warm-up normalisation mismatch matter?** Compare declaring
   `n_samples=buffer_samples` against `valid_samples`.
4. **Does DTW ever beat the time-aware aligner?** Only worth revisiting if
   logged instability shows alignment failures the edit-distance path cannot
   handle.
