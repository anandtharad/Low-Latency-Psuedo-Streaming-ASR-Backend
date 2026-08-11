# Look into later

Analysed, not implemented. Everything here was worked through against the real
code and real sources; none of it has been written or measured on this system.
Deferred deliberately — the reasoning for deferring is recorded alongside each
item, because that reasoning is what will be wrong first if the product changes.

Section 3 is **not** deferred. It is a live defect found while analysing the
rest.

---

## 1. Replace the energy pause detector with CTC blank runs

### 1.1 What runs today

`SpeechDetector` in `streaming_asr/segmented.py` classifies each chunk by RMS
against `energy_threshold` (0.005), with hysteresis at half that value. A
segment closes when `_trailing_silence` reaches `segment_silence` (0.5 s).

This is a loudness measurement. It asks *"is the room quiet?"* as a proxy for
*"did the person stop talking?"*

### 1.2 Why the proxy fails

Three distinct failures, all from the same root:

| failure | what happens |
|---|---|
| **Noise floor** | Room hum sits above the threshold. Nothing ever registers as silence, no segment ever closes, the buffer grows to `max_segment_duration` and force-cuts mid-word. Already documented in `README.md` and `PROJECT_REPORT.md` §7. |
| **Transients** | A cough, a door, a dropped clipboard, paper rustling — all loud, all counted as speech. A transient landing inside a real pause resets `_trailing_silence`, so the segment never closes. |
| **Quiet speakers** | Someone trailing off, or a patient who is unwell, elderly or simply soft-spoken, falls below the line. The detector reads speech as silence and cuts mid-sentence — or drops it entirely (§3). |

There is no value of `energy_threshold` that is correct everywhere, because it
is an absolute measurement of an analog quantity with no reference. It depends
on the room, the microphone, the gain/AGC, the distance, and the speaker's
volume — every one of which changes per deployment.

### 1.3 The proposal

Ask the model instead of the microphone. A CTC model emits one symbol per frame
and emits **blank** where it does not hear speech. A pause is a long run of
blanks. Count the run; threshold it.

Reference implementation is NVIDIA's, in the NeMo Speech repo:

- `nemo/collections/asr/inference/streaming/endpointing/greedy/greedy_endpointing.py`
  — `GreedyEndpointing.detect_eou_given_emissions()`, ~45 lines, pure Python,
  no torch, no NeMo model objects.
- `nemo/collections/asr/inference/streaming/endpointing/greedy/greedy_ctc_endpointing.py`
  — supplies the two predicates.

Shipped defaults there: `stop_history_eou` 800 ms, `residue_tokens_at_end` 2.
It is the same mechanism as Riva's `stop_history`.

The algorithm:

1. Take the per-frame argmax — the raw CTC path, blanks included. **Not** the
   collapsed token list.
2. Walk backwards from the end, counting consecutive blanks.
3. If the run exceeds the limit, that is an endpoint.
4. Guard: the nearest non-blank token to the *right* of the run must start a
   word (`▁`). Otherwise the gap is inside a word and cutting would split it.
5. Cut at `silence_start + limit // 2` — the middle of the gap. Cutting at the
   start clips the previous word's tail; at the end, the next word's opening.

### 1.4 It costs nothing

`GreedyCTCDecoder.decode_text()` already computes `window.argmax(axis=-1)` at
`streaming_asr/decoding/greedy_ctc.py:195` and passes it straight to
`ctc_collapse`, discarding the frame path. Every partial already computes
exactly the array this needs. Exposing it is a one-line change.

### 1.5 What a direct copy would get wrong

| | NeMo | here |
|---|---|---|
| blank index | `len(vocabulary)` — blank excluded from the list | `len(vocabulary) - 1` — `"__"` is appended |
| frame duration | 80 ms (FastConformer, 8× subsampling) | 40 ms (Conformer, 4×) |
| 800 ms in frames | 10 | **20** |
| search scope | a fixed 8 s buffer, with a `pivot_point` from sliding-window bookkeeping | the open segment; `search_start_point` is 0 |

Do not hardcode the frame duration. Both engines already expose
`ctc_frame_duration(hop_duration)` read from the graph, so deriving it keeps the
count correct if the checkpoint is ever swapped for an 8× model.

The word-start predicate works directly — the vocabulary already uses the
SentencePiece `▁` marker, and `ARCHITECTURE.md` §3 already relies on that same
marker for the windowed pipeline's truncated-word rule.

### 1.6 The thing that will bite

**A blank does not mean silence.** CTC is peaky: the model emits a token on one
or two frames and blanks everywhere else, *including mid-word and mid-phrase
during fluent speech*. The limit has to sit above the natural inter-token gap.

Worse, and specific to this design: the segmented pipeline re-decodes the open
segment from its start, and the segment **ends at "now"** with no audio after
it. A full-context model at the right edge of its input has nothing to look
ahead to and tends to emit blanks there regardless. So every decode shows
trailing blanks as a boundary artefact.

NeMo does not hit this — its buffer carries 1.6 s of right padding and it
discards the tail. There is no padding here; the tail is the newest audio, which
is the entire point of the design.

**Consequence:** the measured trailing-blank count is *real silence + boundary
artefact*. Taking 20 frames off the shelf will likely cut early and often.
`residue_tokens_at_end` is a partial mitigation, not a fix.

### 1.7 On "isn't this just another threshold to tune?"

Fair objection, and the answer is that the two thresholds do not vary along the
same axis:

| | loudness line | blank-run limit |
|---|---|---|
| room noise floor | **yes** | no |
| microphone / gain / AGC | **yes** | no |
| distance from the mic | **yes** | no |
| how loudly the person speaks | **yes** | no |
| the checkpoint | no | **yes** |
| frame rate | no | **yes** — but readable from the graph |
| speaking rate | no | somewhat |

It becomes a measured constant, but one measured **once per checkpoint**, not
once per deployment. That slots into an existing ritual: `extract_vocabulary.py`
is already mandatory with a new checkpoint, and `check_alignment_fidelity.py`
already exists to re-verify another per-checkpoint property before trusting the
defaults.

### 1.8 An adaptive variant — considered and rejected for this workload

Rather than a fixed limit, compare the trailing gap against the gaps *this
speaker* left between their own words in the same decode. The blank runs
*between* emitted tokens are by construction intra-speech gaps, so they give a
non-circular baseline. Ratio instead of absolute; self-calibrating for speaking
rate; absorbs the checkpoint's peakiness because peakiness stretches baseline
and candidate equally.

**Rejected here** because a one-word answer has one or two tokens and therefore
no between-token gaps to build a baseline from. With this workload (§2.2) that
is the common case, not an edge case, so the fallback floor would do nearly all
the work. A fixed per-checkpoint limit is the right shape and is simpler.

Revisit if the workload shifts to longer, free-form speech.

### 1.9 How to introduce it

Wire as **agreement, not replacement**: close a segment when the energy meter
*or* the blank run says so, and log the disagreement rate. Quiet room, they
agree and nothing changes; noisy room, the meter goes silent and the model
carries it. That gets the noise-floor fix without betting segmentation
behaviour on an untested predicate, and the disagreement rate is the evidence
for whether to drop the meter entirely.

Once `refactor/single-segmentation-loop` lands, the segmentation loop is a
single implementation, so this is one edit in `segmented.py` and both runtimes
inherit it. Before that, it is two edits that must be kept in step.

### 1.10 Worth measuring once, if this is picked up

Not to choose a shipping number — to check the signal exists at all. On real
audio and the real checkpoint, for every partial, record the trailing blank-run
length alongside the energy meter's verdict. Two histograms: runs when the meter
says speech, runs when it says silence.

Clean separation → the gap between them is the limit, and the meter can go.
Heavy overlap → the boundary artefact is swamping the signal, and both stay.
One-off question, yes/no answer, ~30 lines in `tools/`, same spirit as
`check_alignment_fidelity.py`.

---

## 2. Semantic turn detection — deferred, probably indefinitely

### 2.1 What the industry actually does

Silence duration has been abandoned as the primary end-of-turn signal across
LiveKit, Daily/Pipecat, AssemblyAI and Deepgram. The reason it cannot be tuned:
*"I would like to order one large pizza…"* plus 500 ms could be finished or
could continue with "and garlic bread." **The information needed to tell those
apart is not in the audio envelope**, so no threshold, adaptive or otherwise,
recovers it.

The current stack is three layers doing three different jobs:

| layer | job | typical |
|---|---|---|
| Acoustic VAD | is there speech at all? mic gating, barge-in | Silero VAD |
| Silence timer | trigger and fallback | 160 ms trigger, 2400 ms hard fallback (AssemblyAI defaults) |
| Turn model | did this person finish their thought? | dedicated small model |

Note layer 2 is not a single threshold — a *short* silence triggers evaluation,
a model decides, and a long fallback fires if the model will not commit.

Benchmarks worth carrying (LiveKit Turn Detector v1.0):

| | |
|---|---|
| 300 ms latency budget | 9.9 % false-cutoff |
| 600 ms latency budget | 4.5 % false-cutoff |
| 5 % false-cutoff target | 543 ms mean latency |

**State of the art still cuts people off ~5 % of the time and needs ~half a
second to decide.** `segment_silence` at 0.5 s is where the industry sits.
That knob was never the problem — consistent with the "500 ms policy floor" in
`PROJECT_REPORT.md` §6.8.

### 2.2 Why it does not fit here

**The workload inverts the value.** Intake-style Q&A produces short, often
one-word answers. Short complete answers are the *easy* case for a silence
timer — "three days." plus half a second is unambiguous. The ambiguity these
models exist to resolve is largely absent. Meanwhile single-word answers are
the *hard* case for a semantic model: least context, least prosody, smallest
evidence.

**The cost of being wrong is asymmetric.** For a voice agent a false cutoff
means the bot talks over the user — catastrophic, and the entire reason for the
industry's investment. Here an early cut produces two `segment` events instead
of one. They append. **The joined transcript is identical.** Nothing responds at
a segment boundary; nothing talks back at all.

### 2.3 The candidate, if it is ever needed

`pipecat-ai/smart-turn-v3` fits the constraints unusually well: Whisper Tiny
encoder plus a linear classifier, ~8 M parameters, int8 ONNX at 8 MB, CPU
inference, analyses the raw waveform rather than the transcript (so it does not
depend on the decoder), open source, multilingual. It would run inside the
torch-free `streaming_asr_lite` runtime.

Input handling: 16 kHz mono, **up to** 8 s — shorter audio is zero-padded at the
*beginning* so the speech sits at the end of the vector. So short utterances
work mechanically. But the project documents that it *"works best when given
sufficient context, and is not designed to run on very short audio segments"* —
i.e. this workload would run it in its weakest regime.

`livekit/turn-detector` is the other open-weights option (Qwen2.5-0.5B
fine-tune, CPU-targeted) but is considerably heavier.

### 2.4 When this becomes real

At `final`, not `segment` — and only when something downstream acts on a
completed turn. That is the `CONVERSATIONAL_LOOP.md` design, which is
deliberately not built. If barge-in or a spoken response is ever added, revisit
this section first, because the cost asymmetry in §2.2 reverses completely.

---

## 3. Live defect: quiet short utterances are silently dropped

**Not deferred.** Found while analysing the above.

```python
has_speech = self._speech_samples >= int(
    self.settings.min_segment_speech * self.config.sample_rate
)
```

`_close_segment` only runs when `has_speech` is true, and `finalize()` guards on
the same condition. `min_segment_speech` is 0.2 s, and `_speech_samples`
accumulates only on chunks the **energy meter** classified as speech — so at
0.16 s chunks it takes two consecutive chunks above the loudness line.

A quiet, brief "yes" from a soft-spoken patient may never accumulate 0.2 s of
above-threshold audio. No segment is emitted, no error is raised, and **the word
is absent from the transcript.**

This is worse than a badly placed cut: it is silent data loss, on exactly the
utterances this workload is full of. It is also the strongest argument for §1 —
a blank-run detector does not care how loud "yes" was.

**Reproduce before fixing:** feed several quiet one-word answers through the
real checkpoint and confirm whether they survive.

---

## 4. Latent, noticed in passing

`_close_segment` computes retained audio for a forced cut as:

```python
retained = full[max(keep_from - tail, keep_from):]
```

`tail` is non-negative, so `max(keep_from - tail, keep_from)` is always
`keep_from` and the `- tail` is dead. The intent — from the comment "retain a
little audio so the next segment does not start clipped" — appears to have been
`max(keep_from - tail, 0)`, keeping a little audio from *before* the cut.

Identical in both copies of the loop, so the `refactor/single-segmentation-loop`
merge preserved it exactly rather than quietly correcting it. Left alone
deliberately: changing it alters segmentation behaviour and belongs in its own
change with its own measurement, not folded into a refactor.

---

## Sources

Industry practice, retrieved August 2026:

- [Solving end-of-turn detection: LiveKit Turn Detector v1.0](https://livekit.com/blog/solving-end-of-turn-detection)
- [Using a transformer to improve end of turn detection — LiveKit](https://blog.livekit.io/using-a-transformer-to-improve-end-of-turn-detection)
- [LiveKit turn detector plugin docs](https://docs.livekit.io/agents/build/turns/turn-detector/)
- [How intelligent turn detection (endpointing) solves the biggest challenge in voice agent development — AssemblyAI](https://www.assemblyai.com/blog/turn-detection-endpointing-voice-agent)
- [Smart Turn v2: Open source semantic VAD for voice AI — Daily](https://www.daily.co/blog/smart-turn-v2-faster-inference-and-13-new-languages-for-voice-ai/)
- [pipecat-ai/smart-turn](https://github.com/pipecat-ai/smart-turn) · [smart-turn-v3 model card](https://huggingface.co/pipecat-ai/smart-turn-v3)
- [Deepgram: Endpointing](https://developers.deepgram.com/docs/endpointing) · [Utterance End](https://developers.deepgram.com/docs/utterance-end)

Reference implementation: `NVIDIA-NeMo/Speech`, paths in §1.3.
