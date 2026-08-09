# The conversational loop: audio capture, barge-in and turn boundaries

> **Status: design note. None of this is implemented.**
> One item below (§4, the timeline discontinuity) describes a latent bug in the
> *current* server that this design would trigger. It is not yet fixed.

Companion to [`PROGRESSIVE_CONSUMPTION.md`](PROGRESSIVE_CONSUMPTION.md), which
covers what the downstream layers do with ASR events. This one covers the audio
path and the client contract.

## 0. The loop

```
patient speaks → ASR → NLU/retrieval → question → TTS → patient hears →
patient speaks → …
```

with NMT somewhere in the middle (see §7) and report generation running off the
accumulated transcript.

The latency the patient feels is the **whole chain**, not any single stage. That
is what justifies the speculative machinery in the companion document: the fetch
is a small term in a budget that plausibly runs 1.5–3 s per turn.

---

## 1. One ring buffer, two problems

The client should capture into a bounded circular buffer from the moment the
microphone opens, and keep a separate flag deciding whether a pump drains it to
the socket. **Capture always; gate sending.**

That single mechanism covers two situations that look unrelated:

**Cold start.** The patient begins speaking before the WebSocket is up. Without
a buffer the opening words are gone. With one, the pump flushes the backlog on
connect and then switches to live streaming.

**Barge-in.** The patient interrupts while TTS is playing. Detection takes
200–300 ms, by which point the first word has already been spoken. The ring
holds it, so on barge-in the client flushes the last ~1 s and continues live —
nothing is lost.

### Sizing and overflow

Size for the *cold start* case, which is the longer of the two — barge-in needs
about a second of lookback, a bad network could take several. Around 10 s is
reasonable.

When it overflows it **must say so**. Silently dropping the oldest audio yields
a transcript with an invisible hole; the app has to be able to mark that turn as
incomplete rather than trusting it. Emit a "dropped N seconds" signal.

---

## 2. Echo cancellation is not optional

If the ring captures during playback and the microphone hears the speaker, the
flushed buffer contains the system's own question. The ASR transcribes it
faithfully and the NLU treats it as patient speech.

**The ring must sit after AEC, not before.** Platform implementations are the
practical answer:

| Platform | Mechanism |
|---|---|
| iOS | `kAudioUnitSubType_VoiceProcessingIO` |
| Android | `AcousticEchoCanceler` |
| Browser | `getUserMedia({audio: {echoCancellation: true}})` |

Headsets sidestep the problem. Speakerphone in a clinic room does not.

---

## 3. Detect barge-in locally

Cancel TTS the moment the patient speaks. A server round trip adds 100–200 ms
to something the user experiences as the system talking over them, so the VAD
that triggers the cancel should run client-side. The server's own VAD stays
responsible for segmentation, not for interruption.

---

## 4. Timeline discontinuity — a live bug against the current server

The pipeline derives `audio_time` from **cumulative samples received**, not from
wall clock. If the client stops sending during TTS and resumes afterwards, the
server sees no gap at all.

Consequences, given the segmented pipeline as it stands:

* a 4-second TTS pause registers as **zero** silence;
* `trailing_silence` never accumulates across it;
* the segment open before the TTS **never closes**, and the patient's pre-TTS
  and post-TTS speech merge into one segment.

Two fixes, in order of preference:

1. **Explicit turn boundaries.** The client sends `{"type":"end"}` when the
   patient's turn ends and `{"type":"reset"}` before the next. The server
   finalises cleanly and starts fresh. In a turn-taking agent the boundaries are
   *known*; inferring them from silence when ground truth is available is
   strictly worse. `turn_silence` then degrades to a fallback for clients that
   do not signal.
2. **Client sends silence during TTS.** Keeps the timeline honest with no
   protocol change, at the cost of bandwidth.

### Related: metrics pollution

`AudioChunk.capture_time` feeds the `update_latency` metric. A flushed backlog
arrives faster than real time, so those samples are meaningless for the burst.
Backlog chunks should be flagged and excluded from latency statistics, or the
first turn of every session will report an inflated latency that never happened.

---

## 5. Recommended client state machine

```
        ┌───────────── capture always (post-AEC) → ring ──────────────┐
        │                                                             │
   [connecting]  ──connect──▶  [streaming]  ──turn end──▶  [awaiting reply]
        │   flush backlog          │  send live              │  pump gated
        │                          │                         │  local VAD armed
        └──────────────────────────┴──◀── barge-in: cancel TTS,
                                          flush ~1s, send {"reset"}
```

The pump gate is the only thing that changes between states. Capture never
stops.

---

## 6. What this asks of the server

Nothing that exists today needs to change for the ring buffer itself — the
WebSocket endpoint already accepts arbitrarily-sized frames and re-blocks them,
so a flushed backlog is handled correctly.

Outstanding items:

* fix or document the timeline discontinuity in §4;
* flag backlog chunks so they do not pollute latency metrics;
* the `seq` and `segment_id` fields proposed in the companion document.

---

## 7. Where NMT belongs

The source is an Indic language; the NLU works in English. MT therefore has
**two consumers with completely different latency requirements**, and they
should not share a path:

| Consumer | Budget | Notes |
|---|---|---|
| Next-question fetching | latency-critical, inside the loop | the problem |
| Report generation / conversation tracking | batch, after the session | not the problem |

### Do not translate partials

Partial *transcription* degrades by truncation, which is safe. Partial
*translation* degrades by **hallucination**, which is not.

Most Indic languages are SOV — the verb lands last. A partial such as

> "मुझे तीन दिन से सीने में…"   ("for three days, in my chest…")

has no verb yet. MT will not stop short; it will invent one, and the invented
verb can flip clinical meaning (`is` vs `is not`, `started` vs `stopped`).

So: keep MT out of the speculative path. Run concept extraction directly on the
source-language partial — entity spotting for symptom terms is far more robust
to incomplete input than full translation, because it does not require syntax.
Reserve MT for `segment` and `final`, where the span is complete.

### Better: consider removing MT from the loop entirely

A multilingual sentence embedding model (LaBSE, multilingual-E5) can match an
Indic-language utterance directly against an English question bank **without
translating it**. If that works for this domain, MT leaves the critical path
altogether and runs only asynchronously for reports — which also disposes of the
partial-translation hazard above, because no incomplete utterance is ever
translated.

**Worth prototyping before committing to MT-in-the-loop:** embed ~50 real
patient utterances and check whether the correct question ranks top-3 against
the bank. About a day of work, and it decides the shape of the whole loop.

---

## 8. And if the question bank is bounded

**Pre-render TTS for every question offline.** TTS then leaves the runtime
budget entirely and the loop becomes ASR → NLU → select → play cached audio.

For a fixed bank this is usually a larger saving than anything speculative
execution buys, at a fraction of the complexity. It is complementary rather than
alternative: pre-rendering removes TTS latency, speculation removes NLU and
retrieval latency.
