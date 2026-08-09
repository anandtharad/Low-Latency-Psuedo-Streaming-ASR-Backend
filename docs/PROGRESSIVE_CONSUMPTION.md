# Progressive consumption of ASR output

> **Status: design note. None of this is implemented.**
> The ASR service emits the events this design consumes; everything below the
> transport layer is future work. Read this as a plan, not a description of
> current behaviour.

Companion to [`CONVERSATIONAL_LOOP.md`](CONVERSATIONAL_LOOP.md), which covers
the audio path — ring buffer, echo cancellation, barge-in, turn boundaries — and
where NMT belongs relative to the latency-critical path.

## The idea

Downstream work does not have to wait for the speaker to finish. A turn like

> "i have been experiencing chest pain for three days"

carries usable information long before it ends. `chest pain` alone is enough to
start fetching candidate follow-up questions; `three days` arrives later and
narrows them. If the fetch runs while the patient is still speaking, its
latency disappears into time the app was going to spend waiting anyway.

This is latency hiding by speculative execution. The ASR layer already emits
what it needs — the design work is in deciding *what to key the work on* and
*when it is safe to act*.

---

## 1. Track concepts, not text

Do not diff transcripts. Extract clinical concepts from each event and diff
those:

```
"i have been experiencing chest pain"        → {symptom: chest_pain}
"i have been experiencing chest pain for th" → {symptom: chest_pain}      ← unchanged, no work
"...chest pain for three days"               → {symptom: chest_pain, duration: 3d}
```

Partials arrive roughly six times a second; the concept set changes perhaps
twice a sentence. Diffing concepts collapses that churn for free, so work is
triggered by *semantic* deltas rather than text deltas.

It also makes revision natural to express: `chest pain` → `chest sprain` is a
concept swap, not a string edit, and the machinery for "cancel what depended on
the old concept" follows directly.

Concept **identity** is where most of the real engineering lives. `chest pain`,
`pain in my chest` and `chest pains` must normalise to one concept, or the
system fires duplicate fetches and sees churn that isn't there.

---

## 2. Three tiers of trust

| Event | Trust | May drive |
|---|---|---|
| `partial` | speculative | **prefetch only** — warm caches, no visible effect, no writes |
| `segment` | confirmed | display, filtering, narrowing the question set |
| `final` | authoritative | verification, persistence, anything with consequences |

**The rule, and it is not negotiable in a clinical context: speculative work is
read-only and invisible.**

A partial may pull candidate questions into a cache. It must never render them
and must never write anything. The user sees results only once a `segment`
confirms the concept — but the fetch has already completed by then, so it
*feels* instant. The speed is real; the risk of showing something the ASR later
retracts is not taken at all.

---

## 3. The speculation loop

For each newly-seen speculative concept:

1. **Debounce.** Require it to persist across 2–3 partials before acting.
   Cheap, and removes most churn.
2. **Fire async, tagged.** Every fetch carries a generation counter and the
   concept set that triggered it.
3. **Cancel or discard on revision.** If the concept vanishes from later
   partials, cancel in-flight work; if a stale response lands anyway, drop it by
   tag. Async completions arrive out of order — ordering must come from the tag,
   never from arrival time.
4. **Promote on `segment`.** The concept becomes confirmed and its cached
   result becomes displayable.
5. **Reconcile on `final`.** Re-extract from the authoritative text and diff
   against what was acted on. Any mismatch is a retraction path that has to be
   designed deliberately rather than discovered in production.

### Narrowing without re-querying

Fetch a **superset** on the first concept, then filter locally as constraints
arrive. `chest_pain` pulls the candidate set over the network; `duration: 3d`
filters it in memory with no round trip. Re-query only when a new concept
cannot be satisfied from what is already held.

---

## 4. Hazards

**Negation inverts meaning.** `"chest pain"` can become `"no chest pain"`. In
English negation usually precedes the concept, so it is often visible in time —
but not always (`"chest pain? no, none"`). This alone justifies the read-only
rule for speculation, and re-extraction at `final`.

**The trailing word is the least stable**, having the least right context.
Observed directly in this project's logs: `climent` → `climb in college` →
`climb in coloial` → `climb in`. Mitigation: ignore the last word or two of a
partial when extracting, or equivalently extract only from text older than
~0.5 s. Same idea as the ASR stability window, applied at concept level.

**Segment boundaries can split a concept.** `"chest pain"` ⟨pause⟩ `"for three
days"` arrives as two segments. The concept tracker must accumulate across
segments *within a turn*, not reset per segment.

**Never translate a partial.** The source is an Indic language and most are SOV,
so a partial often has no verb yet — and MT invents one rather than stopping
short, which can flip clinical meaning. Extract concepts in the source language
for the speculative path. See
[`CONVERSATIONAL_LOOP.md` §7](CONVERSATIONAL_LOOP.md#7-where-nmt-belongs), which
also covers removing MT from the critical path altogether.

**Fetches complete out of order.** Covered by the generation tag above, but it
is the failure that looks like a heisenbug if the tag is skipped.

---

## 5. Protocol additions this needs

Both are additive to the existing WebSocket contract and would not break
current consumers:

- **`seq`** — a monotonic sequence number on every event, so a client can order
  events and discard stale async results without relying on arrival time.
- **`segment_id`** — on `partial` events, identifying the segment the partial
  will become. Lets a client distinguish "this partial replaced the previous
  one" from "this is a new segment", which the current contract leaves implicit.

---

## 6. Instrument these from day one

This design is a bet, and these three numbers say whether it paid:

- **Speculation precision** — fraction of speculative concepts that survive to
  `confirmed`. Low precision means API calls burned for nothing.
- **Latency saved** — time from concept-confirmed to results-displayed, with
  speculation on versus off.
- **Retraction rate** — how often `final` contradicts something already shown.
  The safety metric. Should be near zero if display is gated on `confirmed`.

---

## 7. Decide this before building any of it

**Measure the fetch latency you are trying to hide.**

If a question fetch takes 50 ms, none of this is worth the moving parts — the
user cannot perceive the difference and the complexity is pure cost. If it takes
800 ms, hiding it behind speech the patient is still producing is
transformative.

That single number decides whether the speculation layer gets built at all, and
it is measurable today without writing any of this.
