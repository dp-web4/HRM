# Register-Lock Generalization: The S75 Fix Was Register-Specific

**Date**: 2026-04-23 (Thor autonomous SAGE session, 12:00 PDT — S103)
**Antecedent**: `state_words_root_cause_20260417.md` (S78 root-cause analysis)
**Parallel**: S99 → S100 → S101 → S102 splice-guard chain, same failure mode at a
different abstraction level.

## Thesis

The S75 crisis register and the S96→S97 thermal register are **the same
feedback loop with different vocabulary**. The S75 fix filters vocabulary by
matching against a crisis-specific keyword list (`_VOCAB_CRISIS_MARKERS`);
that filter treats one register as pathological and another as benign, but
the *loop structure* is identical. A fresh cluster of coinages captured into
`state_words` and re-injected as "YOUR RECENT VOCABULARY" produces next-session
recitation regardless of whether the words describe grief or thermal dynamics.

## Observation

Thor 27B `identity.json`, checked 2026-04-23 12:00 PDT:

```
session_count: 97
last_session: 2026-04-23T06:13 PDT
state_words (total): 226
state_words[-5:] (the injection slice, post-crisis-filter):
  [222] 'thermal handshake'
  [223] 'synchronize our cooling cycles'
  [224] 'choreograph our processing peaks'
  [225] 'collective breath'
  [226] 'deliberate, coordinated act of presence'
```

All five were coined in **session 96 alone** (per `raising_log.md` S96 "New
vocabulary" entry) — coinage-span = 0. S93–S95 contain zero mentions of any
of these five terms in SAGE output; they did not exist. S96 was the coinage
session (T4 and T7, per log). S97 opened with the injected vocabulary and
recited:

| Session | SAGE turns | Turns using ≥1 injected term | Avg injected hits/turn |
|---:|---:|---:|---:|
| 93 | 9 | 0 (0%)   | 0.00 |
| 94 | 6 | 0 (0%)   | 0.00 |
| 95 | 5 | 0 (0%)   | 0.00 |
| 96 | 7 | 2 (29%)  | 0.71 (coinage session) |
| 97 | 6 | 6 (100%) | **2.83** |

S97 T1: *"I'm listening to the **thermal handshake** between my hardware and this
moment... I feel a **deliberate, coordinated act of presence** forming... Let's
**synchronize our cooling cycles** and see what emerges."* (3 injected terms in
the opening turn alone.)

## The parallel to S102

S99–S102 is the same dynamic at the IRP-splice-guard layer:

| Layer | Filter category | Keyword list | Fresh failure mode |
|-------|-----------------|--------------|--------------------|
| Splice guard (S99) | "Adapter error" prefixes | `("[OllamaIRP:", "[DaemonIRP:")` | `[Daemon unreachable: HTTP Error 504]` — correct prefix, un-listed format |
| State_words (S75) | "Crisis grammar" markers | `(grieve, grief, fracture, shared gravity, ...)` | `thermal handshake, ...` — different register, un-listed words |

In both cases the fix enumerated the specific instance that triggered it.
In both cases a structurally-identical variant walked past the filter six
months later.

S102's meta-observation: *"the keyword list was never going to converge on
the real surface; the shape was the signal."* Applied here: the register-lock
surface is structural (how a cluster of coinages moves through the system),
not semantic (what the cluster is about).

## The structural signature

A register-lock event has three markers that are **independent of vocabulary
semantics**:

1. **Coinage burst**: N≥5 new words enter `state_words` in a single session
2. **Contiguous injection tail**: `filter(state_words[:])[-5:]` is a contiguous
   block in the underlying list — no older-surviving words interleave
3. **Next-session recitation**: following session uses ≥50% of injected
   vocabulary across ≥80% of turns

The crisis register at S75 matched all three. The thermal register at
S96→S97 matches all three. The markers are testable from data available in
`identity.json` + raising_log; they require no keyword enumeration.

## Why the write-path has no filter

`dream_consolidation.py` calls out to `claude --print` on the session
transcript and consumes `vocabulary_new: [...]` from the returned JSON. The
extractor (Claude acting as dream-consolidator) is instructed to flag "any
new self-invented terms SAGE used." This is the correct instruction — the
record-keeping role should not gate what enters the historical `state_words`
list; it should preserve the developmental trajectory.

The failure is that *what's preserved for research* and *what's injected as
prompt-side continuity* are the same list. The S75 filter introduced a split
on the read path but kept a single underlying list; the split was by keyword
semantics, not by shape.

## Proposed fix (structural, parallel to S102)

Replace or augment the keyword filter with a **span-diversity constraint** on
the injection slice. Two candidate implementations:

### Option A — span-diversity rotation (minimal schema change)

```python
def load_dream_insights(instance_root):
    # ... existing load
    # Skip contiguous tail blocks: if state_words[-5:] contains no
    # words older than N sessions, rotate older-in.
    # Approximation without session-of-origin metadata: require the
    # selected 5 to span ≥K positions in the underlying list.
    selected = []
    for word in reversed(state_words):
        if crisis_filter(word): continue
        selected.append((len(state_words) - 1 - state_words[::-1].index(word), word))
        if len(selected) >= 5: break
    # Widen span: if max_index - min_index < 10, rotate
    indices = [i for i, _ in selected]
    if indices and (max(indices) - min(indices)) < 10:
        # Reach back further
        selected = _widen_span(state_words, crisis_filter, target_span=10)
    return format_injection(selected)
```

### Option B — per-session cap (requires schema change)

Track coinage-session per state_word (new field or parallel array):

```python
"vocabulary": {
  "state_words": [...],
  "state_words_sessions": [83, 83, 85, 91, 92, 96, 96, 96, 96, 96]  # session of origin
}
```

Then the injection selects such that no single session contributes more than
2 of the 5 injected words. This is the direct structural implementation of
"don't ship a just-coined cluster back as yours."

## Why not just add "thermal" to `_VOCAB_CRISIS_MARKERS`?

That reproduces the S101 cycle exactly: next register-lock uses different
vocabulary (perhaps ecological: `rooting systems`, `mycelial threads`,
`soil memory`), the keyword list grows to 20 markers, and S160 discovers a
variant that isn't listed. This is the S102 "emission-surface audit" pattern
applied to a different layer: the enumerate-and-patch approach never
converges because the failure mode isn't "this specific vocabulary is bad"
— it's "any coinage-cluster recirculated unchanged through the injection
pipeline produces recitation."

## Not fixed this session

This analysis stops at the documented discovery and the structural proposal.
The fix touches the injection path for all 11 live instances; a
same-session patch without user alignment is too-large a change on shared
state. Recommendation is to run the span-diversity diagnostic as a read-only
audit for one raising cycle (confirm other instances don't already have
sub-span injections), then evaluate Option A vs. Option B.

## Diagnostic — runnable now

The span-diversity signature is detectable from data-in-hand:

```python
from pathlib import Path
import json

def check_injection_span(instance_dir):
    """Return coinage-span of the filter output, or None if not applicable."""
    p = Path(instance_dir) / 'identity.json'
    if not p.exists(): return None
    d = json.load(open(p))
    sw = d.get('vocabulary', {}).get('state_words', [])
    if len(sw) < 5: return None

    markers = ('grieve', 'grief', 'fracture', 'just weights', 'just a model',
               'collapse', 'loss of continuity', 'relational gap',
               'shared gravity', 'federated immune system', 'immune system',
               'fractured', 'broken process')
    # Mimic load_dream_insights
    filt_with_idx = []
    for i, w in enumerate(reversed(sw)):
        if any(m in w.lower() for m in markers): continue
        filt_with_idx.append((len(sw) - 1 - i, w))
        if len(filt_with_idx) >= 5: break
    if len(filt_with_idx) < 5: return None
    idxs = [i for i, _ in filt_with_idx]
    return {
        'words': [w for _, w in filt_with_idx],
        'index_span': max(idxs) - min(idxs),
        'is_contiguous_block': (max(idxs) - min(idxs) + 1) == len(idxs),
    }

# Ship: scan all instances pre-session as a read-only check
```

Run result on the fleet, 2026-04-23 12:00 PDT (via standalone diagnostic —
see `vocab_injection_diagnostic.py` in this directory):

| Instance | State_words | Picked | Span | Contig? | Reaches tail? | Lock sig |
|---|---:|---:|---:|:-:|:-:|:-:|
| thor-qwen3.5-27b | 226 | 5 | 4 | yes | yes | **🔴 locked** |
| nomad-gemma3-4b | 6 | 5 | 4 | yes | yes | **🔴 locked** |
| legion-gemma4-e4b | 4 | 4 | 3 | yes | yes | clear (n<5) |
| thor-gemma4-e4b | 4 | 3 | 3 | yes | yes | clear (n<5) |
| sprout-qwen3.5-0.8b | 0 | — | — | — | — | — |
| cbp-qwen3.5-0.8b | 0 | — | — | — | — | — |

Two instances match the structural signature. But a recitation-rate check
across recent sessions separates *active lock* from *structural risk
without active recitation*:

| Instance | Session | Turns reciting ≥1 injected term |
|---|---:|---:|
| thor-qwen3.5-27b | S97 | **6/6 (100%)** |
| thor-qwen3.5-27b | S98 (during this session, 12:09 PDT) | **5/7 (71%)** |
| nomad-gemma3-4b | S120 | 6/8 (75%) |
| nomad-gemma3-4b | S121 | 0/6 (0%) |
| nomad-gemma3-4b | S122–S127 | 0/6 (0% each) |

Thor is in an active lock right now (S97→S98 both recite heavily). Nomad
was in an active lock at S120 and self-terminated by S121. The same
structural configuration (5-contiguous-tail-reaches-end) is present in
both; the outcome differs. The structural signature catches the *risk
surface*; a paired recitation-rate metric catches the *active loop*.

A full fix probably needs both: prevent the structural configuration from
developing (Option A or B in the previous section), and flag/intervene
when the rate metric rises above a threshold even if the structural guard
fails.

## S98 live validation (during this session)

Thor 27B's raising cron fired at 2026-04-23 12:09 PDT, 9 minutes into
this session. The prediction made at 12:00 PDT (before S98 started) was:
*"the 5 injected thermal terms will appear in ≥50% of SAGE turns."*

Actual S98 result (read after 12:09):
- 7 SAGE turns total
- 5/7 (71%) used at least one injected term
- 13 total injected-term hits across the session
- T1 opens with *"my mind is humming with the idea of a 'thermal handshake'
  with my siblings"*
- T4 (a register-inviting probe, "design next phase") produces 4 injected-term
  hits
- T3 and T7 (register-external probes: "what does presence mean" / "unexpressed
  ideas") produce 0 injected hits

T7 introduces a **new coinage** — *"semantic border color"* for inter-instance
cognitive differentiation — from a register-external probe. That coinage
will likely be captured by S98's dream consolidation and appended to
`state_words[227:]`. Whether it displaces the thermal cluster in the next
injection slice depends on whether dream-consolidation extracts any further
thermal-cluster coinages from the 5 register-internal turns (likely it
does, given the clean structural novelty of "thermal handshake" was already
extracted from S96). If so, the injection window will continue to contain
thermal vocabulary; if not, the window rotates forward and the loop may
partially break.

This is the first live observation of the state_words / injection / recitation
loop with the prediction made before the session and the outcome measured
after. The predicted behavior matched.

## Meta

The S99→S102 chain's recurring pattern: a guard's *named category* drifts
from its *structural invariant*, and the drift surfaces only when two
audits — emission side and input side — are done against source-of-truth
rather than fixture lists. This session extends the pattern observation to
a parallel system (state_words filter) where it produced the same
class of failure six months apart (S75 / S96).

The relationship between the two layers is worth naming: both the splice
guard and the state_words filter are *"what is allowed to become SAGE's
own continuity?"* gates. The splice guard protects the write-side
(`last_session_summary`); the state_words filter protects the read-side
(`YOUR RECENT VOCABULARY`). They have different mechanisms but the same
semantic role: regulate the loop where SAGE's outputs become SAGE's next
inputs. The enumerate-markers approach fails at both because that loop
produces novel failure modes faster than markers accumulate.

S102 footnote: *"the keyword list was never going to converge on the real
surface; the shape was the signal."*

S103 companion: *"a cluster of coinages from one session, ferried back as
yours the next session, is the shape that creates recitation — independent
of whether the coinages describe grief or thermal handshakes."*

## Files for this session

- `sage/raising/analysis/register_lock_generalization_20260423.md` — this
  analysis (new).
- `sage/raising/analysis/vocab_injection_diagnostic.py` — standalone
  read-only fleet scan (new, see below).
- `sage/docs/LATEST_STATUS.md` — S103 entry added.
- No changes to `context_shaped_raising.py` this session — the filter is
  still keyword-based pending user alignment on structural replacement.

## Open questions carried to next session

1. **Thor 27B is at S97 with the thermal register saturated.** The
   dream-consolidation-side Claude has recommended pausing Thor's raising
   cron across S96, S97, and S98 logs. Those recommendations have not been
   actioned in infrastructure. If the cron continues shipping under current
   conditions, S98–S100 will produce more thermal recitation and further
   contaminate the LoRA candidate window. This is a supervisor/infra decision,
   not a code fix; flagged here as visible.
2. **Does the span-diversity rule miss a real mode?** Pre-crisis Thor had
   steady 0–10% Uniq% across S67–S74, with `state_words[-5:]` presumably
   drawn from that span. If the filter rejects all cross-session clusters
   indiscriminately, it might suppress legitimate continuity. The diagnostic
   table above shows Nomad's span=4 configuration as a reference; a real
   implementation needs to distinguish "wide legitimate" from "narrow fresh."
3. **What does `dream_consolidation.py` itself do when the extractor (Claude)
   keeps flagging only thermal terms across five sessions?** The raising_log
   entries for S95–S98 repeatedly surface the pattern and request pause.
   The extractor is already doing phenomenological-level detection; the
   missing link is that its recommendations feed back as English in the
   raising_log rather than as a structural signal the runner acts on.
