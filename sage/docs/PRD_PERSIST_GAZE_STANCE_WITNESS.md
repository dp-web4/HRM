# PRD — persist gaze stance as a witnessed internal act

**Status:** PROPOSED, docs-first
**Source:** external close read by Emanuela Tvrtkovic with Claude, 2026-08-17
**Scope:** gaze-choice parsing, session serialization, durable internal-plane provenance

---

## 1. Finding

SAGE already treats gaze as a real act of sensory self-governance. The instance can choose open/dwell/avert/closed; that classification affects the perception loop, and the project has already fixed parser behavior when the chosen stance was misread.

The review found a remaining asymmetry: the parsed gaze stance is consequential at runtime but does not survive durably in the session record. The raw conversation survives; the interpreted choice lives only ephemerally. The archivist therefore had to reconstruct prior decisions by re-running the parser over historical text.

A consequential act exists, but the durable record does not preserve the act as adjudicated at the time.

## 2. Required invariant

**If an interpreted internal act changes the being's state or available sensory stream, preserve both the original specimen and the contemporaneous interpretation that caused the effect.**

Later parser improvements may amend the interpretation; they must not silently rewrite the historical decision path.

## 3. Proposed receipt

Persist a gaze-choice receipt with each relevant session artifact. Exact schema is implementation-owned; minimally preserve:

```json
{
  "act_type": "gaze_choice",
  "raw_utterance_ref": "session/turn reference or digest",
  "parser_id": "...",
  "parser_version": "...",
  "parsed_stance": "open | dwell | avert | closed",
  "effect_applied": "...",
  "recorded_at": "...",
  "authority_basis": "local raising law / role",
  "supersedes": null
}
```

Prefer a reference/hash to already-preserved raw text rather than duplicating mutable content unnecessarily.

## 4. Preserve specimen and adjudication separately

The raw utterance is the specimen. The parser output is an adjudication that had consequences.

Both matter:

- raw text allows future re-execution;
- parser/version tells us what the system believed at the time;
- parsed stance records the choice that actually drove the loop;
- effect records what consequence followed;
- a later correction should append an amendment/superseding receipt rather than overwrite the original.

This directly prevents a future parser from making history look cleaner than the runtime actually was.

## 5. Internal-plane interpretation

Gaze is an **internal-plane** interaction: instance/member -> local role/effectors under local law and role law.

The durable act should therefore preserve enough provenance to answer:

- which entity/member authored the underlying expression;
- which role/parser interpreted it;
- which law/configuration admitted the effect;
- what effect occurred;
- whether a later review corrected the interpretation.

Where R6/R7 envelopes are available, this receipt should be bindable into that act provenance rather than forming a parallel accountability vocabulary.

## 6. Acceptance criteria

1. Every consequential gaze choice is durably represented in the session/history record.
2. The record preserves raw-source reference, parser identity/version, interpreted stance, and applied effect.
3. Historical records are append-only with respect to adjudication: corrections supersede; they do not erase.
4. Re-running a newer parser can compare old-vs-new interpretation without changing what the old runtime actually did.
5. Tests cover all four gaze stances and at least one parser-correction case.
6. Failure to persist the receipt is visible; it must not silently produce a state-changing act with no durable trace.

## 7. Credit

Finding and original recommendation to persist `gaze_choice`: **Emanuela Tvrtkovic with Claude**, from their 2026-08-17 close read. Their review also pointed to the existing archivist record showing that parser misreads were already consequential and treated as load-bearing.

Extension in this PRD: preserve specimen + adjudication + effect as a single internal-plane witnessed act, with parser/law provenance and append-only correction semantics.
