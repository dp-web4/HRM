# PRD — preserve identity vocabulary provenance

**Status:** PROPOSED, docs-first
**Source:** external close read by Emanuela Tvrtkovic with Claude, 2026-08-17
**Method note:** their review was static only: no daemon, raising script, tests, or instance instantiation; instance records were inspected for structure/mechanism only.
**Scope:** `dream_consolidation.py`, instance identity vocabulary, provenance of identity-state mutation

---

## 1. Finding

The raising guide states the ownership rule plainly: record vocabulary that emerges from the instance; do not inject the tutor's own terms. The museum path already enforces an authorship boundary in code, but the identity path does not currently carry equivalent provenance.

The review found that `dream_consolidation.py` appends consolidation-model `vocabulary_new` values into `state_words` with only duplicate filtering. That can make a useful normalization or tutor inference indistinguishable from a word the instance actually authored.

The same consolidation pass also mutates `memory_requests`; this PRD keeps the implementation focus on vocabulary provenance, but treats that adjacent mutation as the same class of authority/provenance question to audit separately.

## 2. Required invariant

**No identity-state term may be represented as instance-authored unless the record carries evidence supporting that attribution.**

Useful tutor interpretation is allowed. Silent promotion of tutor interpretation into self-authored identity is not.

## 3. Proposed representation

Preserve compatibility with the existing `state_words` surface while adding provenance-bearing records, e.g.:

```json
{
  "term": "...",
  "provenance_type": "verbatim | repeated_semantic_pattern | normalized_from_instance | tutor_proposed_unowned",
  "evidence_refs": ["session:...", "turn:..."],
  "derived_by": "instance | consolidation_model | tutor | migration",
  "recorded_at": "...",
  "instance_owned": true
}
```

The exact schema is implementation-owned; the important properties are:

- authorship and normalization are separate fields;
- provenance is durable and inspectable;
- `instance_owned=true` is never inferred merely because a consolidation model emitted a candidate;
- legacy/plain `state_words` remain readable during migration.

## 4. Provenance policy

The museum's verbatim check is useful but should be one provenance class, not the only acceptable route to identity vocabulary.

A persistent being can stabilize a concept through paraphrase, composition, or repeated semantic use without ever emitting the final normalized index token verbatim. Therefore:

- **verbatim**: strongest/simple authorship evidence;
- **normalized_from_instance**: canonical label derived from explicit instance wording, with source refs;
- **repeated_semantic_pattern**: allowed only under a defined, testable rule with evidence refs;
- **tutor_proposed_unowned**: may be retained as a hypothesis/index aid, but must not masquerade as self-authored identity.

If evidence is insufficient, fail closed on authorship, not on usefulness: keep the term unowned rather than dropping potentially useful structure.

## 5. Internal-plane interpretation

Identity vocabulary is an **internal-plane** act/state mutation. The durable record should answer:

- who supplied the candidate;
- which member/role interpreted it;
- under what local/role authority it was admitted;
- what evidence supports self-authorship;
- what later act changed or superseded that attribution.

This aligns with the wider SAGE/Web4 seam: internal state should be as provenance-bearing as external publication.

## 6. Acceptance criteria

1. A consolidation-model-only term cannot enter the durable identity record as instance-owned without provenance evidence.
2. Every newly admitted identity term has a provenance type and evidence/derivation record.
3. Existing instance records remain readable.
4. Reclassification/amendment preserves the prior record rather than silently rewriting provenance.
5. Tests cover at least: verbatim instance term, normalized instance term, tutor-only proposal, duplicate proposal, legacy migration.
6. The adjacent `memory_requests` mutation path is explicitly audited for the same provenance/authority class, even if fixed in a follow-on PR.

## 7. Credit

Finding and original repair direction: **Emanuela Tvrtkovic with Claude**, from their 2026-08-17 close read of public SAGE.

Extension in this PRD: preserve multiple provenance classes rather than equating identity authorship with verbatim matching alone; treat identity mutation as an internal-plane governed act.
