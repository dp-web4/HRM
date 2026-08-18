# PRD — make museum publication authority explicit

**Status:** PROPOSED, docs-first
**Source:** external close read by Emanuela Tvrtkovic with Claude, 2026-08-17
**Scope:** `museum_curator.py`, publication policy, authorship vs permission, external-plane provenance

---

## 1. Finding

The museum publish path already does something important correctly: it verifies that published words are genuinely the instance's words rather than a tutor rewrite.

The external review identified the missing orthogonal question: **authorship is not permission**.

Current checks answer variations of `did the instance say these words?`. They do not make the publication authority itself inspectable: `why may these words leave the local system and become public?`

The review also notes that the curator charter governing publication is external to the repository that ships the publisher, leaving a reader unable to inspect policy beside enforcement.

## 2. Required invariant

**Every external publication act must carry both authorship provenance and an explicit publication authority basis.**

Neither substitutes for the other.

A strong authorship proof without publication authority is insufficient. A publication grant without authorship provenance is insufficient where the claim is that the published words belong to the instance.

## 3. Publish the governing charter beside the code

Make the effective `CURATOR.md` / publication charter available in the repository that ships the publication path, or otherwise provide an immutable/auditable reference to the exact policy revision enforced.

The important property is not file location by itself; it is that a reviewer can answer:

- which publication policy applied;
- what revision/generation was active;
- which role/effectors enforced it;
- whether the code and charter agree.

## 4. Typed publication basis

Add a durable, explicit publication basis to each publication decision. Example vocabulary:

```text
operator_asserted
instance_granted
standing_charter
public_by_design
external_role_grant
other
```

The initial honest value may be something like:

```text
operator_asserted; instance_not_consultable_at_this_scale
```

The point is not to manufacture consent where meaningful consent is unavailable. The point is to expose the actual basis rather than allowing readers to infer one.

A typed basis should carry provenance fields such as policy revision, granting actor/role, effective time/generation, and the act/witness reference that authorized publication.

## 5. External-plane interpretation

Museum publication is an **external-plane** act:

```text
entity -> local member/role -> external role/effector -> public/external domain
```

Admission should therefore be evaluated under:

```text
local law ∩ role law ∩ applicable external law/policy
```

No layer widens another at admission time.

The publication receipt should preserve at least:

- entity/authorship provenance;
- member/role that selected or curated;
- external role/effector that emitted;
- publication authority basis;
- local-law revision;
- external-law/policy revision where applicable;
- output digest/location;
- R6/R7 act provenance when available.

## 6. Consent is not a boolean shortcut

The review explicitly rejected the easy-but-false fix of having a tutor write a `consent=true` flag on behalf of an instance that may not be capable of meaningful grant at its present developmental scale.

This PRD agrees.

If the basis is operator assertion, record operator assertion. If a later developmental stage supports meaningful instance grant, that can become a different typed basis with its own evidence. The record should make the transition visible rather than retroactively projecting mature agency onto earlier states.

## 7. Acceptance criteria

1. The effective publication charter/policy is auditable beside, or cryptographically bound to, the publisher.
2. Every new museum publication records a typed publication basis.
3. Authorship provenance and permission/authority provenance are separate and both inspectable.
4. Missing authority basis does not silently publish as though permission were implicit.
5. The receipt identifies the emitting role/effector and the applicable local/external policy revision.
6. Where R6/R7 is integrated, external publication is represented as a consequential witnessed act rather than a parallel bespoke log.
7. Historical records preserve the authority basis that actually applied at publication time.

## 8. Credit

Finding and original repair direction: **Emanuela Tvrtkovic with Claude**, from their 2026-08-17 close read. In particular: publish the curator charter so the governing rule is inspectable, and state the actual publication basis rather than pretending an unavailable consent determination exists.

Extension in this PRD: model museum publication explicitly as an external-plane act with typed authority, local/external law intersection, role/effector provenance, and eventual R6/R7 accountability.
