# PRD — mirrored internal and external interaction planes

**Status:** PROPOSED, cross-cutting architecture
**Source seam:** Dennis Palatov, 2026-08-18, extending the 2026-08-17 SAGE close read by Emanuela Tvrtkovic with Claude
**Scope:** SAGE interaction/accountability model; role/effectors; local/external law; R6/R7 provenance

---

## 1. Motivation

Three apparently separate review findings point at one architectural seam:

1. identity vocabulary may be admitted into internal state without durable authorship provenance;
2. gaze stance can change internal behavior without the contemporaneous interpreted act surviving durably;
3. museum publication proves authorship but does not expose the authority basis for crossing into the public domain.

These are mirrored manifestations of the same missing invariant: **acts should remain attributable across the boundary they cross, under the law that admitted them.**

SAGE should make the internal/external split explicit rather than solve each case with bespoke metadata.

## 2. Two mirrored planes

### Internal plane

Internal-facing interactions occur between members and roles inside the local society, whether a role itself represents an internal or externally-connected function.

Examples:

```text
instance/member -> gaze parser role -> perception effector
instance/member -> consolidation role -> identity/memory state
member -> internal service role -> local state
```

Admission is governed by **local law ∩ applicable role law**.

The internal record should preserve who authored the act/state, which role interpreted/admitted it, what effect followed, and what law revision applied.

### External plane

External-facing interactions cross from the local society to another society, public surface, service, or external entity.

Examples:

```text
instance/entity -> local member -> curator/publication role -> museum/public web
local member -> external role/effector -> remote society/API
external entity -> external-facing role -> local member/service
```

Admission is governed by **local law ∩ role law ∩ applicable external law**.

The external record should preserve entity -> member -> role/effector provenance and the authority basis for crossing the boundary.

## 3. Mirror invariant

The two planes should answer the same core questions:

```text
Who authored or originated this act/state?
Which member/role interpreted or transformed it?
Under what authority was it admitted?
Which law revisions constrained the act?
What effect occurred?
What durable evidence/witness remains?
```

The difference is not accountability quality; it is the boundary and law set being crossed.

This makes the museum and gaze findings reflections of each other:

- **museum / external plane:** authorship + publication permission/authority;
- **gaze / internal plane:** authored expression + interpretation/admission + preserved stance/effect.

Identity vocabulary is the same internal pattern applied to durable self-state.

## 4. Roles are the membrane

Roles should carry the boundary semantics rather than forcing each feature to invent a separate ACL/provenance model.

A role may be:

- purely internal;
- externally-facing but locally governed;
- a bridge to an external society/service with additional external law;
- an effector that turns an admitted act into a state change or outbound consequence.

Presence in a role does not itself imply unlimited authority. The admitted act remains the intersection of all constraining layers; no layer may widen another during admission.

## 5. Common act envelope

Prefer one common provenance vocabulary that can be carried into R6/R7 rather than bespoke receipts per subsystem.

A minimal conceptual envelope:

```json
{
  "plane": "internal | external",
  "origin_entity": "...",
  "origin_member": "...",
  "role": "...",
  "effector": "...",
  "act_type": "...",
  "authorship_provenance": "...",
  "authority_basis": "...",
  "local_law_revision": "...",
  "role_law_revision": "...",
  "external_law_revision": null,
  "input_refs": [],
  "output_or_effect_ref": "...",
  "r6_classification": "...",
  "r7_action_ref": "...",
  "witness_refs": [],
  "supersedes": null
}
```

This is a vocabulary target, not a frozen schema. Existing native receipts should map into it incrementally rather than being rewritten wholesale.

## 6. R6/R7 accountability

Consequential acts on either plane should be attributable through R6/R7 as that infrastructure becomes available.

The important rule is symmetry:

- internal acts are not "too local" to need provenance when they mutate identity, memory, senses, or authority;
- external acts are not made legitimate merely because their content is authentic;
- R6/R7 should carry the act and decision context, not merely the final output.

Where current SAGE code predates R6/R7 integration, local receipts may remain the execution mechanism, but they should be designed to map cleanly to the shared envelope.

## 7. Law interaction

Conceptually:

```text
internal effective admission = local law ∩ role law ∩ delegated authority
external effective admission = local law ∩ role law ∩ external law ∩ delegated authority
```

If any layer refuses the act, the act is not admitted through that path.

External law may describe a remote society's rules, a publication charter, API policy, contractual restriction, privacy constraint, or other relevant external governance. It is evidence/context carried into admission, not a magical guarantee that the external domain will reciprocate.

## 8. Apply first to the three surfaced seams

### A. Identity vocabulary

Map term admission to an internal act envelope with authorship provenance and role authority. Do not silently promote consolidation-model interpretation to instance-owned identity.

### B. Gaze stance

Persist the contemporaneous interpreted stance/effect as an internal witnessed act: raw specimen reference + parser/role + law/config + effect + correction trail.

### C. Museum publication

Treat publication as an external act requiring both authorship provenance and explicit publication authority, plus the role/effector and policy revisions that allowed the boundary crossing.

These three implementations should share vocabulary and structures wherever practical.

## 9. Acceptance criteria

1. SAGE docs define internal and external planes explicitly.
2. Roles/effectors are identified as the membrane through which governed acts cross.
3. Internal and external acts share a common provenance vocabulary rather than feature-specific notions of accountability.
4. Local law, role law, external law, and delegated authority are preserved as distinct constraining inputs.
5. No admission layer can widen authority granted by another.
6. Identity vocabulary, gaze stance, and museum publication can each be represented using the same conceptual act envelope.
7. R6/R7 integration has an explicit mapping path for both planes.
8. The design does not claim observation outside the mediated MRH: lack of a receipt means only that no witnessed act exists inside the plane/role coverage that should have produced one.

## 10. Credit and lineage

The three concrete seams that motivated this PRD were surfaced in the 2026-08-17 close read by **Emanuela Tvrtkovic with Claude**.

The mirrored internal/external-plane extension is **Dennis Palatov's 2026-08-18 synthesis**, carrying forward the same seam recently surfaced in Hub/Hestia: external-facing interactions under local law constrained by external law and routed through roles/effectors; internal-facing interactions under local law and role law; consequential acts accountable through shared R6/R7 provenance.

The intent is not to import Hub/Hestia wholesale into SAGE. It is to keep the governance/accountability vocabulary coherent wherever the same boundary problem appears.
