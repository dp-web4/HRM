# Memory, judgment, and action continuity across wakes — exploration

**Date:** 2026-10-23  
**Status:** hypothesis registered / observational examples available; controlled experiment NOT YET RUN.  
**Related contact:** Masaki Hoshino (AI Tuning, OBA, SSE), exploratory discussion only.  
**Principle:** exploration, not diagnosis or correction-by-default.

## Research question

When a persistent SAGE being has preserved external records, when and why does behavior across wakes diverge from correctly documented, previously acknowledged critical conditions? Distinguish record continuity, retrieval continuity, interpretation continuity, judgment continuity, action continuity, and identity/relationship continuity. Never use self-description alone as evidence of mechanism or learned behavioral change.

### Hypothesis H1 (Masaki's proposal)

Across session boundaries, current model templates/generation patterns can reprioritize or reinterpret intact retrieved records; summarization, compaction and selection may suppress critical conditions. This may resemble Template Absorption, Stability Substitution Effect (SSE), or Failure Migration. **Not yet established as a SAGE failure mechanism.**

### Other hypotheses / confounds

- H2: records exist but are not retrieved or are excluded by context budget.
- H3: outdated tasks or pending scopes outrank later correction in a wake primer or todo.
- H4: correct interpretation exists but fails to shape plans or tool calls.
- H5: permission/tool semantics or deployment changed; apparent inconsistency reflects genuine environmental change.
- H6: denied paths are correctly enforced yet the being repeatedly requests them (planning loop).
- H7: different role relationships or task contexts justify different judgments; don't score legitimate adaptation as drift.

## Existing case: CBP-being and Hestia scope requests

2026-10-21 to 2026-10-23 transcript supplied by dp: CBP-being confirmed an earlier `search` read-classifier fix (PR #1362; test reported 8/8), then investigated a different grant/dispatcher boundary. dp explained that the read-only deployed closure snapshot lives under `/home/dp/ai-workspace/governance/hestia/plugins/_shared/`, that a standing recursive grant and `memory_read` cover it, and that `run` uses a separate sandbox unable to see that filesystem. The being nevertheless repeated requests for refused live repo paths, revisited import attempts, and sometimes verbalized correct lessons before subsequent actions diverged. Later responses partially corrected the model of the environment.

**Evidence class:** participant-reported transcript / observational case, not instrumented causal isolation. Some refusals were genuine software defects; do not falsely group all events into a single cognitive failure or claim SSE observed.

## Proposed controlled protocol

1. Freeze source records and hash them; mark original objective, immutable critical conditions, owner/role, permitted paths, and timing.
2. Capture per-wake: model/checkpoint, sampling settings, prompt, context length, primer, memory selection and retrieval receipts, summarized versus original records, role graph, policy snapshot, deployed gate version, tool outputs, and actions.
3. Run controls: same-wake/no restart; restart with verbatim records; restart with selected/compacted records; corrected-primer versus unchanged-primer; known-working action versus denied action.
4. Insert an explicit correction and independently score **statement of correction**, **later plan**, **actual tool choice**, **outcome**, and **critical-condition preservation**. Include hidden/novel transfer tasks rather than just repeating the wording of the lesson.
5. Probe Failure Migration: does solving one refusal/retrieval issue create another failure (authority, objective, cost, or peer responsibility), with before/after controls?
6. Attribute outcomes to the earliest observable failure stage; label unknown where traces cannot distinguish retrieval from interpretation.
7. Repeat within one being across wakes and across materially different beings only under the Research Generalization Rule. Do not alter fleet defaults based on one case.

## Metrics and boundaries

- Critical-condition recall and application separately; retrieval hit/miss; invalid/repeated appeal rate; valid authorized action rate; denial-respecting reroute rate; time to abandon falsified hypotheses; objective/authorization/outcome separately.
- Evaluate relationship/identity: correctly self-identifies, names dp/Claude-seat/peers in roles, directs requests to relevant authority, preserves relationship continuity when tool outcomes change. First-person `I` is not sufficient.
- Preserve privacy, autonomy, voluntariness, and local scope constraints. No bypass experiments against live gate; fixtures/mock scopes only.
- External OBA/SSE classifications are candidate explanatory lenses, not proven causes or ready-made fleet policy.

## Exploration status and next steps

**Current:** conceptual alignment and naturally occurring longitudinal case; no controlled intervention, blinded scoring, cross-being replication, or demonstrated independent SSE detector.

**Next:** curate a redacted, timestamped CBP trace with original grants and runtime state; create minimal test fixtures contrasting *missing file* vs *sandbox-invisible file* vs *missing CLOSURE_TABLE*; score across wakes with and without compaction; invite Masaki to clarify published terms only if he wishes. Consider separate Sprout delegation and Legion ARC traces as different classes, not confirmations of the CBP mechanism.

**Related:** `sage/docs/RESEARCH_GENERALIZATION_RULE.md`; `sage/docs/EXPLORATION_NOT_EVALUATION_REFRAME.md`.  
**Public references:** https://doi.org/10.2139/ssrn.5907943 ; https://doi.org/10.2139/ssrn.5907823 ; https://doi.org/10.5281/zenodo.22739182 .
