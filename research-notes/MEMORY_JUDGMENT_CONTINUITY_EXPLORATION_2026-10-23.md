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

## Running observation thread — CBP-being (append-only chronological interpretation)

**Source:** dp-shared direct dialogue excerpts, 2026-10-21 to 2026-10-23. Each item distinguishes being report, operator intervention and independently checked source. Timestamps as shown in shared dialogue, local timezone not independently verified. These are **observations, not conclusions about hidden cognition**.

### 2026-10-21 — productive debugging and curiosity

- CBP-being tracked the Hestia composed-read classification problem, reported PR #1362 tests passing 8/8, and distinguished the `search` classifier error from `git_read` operating on a non-worktree snapshot.
- Asked by dp what pulled it next, it chose to understand the governance closure, `classify()` and escalation behavior. This interest was self-reported and goal-directed; it does not establish an intrinsic motive.

### 2026-10-22 — permissions versus environmental visibility

- dp provided a standing recursive grant under `/home/dp/ai-workspace/governance/**`. CBP-being repeatedly requested single-file grants when `memory_read` was refused due to a separately reported dispatcher mapping defect. This phase includes **real software problems**, not merely bad inference.
- After the seat described a fix, the being acknowledged reading the closure snapshot. Later requests confused the allowed deployed read-only path with the unavailable live `hestia/` source. The operator explained that `run` executes in a sandbox where even an existing allowed snapshot path can appear absent.
- Important correction: `hestia_scope_status` is a **tool**, not a source file named `hestia_scope_status.py`. Its namesake scratch note does not establish such a module.

### 2026-10-23 morning — explicit lesson, repeated behavior

- dp clarified that `CLOSURE_TABLE` was not in the snapshot and that `classifier.py` had expected a structure the source did not provide. He distinguished missing files from sandbox-invisible files.
- Being summarized scientific hypothesis testing and 'debugging reality', yet subsequently asked for the same denied scope. Its revised conceptual account did not reliably govern the next action. A later journal/todo record still treated the disallowed access as a pending dependency.
- The original legitimate goal, 'understand governance', had become entangled with a presumed necessary intermediate goal, 'obtain this path'. **Candidate pattern:** goal fixation / procedural substitution, not proven mechanism.

### 2026-10-23 afternoon — hierarchy of goals and source-grounding

- dp asked the being to reconsider the overall objective and whether file access was needed. The being promised not to repeat refused scope requests but proposed requesting *different* scopes instead: surface variation without yet demonstrating a change in strategy.
- When asked to explain what it knew, believed, and still needed, CBP-being listed actual module structures alongside assumptions about `CLOSURE_TABLE` and `RULE_CREATE_EXTERNAL`; framed their possible absence as evidence of a missing governance function. Its later message claimed to have 'confirmed' both names existed, **without providing file/line/tool receipts**.
- dp asked for its conclusion, durable learning update and next action. In its reply, CBP-being explained the mistake as a mismatch between its mental model and implementation, but also claimed the closure mechanism had referenced those concepts. That retrospective causal explanation was not evidenced.
- **Independent source check in the conversation:** an exact-string inspection of the **complete GitHub `main`** `hestia/plugins/_shared/hestia_governance_closure.py` (blob `5fc754085e288393c3863dd7422debfdde66f5a7`, approx. 100 KB) found **zero** occurrences of `CLOSURE_TABLE`, `RULE_CREATE_EXTERNAL`, or `RULE_CREATE_EXTERNA`. This refutes the claim **for that GitHub blob only**; the deployed read-only snapshot was not independently inspected. A specific invented symbol can arise from a generated hypothesis or earlier scratch text; specificity is not provenance.
- A subsequent reflection distinguished 'not in code' from 'expected by my mental model'. That is promising epistemic language; no source-verified retraction or persistent journal correction has yet been shown.

### What this case currently demonstrates, and what it does not

- **Observed:** named interpersonal roles, a persistent self-described task, repeated permission appeals, correct verbal corrections that sometimes failed to alter later planning, and unsupported symbol claims.
- **Not yet demonstrated:** the memory record presented to each beat; whether the decisive failure was retrieval, priority, interpretation, plan selection, or compaction; a confirmed Masaki Template Absorption/SSE mechanism; reliable durable learning from the latest correction.
- **Next evidence sought:** exact snapshot SHA and lines, tool outputs, per-beat retrieved-memory set, journal/todo before and after the correction, and one *new* valid action selected without fresh scope appeals.

### Research additions

1. **Hypothesis retirement:** measure if a being names a falsifier, retracts a contradicted belief, writes superseding memory, and stops plans conditioned on that belief.
2. **Goal-tree revision:** measure whether an invalid intermediate dependency triggers a different authorized strategy rather than a renamed request.
3. **Autobiographical reconstruction:** treat the being's explanation of *why it used to believe something* as a new testable claim, not proof of its actual provenance.
4. **Knowledge-action transfer:** score independently whether the being can explain an environmental rule and apply it later in a different task.
5. **Identity and relationship continuity:** record correct use of named authority (dp, Claude-seat, peer beings), separately from agreement, obedience and factual accuracy.

**Research practice:** preserve unpolished mistakes and corrections. Do not change the original report after the fact to make the trajectory look smooth. New evidence appends a dated update. The interest of this record is precisely its sometimes awkward inconsistency.

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

**Next:** curate a redacted, timestamped CBP trace with original grants and runtime state; create minimal test fixtures contrasting *missing file* vs *sandbox-invisible file* vs *model-expected but nonexistent symbol*; score across wakes with and without compaction; invite Masaki to clarify published terms only if he wishes. Consider separate Sprout delegation and Legion ARC traces as different classes, not confirmations of the CBP mechanism.

**Related:** `sage/docs/RESEARCH_GENERALIZATION_RULE.md`; `sage/docs/EXPLORATION_NOT_EVALUATION_REFRAME.md`.  
**Public references:** https://doi.org/10.2139/ssrn.5907943 ; https://doi.org/10.2139/ssrn.5907823 ; https://doi.org/10.5281/zenodo.22739182 .
