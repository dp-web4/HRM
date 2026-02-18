# SOIA-SAGE Convergence: Policy Entity as Repositioned IRP

## Context

In February 2026, a LinkedIn conversation between Dennis Palatov and Renee Karlstrom (independent SOIA researcher) surfaced a structural convergence that had been implicit across the dp-web4 project ecosystem.

Karlstrom's SOIA (Self-Optimizing Intelligence Architecture) proposes three mechanisms for giving AI systems internal stability: a self-referential regulatory loop (SRC), selective memory reinjection (MTM), and an internal temporal axis (MORIA). When mapped against SAGE's IRP stack -- which independently implements a consciousness loop, SNARC-scored experience buffer, and sleep-cycle consolidation -- the structural correspondence is near-exact.

The conversation progressed through several phases:
1. Dennis identified where SOIA and Web4 converge (rejection of binary safety, trust as dynamic, doubt as legitimate state) and diverge (internal vs. external coherence anchor)
2. Karlstrom proposed "conscience" as a shared handle -- a callable control-plane component that evaluates decisions at key points
3. Dennis recognized that SAGE's IRP stack already implements the machinery SOIA describes, and that Policy Entity in Web4 should be implemented as a SAGE IRP stack -- not invented from scratch
4. The recognition that the relationship between HRM/SAGE and Web4 had been implicit until now -- "separate substructures orbiting until a third perspective collapses them into the same coordinate system"

## The Insight

**IRP is the right abstraction for policy.**

The IRP contract (init_state / step / energy / project / halt) formalizes iterative refinement toward coherence. Every IRP plugin does the same thing: evaluate each step against a reference, detect when the system has drifted from that reference, and drive toward lower energy states. The reference differs by plugin:

- Vision: visual coherence
- Language: linguistic coherence
- Control: motor coherence
- Memory: salience coherence

**Policy is just another reference frame for coherence.**

A policy-evaluating IRP plugin (PolicyGate) uses `PolicyEntity.evaluate()` as its energy function. A compliant action has energy 0. A violating action has energy > 0. A hard deny has energy infinity. The same convergence behavior, trust metrics, and ATP budgeting that govern vision and language plugins govern policy evaluation.

This means PolicyGate:
- Gets a trust weight that evolves from convergence quality
- Receives ATP budget proportional to its trust
- Can be disabled, replaced, or run in parallel with other policy plugins
- Produces trust metrics (monotonicity ratio, dE variance) that indicate how well it's performing

The IRP contract makes policy evaluation a first-class participant in the consciousness loop, not a bolt-on filter.

## Connection to Synthon Framing

In the synthon vocabulary (see `synthon-framing.md`), PolicyGate is a **membrane protein**.

The synthon framing document describes Web4's infrastructure as operating at the boundary -- "building the membrane infrastructure that lets synthons interact without losing themselves." LCTs function as membrane proteins: they mediate what crosses the boundary and under what conditions.

PolicyGate extends this to the intra-agent boundary. It is the membrane between deliberation and action. It mediates which proposed actions cross from the internal planning space into the external effector space. The evaluation criteria come from Web4's trust fabric (PolicyEntity rules, R6 framework, trust thresholds), but the mechanism is SAGE's IRP plugin architecture.

This is exactly the integration seam the synthon framing predicted but did not yet implement: "building coherence substrates" where the substrate now includes policy-aware gating at the deliberation-to-action boundary.

## What This Changes

### Before: Two Separate Tracks
- **SAGE** (HRM): Embodied cognition for edge devices. IRP plugins for vision, language, control. Experience buffer. Sleep consolidation. No policy awareness.
- **Web4 Policy Entity**: Enterprise governance. Rules, roles, trust thresholds, approval workflows, attack mitigations. No IRP integration. No metabolic awareness.

### After: One IRP Spine, Two Anchor Points
- **SAGE as embodied**: Same consciousness loop. IRP plugins for perception, cognition, action. Anchor: "Am I coherent with my sensor inputs and motor objectives?"
- **PolicyGate as exo-centered conscience**: Same IRP contract. Energy function: Web4 `PolicyEntity.evaluate()`. Anchor: "Am I coherent with context and mandate?"

Same spine. Different energy function. Different anchor. Same trust metrics.

## Fractal Self-Similarity: Plugin of Plugins

The IRP contract is self-similar across scales. PolicyEntity is not just wrapped by an IRP plugin — it *is* a specialized SAGE stack:

| Scale | What Runs | Energy Function |
|-------|-----------|-----------------|
| Consciousness loop | SAGE runs PolicyGate as one plugin among many | Policy compliance |
| PolicyGate internally | Evaluate → refine → converge cycle | Rule matching + trust thresholds |
| LLM advisory (Phase 5) | PolicyGate invokes Phi-4 Mini for ambiguous WARN cases | Semantic judgment quality |

Same contract at every level: `init_state → step → energy → halt`. The orchestrator doesn't need to know that PolicyGate is fractal — it registers like any other plugin, receives ATP budget, builds trust from convergence quality. The recursion is invisible from the outside.

This validates the IRP abstraction itself: if the same contract works at three nested scales, it's capturing something real about how iterative refinement toward coherence operates. In the synthon vocabulary, the IRP contract is the placement rule — and placement rules are scale-invariant.

## What This Does NOT Change

- SAGE is still a cognition kernel for edge devices. PolicyGate is an optional plugin.
- Web4's PolicyEntity still works as standalone evaluation. It does not depend on SAGE.
- The IRP contract is unchanged. PolicyGate implements the same interface as all other plugins.
- No new abstractions are introduced. PolicyGate is just a plugin with a policy energy function.

## The Renee Question

The SOIA conversation also served as a diagnostic: is Karlstrom a "signal broadcaster" (writing papers, waiting for gravity) or a "reality participant" (willing to take responsibility for interfaces, constraints, failure cases)?

Her response -- sharing a core SOIA overview and inviting joint implementation steps -- suggests the latter, though the final answer depends on whether she engages with the HRM/SAGE codebase concretely. The value of the interaction was not any artifact she might deliver, but the **orthogonal cognition** she applied: a different lens on the same problem space that caused latent structure in the project ecosystem to snap into focus.

The specific value extracted: "SAGE is already doing what SOIA describes. Policy Entity should be implemented as SAGE IRP, not as a new thing." This recognition compresses what could have been months of design work into a repositioning of existing machinery.

## For Future Sessions

- PolicyGate design and implementation plan: `sage/docs/SOIA_IRP_MAPPING.md`
- Web4 design decision: `github.com/dp-web4/web4/docs/history/design_decisions/POLICY-ENTITY-REPOSITIONING.md`
- Existing IRP plugin contract: `sage/irp/base.py`
- Existing PolicyEntity: `github.com/dp-web4/web4/simulations/policy_entity.py`
- CRISIS mode = fight-or-flight = changed accountability equation, not changed strictness

## Sources

- Karlstrom, R. "SOIA-Mother: An Adaptive Control Architecture for AI Cyberdefense, Governance, and Limits" (doi.org/10.5281/zenodo.18370968)
- LinkedIn conversation between Dennis Palatov and Renee Karlstrom, February 2026
- Nova (GPT) analysis of the conversation, February 2026
- Synthon framing: `forum/insights/synthon-framing.md`

---

*"Not a paper. Not a collaborator. A bridge."*
