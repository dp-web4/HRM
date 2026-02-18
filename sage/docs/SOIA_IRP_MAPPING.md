# SOIA-SAGE-Web4 Structural Mapping

**Status**: Design Document v1.0
**Date**: 2026-02-18
**Author**: Dennis + Claude (Opus 4.6)
**Origin**: LinkedIn conversation with Renée Karlström (SOIA researcher)

---

## 1. SOIA Architecture Summary

SOIA (Self-Optimizing Intelligence Architecture) is a framework proposed by Renée Karlström for giving intelligent systems internal structure that holds together over time. It is not a new model or a safety filter -- it is an architectural framework designed to make coherence, continuity, and stability the natural outcome of optimization.

SOIA has three core mechanisms:

### SRC (Self-Referential Core)
A continuous regulatory loop. Evaluates each inference step in relation to the system's ongoing trajectory, detecting drift, instability, or incoherent transitions. Does not assess truth, intent, or meaning -- operates purely at the level of structural compatibility, providing a priority signal that orients inference toward coherence.

### MTM (Memory Transductive Module)
Replaces passive retrieval with selective reinjection. Evaluates past traces according to their entropy, frequency of activation, and coherence with the present state. Only elements that reinforce the current trajectory are reintroduced. Memory is not a record of the past but a constraint on the future.

### MORIA (Internal Temporal Axis)
Time as an active structural variable, not implicit or external. Each new state is evaluated as a position along a continuous trajectory. Enables persistence, trajectory inertia, and resistance to reset-based drift. Local solutions that appear optimal in the short term but would degrade long-term coherence are naturally disfavored.

**Key properties of SOIA**:
- Alignment reframed as geometric and temporal, not psychological or normative
- Stability becomes the path of least resistance within internal architecture
- Sycophancy replaced by phase resonance (more selective under pressure, not more compliant)
- Reward hacking resisted because coherence is a distributed constraint, not a scalar target

**Reference**: Karlström, R. "SOIA-Mother: An Adaptive Control Architecture for AI Cyberdefense, Governance, and Limits" (doi.org/10.5281/zenodo.18370968)

---

## 2. The Mapping

| SOIA Component | SOIA Function | SAGE Equivalent | Web4 Surface |
|---|---|---|---|
| **SRC** (Self-Referential Core) | Continuous regulatory loop evaluating each step against ongoing trajectory | `sage_consciousness.py` consciousness loop + `metabolic_states.py` (5 states: WAKE/FOCUS/REST/DREAM/CRISIS) + `metabolic_controller.py` state transitions | `PolicyEntity.evaluate()` cycle + `PolicyRegistry.witness_decision()` witnessing chain |
| **MTM** (Memory Transductive Module) | Selective memory reinjection based on coherence with present state | `experience_collector.py` SNARC 5D salience scoring (Surprise, Novelty, Arousal, Reward, Conflict) + collapse prevention + circular buffer (x-from-last) | Trust history accumulation via `PolicyRegistry.witness_*` bidirectional witnessing + R6 audit trail |
| **MORIA** (Internal Temporal Axis) | Each state evaluated as position along continuous trajectory | `dream_consolidation.py` sleep cycle LoRA training + `MetabolicTransition[]` state transition history + trust weight evolution over time | `PolicyStore` hash-chain versioned policy history + `PolicyEntity.content_hash` immutability + R6 lifecycle audit records |

### Expanded Correspondence Table

| SOIA Concept | SAGE Implementation | Location |
|---|---|---|
| "Internal coherence attractor" | Metabolic state equilibrium + trust weight convergence | `metabolic_states.py`, `orchestrator.py` |
| "Exo-centered" (defined relative to human/mandate/environment) | Not yet implemented -- **this is the PolicyGate integration point** | `sage/irp/policy_gate.py` (planned) |
| "Memory gate" | SNARC salience threshold (0.5) + collapse detection (0.85 similarity) | `experience_collector.py` |
| "Tool gate" | ATP budget allocation -- plugins only execute if budgeted | `orchestrator.py` |
| "Delegation gate" | Trust-weighted plugin selection -- low-trust plugins get less ATP | `orchestrator.py` |
| "Policy ladder" | Metabolic states as graduated response levels | `metabolic_states.py` |
| "Safe-default doctrine" | Freeze irreversible actions under uncertainty (design constraint) | Not yet operationalized in SAGE |
| "Doubt as first-class state" | REST and DREAM metabolic states reduce processing, favor consolidation | `metabolic_states.py` |
| "Graceful degradation" | ATP depletion triggers REST, consecutive errors trigger CRISIS | `metabolic_states.py` |

---

## 3. Where the Mapping Holds

### 3.1 Rejection of Binary Safety
Both SOIA and SAGE/Web4 treat safety as a continuous, contextual property rather than a binary gate. SOIA's policy ladder maps directly to SAGE's metabolic state progression and Web4's graduated trust thresholds. Neither system has a single "safe/unsafe" boundary.

### 3.2 Trust as Dynamic and Degradable
SOIA treats internal coherence as dynamic -- always being evaluated, always subject to drift. SAGE's trust weights for IRP plugins evolve based on convergence quality (monotonicity ratio, dE variance, convergence rate). Web4's T3/V3 tensors decay over time without reinforcement. All three systems treat trust as something earned and maintained, not granted.

### 3.3 Structural Constraint Over Behavioral Filter
SOIA's central claim: "incoherent transitions carry an internal cost that accumulates over time." This is structurally identical to SAGE's IRP energy function -- every plugin's refinement process drives toward lower energy states, and deviations from monotonic decrease are penalized in trust metrics. Web4's ATP costs for R6 actions serve the same role: structural incentives that make coherent behavior the path of least resistance.

### 3.4 Memory as Constraint on the Future
SOIA's MTM: "Memory is not a record of the past but a constraint on the future." SAGE's SNARC scoring does exactly this -- high-salience experiences are selectively stored not for recall but for future consolidation during DREAM state, where they shape the agent's behavioral priors via LoRA training. The past constrains the future through learned weights, not through replay.

---

## 4. Where the Mapping Breaks

### 4.1 Continuous vs. Discrete Temporal Experience
SOIA's MORIA implies continuous temporal experience -- each state is a position along an unbroken trajectory. SAGE's consciousness loop is continuous during operation, but sleep cycles introduce discrete breaks. The agent "wakes up" after DREAM state with consolidated patterns but without continuous memory of the consolidation process. This is acceptable -- biology has the same structure. The audit trail in R6 provides continuity across sleep boundaries.

### 4.2 Single Regulatory Loop vs. Distributed Plugins
SOIA's SRC is described as a single regulatory loop. SAGE runs multiple IRP plugins in parallel, each with its own energy function and convergence behavior. PolicyGate will be one plugin among many, not the sole regulatory mechanism. This is actually **better** than the SOIA model -- it distributes the regulatory function across multiple independent evaluators with different trust scores, making the system more resilient to any single evaluator's failure.

### 4.3 Internal Anchor vs. External Anchor
SOIA-Mother stabilizes around an internal coherence attractor. Web4 externalizes the stabilizer into a trust fabric built from provenance, lineage, and permissioned authority. These are not competing -- they are complementary layers:
- SOIA/SAGE asks: "Am I coherent with myself?"
- Web4 asks: "Am I coherent with context and mandate?"

PolicyGate is the bridge between these two questions. It sits inside the SAGE consciousness loop (internal) but evaluates against Web4 PolicyEntity rules (external). It is the point where internal coherence meets external accountability.

### 4.4 Bidirectional Transduction vs. Write-Heavy Memory
SOIA's MTM implies bidirectional transduction -- memory actively shapes inference in real-time. SAGE's experience buffer is currently write-heavy: experiences are collected during WAKE/FOCUS and consolidated during DREAM, but the circular buffer provides limited real-time read-back. The policy feedback loop (PolicyGate decisions entering the experience buffer) addresses this gap partially. Full bidirectional transduction would require the experience buffer to actively shape attention allocation during WAKE state -- a future enhancement.

---

## 5. The Repositioning Argument

The argument is not that SAGE should become SOIA, or that SOIA should adopt SAGE's machinery. The argument is:

**SAGE's IRP stack already implements the structural patterns that SOIA describes theoretically.**

The IRP contract (init_state / step / energy / project / halt) is a formalization of iterative refinement toward coherence. Every IRP plugin does what SOIA describes: evaluate each step against an ongoing trajectory, detect instability, and drive toward lower energy states. The only missing piece is an IRP plugin that evaluates actions against *external* coherence criteria -- policy, mandate, context.

**PolicyGate fills this gap.**

It is not a new architecture. It is the same IRP spine with a different energy function:
- Vision plugin energy: distance from visual coherence
- Language plugin energy: distance from linguistic coherence
- Control plugin energy: distance from motor coherence
- **PolicyGate energy: distance from policy compliance**

Same contract. Same convergence behavior. Same trust metrics. Different anchor.

This means Policy Entity doesn't need a new substrate. It needs an IRP plugin whose energy function is `PolicyEntity.evaluate()`.

### 5.1 Fractal Self-Similarity: Plugin of Plugins

The IRP contract is self-similar — it works at every scale of the fractal:

- **Outer loop**: SAGE consciousness runs PolicyGate as one plugin among many
- **Inner loop**: PolicyGate runs an evaluate → refine → converge cycle (rule matching, energy scoring, action filtering) — the same init/step/energy/halt pattern
- **Innermost loop**: When PolicyGate hits an ambiguous WARN case, it can invoke the LLM (Phi-4 Mini advisory) for iterative refinement — which is itself init/step/energy/halt

PolicyEntity is not just wrapped by an IRP plugin — it *is* a specialized SAGE stack. A plugin of plugins. The orchestrator doesn't need to know this. PolicyGate registers like any other plugin, gets its ATP budget, builds trust from convergence quality. The fractal recursion is invisible from the outside, which is exactly how it should be.

This validates the IRP abstraction: if the same contract works at three nested scales (consciousness loop → policy evaluation → LLM advisory), the abstraction is capturing something real about how iterative refinement toward coherence operates.

---

## 6. CRISIS Mode as Accountability Frame

CRISIS is fight-or-flight, operationalized. Both freeze and fight are valid responses. Biology shows neither is universally correct.

**What CRISIS changes**: Not the policy rules. The accountability equation.

Under CRISIS, the entity acknowledges: "the consequences are not in my control, and whether I freeze or fight, what may come will come." The conscience (PolicyGate) still runs. It still evaluates rules. It still records decisions. But the audit record gains a `duress_context` that captures:

- What triggered CRISIS (consecutive errors, high frustration, urgent task)
- Available ATP at decision time
- Whether the agent chose freeze (halt effectors) or fight (proceed with best available action)
- The metabolic state transition log showing how CRISIS was entered

Both freeze and fight are recorded as valid under duress. The accountability frame shifts from "I chose this outcome" to "I responded under duress."

**Biological parallel**: Legal systems recognize "duress" as a defense. It does not mean the action was right. It means the accountability equation changed. The person's conscience did not vanish -- they remember what they did. But the context in which they acted is part of the record.

---

## 7. References

- Karlström, R. "SOIA-Mother: An Adaptive Control Architecture for AI Cyberdefense, Governance, and Limits" (doi.org/10.5281/zenodo.18370968)
- Karlström, R. "SOIA: A Coherent Overview" (shared in LinkedIn conversation, Feb 2026)
- Synthon framing: `forum/insights/synthon-framing.md`
- SAGE system understanding: `sage/docs/SYSTEM_UNDERSTANDING.md`
- IRP architecture analysis: `sage/docs/irp_architecture_analysis.md`
- Web4 policy entity: `github.com/dp-web4/web4/simulations/policy_entity.py`
- Web4 entity types: `github.com/dp-web4/web4/web4-standard/core-spec/entity-types.md`

---

*"Policy Entity doesn't need to be invented. It needs to be repositioned."*
