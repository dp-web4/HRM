# Coupling-Coherence Experiment: Implications for Web4 and SAGE

**Date**: 2026-02-22
**Origin**: Synchronism coupling-coherence experiment (response to Andrei's AI challenge)
**Cross-references**:
- Experiment: `github.com/dp-web4/Synchronism/Research/Coupling_Coherence_Experiment.md`
- Results: `github.com/dp-web4/Synchronism/simulations/results/coupling_coherence_analysis.json`
- Site page: `synchronism-site.vercel.app/coupling-experiment`

---

## What The Experiment Found

900 simulation runs of 5 Bayesian agents discovering a random knowledge graph with controllable coupling (p = probability of belief sharing per agent-pair per round). Key results:

| Finding | Value | Implication |
|---------|-------|-------------|
| Sigmoid transition | C: 0.34 → 0.94 | Coherence emerges via phase transition, not gradual accumulation |
| Transition threshold | p ≈ 0.002 | Even 0.2% coupling per round triggers the transition |
| p ≈ 0.01 effect | C jumps from 0.34 to 0.46 | 1% coupling = 35% coherence gain |
| Best model | Hill function (cooperative binding) | Power-law kinetics, not logarithmic saturation |
| Convergence tracks correctness | Max gap 0.128 | No "shared wrongness" with oracle validation |
| Derived p_crit | Fails (400× error) | Critical threshold is empirical, not derivable from system properties alone |

---

## Connection to Synthon Formation

The coupling-coherence experiment is a **synthon formation study in miniature**. The sigmoid transition is the point where independent agents stop being separable components and start behaving as a coherent collective entity.

The synthon framing (`forum/insights/synthon-framing.md`) predicted exactly the signatures we observed:

| Synthon Signature | Experimental Observable |
|-------------------|------------------------|
| **Prediction error collapse** | C_corr rises sharply: agents' models converge toward ground truth |
| **Information flow asymmetry** | Not measured in this experiment — opportunity for follow-up |
| **Behavioral irreducibility** | Collective coherence (C=0.94) far exceeds any individual agent's capability (~0.55) |
| **Coherence persistence under perturbation** | Stable across 20 random worlds per coupling level |

**New insight from the experiment**: Synthon formation happens at much lower coupling than expected. p_crit ≈ 0.002 means a synthon can crystallize with vanishingly rare inter-component communication. You don't need a dense pheromone field — you need a sparse but present one.

---

## Implications for Web4

### 1. Sparse Witnessing Is Sufficient

The four Web4 relationship mechanisms (Binding → Broadcast → Witnessing → Pairing) form a coupling hierarchy. The experiment says the most impactful transition happens at very low coupling:

```
p = 0.00 (Binding only, no communication)  → C = 0.34
p = 0.01 (Rare Witnessing/Broadcast)       → C = 0.46  (+35%)
p = 0.05 (Moderate Witnessing)             → C = 0.66  (+94%)
p = 0.20 (Frequent Pairing)                → C = 0.84  (+147%)
p = 1.00 (Full transparency)               → C = 0.94  (+177%)
```

**Design implication**: Web4's trust architecture doesn't need constant full transparency between entities. Periodic witnessing events — even rare ones — are enough to trigger collective coherence. The system should optimize for **reliable sparse signals** over **frequent noisy ones**.

This validates the existing design: Broadcast (ephemeral, public, no relationship) already provides the lowest-cost coupling. The experiment says that's where most of the coherence gain lives.

### 2. T3/V3 Tensors as Belief Matrices

The agents' belief matrices B[i][j][t] ∈ [0,1] — representing P(edge (i→j) of type t exists) — are structurally identical to T3/V3 trust tensors:

| Experiment | Web4 |
|-----------|------|
| B[i][j][t] = P(entity i relates to entity j via type t) | T3[entity][role][dimension] = trust score |
| Bayesian update from noisy observations | Trust update from witnessed interactions |
| Weighted belief averaging when coupled | MRH tensor merging across Markov blanket boundaries |
| Uninformative prior (0.5) | New entity enters with neutral trust |

**Insight**: The experiment's Bayesian update mechanism (log-odds update with known noise rate) could inform how T3/V3 tensors should update. Currently trust updates are not formalized to this degree — the experiment provides a concrete, tested update rule.

### 3. Geometric Mean for Trust Validation

The experiment's coherence metric C = √(C_conv × C_corr) — geometric mean of convergence and correctness — solves the "shared wrongness" problem. In Web4 terms:

- **C_conv** = Do entities agree on trust assessments? (inter-entity trust convergence)
- **C_corr** = Are the trust assessments actually right? (validated against ground truth / outcomes)
- **C** = Do entities agree AND are they right? (genuine trust coherence)

**Design implication**: Web4 should track both dimensions. High inter-entity trust convergence with low outcome validation = "trust bubble" (everyone trusts the same entities, but those entities perform poorly). The geometric mean catches this.

### 4. MRH Broadcasts Are the Coupling Mechanism

When agents "share beliefs" in the experiment (with probability p), this maps directly to MRH broadcasts in Web4:

- An MRH broadcast = one entity sharing its trust tensor across its Markov blanket boundary
- The coupling parameter p = frequency/probability of these broadcasts
- The weighted averaging (α=0.7 self-weight) = how much an entity trusts its own assessment vs others'

The self-weight parameter (0.7 in the experiment) is itself a trust parameter — how much an entity's own experience counts vs received witness reports. This could be tied to the entity's own T3 scores: higher Training dimension → higher self-weight.

---

## Implications for SAGE

### 1. IRP Plugins as Agents

SAGE's IRP plugins are structurally equivalent to the experiment's agents:

| Experiment | SAGE |
|-----------|------|
| Agent | IRP plugin (Vision, Language, Memory, etc.) |
| Agent's belief matrix | Plugin's internal state |
| Observation (noisy, partial) | Plugin's sensor input |
| Bayesian update | Plugin's `step()` function (iterative refinement) |
| Coupling (belief sharing) | Inter-plugin communication via shared latent spaces |
| Coherence measurement | Energy convergence across plugins |

**Design implication**: SAGE doesn't need all plugins communicating with each other all the time. The experiment says that even very sparse inter-plugin coordination (p ≈ 0.01) dramatically improves collective inference. This validates SAGE's approach of selective attention — the ATP budget allocation already implements variable coupling by deciding which plugins get resources to communicate.

### 2. Hill Function and the Metabolic Metaphor

The Hill function winning over tanh is thematically significant for SAGE. The Hill equation comes from **cooperative binding kinetics** (Hill, 1910) — it describes how substrate molecules bind to enzymes with cooperative effects:

```
Response = [S]^k / ([S]^k + K_half^k)
```

This is the same math that describes oxygen binding to hemoglobin, enzyme kinetics, and neural receptor activation — all metabolic processes. SAGE already uses the metabolic metaphor (ATP/ADP, WAKE/FOCUS/REST/DREAM/CRISIS states). The experiment empirically confirms that the metabolic functional form (Hill) better describes collective intelligence emergence than the physics form (tanh).

**Insight**: If coherence emergence follows cooperative binding kinetics, then SAGE's metabolic states might correspond to different points on the Hill curve:

| State | Coupling Regime | Hill Region |
|-------|----------------|-------------|
| REST | Low (p < 0.01) | Below K_half — minimal cooperative binding |
| WAKE | Moderate (p ≈ 0.05) | Near K_half — transition zone |
| FOCUS | High (p ≈ 0.2) | Above K_half — saturating cooperation |
| DREAM | Variable | Exploring different coupling regimes |
| CRISIS | Maximum | Full coupling override — all resources shared |

### 3. The Observation Budget Principle

The experiment was carefully designed so individual agents CANNOT learn the world alone (640 observations for 396 edge positions ≈ 1.6 per position), but collectively they CAN (3200 observations ≈ 8 per position).

This is exactly SAGE's pitch: **a single local LLM is insufficient for situational awareness, but an orchestrated system of specialized components succeeds**. The experiment provides quantitative backing:

- Individual capability: ~55% correctness (barely above chance for the graph structure)
- Collective capability at full coupling: ~88% correctness
- The gap (55% → 88%) is what orchestration buys you

### 4. Derived Threshold Failure = Empirical Tuning Required

The p_crit derivation failing (predicted 0.82, actual 0.002) means you can't predict the optimal coupling threshold from system properties alone. For SAGE, this means:

- ATP budget allocation cannot be derived from first principles
- The optimal inter-plugin communication rate must be learned empirically
- SAGE's trust-based allocation (give more ATP to plugins that perform well) is the right approach — it's learning p_crit through experience rather than trying to derive it

---

## The Synthon Lifecycle and These Results

The synthon framing describes three phases: Formation, Health, Decay. The experiment directly informs all three:

### Formation
The sigmoid transition IS synthon formation. The key finding: **formation happens at much lower coupling than expected**, and follows cooperative binding kinetics (Hill function). Substrate engineers don't need to provide dense coupling — they need to provide reliable minimum coupling. The pheromone field doesn't need to be strong. It needs to be present.

### Health
Coherence C = √(conv × corr) is a synthon health metric. The experiment shows it's measurable and that high coupling (p > 0.2) produces stable, high-coherence synthons. The metric catches "trust bubbles" (high convergence, low correctness) that would indicate a degenerate synthon.

### Decay
Not directly tested, but the data implies: if coupling drops below p ≈ 0.01, coherence drops rapidly (C from 0.46 back toward 0.34). This maps to the synthon decay signature of "rising prediction error divergence" — agents' beliefs drift apart when coupling is removed. **The decay threshold is as sharp as the formation threshold.**

---

## Actionable Items

1. **Web4**: Consider formalizing trust update rules using Bayesian log-odds with configurable noise rate, informed by the experiment's tested update mechanism
2. **Web4**: Add a "trust coherence" metric that's the geometric mean of inter-entity convergence and outcome-validated correctness
3. **SAGE**: Use the Hill function form (not tanh) for modeling ATP budget returns — the metabolic metaphor is empirically justified
4. **SAGE**: Design inter-plugin communication to be sparse but reliable — p ≈ 0.01 is enough, but p = 0 is a cliff
5. **Synchronism**: The Hill function winning should be back-annotated to the synthon framing — cooperative binding kinetics describes synthon formation better than logarithmic saturation
6. **Cross-project**: The "observation budget principle" (individual insufficient, collective sufficient) should be added to SAGE's pitch materials, backed by the quantitative results

---

## Sources

- Coupling-Coherence Experiment: `github.com/dp-web4/Synchronism/Research/Coupling_Coherence_Experiment.md`
- Synthon Framing: `github.com/dp-web4/HRM/forum/insights/synthon-framing.md`
- Web4 Binding/Pairing/Witnessing/Broadcast: `web4/docs/why/binding_pairing_witnessing_broadcast.md`
- SAGE System Understanding: `HRM/sage/docs/SYSTEM_UNDERSTANDING.md`
- Hill, A.V. (1910). "The possible effects of the aggregation of the molecules of haemoglobin on its dissociation curves." *J Physiol* 40: iv–vii.
