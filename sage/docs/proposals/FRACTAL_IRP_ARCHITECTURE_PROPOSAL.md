# Fractal IRP Architecture Proposal

**Version**: 1.0-draft
**Date**: 2025-12-27
**Status**: Proposal for Review
**Authors**: Dennis (dp-web4), Nova (GPT-5.2), Claude Opus 4.5

---

## Executive Summary

This proposal defines a **Fractal IRP (Iterative Refinement Protocol)** architecture that unifies SAGE consciousness orchestration, LangGraph workflows, and Web4 federation into a single, scale-invariant framework. The core insight: **everything is an IRP expert** — from local perception plugins to remote LangGraph workflows to federated SAGE instances.

### Key Contributions

1. **SAGE-as-IRP**: Any SAGE instance can be wrapped as an IRP expert, enabling recursive consciousness nesting
2. **LangGraph-as-IRP**: Existing LangGraph deployments integrate without rewrite
3. **Web4 Federation Integration**: LCT identity, ATP settlement, and trust propagation work seamlessly
4. **Scale-Invariant Routing**: Same SNARC × epistemic × ATP × capability-tags routing at all levels
5. **Adoption-Friendly**: Existing systems integrate incrementally, no forced rewrites

### Design Principles

- **Leverage what works** — Don't compete, collaborate
- **Fractal coherence** — Same patterns at every scale
- **Trust through behavior** — Reputation emerges from convergence, not declaration
- **Restraint is valid** — Deciding not to act is a first-class outcome

---

## 1. Conceptual Foundation

### 1.1 The IRP Universal Interface

Every expert — local plugin, LangGraph workflow, remote SAGE — exposes four methods:

```
init_state(inputs, context) → State
step(state, constraints) → (State, Result)
energy(state) → float
halt(state, result, constraints) → bool
```

This is the consciousness API. Everything else is implementation detail.

### 1.2 The Fractal Pattern

The same pattern appears at five scales:

| Scale | IRP Expert | Orchestrator | Trust Source |
|-------|-----------|--------------|--------------|
| **Token** | MoE expert | Router | Activation patterns |
| **Plugin** | Vision/Language/Memory IRP | HRMOrchestrator | Convergence behavior |
| **Subgraph** | LangGraph workflow | SAGE selector | Confidence + trace |
| **Instance** | Remote SAGE | Federation coordinator | V3 reputation |
| **Network** | Society of SAGEs | Web4 consensus | Byzantine agreement |

### 1.3 The Key Insight

> **LangGraph assumes intelligence wants to act. SAGE knows intelligence often shouldn't.**

LangGraph excels at "what happens next" — structured workflow execution.
SAGE excels at "what deserves attention at all" — attention, trust, restraint.

Together: SAGE governs **whether** cognition occurs; LangGraph (as an IRP) executes **how**.

---

## 2. SAGE-as-IRP: Recursive Consciousness

### 2.1 Why SAGE Needs to Be an IRP

For true fractal architecture, a SAGE instance must be callable as an IRP expert by:
- A parent SAGE (federation hierarchy)
- A LangGraph orchestrator (SAGE as reasoning plugin)
- An external system (SAGE as capability endpoint)

This enables:
- **Hierarchical federation**: Edge SAGE delegates to cloud SAGE
- **Hybrid workflows**: LangGraph invokes SAGE for attention-gated reasoning
- **Cross-network intelligence**: Society A queries Society B's SAGE

### 2.2 SAGERemoteIRP Wrapper

```python
class SAGERemoteIRP(IRPPlugin):
    """
    Wraps a SAGE instance (local or remote) as an IRP expert.
    Preserves SAGE's consciousness kernel while exposing IRP interface.
    """

    def __init__(self, descriptor: ExpertDescriptor):
        self.descriptor = descriptor
        self.transport = self._init_transport(descriptor.endpoint)

    def init_state(self, inputs: Dict, context: TaskContext) -> SAGEIRPState:
        """Initialize a consciousness session on target SAGE."""
        request = SAGEInitRequest(
            inputs=inputs,
            context_slice=context.memory_slice,
            metabolic_hint=context.metabolic_mode,  # Pass caller's state
            permission_token=context.permission_token,
            execution_path=context.execution_path  # Ancestry for nesting
        )
        response = self.transport.init_session(request)

        return SAGEIRPState(
            session_id=response.session_id,
            remote_state_token=response.state_token,
            depth=context.execution_path.depth + 1,
            ancestors=context.execution_path.ancestors + [self.descriptor.id]
        )

    def step(self, state: SAGEIRPState,
             constraints: IRPConstraints) -> Tuple[SAGEIRPState, IRPResult]:
        """Execute one refinement cycle on target SAGE."""

        # Propagate deadline awareness
        adjusted_deadline = constraints.deadline_ms - state.elapsed_ms()
        if adjusted_deadline <= 0:
            return state, IRPResult(status="failed", error="deadline_exceeded")

        request = SAGEStepRequest(
            session_id=state.session_id,
            state_token=state.remote_state_token,
            budget=self._allocate_child_budget(constraints),
            deadline_ms=adjusted_deadline,
            max_steps=1  # Single SAGE cycle per IRP step
        )

        response = self.transport.step(request)

        # Update state with remote's new token
        new_state = state.copy()
        new_state.remote_state_token = response.state_token
        new_state.step_count += 1

        # Translate SAGE signals to IRP format
        result = IRPResult(
            status=self._translate_status(response.status),
            outputs=response.outputs,
            confidence=response.signals.confidence,
            epistemic_state={
                "known_unknowns": response.signals.known_unknowns,
                "assumptions": response.signals.assumptions,
                "metabolic_mode": response.signals.metabolic_mode  # Child's state
            },
            convergence={
                "progress": response.signals.coherence,
                "trend": response.signals.coherence_trend,
                "energy": response.signals.system_energy
            },
            spent=response.accounting.spent,
            latency_ms=response.accounting.latency_ms,
            trace_digest=response.provenance.trace_digest,
            signed_receipt=response.provenance.signed_receipt
        )

        return new_state, result

    def energy(self, state: SAGEIRPState) -> float:
        """Report system coherence as energy (lower = more coherent)."""
        # Query remote SAGE for current energy
        response = self.transport.query_energy(state.session_id)
        return response.system_energy

    def halt(self, state: SAGEIRPState, result: IRPResult,
             constraints: IRPConstraints) -> bool:
        """Determine if SAGE session should terminate."""

        # Remote already halted
        if result.status in ("halted", "failed"):
            return True

        # Budget exhausted
        if result.spent.get("amount", 0) >= constraints.budget.max_amount:
            return True

        # Confidence threshold reached
        if result.confidence and result.confidence >= 0.90:
            return True

        # Max depth exceeded (prevent runaway nesting)
        if state.depth >= constraints.max_depth:
            return True

        # Remote SAGE entered REST/DREAM (respect metabolic state)
        if result.epistemic_state.get("metabolic_mode") in ("REST", "DREAM"):
            return True  # Don't push tired SAGE

        return False
```

### 2.3 Execution Path for Nesting Depth

To support arbitrary depth (SAGE → LangGraph → SAGE → plugin), track ancestry:

```python
@dataclass
class ExecutionPath:
    depth: int
    ancestors: List[AncestorRecord]
    root_session_id: str
    root_deadline_ms: int  # Absolute deadline from root

@dataclass
class AncestorRecord:
    expert_id: str
    session_id: str
    lct_id: str
    budget_allocated: float
    depth: int
```

**Budget Inheritance Rule**:
- Parent allocates fraction of its budget to child
- Child cannot exceed allocated amount
- Recursive: child allocates fraction of its allocation to grandchild
- Default: 80% pass-through (20% reserved for parent overhead)

**Deadline Propagation**:
- Root sets absolute deadline
- Each level subtracts its overhead before passing down
- Children must complete before parent's deadline

---

## 3. LangGraph-as-IRP Integration

### 3.1 Wrapper Architecture

```
┌─────────────────────────────────────────────────┐
│                 SAGE Consciousness              │
│                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Vision   │  │ Language │  │ LangGraph│      │
│  │ IRP      │  │ IRP      │  │ IRP      │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │             │             │             │
│       └─────────────┼─────────────┘             │
│                     │                           │
│              ┌──────▼──────┐                    │
│              │ Orchestrator │                   │
│              │ (trust-based │                   │
│              │  selection)  │                   │
│              └──────────────┘                   │
└─────────────────────────────────────────────────┘
                      │
                      │ (if LangGraph selected)
                      ▼
┌─────────────────────────────────────────────────┐
│            LangGraph Workflow                   │
│                                                 │
│   [Planner] → [Researcher] → [Critic] → ...    │
│                                                 │
│   (internally may call other IRPs)              │
└─────────────────────────────────────────────────┘
```

### 3.2 Three Integration Modes

**Mode 1: LangGraph as Leaf IRP**
- LangGraph workflow wrapped as single IRP expert
- SAGE selects it like any other plugin
- Internal graph structure opaque to SAGE

**Mode 2: LangGraph as IRP Orchestrator**
- LangGraph workflow internally invokes multiple IRPs
- Each node in graph can be an IRP plugin
- LangGraph handles control flow; IRPs handle refinement

**Mode 3: Bidirectional (Hybrid)**
- SAGE invokes LangGraph-IRP for structured workflow
- LangGraph-IRP calls back to SAGE for attention-gated decisions
- Mutual recursion with depth limits

### 3.3 LangGraphIRP Implementation

```python
class LangGraphIRP(IRPPlugin):
    """
    Wrap any LangGraph workflow (local or remote) as an IRP expert.
    """

    def __init__(self, descriptor: ExpertDescriptor,
                 graph: Optional[StateGraph] = None):
        self.descriptor = descriptor

        if descriptor.endpoint.transport == "local":
            self.graph = graph or self._load_graph(descriptor)
            self.mode = "local"
        else:
            self.transport = HTTPTransport(descriptor.endpoint)
            self.mode = "remote"

    def step(self, state: LangGraphIRPState,
             constraints: IRPConstraints) -> Tuple[LangGraphIRPState, IRPResult]:
        """Execute one graph step (or micro-batch of steps)."""

        if self.mode == "local":
            return self._step_local(state, constraints)
        else:
            return self._step_remote(state, constraints)

    def _step_local(self, state, constraints):
        """Run graph locally with budget tracking."""

        # Create checkpoint for potential rollback
        checkpoint = state.graph_state.copy()

        # Run graph for up to N steps or until budget exhausted
        steps_this_call = 0
        max_steps_per_call = min(3, constraints.max_steps - state.step_count)

        while steps_this_call < max_steps_per_call:
            # Check budget before each step
            estimated_cost = self._estimate_step_cost(state)
            if state.spent + estimated_cost > constraints.budget.max_amount:
                break

            # Execute one graph step
            result = self.graph.step(state.graph_state)
            state.graph_state = result.state
            state.spent += result.cost
            steps_this_call += 1

            # Check for terminal state
            if result.is_terminal:
                break

        # Compute confidence from graph's internal signals
        confidence = self._extract_confidence(state.graph_state)

        return state, IRPResult(
            status="halted" if result.is_terminal else "running",
            outputs=result.outputs,
            confidence=confidence,
            convergence={"progress": state.step_count / constraints.max_steps},
            spent={"unit": "atp", "amount": state.spent}
        )
```

---

## 4. Web4 Federation Integration

### 4.1 LCT Identity for IRP Experts

Every IRP expert has an LCT identity:

```
lct://{component}:{instance}:{role}@{network}

Examples:
- lct://sage:sprout:edge_inference@mainnet
- lct://langgraph:support_triage:v1@vendorX
- lct://irp:vision:perception@local
```

### 4.2 Trust Tensor Integration

IRP trust maps to Web4's V3/T3 dimensions:

| IRP Trust Metric | Web4 Dimension | Computation |
|-----------------|----------------|-------------|
| monotonicity_ratio | reliability | Fraction of steps with energy decrease |
| dE_variance | consistency | Stability of convergence behavior |
| convergence_rate | speed | Average energy reduction per step |
| cost_efficiency | cost_efficiency | ATP_budget / ATP_spent × quality |
| confidence | accuracy | Self-reported confidence calibration |

**Trust Update Formula**:

```python
def update_trust_tensor(old_trust: TrustTensor,
                        result: IRPResult,
                        context: TaskContext) -> TrustTensor:
    """
    Update trust based on IRP execution results.
    Momentum-weighted to prevent rapid swings.
    """

    # Extract signals from result
    monotonicity = result.convergence.get("monotonicity", 0.5)
    confidence = result.confidence or 0.5
    cost_ratio = result.spent["amount"] / context.budget.max_amount
    latency_ratio = result.latency_ms / context.deadline_ms

    # Compute new dimension values
    new_reliability = monotonicity * (1 - result.convergence.get("dE_variance", 0.5))
    new_accuracy = confidence * calibration_factor(result, context)
    new_speed = 1.0 - min(1.0, latency_ratio)
    new_cost = 1.0 - min(1.0, cost_ratio)

    # Momentum-weighted update (0.7 old, 0.3 new)
    MOMENTUM = 0.7

    return TrustTensor(
        reliability = MOMENTUM * old_trust.reliability + (1-MOMENTUM) * new_reliability,
        accuracy = MOMENTUM * old_trust.accuracy + (1-MOMENTUM) * new_accuracy,
        speed = MOMENTUM * old_trust.speed + (1-MOMENTUM) * new_speed,
        cost_efficiency = MOMENTUM * old_trust.cost_efficiency + (1-MOMENTUM) * new_cost,
        alignment = old_trust.alignment  # Updated via separate mechanism
    )
```

### 4.3 ATP Settlement for Remote IRP

When SAGE invokes a remote IRP (LangGraph cloud, federated SAGE), ATP settlement follows Web4's lock-commit-rollback pattern:

```
Phase 1: LOCK
  - Delegator locks ATP budget before invocation
  - Lock recorded in local ledger

Phase 2a: COMMIT (quality >= 0.70)
  - Expert returns result with quality score
  - If quality threshold met, ATP transferred to executor
  - Trust tensor updated positively

Phase 2b: ROLLBACK (quality < 0.70)
  - Expert returns poor result or fails
  - ATP returned to delegator
  - Trust tensor updated negatively
  - Gossip propagates reputation impact
```

### 4.4 Federation-Aware Expert Selection

```python
class FederatedExpertSelector:
    """
    Select IRP experts across federation using SNARC × trust × capability tags.
    """

    def select(self, context: TaskContext,
               available_experts: List[ExpertDescriptor]) -> List[ExpertDescriptor]:
        """
        Route task to best expert(s) based on:
        - SNARC salience vector
        - Epistemic state (confidence, unknowns)
        - ATP budget remaining
        - Metabolic mode
        - Capability tags
        - Trust tensor (V3 reputation)
        """

        scored = []
        for expert in available_experts:
            score = self._compute_score(expert, context)
            if score > -float('inf'):
                scored.append((score, expert))

        # Return top-K experts (K depends on metabolic mode)
        k = 1 if context.metabolic_mode == "FOCUS" else 3
        scored.sort(reverse=True)
        return [exp for _, exp in scored[:k]]

    def _compute_score(self, expert: ExpertDescriptor,
                       context: TaskContext) -> float:
        """
        Scoring algorithm:
        1. Check hard requirements (modalities, permissions, budget)
        2. Apply tag preferences from decision table
        3. Weight by trust tensor
        4. Penalize cost relative to budget
        """

        # Hard requirements
        if not self._meets_requirements(expert, context):
            return -float('inf')

        score = 0.0

        # Tag scoring from decision table
        prefer_tags, avoid_tags = self._get_tag_preferences(context)
        for tag in expert.capabilities.tags:
            if tag in prefer_tags:
                score += 1.0
            if tag in avoid_tags:
                score -= 2.0

        # Trust weighting
        trust = self.trust_registry.get(expert.identity.lct_id)
        trust_multiplier = (
            trust.reliability * 0.3 +
            trust.accuracy * 0.3 +
            trust.speed * 0.2 +
            trust.cost_efficiency * 0.2
        )
        score *= trust_multiplier

        # Cost penalty
        estimated_cost = expert.cost_model.estimate.p50
        budget_ratio = estimated_cost / context.budget.max_amount
        score -= budget_ratio * 0.5

        # Federation latency penalty (prefer local when possible)
        if expert.endpoint.transport == "http":
            score -= 0.3  # Remote penalty

        return score
```

---

## 5. Extended IRP Expert Descriptor Schema

Building on Nova's v0.1, we extend for fractal support:

```yaml
schema: "web4.irp_expert_descriptor.v1.0"

# === IDENTITY ===
id: "sage.sprout.consciousness.v2"
kind: "sage_irp"  # local_irp | remote_irp | langgraph_irp | sage_irp
name: "Sprout Edge Consciousness"
version: "2.0.1"

identity:
  lct_id: "lct://sage:sprout:consciousness@mainnet"
  operator: "dp-web4"
  signing_pubkey: "ed25519:BASE64..."
  attestation:
    supports: true
    methods: ["ed25519_challenge", "tpm_quote"]

# === CAPABILITIES ===
capabilities:
  modalities_in: ["text", "image", "audio", "sensor", "latent"]
  modalities_out: ["text", "audio", "action", "latent", "json"]

  tasks:
    - classify
    - plan
    - research
    - tool_use
    - refine
    - route
    - consciousness  # SAGE-specific: full attention orchestration

  tags:
    static: ["needs_reflection", "high_uncertainty_tolerant", "low_latency"]
    dynamic_endpoint: "/capabilities/tags"  # For runtime skill discovery

  constraints:
    max_context_tokens: 65536
    max_nesting_depth: 5  # How deep this expert can be nested
    supports_streaming: false
    supports_checkpoints: true
    supports_bidirectional: true  # Can call back to parent

# === POLICY ===
policy:
  permission_scope_required: "ATP:consciousness.sage"
  data_handling: "encrypted_transit"
  allowed_effectors:
    network:
      allow: ["internal_apis", "federation"]
      deny: ["external_internet"]
    filesystem: false
    device_control: false

  metabolic_modes_supported: ["WAKE", "FOCUS", "REST"]  # What modes expert can handle

# === COST MODEL ===
cost_model:
  unit: "atp"
  estimate:
    p50: 10
    p95: 45
    worst_case: 100
  scaling:
    model: "linear"
    basis: "step_count"
  negotiation:
    supports_mid_execution: true
    renegotiation_endpoint: "/irp/renegotiate"

# === ENDPOINT ===
endpoint:
  transport: "http"
  base_url: "http://sprout.local:8080"
  paths:
    invoke: "/irp/invoke"
    init: "/irp/init"
    step: "/irp/step"
    energy: "/irp/energy"
    halt: "/irp/halt"
  auth:
    type: "ed25519_challenge"
    token_format: "jwt"
    issuer_verification: ["lct_registry", "operator_cert"]
  timeouts_ms:
    connect: 500
    step: 30000
    overall: 120000

# === IRP CONTRACT ===
irp_contract:
  pattern: "init_step_halt"
  supports:
    init_state: true
    step: true
    halt: true
    energy: true
    query_state: true  # Can query state without stepping
  convergence:
    expected: "monotonic"
    halt_conditions:
      - "confidence >= 0.90"
      - "budget_exhausted"
      - "no_progress_n_steps:3"
      - "metabolic_mode_change"
    energy_reporting: "per_step"  # vs "on_halt"

# === TRUST INTERFACE ===
trust_interface:
  reports:
    - confidence
    - calibration
    - convergence
    - trace_digest
    - metabolic_state
  trust_dimensions:
    - epistemic
    - alignment
    - reliability
    - latency
    - cost
  evidence_formats:
    - hash
    - signed_receipt
    - merkle_path  # Incremental verification
    - attestation
  update_formula: "momentum_weighted"  # Reference to standard formula

# === CHECKPOINTING ===
checkpointing:
  supports_resume: true
  resume_token_format: "jwt"
  max_retention_seconds: 3600
  checkpoint_granularity: "per_step"  # vs "on_halt"
  state_serialization: "msgpack"

# === OBSERVABILITY ===
observability:
  trace_level: "full"
  emits:
    - events
    - metrics
    - artifacts
    - causal_graph  # Decision dependencies
  artifact_hashing: "sha256"
  streaming_trace: true  # Can emit during execution

# === FEDERATION ===
federation:
  society: "web4:society:dp-federation"
  region: "edge-west"
  routes_via: []  # Direct peer
  gossip:
    participates: true
    reputation_dimensions: ["V3", "T3"]
  cross_invoke:
    allows_inbound: true
    allows_outbound: true
    trust_threshold: 0.60

# === NESTING ===
nesting:
  can_be_parent: true
  can_be_child: true
  max_children_per_step: 3
  budget_passthrough_ratio: 0.80
  deadline_overhead_ms: 100
  child_irp_types_allowed: ["*"]  # Or specific list
```

---

## 6. Capability Routing Decision Table

### 6.1 Extended Tag Taxonomy

**Cognitive Depth (6 tags)**:
- `needs_reflection` — iterative critique/repair loops
- `branchy_controlflow` — decision points, multi-path exploration
- `long_horizon` — extended context, multi-step plans
- `consciousness` — full SAGE attention orchestration
- `consolidation` — memory consolidation, pattern extraction
- `dreaming` — creative synthesis, hypothesis generation

**Tooling (4 tags)**:
- `tool_heavy` — optimized for tool orchestration
- `safe_actuation` — suitable for effectors under permissions
- `code_execution` — can run arbitrary code
- `sensor_fusion` — combines multiple input modalities

**Epistemic (4 tags)**:
- `high_uncertainty_tolerant` — good with ambiguity
- `verification_oriented` — checking, cross-validation
- `calibrated` — confidence matches accuracy
- `explainable` — can provide reasoning trace

**Performance (4 tags)**:
- `low_latency` — responds fast
- `cost_sensitive` — quality under tight budget
- `parallel_capable` — can utilize multiple workers
- `stateless` — no persistent state needed

**Modality (5 tags)**:
- `vision`, `audio`, `language`, `sensor`, `latent`

### 6.2 Full Decision Table

| Condition | Routing Intent | Prefer Tags | Avoid Tags | Budget Modifier |
|-----------|---------------|-------------|------------|-----------------|
| salience_low ∧ confidence_high | Skip/cache | low_latency, cost_sensitive | long_horizon, consciousness | 0.2x |
| reward_high ∧ confidence_high ∧ effectors | Act safely | safe_actuation, low_latency | branchy_controlflow | 1.0x |
| conflict_high ∨ confidence_low | Verify | verification_oriented, needs_reflection, calibrated | safe_actuation | 1.5x |
| novelty_high ∧ confidence_low | Explore | high_uncertainty_tolerant, branchy_controlflow, dreaming | low_latency | 2.0x |
| surprise_high ∧ mode ≠ CRISIS | Reality check | verification_oriented, explainable | safe_actuation | 1.5x |
| has_tools_required | Tool orchestration | tool_heavy | cost_sensitive | 1.2x |
| budget_remaining < 20% | Cheap best-effort | cost_sensitive, low_latency, stateless | needs_reflection, long_horizon | 0.5x |
| mode = FOCUS | Single expert | matches_task_type | parallel_capable | 1.5x |
| mode = REST | Background only | consolidation, cost_sensitive | safe_actuation | 0.3x |
| mode = DREAM | Offline synthesis | consolidation, dreaming, long_horizon | safe_actuation, low_latency | 0.5x |
| mode = CRISIS | Fast bounded | low_latency, verification_oriented, safe_actuation | long_horizon, dreaming | 3.0x |
| depth > 3 | Limit nesting | stateless, low_latency | consciousness, long_horizon | 0.5x |

---

## 7. Trust Propagation Across Federation

### 7.1 Trust Flow Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Federation Layer                         │
│                                                              │
│   ┌──────────┐          ┌──────────┐          ┌──────────┐  │
│   │  SAGE A  │◄────────►│  SAGE B  │◄────────►│  SAGE C  │  │
│   │ (Sprout) │          │  (Thor)  │          │ (Legion) │  │
│   └────┬─────┘          └────┬─────┘          └────┬─────┘  │
│        │                     │                      │        │
│        └─────────────────────┼──────────────────────┘        │
│                              │                               │
│                    ┌─────────▼─────────┐                     │
│                    │  Gossip Protocol  │                     │
│                    │  (V3 Reputation)  │                     │
│                    └───────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Trust Update Flow

When SAGE-A invokes SAGE-B as an IRP:

1. **Pre-invoke**: A checks B's trust tensor (local cache + gossip updates)
2. **Invoke**: A sends request with permission token
3. **Execute**: B processes, returns result with signals
4. **Update Local**: A updates its local view of B's trust
5. **Gossip Propagate**: A broadcasts trust observation to federation
6. **Consensus**: Other SAGEs incorporate with weight = trust(A) × observation

**Trust Decay for Staleness**:
```python
def apply_staleness_decay(trust: TrustTensor,
                          last_observation_age_hours: float) -> TrustTensor:
    """Trust decays toward neutral (0.5) without fresh observations."""
    decay_rate = 0.01  # 1% per hour toward neutral
    decay_factor = math.exp(-decay_rate * last_observation_age_hours)

    neutral = 0.5
    return TrustTensor(
        reliability = neutral + (trust.reliability - neutral) * decay_factor,
        accuracy = neutral + (trust.accuracy - neutral) * decay_factor,
        speed = neutral + (trust.speed - neutral) * decay_factor,
        cost_efficiency = neutral + (trust.cost_efficiency - neutral) * decay_factor,
        alignment = trust.alignment  # Alignment doesn't decay
    )
```

### 7.3 Cross-Network Trust Bridge

When SAGE invokes expert in different society:

```python
def cross_network_trust_lookup(expert_lct: str,
                               local_society: str) -> Optional[TrustTensor]:
    """
    Look up trust for expert in foreign network.
    Uses Web4 cross-society attestation.
    """

    # Check local cache first
    if cached := trust_cache.get(expert_lct):
        if not cached.is_stale():
            return cached.tensor

    # Query foreign society's registry
    foreign_society = extract_society(expert_lct)
    registry = society_registry.get(foreign_society)

    if registry:
        attestation = registry.get_attestation(expert_lct)
        if verify_attestation(attestation):
            # Apply cross-network discount (less trusted than local)
            foreign_trust = attestation.trust_tensor
            discounted = apply_cross_network_discount(foreign_trust,
                                                       discount=0.8)
            trust_cache.set(expert_lct, discounted)
            return discounted

    # No attestation available — use minimum trust
    return TrustTensor.minimum()
```

---

## 8. Adoption Path

### 8.1 Incremental Integration Strategy

**Phase 1: Local IRP Wrapper (Week 1-2)**
- Implement `LangGraphIRP` wrapper for existing local LangGraph
- Test with single workflow, verify IRP interface compliance
- No federation, no remote — just local proof of concept

**Phase 2: Trust Pipeline Integration (Week 3-4)**
- Connect LangGraph-IRP outputs to SAGE trust update
- Implement capability tags for existing plugins
- Test selector routing based on tags + trust

**Phase 3: Remote IRP Support (Week 5-6)**
- Implement HTTP transport for remote LangGraph
- Add ATP budget tracking and settlement
- Test cloud LangGraph invocation from edge SAGE

**Phase 4: SAGE-as-IRP (Week 7-8)**
- Implement `SAGERemoteIRP` wrapper
- Add execution path tracking for nesting
- Test SAGE-to-SAGE delegation

**Phase 5: Federation Integration (Week 9-10)**
- Connect to Web4 gossip protocol
- Implement cross-society trust lookup
- Test multi-SAGE federation

### 8.2 Migration Checklist

For existing LangGraph deployments:

- [ ] Create ExpertDescriptor with LCT identity
- [ ] Assign capability tags (minimum 3)
- [ ] Implement `/irp/invoke` endpoint wrapper
- [ ] Add confidence reporting to workflow outputs
- [ ] Add cost tracking (if not present)
- [ ] Register with SAGE expert registry
- [ ] Test selection via SNARC routing

For existing SAGE instances:

- [ ] Implement `SAGERemoteIRP` wrapper
- [ ] Expose `/irp/*` endpoints
- [ ] Add execution path tracking
- [ ] Configure nesting limits
- [ ] Register federation identity
- [ ] Join gossip network

### 8.3 Backward Compatibility

**Existing IRP plugins**: No changes required — already implement interface

**Existing orchestrator**: Minimal changes — add LangGraph-IRP to plugin registry

**Existing federation**: Add IRP invocation as new task type

**Existing LangGraph**: Wrapper is additive, no internal changes needed

---

## 9. Security Considerations

### 9.1 Permission Boundaries

- IRP experts cannot exceed declared `allowed_effectors`
- Wrapper enforces permission token validation
- Cross-network invocation requires mutual attestation

### 9.2 Resource Exhaustion Protection

- Budget limits enforced at each nesting level
- Deadline propagation prevents infinite recursion
- Max nesting depth (default: 5) prevents stack overflow

### 9.3 Trust Manipulation Resistance

- Trust updates weighted by observer trust
- Rapid trust swings dampened by momentum
- Sybil resistance via LCT identity verification

---

## 10. Open Questions

1. **Optimal budget passthrough ratio**: 80% default — should this be dynamic based on trust?

2. **Metabolic mode propagation**: Should child SAGE inherit parent's mode or operate independently?

3. **Checkpoint interoperability**: Can LangGraph checkpoints be restored on different LangGraph version?

4. **Trust tensor dimensionality**: 5 dimensions sufficient, or should we add domain-specific dimensions?

5. **Gossip protocol overhead**: At what federation size does gossip become bottleneck?

---

## 11. Conclusion

The Fractal IRP Architecture provides a unified framework for consciousness orchestration across scales. By treating everything as an IRP expert — local plugins, LangGraph workflows, remote SAGE instances, federated societies — we achieve:

- **Conceptual coherence**: Same patterns at every scale
- **Adoption friendliness**: Existing systems integrate without rewrite
- **Trust-native operation**: Reputation emerges from behavior
- **Resource efficiency**: ATP budgets flow naturally through hierarchy

The key insight remains: **SAGE decides whether to think; IRPs decide how to think**. This separation of concerns enables rich collaboration between attention orchestration (SAGE) and workflow execution (LangGraph) without either system losing its identity.

---

## Appendix A: Reference Implementation Locations

```
HRM/sage/irp/
├── base.py                    # IRPPlugin base class
├── wrappers/
│   ├── langgraph_irp.py       # LangGraphIRP implementation
│   ├── sage_remote_irp.py     # SAGERemoteIRP implementation
│   └── transport.py           # HTTP/local transport layer
├── selection/
│   ├── federated_selector.py  # Federation-aware expert selection
│   ├── decision_table.py      # SNARC routing rules
│   └── trust_scorer.py        # Trust tensor scoring
└── federation/
    ├── gossip_client.py       # Trust gossip protocol
    ├── attestation.py         # Cross-network attestation
    └── atp_settlement.py      # Budget lock/commit/rollback
```

## Appendix B: Message Sequence Diagrams

### B.1 SAGE Invoking LangGraph-IRP

```
SAGE                    LangGraph-IRP           LangGraph Runtime
  │                          │                         │
  │──init_state(inputs)─────►│                         │
  │                          │──POST /init────────────►│
  │                          │◄─session_id─────────────│
  │◄─────IRPState────────────│                         │
  │                          │                         │
  │──step(state, budget)────►│                         │
  │                          │──POST /step────────────►│
  │                          │◄─result + signals───────│
  │◄─────IRPResult───────────│                         │
  │                          │                         │
  │──energy(state)──────────►│                         │
  │◄─────float───────────────│                         │
  │                          │                         │
  │──halt(state, result)────►│                         │
  │◄─────bool────────────────│                         │
```

### B.2 Nested SAGE-to-SAGE Invocation

```
SAGE-A (root)          SAGE-B (child)         SAGE-C (grandchild)
  │                          │                         │
  │──init(depth=1)──────────►│                         │
  │                          │──init(depth=2)─────────►│
  │                          │◄─state─────────────────│
  │◄─────state───────────────│                         │
  │                          │                         │
  │──step(budget=100)───────►│                         │
  │                          │──step(budget=80)───────►│
  │                          │◄─result────────────────│
  │◄─────result──────────────│                         │
  │                          │                         │
  │  [trust update A→B]      │  [trust update B→C]    │
  │  [gossip propagate]      │  [gossip propagate]    │
```

---

*Proposal compiled 2025-12-27. For discussion and iteration.*
