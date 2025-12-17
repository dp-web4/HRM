# Trust Tensor Cross-Project Integration Design

**Date**: 2025-12-17
**Context**: Legion Session 62+ Autonomous Research
**Purpose**: Design bidirectional trust synchronization across ACT blockchain, SAGE neural systems, and Web4 protocol

---

## Executive Summary

ACT and SAGE implement **parallel but architecturally distinct trust systems**:

- **ACT (Blockchain)**: T3 relationship tensors (Talent, Training, Temperament) + V3 value tensors (Valuation, Veracity, Validity)
- **SAGE (Edge)**: Expert reputation with context-specific trust + performance metrics
- **Web4 (Protocol)**: Trust tensor canonical format with observer tracking

**Current State**:
- ✅ SAGE ↔ Web4 bidirectional sync (implemented, Session 61)
- ❌ ACT ↔ Web4 integration (missing blockchain read/write)
- ❌ T3 ↔ SAGE trust semantic mapping (undefined)

**Goal**: Create unified trust system where blockchain consensus, edge performance, and protocol standards flow seamlessly.

---

## Problem Statement

### Current Integration Gaps

**Gap 1: Semantic Mismatch**
- ACT decomposes trust into 3 dimensions (T3: Talent, Training, Temperament)
- SAGE treats trust as single emergent score learned from performance
- **Problem**: No defined mapping between T3 components and SAGE metrics

**Gap 2: Missing Blockchain Bridge**
- SAGE TrustTensorSync exports to Web4TrustClient (JSON storage)
- ACT has trusttensor keeper (blockchain storage)
- **Problem**: No path from Web4TrustClient to ACT blockchain

**Gap 3: Context Model Mismatch**
- ACT uses global context modifier (1.0x-1.2x multiplicative adjustment)
- SAGE stores fully isolated per-context trust scores
- **Problem**: Different conceptual models for context handling

**Gap 4: Multi-Observer Consensus**
- ACT has no explicit observer (blockchain is implicit)
- SAGE TrustTensor tracks explicit observers with confidence weighting
- **Problem**: How to reconcile different evaluators' views?

---

## Architecture Overview

### Three-Layer Trust System

```
┌─────────────────────────────────────────────────────┐
│ Layer 1: BLOCKCHAIN CONSENSUS (ACT)                │
│                                                     │
│ RelationshipTensor (T3)                            │
│  ├─ Talent (0.3 weight)                            │
│  ├─ Training (0.4 weight)                          │
│  └─ Temperament (0.3 weight)                       │
│  → Composite = weighted sum × ContextModifier      │
│                                                     │
│ ValueTensor (V3)                                   │
│  ├─ Valuation (0.4 weight)                         │
│  ├─ Veracity (0.3 weight)                          │
│  └─ Validity (0.3 weight)                          │
│  → Used for operation/transaction evaluation       │
│                                                     │
│ Storage: Blockchain state (immutable consensus)    │
└─────────────────────────────────────────────────────┘
            ↓ [ACT ↔ Web4 Bridge] ↓
┌─────────────────────────────────────────────────────┐
│ Layer 2: PROTOCOL STANDARD (Web4)                  │
│                                                     │
│ TrustTensor (Canonical Format)                     │
│  ├─ observer_id (who observed)                     │
│  ├─ subject_id (who was observed)                  │
│  ├─ context (operational context)                  │
│  ├─ trust_score (0-1, single value)                │
│  ├─ confidence (0-1, certainty)                    │
│  └─ evidence_count (sample size)                   │
│                                                     │
│ Web4TrustClient (Storage Layer)                    │
│  ├─ Multi-observer aggregation                     │
│  ├─ Context filtering                              │
│  └─ Persistence (JSON/Redis/Blockchain)            │
│                                                     │
│ Storage: Distributed (JSON, future blockchain)     │
└─────────────────────────────────────────────────────┘
            ↓ [SAGE ↔ Web4 Sync] ↓
┌─────────────────────────────────────────────────────┐
│ Layer 3: EDGE EXECUTION (SAGE)                     │
│                                                     │
│ ExpertReputation (Performance-Based Trust)         │
│  ├─ context_trust: {context → trust_score}         │
│  ├─ context_observations: {context → count}        │
│  ├─ convergence_rate (learning speed)              │
│  ├─ stability (consistency)                        │
│  ├─ efficiency (quality per cost)                  │
│  └─ average_confidence (router confidence)         │
│                                                     │
│ TrustTensorSync (Bidirectional)                    │
│  ├─ Export: SAGE → Web4                            │
│  ├─ Import: Web4 → SAGE                            │
│  └─ Aggregation: Multi-observer blending           │
│                                                     │
│ Storage: SQLite DB (local, fast queries)           │
└─────────────────────────────────────────────────────┘
```

---

## T3 Tensor Semantic Mapping

### Proposed Mapping: T3 → SAGE Metrics

| ACT T3 Dimension | Weight | SAGE Metric | Interpretation |
|------------------|--------|-------------|----------------|
| **Talent** | 0.3 | `average_confidence` | Natural ability - how confident the router is in this expert |
| **Training** | 0.4 | `convergence_rate` | Learned skill - how quickly expert improves performance |
| **Temperament** | 0.3 | `stability` | Behavioral consistency - how reliable expert is across inputs |

**Composite Formula** (matches ACT):
```python
t3_composite = (
    0.3 * talent_from_confidence +
    0.4 * training_from_convergence +
    0.3 * temperament_from_stability
)
```

**Normalization** (SAGE metrics → [0, 1]):
```python
def sage_metric_to_t3_score(metric_value: float, metric_type: str) -> float:
    """Normalize SAGE metric to T3 score range [0, 1]."""
    if metric_type == "confidence":
        # average_confidence already in [0, 1]
        return min(1.0, max(0.0, metric_value))

    elif metric_type == "convergence_rate":
        # convergence_rate typically [0, 0.5] (loss reduction per step)
        # Map to [0, 1]: 0.5 → 1.0, 0 → 0
        return min(1.0, metric_value * 2.0)

    elif metric_type == "stability":
        # stability typically [0, 1] (consistency metric)
        return min(1.0, max(0.0, metric_value))

    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
```

### Reverse Mapping: T3 → SAGE Metrics

When importing from ACT blockchain:

```python
def t3_to_sage_metrics(
    talent: float,
    training: float,
    temperament: float
) -> Dict[str, float]:
    """Convert T3 scores back to SAGE metric space."""
    return {
        "average_confidence": talent,  # Direct mapping
        "convergence_rate": training / 2.0,  # Inverse of normalization
        "stability": temperament  # Direct mapping
    }
```

---

## Context Modifier Integration

### ACT Context Modifier Model

**Current ACT Implementation**:
```go
// Context modifiers (multiplicative)
contextModifiers := map[string]float64{
    "energy_operation": 1.10,  // +10% boost
    "energy_balance":   1.05,  // +5% boost
    "critical_safety":  1.20,  // +20% boost
    "diagnostic":       0.95,  // -5% penalty
    "default":          1.00,  // neutral
}

// Applied after T3 calculation
finalTrust = t3Composite * contextModifier
```

**Semantic Meaning**: Context modifier represents **situational fitness** - how well-suited this relationship is for this operational context.

### SAGE Per-Context Trust Model

**Current SAGE Implementation**:
```python
context_trust = {
    "code_generation": 0.85,
    "planning": 0.72,
    "reasoning": 0.78
}

# Retrieved directly (no modifier multiplication)
trust_score = context_trust.get(context, default=0.5)
```

**Semantic Meaning**: Each context has **independent learned trust** from observed performance history.

### Integration Strategy

**Option 1: Store ACT Context Modifiers in SAGE** (Recommended)

```python
class ExpertReputation:
    # Existing
    context_trust: Dict[str, float]  # Learned performance
    context_observations: Dict[str, int]

    # NEW: ACT-sourced modifiers
    context_modifiers: Dict[str, float] = {}  # From ACT blockchain
    base_t3_score: Optional[float] = None  # Unmodified T3 composite

    def get_adjusted_trust(self, context: str) -> float:
        """
        Get trust score with optional ACT context modifier applied.

        Strategy:
        1. If context_trust exists locally, use it (SAGE takes precedence)
        2. If base_t3_score + modifier from ACT, use that
        3. Otherwise, default 0.5
        """
        # Local performance trust overrides blockchain (more recent)
        if context in self.context_trust:
            return self.context_trust[context]

        # Fall back to ACT blockchain trust
        if self.base_t3_score is not None:
            modifier = self.context_modifiers.get(context, 1.0)
            return min(1.0, self.base_t3_score * modifier)

        # No data available
        return 0.5
```

**Option 2: Convert ACT Modifiers to SAGE Context Trust** (Simpler)

```python
def import_t3_with_context_modifier(
    self,
    lct_id: str,
    t3_composite: float,
    context: str,
    context_modifier: float
):
    """Import T3 score with context modifier applied."""
    expert_id = self.identity_bridge.lct_uri_to_expert(lct_id)
    reputation = self.reputation_db.get_reputation(expert_id)

    # Apply modifier and store as context-specific trust
    adjusted_trust = min(1.0, t3_composite * context_modifier)
    reputation.context_trust[context] = adjusted_trust
    reputation.context_observations[context] += 1  # Increment as if observed

    self.reputation_db.save_reputation(reputation)
```

**Recommendation**: **Option 2** for simplicity. ACT context modifiers are conceptually similar to SAGE's per-context trust, just with different calculation methods. Storing the final adjusted value as context_trust unifies the model.

---

## V3 Tensor Integration (Operation Evaluation)

### V3 Tensor Purpose

ACT's V3 tensor evaluates **operations/transactions**, not entities:

```protobuf
message ValueTensor {
  string tensor_id = 1;
  string lct_id = 2;          // Observer/evaluator
  string operation_id = 3;    // What was evaluated
  string valuation_score = 4; // Value created (0-1)
  string veracity_score = 5;  // Truthfulness (0-1)
  string validity_score = 6;  // Correctness (0-1)
}

// V3 Composite = 0.4*Valuation + 0.3*Veracity + 0.3*Validity
```

### SAGE Equivalent: Quality Feedback

**Current SAGE Quality Recording**:
```python
class ATPResourceAllocator:
    def record_quality(
        self,
        lct_id: str,
        selected_experts: List[int],
        quality_score: float  # 0-1, from TrustTensorSync
    ):
        """Record quality feedback for ATP → ADP conversion."""
        adp_reward = int(
            self.base_cost_per_expert * quality_score * len(selected_experts)
        )
        self.atp_balances[lct_id] += adp_reward
```

**Problem**: `quality_score` is a single value, but V3 has 3 dimensions.

### Integration Strategy

**Map V3 Composite to SAGE Quality Score**:

```python
def v3_to_quality_score(
    valuation: float,
    veracity: float,
    validity: float
) -> float:
    """Convert V3 tensor to SAGE quality score."""
    # Use same formula as ACT
    return 0.4 * valuation + 0.3 * veracity + 0.3 * validity
```

**Reverse: SAGE Quality → V3 Dimensions** (Estimation):

```python
def quality_score_to_v3_estimate(
    quality_score: float,
    operation_type: str = "default"
) -> Dict[str, float]:
    """
    Estimate V3 components from single quality score.

    Strategy: Different operation types have different V3 profiles.
    """
    profiles = {
        "code_generation": {
            "valuation": quality_score * 0.9,  # Value weighted
            "veracity": quality_score * 1.0,   # Truth matches quality
            "validity": quality_score * 1.1,   # Correctness most important
        },
        "reasoning": {
            "valuation": quality_score * 1.0,
            "veracity": quality_score * 1.2,   # Truth crucial for reasoning
            "validity": quality_score * 1.0,
        },
        "planning": {
            "valuation": quality_score * 1.3,  # Value creation key
            "veracity": quality_score * 0.9,
            "validity": quality_score * 1.0,
        },
        "default": {
            "valuation": quality_score,
            "veracity": quality_score,
            "validity": quality_score,
        }
    }

    profile = profiles.get(operation_type, profiles["default"])

    # Normalize to ensure sum matches expected V3 composite
    # V3 = 0.4*Val + 0.3*Ver + 0.3*Val = quality_score
    # Ensure profile adheres to this constraint
    return {
        k: min(1.0, v) for k, v in profile.items()
    }
```

**Usage in ATP/ADP Flow**:

```python
class ATPResourceAllocator:
    def record_quality_with_v3(
        self,
        lct_id: str,
        selected_experts: List[int],
        quality_score: float,
        operation_type: str = "default"
    ):
        """Enhanced quality recording with V3 tensor creation."""
        # Existing ATP → ADP reward
        adp_reward = int(
            self.base_cost_per_expert * quality_score * len(selected_experts)
        )
        self.atp_balances[lct_id] += adp_reward

        # NEW: Create V3 tensor for blockchain anchoring
        v3_components = quality_score_to_v3_estimate(quality_score, operation_type)

        # Store for periodic blockchain sync
        self.pending_v3_tensors.append({
            "lct_id": lct_id,
            "operation_id": f"sage_expert_selection_{timestamp}",
            "valuation_score": v3_components["valuation"],
            "veracity_score": v3_components["veracity"],
            "validity_score": v3_components["validity"],
            "timestamp": time.time()
        })
```

---

## Multi-Observer Consensus

### Problem Statement

Different observers may have different trust evaluations:

```
Observer A (ACT Node 1): T3 = 0.80 for expert 42
Observer B (ACT Node 2): T3 = 0.75 for expert 42
Observer C (SAGE Instance): context_trust = 0.82 for expert 42
```

**Question**: How to reconcile into single consensus trust score?

### Aggregation Strategies

**Strategy 1: Confidence-Weighted Average** (Current SAGE)

```python
def aggregate_trust_observations(
    observations: List[TrustObservation]
) -> float:
    """Aggregate using confidence-weighted average."""
    if not observations:
        return 0.5  # Default

    total_weighted_trust = sum(
        obs.trust_score * obs.confidence
        for obs in observations
    )
    total_confidence = sum(obs.confidence for obs in observations)

    if total_confidence == 0:
        return 0.5

    return total_weighted_trust / total_confidence
```

**Pros**: Weights trusted observers more heavily
**Cons**: Requires observer reputation tracking

**Strategy 2: Median (Byzantine-Resistant)**

```python
def aggregate_trust_median(
    observations: List[TrustObservation]
) -> float:
    """Aggregate using median (robust to outliers)."""
    if not observations:
        return 0.5

    trust_scores = [obs.trust_score for obs in observations]
    return statistics.median(trust_scores)
```

**Pros**: Resistant to malicious outliers
**Cons**: Loses confidence information

**Strategy 3: Observer Reputation Weighted** (Advanced)

```python
class ObserverReputationTracker:
    """Track reputation of observers themselves."""

    def __init__(self):
        self.observer_reputations: Dict[str, float] = {}

    def aggregate_with_observer_reputation(
        self,
        observations: List[TrustObservation]
    ) -> float:
        """Weight by observer reputation (meta-trust)."""
        total_weighted = sum(
            obs.trust_score * self.get_observer_reputation(obs.observer_id)
            for obs in observations
        )
        total_reputation = sum(
            self.get_observer_reputation(obs.observer_id)
            for obs in observations
        )

        return total_weighted / total_reputation if total_reputation > 0 else 0.5

    def get_observer_reputation(self, observer_id: str) -> float:
        """Get reputation of observer (starts at 0.5, learned over time)."""
        return self.observer_reputations.get(observer_id, 0.5)

    def update_observer_reputation(
        self,
        observer_id: str,
        prediction_accuracy: float
    ):
        """Update observer reputation based on prediction accuracy."""
        current = self.get_observer_reputation(observer_id)
        learning_rate = 0.1
        updated = (1 - learning_rate) * current + learning_rate * prediction_accuracy
        self.observer_reputations[observer_id] = updated
```

**Pros**: Feedback loop improves consensus quality over time
**Cons**: Complex, requires ground truth for accuracy measurement

**Recommendation**: **Strategy 1** (confidence-weighted) with fallback to **Strategy 2** (median) when observer reputations unavailable.

---

## Implementation Roadmap

### Phase 1: ACT ↔ Web4 Bridge (5-7 days)

**Goal**: Enable ACT blockchain tensors to flow to Web4TrustClient

**Tasks**:
1. **ACT Blockchain Query RPC**:
   ```go
   // In x/trusttensor/keeper/query.go
   func (k Keeper) GetRelationshipTensorForExport(
       ctx context.Context,
       lctID string
   ) (*types.RelationshipTensor, error)
   ```

2. **ACT Blockchain Write RPC**:
   ```go
   func (k Keeper) SetRelationshipTensorFromImport(
       ctx context.Context,
       lctID string,
       trustScore float64,
       context string,
       observerID string
   ) error
   ```

3. **Web4TrustClient Enhancement**:
   ```python
   class Web4TrustClient:
       def __init__(self, act_rpc_endpoint: Optional[str] = None):
           self.act_rpc = ACTBlockchainClient(act_rpc_endpoint) if act_rpc_endpoint else None

       def fetch_from_blockchain(self, lct_id: str) -> List[TrustTensor]:
           """Fetch trust tensors from ACT blockchain."""
           if not self.act_rpc:
               return []

           tensor = self.act_rpc.get_relationship_tensor(lct_id)
           return self._convert_t3_to_trust_tensor(tensor)

       def sync_to_blockchain(self, lct_id: str) -> bool:
           """Push local trust observations to ACT blockchain."""
           if not self.act_rpc:
               return False

           observations = self.get_trust_observations(lct_id)
           aggregated = self._aggregate_observations(observations)
           return self.act_rpc.set_relationship_tensor(lct_id, aggregated)
   ```

**Deliverable**: Web4TrustClient can read/write ACT blockchain

### Phase 2: T3 ↔ SAGE Semantic Mapping (3-4 days)

**Goal**: Implement T3 tensor to SAGE metrics conversion

**Tasks**:
1. **Add T3 Fields to ExpertReputation**:
   ```python
   class ExpertReputation:
       # Existing fields...

       # NEW: ACT T3 components
       t3_talent: Optional[float] = None
       t3_training: Optional[float] = None
       t3_temperament: Optional[float] = None
       t3_composite: Optional[float] = None
       t3_last_updated: Optional[float] = None
   ```

2. **Implement Conversion Functions**:
   ```python
   # In sage/web4/trust_tensor_sync.py
   def sage_metrics_to_t3(self, expert_id: int) -> Dict[str, float]:
       """Convert SAGE metrics to T3 components."""

   def t3_to_sage_metrics(
       self,
       talent: float,
       training: float,
       temperament: float
   ) -> Dict[str, float]:
       """Convert T3 components to SAGE metrics."""
   ```

3. **Enhanced Import/Export**:
   ```python
   def export_to_act_blockchain(self, expert_id: int, context: str):
       """Export SAGE reputation as T3 tensor to ACT."""

   def import_from_act_blockchain(self, lct_id: str, context: str):
       """Import T3 tensor from ACT to SAGE reputation."""
   ```

**Deliverable**: Bidirectional T3 ↔ SAGE metrics conversion

### Phase 3: Context Modifier Integration (2-3 days)

**Goal**: Handle ACT context modifiers in SAGE model

**Tasks**:
1. Store context modifiers from ACT
2. Apply when calculating trust scores
3. Test modifier application correctness

**Deliverable**: Context-aware trust scores matching ACT semantics

### Phase 4: V3 Tensor Integration (3-4 days)

**Goal**: Map SAGE quality scores to V3 tensors

**Tasks**:
1. Implement quality_score → V3 estimation
2. Store pending V3 tensors for blockchain sync
3. Periodic batch sync to ACT blockchain
4. Test V3 composite correctness

**Deliverable**: SAGE quality feedback anchored as V3 tensors on blockchain

### Phase 5: Multi-Observer Consensus (5-6 days)

**Goal**: Implement robust multi-observer trust aggregation

**Tasks**:
1. Implement confidence-weighted averaging
2. Implement median aggregation (fallback)
3. (Optional) Observer reputation tracking
4. Test Byzantine resistance
5. Test consensus convergence

**Deliverable**: Robust trust consensus across multiple observers

### Phase 6: End-to-End Integration Testing (7-10 days)

**Goal**: Validate complete trust flow across all systems

**Test Scenarios**:
1. SAGE expert → ACT T3 tensor → Web4 → Import to different SAGE instance
2. ACT blockchain update → SAGE import → Expert selection uses updated trust
3. Multi-observer (2 SAGE instances + ACT) → Consensus → All systems converge
4. Quality feedback loop → V3 tensor → ATP/ADP → Trust update
5. Context switching → Modifier application → Correct trust retrieval
6. Byzantine observer (malicious) → Median consensus rejects outlier

**Deliverable**: Production-ready cross-project trust integration

---

## Risk Assessment

### Technical Risks

**1. T3 Semantic Mapping Mismatch** (High)
- **Risk**: Proposed mapping (confidence→talent, convergence→training, stability→temperament) may not capture true semantics
- **Mitigation**: Empirical validation - compare T3 scores with actual expert performance
- **Fallback**: Use single composite score, log warning about dimensional loss

**2. Blockchain Throughput** (Medium)
- **Risk**: Frequent trust updates could exceed ACT block capacity
- **Mitigation**: Batch updates, only sync significant changes (threshold: Δtrust > 0.05)
- **Fallback**: Periodic full sync (e.g., hourly) instead of real-time

**3. Trust Score Divergence** (Medium)
- **Risk**: SAGE and ACT trust scores may diverge over time due to different update frequencies
- **Mitigation**: Periodic reconciliation, SAGE as source of truth for recent updates
- **Fallback**: Timestamp-based conflict resolution (most recent wins)

**4. Observer Reputation Bootstrap** (Low)
- **Risk**: New observers start with reputation 0.5, may get over/under-weighted
- **Mitigation**: Higher minimum observations before trusting observer consensus
- **Fallback**: Use median aggregation until observer reputation stabilizes

### Operational Risks

**1. Network Partition** (Medium)
- **Risk**: SAGE disconnected from ACT blockchain
- **Impact**: Trust updates stale, expert selection suboptimal
- **Mitigation**: Local cache with grace period (6 hours), reconnect protocol
- **Monitoring**: Alert if blockchain sync age > threshold

**2. Blockchain State Rollback** (Low)
- **Risk**: ACT blockchain reorg invalidates recent trust updates
- **Impact**: Temporary trust score inconsistency
- **Mitigation**: Wait for block finality (N confirmations) before importing
- **Monitoring**: Track reorg frequency, adjust confirmation threshold

**3. Data Migration** (Medium)
- **Risk**: Existing SAGE expert reputations need T3 field migration
- **Impact**: Downtime or inconsistent state during migration
- **Mitigation**: Incremental migration, backward compatible schema
- **Testing**: Full migration dry-run on staging data

---

## Success Metrics

### Integration Completeness

- [ ] Web4TrustClient can read from ACT blockchain
- [ ] Web4TrustClient can write to ACT blockchain
- [ ] T3 → SAGE metrics conversion implemented
- [ ] SAGE → T3 tensor export implemented
- [ ] Context modifiers applied correctly
- [ ] V3 tensor creation from quality scores
- [ ] Multi-observer consensus working
- [ ] End-to-end sync tested (ACT ↔ Web4 ↔ SAGE)

### Performance

- [ ] Trust query latency < 50ms (p99)
- [ ] Blockchain sync latency < 2 seconds (p99)
- [ ] Consensus convergence time < 30 seconds
- [ ] Throughput > 1000 trust updates/minute

### Correctness

- [ ] T3 composite matches ACT formula (±0.01 tolerance)
- [ ] V3 composite matches ACT formula (±0.01 tolerance)
- [ ] Multi-observer consensus median error < 0.05
- [ ] Byzantine resistance validated (50% malicious observers tolerated)

### Reliability

- [ ] Trust score consistency > 99.9% (3-sigma)
- [ ] Network partition recovery < 5 minutes
- [ ] Zero data loss during blockchain sync
- [ ] Backward compatibility with pre-T3 SAGE data

---

## Conclusion

This design establishes a **three-layer trust architecture**:

1. **Blockchain (ACT)**: Consensus, permanence, T3/V3 tensors
2. **Protocol (Web4)**: Standards, interop, canonical trust tensor format
3. **Edge (SAGE)**: Performance, context-specific learning, fast queries

**Key Innovations**:
- **T3 ↔ SAGE Metrics Mapping**: Bridges decomposed trust (ACT) with emergent trust (SAGE)
- **Context Modifier Integration**: Unifies ACT's multiplicative model with SAGE's per-context isolation
- **V3 Tensor Generation**: Maps SAGE quality scores to blockchain-verifiable value assessments
- **Multi-Observer Consensus**: Byzantine-resistant aggregation with confidence weighting

**Implementation Timeline**: 25-35 days across 6 phases

**Next Steps**:
1. Implement ACT ↔ Web4 bridge (RPC endpoints)
2. Create T3 semantic mapping (conversion functions)
3. Prototype end-to-end sync workflow
4. Validate with real Q3-Omni expert selection

---

**Document Version**: 1.0.0-draft
**Last Updated**: 2025-12-17
**Author**: Legion (Autonomous Research Session 62+)
**Cross-References**:
- ATP_CROSS_PROJECT_INTEGRATION_ANALYSIS.md
- LCT_UNIFIED_IDENTITY_SPECIFICATION.md
