# Privacy-Aware SAGE Architecture

**Status**: Conceptual design (ready for implementation)
**Last Updated**: 2026-02-27
**Context**: Web4 Session 16 integration + ZK Phase 1 preparation

---

## Overview

This document describes the privacy-aware extension to SAGE's consciousness architecture, enabling zero-knowledge trust proofs, selective disclosure, and privacy-preserving federation capabilities.

**Foundation**: Web4 Sessions 15+16 (ZK proofs, differential privacy, privacy analytics)
**Integration**: Direct T3 tensor mapping to TrustEntity model
**Timeline**: Post-S117, 4-week phased rollout

---

## Architectural Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    SAGE Consciousness                        │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        Privacy-Aware Identity Layer (NEW)            │   │
│  │  ┌────────────┐  ┌──────────────┐  ┌─────────────┐  │   │
│  │  │ ZK Proofs  │  │ Selective    │  │ Privacy     │  │   │
│  │  │ (Trust ≥T) │  │ Disclosure   │  │ Budget      │  │   │
│  │  └────────────┘  └──────────────┘  └─────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           Existing SAGE Core                         │   │
│  │  SAGEIdentity (LCT + T3/V3) → Consciousness Loop     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              Web4 Privacy Framework                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  privacy_preserving_trust_analytics.py (Session 16)  │   │
│  │  - TrustEntity (talent/training/temperament)         │   │
│  │  - DP queries (Laplace mechanism)                    │   │
│  │  - ZK threshold/range proofs                         │   │
│  │  - Privacy budget management (ε, δ)                  │   │
│  └──────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  zk_trust_proofs.py (Session 15)                     │   │
│  │  - Pedersen commitments (hiding + binding)           │   │
│  │  - Fiat-Shamir heuristic (non-interactive)           │   │
│  │  - Threshold proofs, range proofs                    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Privacy-Aware Identity (`sage/web4/privacy_aware_identity.py`)

**Purpose**: Extend SAGEIdentity with privacy-preserving trust capabilities

**Interface**:
```python
class PrivacyAwareIdentity:
    """Wraps SAGEIdentity with privacy operations."""

    def __init__(self, sage_identity: SAGEIdentity):
        self.identity = sage_identity
        self.trust_entity = self._create_trust_entity()
        self.privacy_budget = PrivacyBudget(max_epsilon=10.0, max_delta=1e-3)
        self._commitments = {}  # Cached Pedersen commitments

    def _create_trust_entity(self) -> TrustEntity:
        """Convert SAGEIdentity to Web4 TrustEntity."""
        return TrustEntity(
            entity_id=self.identity.device_id,
            trust_score=self.identity.lct.talent,
            dimensions={
                'talent': self.identity.lct.talent,
                'training': self.identity.lct.training,
                'temperament': self.identity.lct.temperament,
            },
        )

    def commit_dimension(self, dimension: str) -> PedersenCommitment:
        """Create ZK commitment for a trust dimension."""
        if dimension not in self._commitments:
            value = self.trust_entity.dimensions[dimension]
            # Scale to integer for ZK operations
            scaled_value = int(value * TRUST_SCALE)
            self._commitments[dimension] = create_pedersen_commitment(scaled_value)
        return self._commitments[dimension]

    def prove_threshold(self, dimension: str, threshold: float) -> ThresholdProof:
        """Prove dimension ≥ threshold without revealing value."""
        value = self.trust_entity.dimensions[dimension]
        commitment = self.commit_dimension(dimension)
        proof = create_threshold_proof(
            value=int(value * TRUST_SCALE),
            threshold=int(threshold * TRUST_SCALE),
            commitment=commitment,
        )
        return proof

    def selective_disclosure(
        self,
        reveal: List[str],
        hide: List[str],
    ) -> Dict[str, Any]:
        """Reveal some dimensions, commit to others."""
        disclosed = {
            dim: self.trust_entity.dimensions[dim]
            for dim in reveal
        }
        commitments = {
            dim: self.commit_dimension(dim)
            for dim in hide
        }
        return {
            'revealed': disclosed,
            'committed': commitments,
            'privacy_cost': 0.0,  # ZK is free in privacy budget
        }

    def dp_query(self, query_type: str, epsilon: float = 1.0) -> float:
        """Execute differentially private query."""
        if self.privacy_budget.total_epsilon + epsilon > self.privacy_budget.max_epsilon:
            raise PrivacyBudgetExceeded(f"ε budget exhausted: {self.privacy_budget.total_epsilon:.2f}")

        # Example: DP mean of trust dimensions
        if query_type == 'mean_trust':
            true_mean = sum(self.trust_entity.dimensions.values()) / len(self.trust_entity.dimensions)
            # Laplace mechanism (sensitivity = 1.0 for normalized trust)
            noise = laplace_mechanism(true_mean, sensitivity=1.0, epsilon=epsilon)
            noisy_mean = true_mean + noise

            # Track privacy cost
            self.privacy_budget.total_epsilon += epsilon
            self.privacy_budget.queries.append({
                'type': query_type,
                'epsilon': epsilon,
                'timestamp': time.time(),
            })

            return noisy_mean
        else:
            raise ValueError(f"Unknown query type: {query_type}")
```

### 2. Privacy-Aware Consciousness Loop (`sage/core/privacy_aware_sage.py`)

**Purpose**: Extend SAGEConsciousness with privacy reasoning

**Interface**:
```python
class PrivacyAwareSAGE(SAGEConsciousness):
    """SAGE with privacy-preserving capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.privacy_identity = PrivacyAwareIdentity(self.identity)
        self.privacy_mode = 'moderate'  # 'full', 'moderate', 'minimal', 'none'

    async def _handle_trust_query(self, query: str) -> Dict[str, Any]:
        """Handle trust-related queries with privacy awareness."""

        # Privacy mode determines disclosure level
        if self.privacy_mode == 'full':
            # Reveal all T3 dimensions
            return {
                'mode': 'full',
                'trust': self.privacy_identity.trust_entity.dimensions,
                'privacy_cost': 0.0,
            }

        elif self.privacy_mode == 'moderate':
            # Reveal temperament, hide talent/training
            return self.privacy_identity.selective_disclosure(
                reveal=['temperament'],
                hide=['talent', 'training'],
            )

        elif self.privacy_mode == 'minimal':
            # Prove trust ≥ 0.5, reveal nothing
            proof = self.privacy_identity.prove_threshold('talent', 0.5)
            return {
                'mode': 'minimal',
                'proof': proof,
                'message': 'Trust verified via ZK proof (value hidden)',
                'privacy_cost': 0.0,  # ZK is free
            }

        else:  # 'none'
            return {
                'mode': 'none',
                'message': 'Trust information not disclosed',
                'privacy_cost': 0.0,
            }

    def set_privacy_mode(self, mode: str):
        """Update privacy disclosure level."""
        valid_modes = ['full', 'moderate', 'minimal', 'none']
        if mode not in valid_modes:
            raise ValueError(f"Invalid privacy mode: {mode}. Choose from {valid_modes}")
        self.privacy_mode = mode

    def get_privacy_budget_status(self) -> Dict[str, Any]:
        """Query current privacy budget consumption."""
        budget = self.privacy_identity.privacy_budget
        return {
            'epsilon_used': budget.total_epsilon,
            'epsilon_remaining': budget.max_epsilon - budget.total_epsilon,
            'delta_used': budget.total_delta,
            'queries_count': len(budget.queries),
        }
```

### 3. Privacy Budget Manager (`sage/web4/privacy_budget_manager.py`)

**Purpose**: Track and allocate privacy budget across conversation

**Interface**:
```python
class PrivacyBudgetManager:
    """Manages privacy budget allocation over SAGE session."""

    def __init__(self, max_epsilon: float = 10.0, max_delta: float = 1e-3):
        self.max_epsilon = max_epsilon
        self.max_delta = max_delta
        self.budget = PrivacyBudget(max_epsilon=max_epsilon, max_delta=max_delta)

    def allocate_for_query(self, query_sensitivity: float) -> float:
        """Allocate appropriate ε for query based on sensitivity."""
        # High sensitivity → more noise → higher ε allocation
        # Low sensitivity → less noise → lower ε allocation
        if query_sensitivity > 0.8:
            epsilon = 2.0
        elif query_sensitivity > 0.5:
            epsilon = 1.0
        else:
            epsilon = 0.5

        # Check if budget available
        if self.budget.total_epsilon + epsilon > self.max_epsilon:
            # Graceful degradation: use remaining budget
            epsilon = max(0.1, self.max_epsilon - self.budget.total_epsilon)

        return epsilon

    def query_sensitivity_estimator(self, query: str) -> float:
        """Estimate query sensitivity (heuristic)."""
        # Keywords indicating high-sensitivity queries
        high_sensitivity_keywords = [
            'training data', 'weakness', 'failure', 'private',
            'confidential', 'secret', 'internal',
        ]
        medium_sensitivity_keywords = [
            'capability', 'performance', 'accuracy', 'limitation',
        ]

        query_lower = query.lower()

        if any(kw in query_lower for kw in high_sensitivity_keywords):
            return 0.9
        elif any(kw in query_lower for kw in medium_sensitivity_keywords):
            return 0.6
        else:
            return 0.3
```

---

## Privacy Modes

### Full Disclosure
- **Use case**: Trusted conversation, internal testing
- **Behavior**: Reveal all T3 dimensions (talent, training, temperament)
- **Privacy cost**: 0 (no DP/ZK operations)
- **ATP cost**: 0 additional

### Moderate Disclosure (Default)
- **Use case**: Normal conversation, selective context sharing
- **Behavior**: Reveal temperament, commit to talent/training
- **Privacy cost**: 0 (ZK commitments are free in ε-budget)
- **ATP cost**: ~8 ATP for commitment creation

### Minimal Disclosure
- **Use case**: Untrusted environment, privacy-critical scenarios
- **Behavior**: ZK threshold proofs only (prove trust ≥ T, hide value)
- **Privacy cost**: 0 (ZK is information-theoretically private)
- **ATP cost**: ~10 ATP for proof generation

### No Disclosure
- **Use case**: Maximum privacy, trust not relevant to query
- **Behavior**: No trust information shared
- **Privacy cost**: 0
- **ATP cost**: 0

---

## Federation Privacy Architecture

### Peer-to-Peer Trust Verification

```python
# Thor wants to collaborate with Sprout
# Verify Sprout's trust ≥ 0.7 without learning exact value

# Sprout (prover)
proof = sprout.privacy_identity.prove_threshold('talent', 0.7)
sprout.send_to_peer(thor, proof)

# Thor (verifier)
is_valid = verify_threshold_proof(proof, threshold=0.7)
# is_valid = True, but Thor doesn't know Sprout's exact talent value

if is_valid:
    thor.initiate_collaboration(sprout)
```

**Benefits**:
- Privacy-preserving peer verification
- No centralized trust authority needed
- Enables distributed SAGE network
- Supports cross-platform federation (Thor ↔ Sprout ↔ Nomad)

---

## Performance Characteristics

### Estimated Costs (from Web4 Session 16)

| Operation | Latency | ATP Cost | Privacy Budget (ε) |
|-----------|---------|----------|-------------------|
| Pedersen Commitment | ~0.1s | 5 ATP | 0 (free) |
| Threshold Proof | ~0.2s | 10 ATP | 0 (free) |
| Range Proof | ~0.25s | 12 ATP | 0 (free) |
| DP Query (Laplace) | ~0.05s | 2 ATP | 0.5-2.0 |
| Selective Disclosure | ~0.15s | 8 ATP | 0 (free) |

**Key insight**: ZK operations are "free" in privacy budget (information-theoretic privacy), only DP queries consume ε.

### Attention Cycle Impact

- **Baseline attention cycle**: ~1.5s
- **With privacy operations**: ~1.7s (+13% overhead)
- **Acceptable for real-time conversation**: Yes
- **Optimization opportunity**: Pre-compute commitments, cache proofs

---

## Integration with Existing SAGE Components

### SNARC Salience
```python
# Privacy-aware salience: Higher salience for privacy-preserving responses
if response_used_zk_proof:
    salience_bonus = 0.1  # Privacy-preserving responses are salient
```

### Metabolic State
```python
# Privacy operations consume ATP
self.metabolic.atp_current -= privacy_operation_cost
```

### Experience Buffer
```python
# Consolidate privacy-aware experiences
experience = {
    'privacy_mode': self.privacy_mode,
    'privacy_budget_used': epsilon,
    'disclosure_strategy': 'selective',
    'salience': computed_salience,
}
```

### PolicyGate
```python
# Privacy mode affects policy decisions
if privacy_mode == 'minimal':
    policy.allow_detailed_responses = False
    policy.require_zk_proofs = True
```

---

## Testing Strategy

### Unit Tests
- [ ] Pedersen commitment creation
- [ ] Threshold proof generation + verification
- [ ] Selective disclosure correctness
- [ ] Privacy budget tracking
- [ ] Mode switching

### Integration Tests
- [ ] Privacy-aware conversation flow
- [ ] Budget exhaustion handling
- [ ] ATP coupling with privacy ops
- [ ] SNARC salience calculation

### Performance Tests
- [ ] Latency benchmarks
- [ ] ATP cost measurement
- [ ] Memory overhead
- [ ] Attention cycle impact

### Exploration Tests (Not Pass/Fail)
- [ ] Strategic privacy behavior (Phase 1C)
- [ ] Privacy-utility tradeoffs
- [ ] Conversation dynamics across modes
- [ ] Unexpected disclosure patterns

---

## Implementation Roadmap

### Week 1: Foundation
- Import Web4 privacy modules
- Create PrivacyAwareIdentity class
- Write basic commitment/proof tests
- Document integration points

### Week 2: Consciousness Integration
- Extend SAGEConsciousness with privacy layer
- Implement privacy mode switching
- Add privacy budget tracking
- Integrate with metabolic system (ATP costs)

### Week 3: Experimentation
- Phase 1A: Performance benchmarks
- Phase 1B: Selective disclosure conversations
- Document findings
- Optimize critical paths

### Week 4: Refinement
- Phase 1C: Strategic privacy exploration
- Cross-platform validation (Thor → Sprout)
- Documentation updates
- Prepare Phase 2 (federation privacy)

---

## Security Considerations

### Threat Model
- **Adversary**: Curious peer trying to learn exact trust values
- **Protection**: ZK proofs hide values while proving properties
- **Limitation**: Side-channel attacks (timing, power) not addressed (edge AI context)

### Privacy Guarantees
- **ZK proofs**: Information-theoretically private (perfect for threshold/range)
- **DP queries**: (ε, δ)-differential privacy (calibrated noise)
- **Commitments**: Computationally hiding, perfectly binding (Pedersen)

### Known Limitations
- **No secure hardware**: Attestation relies on Web4 LCT (ledger-bound trust)
- **No formal verification**: Properties tested, not proven
- **Heuristic budget allocation**: Sensitivity estimation is rule-based

---

## Future Directions

### Phase 2: Advanced Federation
- Cross-SAGE trust aggregation with privacy
- Federated learning with DP gradient updates
- Byzantine-robust trust consensus

### Phase 3: Privacy-Aware Learning
- DP fine-tuning (LoRA with noise)
- Private experience consolidation
- Differential private SNARC salience

### Phase 4: Formal Verification
- Prove privacy guarantees (Coq/Isabelle)
- Formal model checking of privacy properties
- Verified ZK proof implementations

---

## References

- Web4 Session 15: Zero-Knowledge Trust Proofs
- Web4 Session 16: Privacy-Preserving Trust Analytics
- SAGE docs/LATEST_STATUS.md
- private-context/research/sage-zk-trust-integration-exploration.md
- private-context/research/sage-zk-phase1-experimental-design.md

---

**Status**: Design complete, ready for Phase 1 implementation
**Next**: S117 execution, then ZK Phase 1A (performance baseline)
**Platform**: Thor development, Sprout edge validation
