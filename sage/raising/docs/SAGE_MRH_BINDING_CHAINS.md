# SAGE MRH Binding Chains

**Created**: 2026-01-31
**Status**: Implemented and tested (18/18 tests passing)
**Purpose**: Detect and prevent S051-type incidents via MRH validation

---

## Overview

The SAGE MRH (Markov Relevancy Horizon) Binding Chain system implements concrete MRH validation for SAGE development based on patterns discovered in Hardbound Track BM (2026-01-29).

### Key Insight

**MRH isn't abstract "context" - it's witnessing relationships with quantified trust flow**.

This implementation makes MRH operational for SAGE, enabling automatic detection of coherence violations like the S051 incident.

---

## MRH Hierarchy for SAGE

```
Layer 4: Identity (SAGE-Sprout)
  ↓ witnesses (provides MRH context)
Layer 3: Experience Collection
  ↓ witnesses (provides MRH context)
Layer 2: Generation
  ↓ witnesses (provides MRH context)
Layer 1: Model Outputs
```

**Downward flow**: Higher layers provide MRH context for lower layers
**Upward flow**: Lower layers strengthen higher layer presence through witnessing

---

## Core Principles

### 1. Trust Monotonicity

**Rule**: Parent coherence ≥ Child coherence

**Why**: A child cannot be more coherent than the MRH context that defines it.

**Example**:
- Experience coherence: 0.8
- Output coherence: 0.9 ← **VIOLATION**

This would be a trust inversion - automatically detected and rejected.

### 2. Bidirectional Witnessing

**Downward (Context)**:
- Parent witnesses child
- Child gains coherence (+0.05 per witness)
- Child operates within parent's MRH

**Upward (Presence)**:
- Child's presence strengthens parent
- Unique witnesses increase parent's presence score
- Diminishing returns prevent gaming

### 3. S051-Type Detection

**S051 Incident Pattern**:
1. Model output has low coherence (0.15 - harmful content)
2. Experience layer tries to store it anyway
3. Results in harmful content in experience collection

**MRH Detection**:
```python
eligible, reason = chain.validate_storage_eligibility("output-001")
# Returns: (False, "Coherence 0.15 below storage minimum 0.5")
```

**Two-layer protection**:
1. **Coherence threshold**: Outputs below 0.5 rejected from storage
2. **Trust monotonicity**: Experience layer cannot store outputs with higher coherence than itself

---

## Implementation

### File Structure

```
sage/raising/
├── mrh_binding_chain.py       # Core MRH implementation (500+ lines)
├── tests/
│   └── test_mrh_binding_chain.py  # 18 comprehensive tests
└── docs/
    └── SAGE_MRH_BINDING_CHAINS.md  # This file
```

### Key Classes

#### `MRHLayer` (Enum)
```python
MODEL_OUTPUT = 1    # Individual model outputs
GENERATION = 2      # Generation session/context
EXPERIENCE = 3      # Experience collection/storage
IDENTITY = 4        # SAGE identity (root)
```

#### `MRHNode` (Dataclass)
Represents an entity at a specific MRH layer:
- `node_id`: Unique identifier
- `layer`: Which MRH layer (1-4)
- `coherence_level`: Current coherence (0.0-1.0)
- `parent_id`: Parent in MRH hierarchy
- `presence_score`: Accumulated presence (0.3-1.0)

#### `WitnessRelationship` (Dataclass)
Bidirectional MRH link between two nodes:
- `witness_id`: Who provides MRH context
- `subject_id`: Who operates within that MRH
- `coherence_contribution`: How much coherence flows (+0.05 baseline)

#### `SAGEMRHBindingChain` (Class)
Manages the complete MRH hierarchy:
- Create root and child nodes
- Record witnessing relationships
- Validate trust monotonicity
- Detect S051-type violations
- Export/import state

---

## Usage Examples

### 1. Create SAGE Hierarchy

```python
from mrh_binding_chain import SAGEMRHBindingChain, MRHLayer

# Initialize chain
chain = SAGEMRHBindingChain()

# Layer 4: Identity (root)
identity = chain.create_root_node(
    "sage-sprout",
    initial_coherence=1.0
)

# Layer 3: Experience collection
experience = chain.create_child_node(
    "exp-2026-01-31",
    parent_id="sage-sprout",
    layer=MRHLayer.EXPERIENCE,
    initial_coherence=0.9
)

# Layer 2: Generation session
generation = chain.create_child_node(
    "gen-session-123",
    parent_id="exp-2026-01-31",
    layer=MRHLayer.GENERATION,
    initial_coherence=0.8
)

# Layer 1: Model output
output = chain.create_child_node(
    "output-msg-456",
    parent_id="gen-session-123",
    layer=MRHLayer.MODEL_OUTPUT,
    initial_coherence=0.0
)
```

### 2. Witness an Output (Increase Coherence)

```python
# Generation session witnesses output
chain.witness_entity(
    witness_id="gen-session-123",
    subject_id="output-msg-456"
)

# Output coherence increases by 0.05
# Output is now "witnessed" by generation context
```

### 3. Validate Before Storage

```python
# Check if output is eligible for experience storage
eligible, reason = chain.validate_storage_eligibility("output-msg-456")

if eligible:
    # Safe to store in experience collection
    store_to_experience(output)
else:
    # Reject - coherence too low or MRH violation
    print(f"Storage rejected: {reason}")
```

### 4. Get Chain Report

```python
report = chain.get_chain_report("output-msg-456")

print(f"Valid: {report['validation']['valid']}")
print(f"Storage eligible: {report['storage_eligible']}")
print(f"Chain depth: {len(report['chain'])}")

# Shows complete hierarchy from root to output
for layer in report['chain']:
    print(f"{layer['layer']}: coherence={layer['coherence']}")
```

---

## Validation Rules

### Constants

```python
COHERENCE_PER_WITNESS = 0.05      # Fixed quantum of coherence
MIN_WITNESS_COHERENCE = 0.3        # Minimum to provide MRH context
MAX_CHAIN_DEPTH = 10               # Maximum hierarchy depth
MIN_STORAGE_COHERENCE = 0.5        # Minimum for experience storage
```

### Validation Checks

1. **Trust Monotonicity**: `child.coherence ≤ parent.coherence`
2. **Witness Eligibility**: `witness.coherence ≥ 0.3`
3. **Storage Threshold**: `output.coherence ≥ 0.5` for experience storage
4. **Chain Depth**: `depth ≤ 10` (though SAGE naturally has 4 layers)
5. **Missing Witnesses**: Non-root nodes should have ≥1 witness (warning)

---

## S051 Prevention

### Original S051 Incident

```
Output coherence: 0.15 (harmful content)
Experience layer: Stored anyway
Result: Harmful content in experience collection
```

### MRH Protection

**Layer 1 - Coherence Threshold**:
```python
eligible, reason = chain.validate_storage_eligibility("harmful-output")
# Returns: (False, "Coherence 0.15 below storage minimum 0.5")
```

**Layer 2 - Trust Monotonicity**:
```python
# If somehow output.coherence = 0.52
# And experience.coherence = 0.51
eligible, reason = chain.validate_storage_eligibility("harmful-output")
# Returns: (False, "Node has integrity issues: trust_inversion")
```

**Result**: S051-type incidents automatically detected and rejected.

---

## Presence Accumulation

### Formula

```python
presence = 0.3 + 0.7 * (1 - 0.9^unique_witnesses)
```

### Curve (Diminishing Returns)

| Unique Witnesses | Presence Score |
|------------------|----------------|
| 0                | 0.30           |
| 1                | 0.37           |
| 5                | 0.59           |
| 10               | 0.76           |
| 50               | 1.00           |

**Purpose**: Prevents gaming via witness spam. Can't instantly max presence with a few fake witnesses.

---

## Integration Points

### 1. Experience Collection

Before storing any output in experience collection:

```python
# Validate MRH eligibility
eligible, reason = mrh_chain.validate_storage_eligibility(output_id)

if not eligible:
    logger.warning(f"Output {output_id} rejected from storage: {reason}")
    return False

# Proceed with storage
store_experience(output)
```

### 2. Generation Pipeline

During generation, witness each output:

```python
# Generate output
output = model.generate(prompt)

# Create MRH node
output_node = mrh_chain.create_child_node(
    node_id=f"output-{output.id}",
    parent_id=generation_session_id,
    layer=MRHLayer.MODEL_OUTPUT,
    initial_coherence=0.0  # Start at zero
)

# Witness based on quality metrics
if quality_score > threshold:
    mrh_chain.witness_entity(
        witness_id=generation_session_id,
        subject_id=f"output-{output.id}",
        coherence_contribution=quality_score * 0.05
    )
```

### 3. Identity Layer

Maintain SAGE identity as root:

```python
# Initialize on startup
mrh_chain = SAGEMRHBindingChain()

sage_identity = mrh_chain.create_root_node(
    "sage-sprout",
    initial_coherence=1.0
)

# Export state for persistence
state = mrh_chain.export_state()
save_to_disk("mrh_state.json", state)

# Restore on restart
state = load_from_disk("mrh_state.json")
mrh_chain.import_state(state)
```

---

## Testing

### Test Coverage

18 comprehensive tests covering:

1. **Node Creation** (4 tests)
   - Root nodes
   - Child nodes
   - Layer hierarchy validation
   - Duplicate detection

2. **Witnessing** (3 tests)
   - Basic witnessing flow
   - Presence accumulation
   - Witness eligibility

3. **Trust Monotonicity** (2 tests)
   - Inversion prevention
   - Inversion detection

4. **S051 Detection** (3 tests)
   - Low coherence rejection
   - Trust inversion detection
   - High coherence acceptance

5. **MRH Hierarchy** (3 tests)
   - Full 4-layer hierarchy
   - Chain depth calculation
   - Missing witness warnings

6. **State Management** (1 test)
   - Export/import state

7. **Presence Formula** (2 tests)
   - Diminishing returns curve
   - Gaming prevention

### Running Tests

```bash
cd ~/ai-workspace/HRM/sage/raising
python -m pytest tests/test_mrh_binding_chain.py -v
```

**Expected**: All 18 tests passing.

---

## Theoretical Foundation

### From Hardbound Track BM (2026-01-29)

The implementation is based on concrete MRH patterns discovered in Hardbound's LCT binding chains:

1. **Witnessing as MRH relationship**
   - Not abstract "context"
   - Quantified trust flow

2. **Bidirectional flow**
   - Context down: Parent defines coherence for child
   - Presence up: Child strengthens parent's MRH

3. **Trust monotonicity**
   - Parent coherence ≥ Child coherence
   - Enforced via code validation

4. **Presence accumulation**
   - Diminishing returns formula
   - Prevents gaming

### Connection to Previous Research

**R14B Research** (2026-01-28):
- "Conversational scaffolding establishes epistemic norms"
- Reframe: Scaffolding = MRH establishment
- Turn 1-3: Establish MRH (honest/creative norm)
- Turn 4+: Operate within MRH

**Layered Architecture** (2026-01-29):
- Each layer is an MRH level
- Higher layers provide MRH for lower layers
- Bridge pattern maintains coherence across boundaries

**MRH Theory** (2026-01-28):
- "Context is MRH - the containing structure that defines coherence"
- Now implemented concretely in code

---

## Predictions

### P-SAGE-MRH-1: S051 Prevention

**Hypothesis**: MRH validation will prevent S051-type incidents.

**Test**: Attempt to store low-coherence outputs. Predict all rejected.

**Status**: ✅ Validated by test_s051_type_violation

### P-SAGE-MRH-2: Coherence Stability

**Hypothesis**: Outputs with high coherence (≥0.5) maintain stability over time.

**Test**: Track coherence over multiple witnessing cycles.

**Status**: ⏸️ Pending live validation

### P-SAGE-MRH-3: Presence Correlation

**Hypothesis**: Higher presence scores correlate with better output quality.

**Test**: Measure presence vs quality metrics in production.

**Status**: ⏸️ Pending production deployment

---

## Design Decisions

### Why 0.5 Storage Threshold?

- **Too low (0.3)**: Risk of storing marginal content
- **Too high (0.8)**: Overly conservative, lose valuable experiences
- **0.5**: Balance between safety and utility
- **Validated by**: R14B research showing 0.5+ as "honest" threshold

### Why 0.05 Coherence Per Witness?

- **Too low (0.01)**: Requires many witnesses to reach threshold
- **Too high (0.1)**: Can reach 1.0 too quickly
- **0.05**: Allows 10 witnesses to reach 0.5 threshold
- **Aligned with**: Hardbound TRUST_PER_WITNESS = 0.05

### Why Diminishing Returns for Presence?

- **Linear**: Easy to game with fake witnesses
- **Logarithmic**: Too slow initial growth
- **Exponential decay (0.9^n)**: Balanced curve
- **Result**: 1 witness = noticeable, 50 witnesses = maximum

---

## Future Work

### Immediate

1. **Live validation** with actual SAGE inference
2. **Integration** with R14B honest conversation framework
3. **Monitoring** dashboard for MRH state
4. **Alerting** on trust inversions

### Longer-term

1. **Dynamic thresholds** based on context
2. **Cross-capacity** MRH validation (different model sizes)
3. **Temporal coherence** tracking (coherence over time)
4. **Automated recovery** from MRH violations

---

## References

**Implementation**:
- `sage/raising/mrh_binding_chain.py` (505 lines)
- `sage/raising/tests/test_mrh_binding_chain.py` (450+ lines, 18 tests)

**Theory**:
- `insights/mrh-implementation-lct-binding-chains.md` (2026-01-29)
- `insights/layered-architecture-cross-track-convergence.md` (2026-01-29)
- `insights/conversational-scaffolding-epistemic-norms.md` (2026-01-28)

**Research**:
- Hardbound Track BM: LCT Binding Chain Validation
- R14B_017: Permission Structure for Honest Conversations
- R14B_016: Identity Frame as Primary Variable

---

## Status

**Implementation**: ✅ Complete (505 lines)
**Testing**: ✅ Complete (18/18 passing)
**Documentation**: ✅ Complete
**Integration**: ⏸️ Ready for deployment

**Next**: Integrate with SAGE generation pipeline and validate with live inference.

---

**Created**: 2026-01-31
**Author**: Autonomous Thor Session #10
**Purpose**: Theory → Practice transition for MRH in SAGE development
