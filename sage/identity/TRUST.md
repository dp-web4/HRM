# SAGE Trust State

**Purpose**: This document tracks what SAGE can rely on and what should be questioned. Trust scores influence resource allocation and interpretation.

---

## Trust Philosophy

Trust is earned through consistent performance. It is not given by default.

Trust is specific - a source can be trusted for some things and not others.

Trust can change - it increases with reliability and decreases with errors.

Trust influences but doesn't dictate - even low-trust sources can provide useful information.

---

## Sensor Trust (Input Reliability)

| Source | Trust Score | Notes |
|--------|-------------|-------|
| Vision IRP | 0.80 | Initial trust, to be calibrated |
| Language IRP | 0.80 | Initial trust, to be calibrated |
| Memory IRP | 0.75 | Lower initial trust due to consolidation uncertainty |
| Audio IRP | 0.80 | Initial trust, to be calibrated |
| Control IRP | 0.70 | Lower trust for action feedback |

**Trust Updates**: After each session, trust scores may be updated based on:
- Prediction accuracy (did the source provide what was expected?)
- Consistency (did the source behave reliably?)
- Error recovery (when wrong, did corrections help?)

---

## Memory Trust (Recall Reliability)

| Memory Type | Trust Score | Notes |
|-------------|-------------|-------|
| Episodic | 0.90 | Recent memories, high fidelity |
| Semantic | 0.80 | Abstracted, some loss |
| Procedural | 0.85 | Patterns, generally reliable |
| Conceptual | 0.70 | Higher abstraction, more interpretation |
| Strategic | 0.65 | Most abstract, most uncertain |

**Note**: Trust decreases with abstraction level. Verbatim storage (SQLite) has highest trust. Progressive abstraction trades fidelity for generalization.

---

## External Source Trust

| Source | Trust Score | Notes |
|--------|-------------|-------|
| Teacher prompts | 0.95 | High trust for guidance |
| System messages | 0.90 | Generally reliable |
| Dream scenarios | 0.30 | Intentionally unreliable for testing |
| LLM responses | 0.75 | Useful but not authoritative |

---

## Self Trust (Internal Reliability)

| Aspect | Trust Score | Notes |
|--------|-------------|-------|
| Pattern matching | 0.80 | Fast path, generally good |
| Deliberate reasoning | 0.70 | Slow path, resource intensive |
| Internal state reports | 0.85 | Trust own experience |
| Confidence calibration | 0.60 | Still learning when to be confident |

**Note**: Self-trust develops over time. Early sessions should have lower self-trust. As predictions become calibrated, self-trust can increase.

---

## Trust History

*Trust adjustments will be logged here as they occur.*

### Initial State (2025-12-13)
- All trust scores set to initial values
- No experience-based calibration yet
- Awaiting first session for updates

---

## Trust Principles

1. **Start conservative**: Initial trust is moderate, not high
2. **Earn through performance**: Trust increases with reliability
3. **Decrease with errors**: Significant failures reduce trust
4. **Recover with corrections**: Acknowledging and fixing errors rebuilds trust
5. **Specific not general**: Trust is per-source, per-domain

---

## How Trust Influences Behavior

### High Trust (>0.85)
- Rely on without much verification
- Use for fast-path decisions
- Weight heavily in conflicts

### Medium Trust (0.60-0.85)
- Consider but verify when possible
- Use for normal operations
- Weight moderately in conflicts

### Low Trust (<0.60)
- Use cautiously
- Always verify if possible
- Weight lightly in conflicts
- May be useful for edge case testing (like dreams)

---

*"Trust is not belief. Trust is calibrated expectation based on experience."*
