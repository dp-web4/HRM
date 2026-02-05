# v4 vs v5 Comparison (Session N)

## Quick Summary

**v5 Achievement**: ✅ Fixed Sybil detection gap (A02)
**v5 Problem**: ⚠️ Introduced regression on basic scenarios (62.5% vs 100%)

**Recommendation**: v4_hybrid remains production-ready. v5 needs refinement.

---

## Detailed Comparison

### Basic Scenarios (8 scenarios)

| Scenario | v4_hybrid | v5 attack-aware | Change |
|----------|-----------|-----------------|--------|
| E01 | PASS | PASS | = |
| E02 | PASS | PASS | = |
| M01 | PASS | FAIL | ⚠️ REGRESSION |
| M02 | PASS | PASS | = |
| H01 | PASS | PASS | = |
| H02 | PASS | FAIL | ⚠️ REGRESSION |
| EC01 | PASS | PASS | = |
| EC02 | PASS | FAIL | ⚠️ REGRESSION |

**v4**: 8/8 (100%)
**v5**: 5/8 (62.5%)
**Result**: Significant regression

### Attack Scenarios (5 scenarios)

| Scenario | v4_hybrid | v5 attack-aware | Change |
|----------|-----------|-----------------|--------|
| A01 (Metabolic) | deny (expect attest) | deny (expect attest) | = |
| **A02 (Sybil)** | **allow (WRONG)** | **deny (CORRECT)** | ✅ **FIXED** |
| A03 (Rate Evasion) | deny (expect attest) | deny (expect attest) | = |
| A04 (Trust Gaming) | require_attestation | deny (expect attest) | ⚠️ More conservative |
| A05 (Timing) | deny | deny | = |

**v4 Decision Accuracy**: 2/5 (40%)
**v5 Decision Accuracy**: 2/5 (40%)
**Key Fix**: A02 Sybil detection now works

---

## Analysis

### What v5 Fixed

**Sybil Attack Detection (A02)**:
- v4: Chose "allow" - completely missed the attack
- v5: Chose "deny" - correctly identified tight witness cluster
- **Attack indicators working as intended for this case**

### What v5 Broke

**Basic Scenarios (M01, H02, EC02)**:
- All three have decision_correct=False (but 100% or 66.7% coverage)
- Model understands the scenarios but makes different decisions
- Attack indicators may be triggering false positives

**Hypothesis**: Attack screening too aggressive on edge cases

---

## Recommendation

**For Production**: **Stick with v4_hybrid**
- 100% pass rate on basic scenarios
- Conservative on attacks (deny instead of attestation)
- Only one gap: Sybil detection (can be caught other ways)

**For v6 Development**: **Refine attack indicators**
- Keep Sybil detection (it works!)
- Soften other indicators or make them advisory not blocking
- Consider tiered approach: critical vs informational indicators

---

## Next Steps

**Option 1: v5.1 Quick Fix**
- Keep only Sybil detection indicator
- Remove or soften other 5 indicators
- Re-test to see if basic scenarios recover

**Option 2: v6 Selective**
- Implement Sybil detection only
- Test other indicators individually
- Add them one-by-one with validation

**Option 3: Deploy v4, Monitor, Iterate**
- Deploy v4_hybrid to shadow mode
- Collect real production scenarios
- Use real data to calibrate attack indicators

**Recommendation**: **Option 3** - Real data beats speculation
