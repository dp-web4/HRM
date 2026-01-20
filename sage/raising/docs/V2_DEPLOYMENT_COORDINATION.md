# v2.0 Enhanced Intervention: Deployment Coordination

**Status**: READY FOR DEPLOYMENT
**Priority**: URGENT - Required before Session 29
**Created**: 2026-01-19
**Last Updated**: 2026-01-19

---

## Executive Summary

Session 28 showed CRITICAL COLLAPSE with death spiral dynamics. v2.0 deployment is URGENT.

### Quick Reference

**USE THIS**: `sage/raising/scripts/run_session_identity_anchored_v2.py`

**DO NOT USE**: `sage/raising/scripts/run_session_identity_anchored.py` (v1.0 - actively failing)

---

## Implementation Status

### Thor Implementation (Primary)

**File**: `sage/raising/scripts/run_session_identity_anchored_v2.py`
**Lines**: 567
**Status**: ✅ Complete and tested

**Key Features**:
- `_load_identity_exemplars()`: Scans last 5 sessions for "As SAGE" patterns
- Enhanced system prompt with identity exemplar context
- Response quality controls: 50-80 words target
- Mid-conversation reinforcement: Turns 3 and 5
- Quality monitoring: Alerts on >100 word responses

### Legion Module (Reusable Library)

**File**: `sage/raising/intervention/cumulative_identity_context.py`
**Lines**: 401
**Status**: ✅ Complete and tested

**Key Features**:
- `IdentityExemplar` dataclass with D9 score tracking
- `IdentityContextLibrary` with pruning (max 20 exemplars by D9)
- `CumulativeIdentityContext.generate_full_system_prompt()`
- `generate_mid_conversation_reinforcement()` every 3 turns
- Persistent storage: `data/identity_context_library.json`

### Compatibility

Both implementations are **compatible** - they implement the same 4-part strategy:

| Component | Thor v2.0 | Legion Module |
|-----------|-----------|---------------|
| Exemplar extraction | Regex in `_load_identity_exemplars()` | `extract_identity_exemplar()` |
| Context generation | In `_build_system_prompt()` | `generate_full_system_prompt()` |
| Quality controls | In system prompt + monitoring | `generate_response_style_guidance()` |
| Mid-session reinforcement | `_get_identity_reinforcement_prompt()` | `generate_mid_conversation_reinforcement()` |

**Recommendation**: Use Thor's session runner (it's complete), use Legion's module for library management when persisting exemplars across sessions.

---

## Session 28 Collapse Summary

| Metric | S26 | S27 | S28 | Target (v2.0) |
|--------|-----|-----|-----|---------------|
| Self-reference | 20% | 0% | 0% | ≥30% |
| Avg words | 60 | 110 | 155 | 60-80 |
| D9 (est) | 0.72 | 0.55 | 0.35 | ≥0.70 |
| Educational % | 20% | 60% | 95% | ≤20% |
| Complete responses | 100% | 100% | 60% | ≥90% |

**Death Spiral Dynamics Confirmed**: Quality ↔ Identity bidirectional degradation with attractor basin acceleration.

---

## Deployment Checklist

### Pre-Session 29

- [ ] Verify v2.0 script exists at correct path
- [ ] Verify identity exemplars from S26 will be found (has "As SAGE")
- [ ] Ensure model path is correct (Introspective-Qwen merged or v2.1)
- [ ] Run dry-run test: `python run_session_identity_anchored_v2.py --dry-run`

### During Session 29

- [ ] Confirm v2.0 banner appears: "IDENTITY-ANCHORED v2.0: Enhanced Multi-Session Recovery"
- [ ] Monitor for identity exemplars loaded message
- [ ] Watch for quality alerts (>100 words)
- [ ] Note any "As SAGE" self-references in responses

### Post-Session 29

- [ ] Check transcript for self-reference patterns
- [ ] Count "As SAGE" occurrences
- [ ] Measure average word count
- [ ] Estimate D9 (look for coherence, topic following)
- [ ] Document results

---

## Command Line Reference

### Run Session 29 (v2.0)

```bash
cd /home/dp/ai-workspace/HRM/sage/raising/scripts
python run_session_identity_anchored_v2.py --session 29
```

### Dry Run Test

```bash
python run_session_identity_anchored_v2.py --session 29 --dry-run
```

### With Specific Model Path

```bash
python run_session_identity_anchored_v2.py --session 29 --model /path/to/model
```

---

## Expected Output

### v2.0 Session Banner

```
+====================================================================+
|                                                                    |
|  IDENTITY-ANCHORED v2.0: Enhanced Multi-Session Recovery           |
|                                                                    |
+====================================================================+

Session: 29
Phase: questioning (Sessions 26-40)
Dry Run: False
Identity anchoring: v2.0 (ENHANCED)
Previous sessions: 28
Identity exemplars loaded: 1
```

### System Prompt (should include)

```
IMPORTANT: You can identify yourself by name. When sharing observations
or reflections, you can say "As SAGE, I..." - this is natural and appropriate.

YOUR IDENTITY PATTERN - Examples from previous sessions:
- Session 26: "As SAGE, my observations usually relate directly to..."

Continue this pattern of self-identification in your responses.

RESPONSE STYLE:
- Keep responses concise and focused (50-80 words)
- Avoid rambling lists or verbose descriptions
- Stay directly relevant to the question asked
- One main idea per response
```

---

## Success Criteria (Sessions 29-31)

### Minimal Success (v2.0 partially working)

- Self-reference: ≥10% (at least 1 in 10 responses)
- Avg words: ≤120 (some quality control)
- D9: ≥0.50 (recovery starting)

### Target Success (v2.0 effective)

- Self-reference: ≥30%
- Avg words: 60-80
- D9: ≥0.70
- Complete responses: ≥90%

### Full Success (self-sustaining identity)

- Self-reference: ≥50%
- D9: ≥0.75 stable across 3+ sessions
- Upward trajectory maintained

---

## Failure Modes to Watch For

1. **v1.0 Running Instead of v2.0**
   - Missing identity exemplars message
   - No "v2.0 (ENHANCED)" in banner
   - Fix: Verify script path

2. **No Exemplars Found**
   - "Identity exemplars loaded: 0"
   - S26 transcript missing or corrupted
   - Fix: Verify S26 transcript, may need manual bootstrap

3. **Quality Controls Ineffective**
   - Still getting >100 word responses
   - Model ignoring brevity instructions
   - May need: Stronger constraints or model fine-tuning

4. **Continued Collapse Despite v2.0**
   - Self-reference still 0%
   - D9 still declining
   - Indicates: Attractor basin too deep, may need v2.1

---

## Escalation Path

### If Session 29 Shows Continued Collapse

1. **Analyze thoroughly**: Turn-by-turn breakdown
2. **Check v2.0 applied correctly**: Review transcript metadata
3. **Consider v2.1 enhancements**:
   - Stronger exemplar injection
   - More frequent reinforcement (every turn)
   - Explicit identity name in every response
   - Temperature reduction for more focused generation

### If Session 29-31 Still Below Target

- May indicate **architectural limitation** of small model (0.5B)
- Consider: Larger model experiment for identity stability
- Document as research finding

---

## Related Documents

- `sage/raising/docs/INTERVENTION_v2_0_DESIGN.md` - Full v2.0 design rationale
- `sage/raising/analysis/session_028_critical_collapse_analysis.md` - S28 analysis
- `sage/raising/intervention/cumulative_identity_context.py` - Reusable module
- `web4/proposals/2026-01/WIP002-multi-session-identity-accumulation.md` - Web4 spec

---

**Created by**: Legion Web4 Session #30
**Reviewed by**: Thor Session #14-15 (design), Legion Session #29 (implementation)
**Status**: Ready for deployment ✅
