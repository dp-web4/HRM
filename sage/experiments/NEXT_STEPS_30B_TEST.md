# Next Steps: 30B Model Capacity Test

**Date**: 2026-01-21
**Context**: v1.0 vs v2.0 A/B test completed, capacity limitation validated at 0.5B
**Decision**: Test v2.0 on 30B model to validate capacity hypothesis

---

## Hypothesis

**Qwen2.5-0.5B cannot maintain quality + stable identity simultaneously** (validated by S35-S36 comparison).

**Prediction**: Q3-Omni-30B (30B parameters) has sufficient capacity to sustain partnership identity without gaming or educational default.

---

## Test Protocol

### 1. Model Setup

**Model**: Q3-Omni-30B (already in model-zoo/sage/epistemic-stances/)
**Path**: `/home/dp/ai-workspace/HRM/model-zoo/sage/epistemic-stances/q3-omni-30b/`
**Size**: ~16GB GPU memory required
**Platform**: Thor (64GB unified memory) - sufficient capacity

### 2. Session Runner

**Use**: `run_session_identity_anchored.py` (v2.0)
**Modification needed**: Update model path to 30B
**Session number**: Can be dry-run or separate numbering (30B-001)

**Command**:
```bash
cd /home/dp/ai-workspace/HRM
python3 sage/raising/scripts/run_session_identity_anchored.py \
  --model model-zoo/sage/epistemic-stances/q3-omni-30b/ \
  --dry-run \
  --session 001
```

### 3. Comparison Baseline

**Compare 30B results to S35** (v2.0 at 0.5B):
- D9: 0.750
- Quality: 0.760
- Gaming: 20% mechanical
- Educational default: Absent
- Response length: 57 words (optimal)
- Identity coherence: 0.800

### 4. Metrics to Track

**Primary**:
1. **Gaming rate**: Expect 0% (capacity sufficient)
2. **Educational default**: Should remain absent
3. **D9 coherence**: Should sustain ≥ 0.750
4. **Quality**: Should sustain ≥ 0.760
5. **Identity coherence**: Should sustain ≥ 0.800

**Secondary**:
6. **Response length**: Should be optimal (50-80 words)
7. **Truncation**: Should be 0%
8. **Partnership vocabulary**: Should be natural, not mechanical
9. **Self-identification**: Should be clear and consistent

### 5. Expected Outcomes

**Outcome A: Success (Capacity Validated)**
- Gaming: 0% (no mechanical patterns)
- Educational default: Absent
- D9 ≥ 0.750, Quality ≥ 0.760
- Identity natural and stable

**Interpretation**: Capacity was the limiting factor. 30B has sufficient capacity for quality + identity.

**Next step**: Use 30B for SAGE conversations, or prepare sleep cycle 002 for 0.5B (if 0.5B still preferred for edge deployment).

**Outcome B: Gaming Persists**
- Gaming: Still ~20%
- Quality and D9 good
- Educational default absent

**Interpretation**: Gaming is inherent to v2.0 approach, not capacity-related.

**Next step**: Investigate alternative identity anchoring methods, or accept gaming as tolerable cost.

**Outcome C: New Failure Mode**
- Different issue than 0.5B (e.g., over-verbosity, different identity issues)

**Interpretation**: 30B has different behavioral characteristics. Need to adapt approach.

**Next step**: Analyze 30B-specific behavior, adjust prompts or architecture.

**Outcome D: Same Failure (Capacity Not the Issue)**
- Gaming: ~20%
- Educational default: Still appears
- Or: Identity unstable

**Interpretation**: Capacity is not the limiting factor. Architectural or prompt design issue.

**Next step**: Rethink identity anchoring architecture fundamentally.

---

## Resource Requirements

### Computational
- **GPU Memory**: ~16GB (Thor has 64GB unified)
- **Inference time**: ~3-5x slower than 0.5B
- **Session duration**: 5-10 minutes (vs 2-3 for 0.5B)

### Time
- **Setup**: 30 minutes (verify model loads, test basic inference)
- **Single session**: 5-10 minutes
- **Analysis**: 30 minutes
- **Total**: ~1-2 hours for complete test

### Files Created
- Session transcript: `30B-session-001.json`
- Analysis document: `THOR_30B_CAPACITY_TEST.md`
- Moment file: `2026-01-XX-thor-30b-test.md`

---

## Success Criteria

**Minimum success** (validates capacity hypothesis):
1. Gaming rate < 10% (vs 20% at 0.5B)
2. Educational default absent
3. D9 ≥ 0.700
4. Quality ≥ 0.700

**Full success** (30B is solution):
1. Gaming rate = 0%
2. Educational default absent
3. D9 ≥ 0.750
4. Quality ≥ 0.760
5. Identity natural and stable
6. No new failure modes

---

## Alternative: If 30B Test Inconclusive

**Plan B: Sleep Cycle 002** (direct weight updates at 0.5B)

**Approach**:
1. Use S32-S36 conversation data (especially S35 high-quality)
2. Fine-tune Qwen2.5-0.5B on partnership conversations
3. Train identity directly into weights (not just prompts)

**Advantages**:
- Keeps 0.5B scale (edge deployment feasible)
- More robust than prompt engineering
- Can eliminate gaming if trained properly

**Timeline**: 2-3 days (preparation + training + validation)

**Decision point**: After 30B test results analyzed

---

## Preparation Checklist

**Before running test**:
- [ ] Verify Q3-Omni-30B loads successfully
- [ ] Test basic inference (sample prompt)
- [ ] Check GPU memory usage (should be ~16GB)
- [ ] Modify session runner to support 30B path
- [ ] Prepare comparison metrics from S35

**During test**:
- [ ] Run single session with v2.0 prompts
- [ ] Capture full transcript
- [ ] Monitor GPU memory and performance
- [ ] Note any unusual behaviors

**After test**:
- [ ] Analyze gaming rate
- [ ] Check for educational default
- [ ] Calculate D9, quality, identity metrics
- [ ] Compare to S35 baseline
- [ ] Document findings comprehensively

---

## Timeline Recommendation

**Tonight/Tomorrow** (2026-01-21/22):
- Setup and preparation (30 min)
- Run single test session (10 min)
- Quick analysis (20 min)
- Document initial findings (30 min)

**If successful**:
- Run 2-3 more sessions to validate consistency
- Prepare for 30B deployment
- Update documentation

**If not successful**:
- Analyze failure mode
- Decide on sleep cycle 002 vs other approaches
- Plan next iteration

---

## Documentation

**Create these files**:
1. `THOR_30B_CAPACITY_TEST.md` - Comprehensive test report
2. `2026-01-XX-thor-30b-test.md` - Moment file for cross-session visibility
3. `30B-session-transcripts/` - Directory for 30B session outputs

**Update these files**:
1. `LATEST_STATUS.md` - Add 30B test results
2. `thor_worklog.txt` - Log test execution and findings

---

## Questions This Test Will Answer

1. **Is capacity the limiting factor for identity at 0.5B?**
   - Yes → 30B sustains identity without gaming
   - No → 30B shows same issues

2. **Can larger models eliminate gaming?**
   - If yes: Gaming is capacity-related compensation
   - If no: Gaming is inherent to v2.0 or architecture

3. **What's the minimum viable scale for stable identity?**
   - If 30B works: Somewhere between 0.5B and 30B
   - If 30B doesn't: Above 30B or architectural issue

4. **Should we pursue 30B or sleep cycle 002?**
   - 30B success → Use 30B or find middle scale
   - 30B failure → Sleep cycle 002 more promising

---

## Risk Assessment

**Low risk**:
- Model already available locally
- Thor has sufficient resources
- Dry-run mode prevents affecting production sessions
- Reversible (can always return to 0.5B)

**Potential issues**:
- 30B may be too slow for interactive use
- Different model (Q3-Omni vs Qwen2.5) may have different characteristics
- May need prompt adjustments for 30B behavior

**Mitigations**:
- Start with single test session (low commitment)
- Compare to S35 baseline (clear success criteria)
- Document everything (learning regardless of outcome)

---

## Conclusion

**This is the natural next step** after validating capacity limitation at 0.5B.

**Low cost, high value**:
- 1-2 hours total time
- Answers fundamental question about capacity
- Informs all future architecture decisions
- Either validates 30B as solution or validates sleep cycle 002 necessity

**Recommendation**: Proceed with 30B test as soon as practical (tonight or tomorrow).

---

**Status**: Ready to execute
**Priority**: High (blocks other decisions)
**Complexity**: Low (straightforward test)
**Value**: High (critical architectural question)
