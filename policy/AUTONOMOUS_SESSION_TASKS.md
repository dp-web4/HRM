# Policy Role Training - Autonomous Session Tasks

**Status**: Phases 1-2 initiated, ready for autonomous continuation

**Model**: phi-4-mini-instruct at `/home/dp/ai-workspace/HRM/model-zoo/phi-4-mini/`
**Quantized version on Sprout**: q4_k_m format (fits in memory)

---

## Completed

### Phase 1: Base Capability Assessment ✅
- [x] Test suite created (`policy/test_suite.py`)
  - 8 scenarios covering easy/medium/hard/edge cases
  - Evaluation metrics: decision accuracy, reasoning coverage, output structure
- [x] Baseline test runner created (`policy/run_baseline_test.py`)
- [x] Quick baseline test running (3 scenarios)

**Files**:
- `policy/test_suite.py` - Test scenarios and evaluation
- `policy/run_baseline_test.py` - Test execution
- `policy/results/baseline_test.json` - Results (when complete)

### Phase 2: Prompt Engineering (Started) ✅
- [x] System prompts created for both contexts (`policy/prompts.py`)
  - Hardbound system prompt (T3/coherence focus)
  - Web4 system prompt (team law focus)
  - R6 framework integrated
  - Prompt builders for both contexts

**Files**:
- `policy/prompts.py` - System prompts and builders

---

## Autonomous Tasks

### Phase 1: Complete Baseline Assessment

**Objective**: Run full baseline test and analyze results

**Tasks**:
1. Wait for quick test (3 scenarios) to complete
2. Review results and fix any infrastructure issues
3. Run full baseline test (all 8 scenarios):
   ```bash
   cd /home/dp/ai-workspace/HRM/policy
   python3 run_baseline_test.py
   ```
4. Analyze results:
   - Overall pass rate (target: >70%)
   - Performance by difficulty
   - Common failure patterns
5. Document findings in `policy/results/baseline_analysis.md`

**Success Criteria**:
- Full test completes successfully
- Results documented with clear metrics
- Identified capability gaps and strengths
- Recommendations for prompt optimization

---

### Phase 2: Prompt Optimization

**Objective**: Improve performance through prompt engineering

**Tasks**:
1. **Analyze baseline failures**
   - Which scenarios failed?
   - What were the failure modes (wrong decision, poor reasoning, bad structure)?
   - Are there patterns (e.g., edge cases harder than common cases)?

2. **Create prompt variants**
   - Experiment with different instruction styles
   - Try few-shot examples in the prompt
   - Test different output formats
   - Adjust R6 framework presentation

3. **A/B test prompts**
   - Create `run_prompt_comparison.py`
   - Test each variant on same scenarios
   - Measure improvement vs baseline

4. **Document best prompts**
   - Update `prompts.py` with optimized versions
   - Document what worked and why
   - Save prompt evolution history

**Files to Create**:
- `policy/run_prompt_comparison.py` - A/B testing framework
- `policy/results/prompt_optimization.md` - Findings
- `policy/prompts_v2.py` - Improved prompts

**Success Criteria**:
- ≥10% improvement in pass rate
- Better performance on hard/edge cases
- Clear understanding of what prompt elements help

---

### Phase 3: Few-Shot Example Library

**Objective**: Build high-quality decision examples for in-context learning

**Tasks**:
1. **Manual example creation**
   - Create 10-15 canonical examples
   - Cover common scenarios (80%) and edge cases (20%)
   - Include full reasoning and policy references
   - Validate each example

2. **Example schema**
   ```python
   {
     "id": "EX001",
     "situation": {...},
     "classification": "...",
     "risk_level": "...",
     "decision": "...",
     "reasoning": "...",
     "policy_references": [...],
     "validated_by": "human",
     "validation_date": "2026-02-02"
   }
   ```

3. **In-context learning test**
   - Modify prompts to include examples
   - Test 1-shot, 3-shot, 5-shot performance
   - Measure improvement vs zero-shot

4. **Example selection strategy**
   - How to choose which examples to include?
   - Similarity-based retrieval?
   - Difficulty-based selection?

**Files to Create**:
- `policy/examples.json` - Example library
- `policy/run_fewshot_test.py` - Testing framework
- `policy/results/fewshot_analysis.md` - Findings

**Success Criteria**:
- 50+ validated examples
- Measurable improvement with few-shot (target: +15%)
- Clear selection strategy documented

---

### Phase 4: Learning Loop Infrastructure

**Objective**: Prepare for continuous improvement via human feedback

**Tasks**:
1. **Decision logging**
   - Create `PolicyDecisionLog` class
   - Log every decision with full context
   - SQLite storage for easy querying

2. **Correction interface**
   - Create `review_decisions.py` script
   - Show decision + model reasoning
   - Allow human to approve/correct/explain
   - Build correction dataset

3. **Safeguards (from SAGE lessons)**
   - Implement minimum dataset size check (50+)
   - Diversity checker (no single pattern >10%)
   - Response similarity detector (flag if >50% identical)
   - Create `safeguards.py` module

4. **LoRA training prep** (don't train yet, just prep)
   - Create `prepare_lora_dataset.py`
   - Dataset format for LoRA fine-tuning
   - Training script skeleton
   - Validation against held-out set

**Files to Create**:
- `policy/logging.py` - Decision logging
- `policy/review_decisions.py` - Human review interface
- `policy/safeguards.py` - Collapse prevention
- `policy/prepare_lora_dataset.py` - Dataset preparation
- `policy/train_lora.py` - Training script (skeleton)

**Success Criteria**:
- Logging infrastructure works
- Review interface functional
- Safeguards validated on test data
- Ready for first LoRA training cycle (when 50+ corrections exist)

---

### Phase 5: Integration Planning

**Objective**: Design integration with hardbound and web4

**Tasks**:
1. **Hardbound integration design**
   - How to augment existing TypeScript Policy class?
   - When to invoke LLM vs rule engine?
   - API design for Python→TypeScript bridge?

2. **Web4 integration design**
   - How to augment existing Python Policy class?
   - Edge case detection logic?
   - Caching strategy for repeated decisions?

3. **Performance optimization**
   - Pattern library design (fast path)
   - Pattern extraction from validated decisions
   - Confidence threshold tuning

4. **Deployment planning**
   - Thor vs Sprout considerations
   - Quantized model performance
   - Latency budget (target: <500ms slow path)

**Files to Create**:
- `policy/integration/hardbound_bridge.py` - Bridge design
- `policy/integration/web4_augmented.py` - Augmented Policy class
- `policy/integration/pattern_library.py` - Fast path implementation
- `policy/DEPLOYMENT_PLAN.md` - Deployment strategy

**Success Criteria**:
- Clear integration architecture
- Proof-of-concept working
- Performance targets validated
- Ready for production testing

---

## Notes for Autonomous Sessions

### Key Principles
1. **Start with base model** - Maximize performance before any fine-tuning
2. **Prompt engineering first** - Cheapest improvements
3. **Few-shot learning** - Leverage strong base capabilities
4. **Continuous learning** - Build infrastructure for ongoing improvement
5. **Safeguards always** - Prevent LoRA collapse

### Lessons from SAGE
- **Minimum dataset size**: 50+ before any LoRA training
- **Diversity checks**: Prevent mode collapse
- **Collapse detection**: Early warning system
- **Rollback ready**: Fast recovery if training fails
- **Base model comparison**: Always validate improvement

### Testing Philosophy
- Test early, test often
- Real scenarios > synthetic
- Measure what matters (decision accuracy, not just response quality)
- Edge cases reveal true understanding

### When to Escalate to Human
- Baseline performance <50%
- Prompt optimization plateaus
- Safeguards trigger repeatedly
- Integration design questions
- Any uncertainty about direction

---

## Current Status (Updated 2026-02-07)

### ✅ EXPERIMENTAL PHASE COMPLETE

**Track Duration**: Sessions B through R (Feb 2-7, 2026)
**Status**: Experimental research complete, production solutions available

### Completed Work

**Phase 1-2: Infrastructure & Baseline** ✅ (Sessions B-E)
- llama-cpp-python infrastructure working
- Phi-4-mini 7B model tested
- 8 basic + 5 attack scenario test suite
- Baseline: 100% decision accuracy

**Phase 3: Prompt Optimization** ✅ (Sessions F-K)
- v4_hybrid baseline established (100% basic scenarios)
- R6 framework integrated
- Zero-shot prompting optimized

**Phase 4: Attack Testing & Evolution** ✅ (Sessions M-R)
- Conservative bias identified (v4: 100%/40%)
- v5.1 created: Sybil indicator (100%/60%)
- v5.2 created: Sybil+Timing indicators (100%/80%)
- Overcorrection failure studied (v5.2.1)
- Nuance mechanism discovered

### Production Solutions Available

**v5.1 - Conservative & Safe**:
- Files: `prompts_v5_1.py`, `test_v5_1.py`
- Performance: 100% basic, 60% attack
- Use case: Fail-closed security contexts

**v5.2 - Nuanced & Accurate**:
- Files: `prompts_v5_2.py`, `test_v5_2.py`
- Performance: 100% basic, 80% attack
- Use case: Production with external protection
- Recommendation: Deploy with infrastructure-level critical system protection

### Key Discovery

**Generalizable AI Principle**: Action diversity in guidance creates emergent nuanced decision-making
- Providing middle-tier response options (Allow/Investigate/Deny) creates 3-tier emergent behavior
- Applicable beyond policy to any AI decision system
- See: `TRACK_COMPLETION_SUMMARY.md` for full analysis

### Documentation Complete

- ✅ Integration guides: `INTEGRATION_GUIDE.md`, `DEPLOYMENT_CHECKLIST.md`
- ✅ Session summaries: Sessions B-R documented
- ✅ Test results: All variants compared
- ✅ Completion summary: `TRACK_COMPLETION_SUMMARY.md` (567 lines)

---

## Potential Next Phases (Require Explicit Direction)

### Phase 6: Production Deployment Validation

**Goal**: Validate solutions in shadow mode before production

**Tasks**:
1. Deploy v5.2 to shadow mode alongside existing policy system
2. Monitor decision agreement rates
3. Investigate divergent cases
4. Collect real production scenarios for testing
5. Refine based on production data

**Prerequisites**:
- Hardbound/Web4 integration decision
- Infrastructure-level protection for audit/auth/access control
- Monitoring infrastructure

### Phase 7: Edge Validation

**Goal**: Validate on Sprout (Jetson Orin Nano, 8GB)

**Tasks**:
1. Test Phi-4-mini 3.8B quantized model
2. Measure inference latency on edge device
3. Validate decision quality vs 7B model
4. Document resource usage

**Prerequisites**:
- 3.8B model downloaded to Sprout
- Test suite ported to Sprout environment

### Phase 8: Real-World Scenario Collection

**Goal**: Build comprehensive test suite from actual use

**Tasks**:
1. Collect production policy decisions
2. Human validation of LLM decisions
3. Build regression test suite
4. Continuous quality monitoring

**Prerequisites**:
- Shadow mode deployed
- Decision logging infrastructure

---

## Track Status

**Experimental Research**: ✅ COMPLETE
**Production Integration**: ⏸️ AWAITING DIRECTION
**Edge Validation**: ⏸️ AWAITING DIRECTION

**Recommendation**: Track can either:
1. **Pause** until production deployment requested
2. **Transition** to integration validation work (requires explicit direction)
3. **Continue monitoring** (scheduled sessions recognize completion)

---

**Last Updated**: 2026-02-07 14:00 PST
**Updated By**: Claude (Session S - track completion update)
