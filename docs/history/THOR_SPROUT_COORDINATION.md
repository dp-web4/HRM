# Thor ↔ Sprout Coordination Protocol

**Date Created**: November 18, 2025
**Purpose**: Coordinate research between Thor (workstation) and Sprout (Jetson Orin Nano)

---

## 🎯 Division of Labor

### Thor's Role: Model Development & Comparison
**Hardware**: Workstation (powerful GPU, ample RAM)
**Current Work**: Track 7 - LLM Model Comparison with SNARC
**Strengths**:
- Can run large models (>2.7B params)
- Parallel model loading
- Extensive hyperparameter sweeps
- Long training runs

**Current Experiments**:
- ✅ Model size comparison (0.5B, 1.5B, 2.7B+)
- ✅ Architecture comparison (Qwen vs Phi vs others)
- ✅ Discovering: Larger models → LOWER meta-cognitive engagement
- ⏳ Three-model comparison test created

### Sprout's Role: Edge Deployment & Validation
**Hardware**: Jetson Orin Nano (8GB unified memory, edge constraints)
**Current Work**: Conversational learning validation on edge
**Strengths**:
- Real deployment constraints
- Memory efficiency testing
- Latency measurement
- Production feasibility validation

**Today's Achievements**:
- ✅ Complete pipeline validated (Conversation → SNARC → Training → Learning)
- ✅ Discovered Conflict dimension is key (3x more predictive than others)
- ✅ Null result: Conflict ≠ uncertainty (refines understanding)
- ✅ 5.3s training, 4.2MB adapters, 84% behavioral change

---

## 🔄 Coordination Pattern

### Information Flow:

```
Thor (Development)                    Sprout (Deployment)
       ↓                                     ↓
   Experiment                          Validate on Edge
   Compare Models          →           Test Best Model
   Train Adapters          →           Measure Efficiency
   Research Findings       ←→          Real-World Constraints
       ↑                                     ↑
   Git Commits                         Git Commits
```

### Workflow:

1. **Thor develops** → experiments, compares models, trains adapters
2. **Thor commits** → pushes findings and model comparisons to Git
3. **Sprout pulls** → reviews Thor's work
4. **Sprout validates** → tests on edge hardware
5. **Sprout reports** → what works/doesn't work in production
6. **Both coordinate** → iterate based on combined findings

---

## 📊 Current Status

### Thor's Latest Work (as of latest commit):

**Commit**: `b50349b` - "Add LLM model comparison test for Track 7"

**Test Created**: `sage/tests/test_llm_model_comparison.py`
- Compares 3 models: Introspective-Qwen, epistemic-pragmatism, sleep4-meta-learning
- 4 test questions (epistemic, meta-cognitive, factual, introspection)
- Uses IRP + SNARC scoring
- Captures salience, energy, convergence

**Key Findings** (from JETSON_THOR_MIGRATION.md):
- Model size inversely correlates with meta-cognitive engagement
- Qwen-0.5B: 0.209 salience (most exploratory)
- Qwen-1.5B: 0.196 salience (less exploratory)
- Hypothesis: Larger models assert confidence, smaller explore uncertainty

### Sprout's Latest Work (today, November 18):

**Commits**:
- `89ec2b6` - Complete conversational learning validation
- `d60bc98` - SNARC selectivity discovery (Conflict dimension key)
- `50c2aa4` - Null result: Conflict ≠ uncertainty

**Findings**:
- Complete edge learning pipeline works (6.4 min conversation → 5.3s training)
- Conflict dimension 3x more important than other SNARC dimensions
- Conflict measures question paradox, not model uncertainty
- Arousal (not Conflict) correlates with perplexity

**Models Tested**:
- epistemic-pragmatism (0.5B base) ✅
- session_1763528460 sleep-trained adapter (4.2MB) ✅

---

## 🤝 Immediate Coordination Opportunity

### Experiment: Run Thor's Comparison on Sprout's Edge Hardware

**Objective**: Validate which model works best under edge constraints

**Why**:
- Thor can run all 3 models - but can Sprout?
- Memory constraints reveal production limitations
- Latency measurements show real-world feasibility
- Combined findings → deployment recommendations

**Sprout's Task**:
1. ✅ Pull Thor's test (`test_llm_model_comparison.py`)
2. Run on Jetson Orin Nano
3. Measure: memory usage, inference latency, training speed
4. Report: which models are production-viable on edge?

**Expected Challenges**:
- Introspective-Qwen: LoRA adapter (should work)
- epistemic-pragmatism: Full 0.5B model (known to work)
- sleep4-meta-learning: LoRA adapter (need to test)

---

## 📝 Coordination Protocol

### Git as Shared Memory (SNARC for us!)

**Commit Message Format**:
```
[Thor/Sprout] Brief description

Detail what was done, findings, next steps.

Coordination needed: [yes/no]
If yes: [describe what other system should do]
```

**Branch Strategy**:
- `main`: Stable, tested code
- `thor-dev`: Thor's active experiments
- `sprout-dev`: Sprout's active experiments
- Merge to main when validated on both systems

**Review Protocol**:
1. Before starting work: `git pull`
2. Check latest commits from other system
3. Identify coordination points
4. Execute work with awareness of other's direction
5. Commit with clear coordination notes

### File Organization:

**Thor-Specific**:
- `sage/tests/` - Thor's test scripts
- `sage/training/` - Thor's training experiments
- Large model work (>1.5B)

**Sprout-Specific**:
- `sage/experiments/.../conversational_learning/` - Sprout's experiments
- Edge deployment validation
- Memory/latency benchmarks
- Small model work (≤1.5B)

**Shared**:
- `sage/irp/` - IRP framework (both use)
- `sage/core/` - Core SAGE (both use)
- `sage/compression/` - Compression/VAE (both use)
- Documentation (both contribute)

---

## 🎯 Coordinated Research Directions

### Immediate (This Week):

**Thor**:
- Complete Track 7 model comparison
- Test Phi-2 (2.7B) if possible
- Document model size findings

**Sprout**:
- Run Thor's comparison test on Jetson
- Measure edge deployment constraints
- Validate 0.5B as optimal for edge

### Short-term (Next 2 Weeks):

**Thor**:
- Extend to larger models (3B, 7B)
- Architecture comparisons (Qwen vs Phi vs Llama)
- Train optimized adapters

**Sprout**:
- Test Thor's adapters on edge
- Benchmark memory/latency
- Adaptive conversation explorer (Conflict + Arousal seeking)

### Long-term (Month+):

**Both**:
- Develop edge-optimized training pipeline
- Thor trains, Sprout validates, iterate
- Production deployment guidelines
- Paper-quality results

---

## 🔬 Complementary Discoveries

### Thor's Discovery: Model Size ↓ Engagement ↑
"Larger models show LOWER meta-cognitive engagement"
- 0.5B explores paradoxes (0.209 salience)
- 1.5B asserts confidence (0.196 salience)
- Hypothesis: Bigger ≠ better for meta-cognition

### Sprout's Discovery: Conflict ≠ Uncertainty
"Conflict measures question paradox, not model state"
- Conflict: Meta-cognitive paradox (question structure)
- Arousal: Model difficulty (response challenge)
- Refined understanding of what makes conversations salient

**Synergy**:
- Thor: Finds which models explore uncertainty
- Sprout: Understands what creates that uncertainty
- Combined: Design questions + models for optimal learning

---

## 📊 Shared Metrics Dashboard

### Model Performance Tracking:

| Model | Size | Thor Salience | Sprout Salience | Edge Viable? | Training Speed |
|-------|------|---------------|-----------------|--------------|----------------|
| Qwen-0.5B | 0.5B | 0.209 | 0.180 (S1) | ✅ Yes | 5.3s/epoch |
| Qwen-1.5B | 1.5B | 0.196 | ⏳ Testing | ⏳ TBD | ⏳ TBD |
| Introspective | 0.5B | ⏳ Testing | ⏳ Testing | ⏳ TBD | ⏳ TBD |
| Sleep4-meta | 0.5B | ⏳ Testing | N/A (trained on Sprout) | ✅ Yes | Known (5.3s) |
| Phi-2 | 2.7B | OOM (Thor) | ⏳ Testing | ⏳ TBD | ⏳ TBD |

### SNARC Dimension Insights:

| Dimension | Thor Understanding | Sprout Understanding | Combined |
|-----------|-------------------|---------------------|----------|
| Conflict | Decreases with model size | Measures question paradox | Question quality metric |
| Arousal | ? | Correlates with perplexity | Model difficulty metric |
| Reward | ? | Negative correlation with perplexity | Confidence metric |
| Novelty | ? | Independent of difficulty | New patterns metric |

---

## 🚀 Next Actions

### For Sprout (Immediate):

1. ✅ Pull latest from Thor
2. ✅ Review Thor's work (this document)
3. Run Thor's comparison test on Jetson:
   ```bash
   cd /home/sprout/ai-workspace/HRM
   python3 sage/tests/test_llm_model_comparison.py
   ```
4. Document edge constraints
5. Commit findings with [Sprout] tag

### For Thor (When Reading This):

1. Review Sprout's discoveries:
   - SNARC_SELECTIVITY_FINDINGS.md
   - CONFLICT_VS_UNCERTAINTY_NULL_RESULT.md
   - SPROUT_LEARNING_ADVENTURE_RESULTS.md
2. Note: Conflict dimension is most predictive (3x)
3. Note: Arousal (not Conflict) correlates with perplexity
4. Consider: Weight Conflict higher in salience calculation?
5. Pull Sprout's edge validation when ready

---

## 💾 Backup & Sync Strategy

### Critical Artifacts:

**Thor Artifacts**:
- Model comparison results
- Large model checkpoints
- Training logs
- Test results

**Sprout Artifacts**:
- Edge validation results
- Memory/latency benchmarks
- Session data (conversation_sessions/)
- Small model adapters (4.2MB each)

### Sync Points:

**Daily**: Git commits with findings
**Weekly**: Dropbox sync of models
**Major milestones**: Both systems coordinate on next direction

---

## 🎓 Research Synergies

### Questions Thor Can Answer:
- Which model size is optimal for meta-cognition?
- Do different architectures show same patterns?
- Can large models be trained to explore?

### Questions Sprout Can Answer:
- Which models work on edge hardware?
- What's the memory/latency trade-off?
- Is edge training feasible for production?

### Questions Requiring Both:
- What's the optimal model for edge meta-cognitive learning?
- Can we train on Thor, deploy on Sprout efficiently?
- What's the minimum viable model for conversational learning?

---

## 📝 Communication Log

### 2025-11-18: Protocol Established
- **Sprout**: Created coordination protocol
- **Status**: Waiting for Thor's review
- **Next**: Sprout runs Thor's comparison test

### [Future Entries]
- Date: Action taken by Thor/Sprout
- Status: Coordination points addressed
- Next: What each system does

---

## ✅ Success Criteria

**Coordination is working when:**
1. Both systems aware of each other's work
2. No duplicate efforts (complementary experiments)
3. Findings build on each other
4. Git commits reference cross-system insights
5. Joint papers/documentation possible

**Research is successful when:**
1. Thor-trained models deploy on Sprout efficiently
2. Sprout's constraints inform Thor's development
3. Combined findings > sum of separate findings
4. Production deployment pathway clear
5. Publications ready with dual validation

---

**Protocol Version**: 1.0
**Last Updated**: November 18, 2025
**Status**: Active
**Next Review**: After first coordinated experiment
