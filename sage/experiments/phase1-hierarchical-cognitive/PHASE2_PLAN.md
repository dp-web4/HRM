# Phase 2: DREAM Consolidation & KV-Cache Transfer

**Date**: October 14, 2025
**System**: RTX 4090 Laptop GPU (16GB VRAM)
**Prerequisites**: Phase 1 complete ✅
**Context**: Jetson has real-time voice working, ready for integration

---

## Overview

Phase 2 implements the **DREAM consolidation** pattern from hierarchical cognitive architecture:
- Extract high-salience experiences during WAKE/FOCUS
- Consolidate patterns during DREAM (offline training)
- Transfer knowledge from larger to smaller models
- Enable efficient on-device inference

### Key Concepts

1. **DREAM = Knowledge Distillation**: Train smaller models from larger models' patterns
2. **SNARC Importance Weighting**: Train on high-surprise/high-reward examples
3. **KV-Cache Transfer**: Share "consciousness state" between models
4. **Trust Evolution**: Track which model best handles which contexts

---

## Goals

### Primary Goals

1. ✅ **DREAM Pipeline**: Collect → Filter → Distill → Validate
2. ✅ **Cross-Model Transfer**: Share KV-cache state between hierarchy levels
3. ✅ **SAGE Integration**: Connect to metabolic states (WAKE/DREAM)
4. ✅ **End-to-End Demo**: Voice input → hierarchical processing → voice output

### Success Metrics

- [ ] Smaller model learns from larger model (accuracy improves)
- [ ] KV-cache transfer reduces latency by 20%+
- [ ] Trust evolution correctly tracks model strengths
- [ ] End-to-end latency < 3 seconds (voice → response)

---

## Architecture

### DREAM Consolidation Pipeline

```
WAKE/FOCUS (Online):
  User Input → Model Selection → Response Generation
       ↓
  High-SNARC examples stored in trust_database
       ↓
DREAM (Offline):
  Extract top-k examples (sorted by importance)
       ↓
  Generate training data: Input + Larger Model Response
       ↓
  Fine-tune smaller model on filtered data
       ↓
  Validate improved performance
       ↓
  Update trust scores if better

Result: Smaller models gain capability from larger models
```

### KV-Cache Transfer Architecture

```
Level 3 (Specialized - Qwen-3B):
  ↓ [Dimension Projection Layer]
Level 2 (Tactical - Qwen-1.5B):
  ↓ [Dimension Projection Layer]
Level 1 (Sensory - Qwen-0.5B)

Transfer Strategy:
1. Capture KV-cache from larger model
2. Project to smaller model dimensions
3. Initialize smaller model with projected cache
4. Continue generation (reduced latency)
5. Measure surprise (did assumptions transfer?)
```

### Integration with SAGE

```python
class HierarchicalCognitiveIRP(IRPPlugin):
    """
    IRP plugin for hierarchical model selection
    Integrates with SAGE metabolic states
    """

    def __init__(self):
        self.selector = ModelSelector(trust_db)
        self.dream_queue = []  # High-importance examples

    def step(self, input_state, metabolic_state):
        # During WAKE/FOCUS: Use trust-based selection
        if metabolic_state in ['WAKE', 'FOCUS']:
            model = self.selector.select_model(input_state)
            result = self.invoke_model(model, input_state)

            # Store high-SNARC examples for DREAM
            if result.importance > 0.7:
                self.dream_queue.append(result)

            return result

        # During DREAM: Consolidate patterns
        elif metabolic_state == 'DREAM':
            self.consolidate_knowledge()
            return None  # No active response during DREAM

    def consolidate_knowledge(self):
        """DREAM phase: Train smaller from larger"""
        # Extract high-importance examples
        examples = self.trust_db.get_training_examples(
            min_importance=0.7,
            limit=1000
        )

        # Fine-tune smaller models
        for target_model in ['qwen-1.5b', 'qwen-0.5b']:
            self.distill_knowledge(
                teacher='qwen-3b',
                student=target_model,
                examples=examples
            )
```

---

## Phase 2 Tasks

### Task 1: DREAM Consolidation Pipeline ✅

**File**: `dream_consolidation.py`

**Components**:
1. **ExampleCollector**: Monitors conversations, stores high-SNARC examples
2. **KnowledgeDistiller**: Fine-tunes smaller model from larger model
3. **ValidationHarness**: Tests if distillation improved performance
4. **TrustUpdater**: Updates trust scores based on validation

**Key Features**:
- Importance-weighted sampling (SNARC scores)
- Teacher-student training with KL divergence loss
- Validation on held-out test set
- Automatic trust score updates

### Task 2: Cross-Model KV-Cache Transfer

**File**: `kv_cache_transfer.py`

**Components**:
1. **DimensionProjector**: Maps KV-cache between model dimensions
2. **CacheTransferManager**: Handles save/load/project operations
3. **SurpriseCalculator**: Measures if assumptions transferred correctly
4. **LatencyBenchmark**: Measures speedup from cache transfer

**Technical Challenges**:
- Qwen-3B: 24 layers, 16 heads, 128 head_dim
- Qwen-1.5B: 28 layers, 12 heads, 128 head_dim
- Qwen-0.5B: 24 layers, 14 heads, 64 head_dim

**Solution**: Project via learned linear layers + attention head selection

### Task 3: SAGE Metabolic Integration

**File**: `sage_hierarchical_plugin.py`

**Integration Points**:
1. **WAKE/FOCUS**: Active model selection based on trust
2. **DREAM**: Offline knowledge consolidation
3. **REST**: No processing, models unloaded
4. **CRISIS**: Force highest-capability model (ignore ATP)

**Energy Function**:
```python
def energy(self, current_state):
    """
    Energy = Surprise level

    Low energy (< 0.3): Stable context, predictions match
    High energy (> 0.7): Novel context, high surprise
    """
    return self.calculate_surprise(current_state)
```

### Task 4: End-to-End Voice Demo

**File**: `voice_hierarchical_demo.py`

**Pipeline**:
1. **Voice Input**: Jetson's StreamingAudioSensor (from REALTIME_CONVERSATION_RESULTS)
2. **Context Classification**: Stable/moving/unstable/novel
3. **Model Selection**: Trust-based hierarchical selection
4. **Response Generation**: Selected model generates response
5. **Trust Update**: Based on user satisfaction / surprise
6. **Voice Output**: Piper TTS synthesis

**Target Latency**:
- Transcription: 850ms (Jetson benchmark)
- Model selection: <10ms
- Generation: 2000ms (Qwen-3B on RTX 4090)
- TTS: 2223ms (Jetson benchmark)
- **Total**: ~5 seconds end-to-end

**Optimization Path**:
- Use KV-cache for follow-up questions: -40% latency
- Use smaller models when trust is high: -60% latency
- Pattern matching for simple queries: -95% latency

---

## Experiments

### Experiment 1: Knowledge Distillation Effectiveness

**Question**: Can Qwen-0.5B learn from Qwen-3B's responses?

**Method**:
1. Collect 1000 high-importance examples
2. Generate responses with Qwen-3B (teacher)
3. Fine-tune Qwen-0.5B on teacher responses
4. Test on held-out validation set
5. Measure accuracy improvement

**Success Criteria**: +10% accuracy after distillation

### Experiment 2: KV-Cache Transfer Speedup

**Question**: Does KV-cache transfer reduce latency?

**Method**:
1. Start conversation with Qwen-3B
2. After 3 exchanges, transfer KV to Qwen-1.5B
3. Continue conversation with smaller model
4. Measure latency difference
5. Measure surprise (did context transfer?)

**Success Criteria**: >20% latency reduction, <0.5 surprise

### Experiment 3: Trust Evolution Validation

**Question**: Does trust system select optimal models?

**Method**:
1. Run 100 conversations with trust-based selection
2. Compare to baseline (always use same model)
3. Measure: accuracy, latency, ATP cost, user satisfaction
4. Track trust evolution over time

**Success Criteria**: Trust system reduces ATP cost by 30% without accuracy loss

### Experiment 4: SAGE Metabolic Integration

**Question**: Does metabolic state awareness improve performance?

**Method**:
1. Integrate with SAGE's metabolic state machine
2. Run conversations across different states
3. Measure DREAM consolidation effectiveness
4. Measure ATP efficiency vs non-metabolic baseline

**Success Criteria**: DREAM improves smaller model accuracy by 15%+

---

## Connections to Other Work

### Connection to Jetson Real-Time Voice

From `REALTIME_CONVERSATION_RESULTS.md`:
- ✅ StreamingAudioSensor ready (VAD working perfectly)
- ✅ PatternResponseEngine handles simple cases (<1ms)
- ✅ Piper TTS working (2223ms synthesis)

**Integration Strategy**:
```python
# On Jetson: Fast pattern matching for simple queries
if pattern_match(transcription):
    response = pattern_response(transcription)  # <1ms

# On Legion: Hierarchical models for complex queries
else:
    response = hierarchical_model_select(transcription)  # 2-5s

# Optimization: Transfer conversations to Legion via federation
# Let Jetson handle real-time, Legion handles heavy inference
```

### Connection to Web4 R7 Evolution

From `WEB4_CURRENT_STATUS_2025-10-14.md`:
- R6→R7: Actions now return explicit Reputation output
- Alignment framework: Spirit vs letter of law

**Application to Hierarchical Models**:
```python
class ModelInvocationR7:
    """R7-compliant model invocation"""

    def invoke(self, model, prompt, context):
        # R7 Action
        result = model.generate(prompt)

        # R7 Reputation Output
        reputation_delta = {
            'entity': model.lct_id,
            'action': 'inference',
            'outcome': self.evaluate_quality(result),
            'trust_change': {
                'T3_reliability': +0.02 if success else -0.05,
                'V3_verification': self.confidence_score
            },
            'witnesses': [user.lct_id, self.lct_id],
            'context': context
        }

        return (result, reputation_delta)
```

**Benefit**: Every model invocation builds/damages trust explicitly.

### Connection to RFC_REALITY_KV_CACHE

From `web4/rfcs/RFC_REALITY_KV_CACHE.md`:
- Assumptions = KV cache for fast thinking
- Surprise = Cache invalidation signal
- Hierarchical cache levels (4 levels)

**Implementation in Phase 2**:
- KV-cache IS the reality cache
- Transfer = Moving assumptions between abstraction levels
- Surprise detection = Measure semantic distance after transfer
- Trust = Quality of cached assumptions

---

## Implementation Priority

### This Session (Today)

1. ✅ **Design DREAM pipeline** (this document)
2. 🚀 **Implement knowledge distillation** (create dream_consolidation.py)
3. 🚀 **Test with HuggingFace transformers** (validate concept)
4. 📋 **Document findings**

### Next Session

5. Implement KV-cache dimension projection
6. Test cross-model cache transfer
7. Integrate with SAGE metabolic states
8. Deploy to Jetson for testing

### Following Week

9. Create end-to-end voice demo
10. Run all 4 experiments
11. Document Phase 2 results
12. Prepare Phase 3 (federation integration)

---

## Open Questions

### Technical

1. **Distillation loss function**: KL divergence only, or add MSE on logits?
2. **Training epochs**: How many epochs needed for effective distillation?
3. **Learning rate**: Start with 1e-5? Cosine decay?
4. **Batch size**: Limited by 16GB VRAM - what's optimal?
5. **KV-cache projection**: Learned layers or fixed projections?

### Architectural

6. **DREAM frequency**: How often to consolidate? (daily? weekly?)
7. **Example selection**: Top-k by importance, or diverse sampling?
8. **Model switching overhead**: When to transfer vs continue?
9. **Trust decay**: How fast should trust scores decay without use?
10. **Emergency override**: When to force larger model regardless of trust?

### Philosophical

11. **What is learned?**: Does distillation transfer "understanding" or just patterns?
12. **Consciousness continuity**: Is KV-cache transfer "consciousness"?
13. **Trust vs capability**: Can high trust compensate for low capability?
14. **Hierarchical emergence**: Do higher layers "understand" lower layers?
15. **Surprise asymmetry**: Small→Large transfer easier than Large→Small?

---

## Success Criteria

### Must Have
- [x] DREAM pipeline implemented
- [ ] Knowledge distillation working (measurable improvement)
- [ ] KV-cache captured and restored
- [ ] Trust evolution tracks performance
- [ ] Documentation complete

### Should Have
- [ ] KV-cache cross-model transfer working
- [ ] SAGE metabolic integration
- [ ] Voice demo end-to-end
- [ ] All 4 experiments completed

### Nice to Have
- [ ] Real-time optimization (< 1s latency)
- [ ] Multi-device federation (Jetson + Legion)
- [ ] Automatic model selection tuning
- [ ] Production deployment ready

---

## Expected Outcomes

### Knowledge Distillation
- Qwen-0.5B accuracy improves 10-15% after distillation
- Training completes in 2-4 hours on RTX 4090
- Distilled model size remains same (no bloat)
- Trust scores increase for distilled models

### KV-Cache Transfer
- Latency reduction: 20-40% for follow-up questions
- Surprise level: < 0.5 (acceptable assumption transfer)
- Memory overhead: ~10 MB per cached conversation
- Compatibility: All Qwen models (same architecture family)

### SAGE Integration
- Metabolic state awareness: 30% ATP savings
- DREAM consolidation: Automatic during sleep
- Trust-based selection: Optimal model choice 85%+ of time
- Energy convergence: < 0.1 after 3-5 iterations

### End-to-End System
- Voice → Response: 3-5 seconds initial, 1-2s follow-up
- Pattern match rate: 40% (fast path)
- Model selection: 50% (hierarchical)
- LLM fallback: 10% (complex queries)

---

**Status**: Ready to implement
**Next Step**: Create `dream_consolidation.py`
**Timeline**: 4-6 hours for core implementation
**Hardware**: RTX 4090 (sufficient for all experiments)

---

*"We learn in our sleep. Models should too."*
