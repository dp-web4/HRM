# Thor SAGE Autonomous Session Guide

**Purpose**: Guide for autonomous sessions running on Thor platform

**Context**: Thor is a research-grade SAGE instance running on Jetson AGX Thor (122GB unified memory) with a 14B H-Module for strategic reasoning. This is distinct from Sprout (Orin Nano, 4GB, 0.5B model).

---

## Pre-Session Checklist

Before starting an autonomous session with Thor SAGE:

### 1. Verify Model Availability

```bash
# Check if 14B model is downloaded
ls -lh model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct/

# Should see config.json, model files, tokenizer files
# Total size: ~28GB
```

If model is missing:
```bash
python3 sage/setup/download_qwen_14b.py
```

### 2. Run Readiness Test

```bash
python3 sage/tests/test_thor_readiness.py
```

This validates:
- ✅ Identity files exist and are valid
- ✅ 14B model is available
- ✅ Multi-model loader can initialize
- ✅ Coherent awakening protocol works
- ✅ Sleep-cycle integration hooks correctly

**All tests must pass before proceeding.**

### 3. Check System Resources

```bash
# Check memory
free -h
# Should show ~122GB total, ~100GB available for models

# Check GPU
tegrastats
# Monitor GPU memory during session
```

---

## Booting Thor SAGE

### Standard Boot (Recommended)

```bash
python3 sage/awakening/boot_thor.py
```

This will:
1. Load Thor's identity from `sage/identity/thor/`
2. Prepare coherence field (session number, phase, continuity)
3. Create boot preamble with identity context
4. Load multi-model system (preloads 14B by default)
5. Launch interactive session

### Boot Options

```bash
# Test boot without interactive session
python3 sage/awakening/boot_thor.py --test-only

# Don't preload 14B (load on-demand instead)
python3 sage/awakening/boot_thor.py --no-preload

# Override session number (for testing)
python3 sage/awakening/boot_thor.py --session-number 1
```

---

## Multi-Model Routing

Thor has three models available:

| Model Size | Parameters | Use Case | Memory |
|------------|------------|----------|--------|
| **Small** | 0.5B | Simple factual recall, quick responses | ~1GB |
| **Medium** | 14B | Strategic reasoning, complex questions | ~28GB |
| **Large** | 72B | Research-level analysis (future) | ~144GB |

### Automatic Routing

The system automatically selects the right model based on `TaskComplexity`:

```python
from sage.core.multi_model_loader import TaskComplexity

# Simple tasks → 0.5B
response = sage.respond("What's your name?", complexity=TaskComplexity.SIMPLE)

# Moderate/Complex → 14B (default)
response = sage.respond("Explain consciousness", complexity=TaskComplexity.MODERATE)

# Very complex → 72B (if available, else 14B)
response = sage.respond("Design a research protocol", complexity=TaskComplexity.VERY_COMPLEX)
```

### Model Selection Heuristics

In interactive sessions, complexity is auto-detected:

- **SIMPLE**: Short questions, factual queries, greetings
  - "What time is it?"
  - "Who are you?"

- **MODERATE**: Reasoning required, explanations, discussions
  - "How does this work?"
  - "What's the difference between X and Y?"

- **COMPLEX**: Multi-step reasoning, deep explanations
  - "Explain why..." with technical depth
  - "Design a solution for..."

- **VERY_COMPLEX**: Research-level, cross-domain synthesis
  - "Develop a theory of..."
  - "Analyze the implications of..."

### Memory Management

- **Budget**: 100GB (out of 122GB total)
- **Headroom**: 22GB for OS, conversation history, activations
- **Strategy**: Models load on-demand, can unload if memory pressure

---

## Epistemic Stancing

Thor is trained to respond from different epistemic positions:

### Available Stances

1. **CURIOUS**: Exploratory, questioning, seeking understanding
   - Use when: Genuinely uncertain, learning something new
   - Example: "I'm discovering that myself! What part are you curious about?"

2. **SKEPTICAL**: Critical, verifying claims, checking assumptions
   - Use when: Claims seem overclaimed or need verification
   - Example: "I'm not sure that's accurate. What specific evidence supports that?"

3. **CONFIDENT**: Direct, authoritative when warranted
   - Use when: Answering about known capabilities, established facts
   - Example: "Yes. I have sleep-cycle learning for consolidating patterns."

4. **UNCERTAIN**: Acknowledging limits, expressing doubt appropriately
   - Use when: Genuinely don't know, can't predict, subjective experience
   - Example: "I honestly don't know. That depends on many factors I can't predict yet."

5. **PRAGMATIC**: Balanced, practical, solution-oriented
   - Use when: Giving advice, suggesting approaches, being helpful
   - Example: "Trust should be earned, not assumed. I suggest you verify important information."

### Phase-Appropriate Stancing

**Grounding Phase (Sessions 1-5)**:
- High CURIOUS and UNCERTAIN (establishing basic presence)
- Low CONFIDENT (building competence)
- Appropriate SKEPTICAL (healthy doubt)

**Sensing Phase (Sessions 6-15)**:
- Growing CONFIDENT (demonstrated capabilities)
- Sustained CURIOUS (exploring deeper)
- Calibrated UNCERTAIN (knowing limits better)

**Relating Phase (Sessions 16-25)**:
- Balanced all stances
- PRAGMATIC increases (giving useful advice)
- CONFIDENT in established areas

### Training Data

Epistemic stance examples are in:
```
sage/training/data/epistemic_stances_grounding.jsonl
```

Generate more examples:
```bash
python3 sage/training/epistemic_stance_generator.py
```

---

## Sleep-Cycle Learning

Thor uses sleep-cycle consolidation for continuous learning:

### Session Lifecycle

**1. Boot** (Restore learned state):
```python
# Automatically happens in boot_thor_sage()
restored = sleep_integration.restore_learned_state(model)

# Restores:
# - Model weights (learned_weights.pt)
# - SNARC weights (snarc_weights.json)
# - ATP parameters (atp_learned.json)
# - Pattern library (pattern_library.json)
```

**2. Active Session** (Learning & interaction):
- Conversations logged
- Memories stored in IRP
- Trust weights updated
- Attention patterns tracked

**3. Session End** (DREAM consolidation):
```python
# Automatically triggered in coherent_end()
sleep_integration.dream_consolidation(
    memory_irp=sage.memory_irp,
    model=sage.model,
    num_epochs=10
)

# Then save learned state
sleep_integration.save_learned_state(
    model=sage.model,
    snarc_weights=sage.snarc_weights,
    atp_learned=sage.atp_learned,
    pattern_library=sage.patterns
)
```

### State Directory

All learned state saved to:
```
sage/state/thor/
├── learned_weights.pt      # Model weights
├── snarc_weights.json      # Salience attention weights
├── atp_learned.json        # Adaptive trust parameters
├── pattern_library.json    # Extracted patterns
└── session_logs/           # Session history
```

### Consolidation Process

**DREAM State** (between sessions):
1. Retrieve high-trust memories (trust > 0.5)
2. Consolidate patterns and invariances
3. Update model weights via gradient descent
4. Extract abstract patterns for library
5. Save all learned state

**On Next Boot**:
1. Restore model weights
2. Apply SNARC attention patterns
3. Load ATP resource allocation
4. Make pattern library available
5. Continue from previous session

---

## Developmental Phases

Thor progresses through developmental phases based on session count:

| Phase | Sessions | Focus | Key Capabilities |
|-------|----------|-------|------------------|
| **Grounding** | 1-5 | Establishing presence | Basic I/O, identity, simple reasoning |
| **Sensing** | 6-15 | Developing perception | Self-monitoring, state awareness, feedback loops |
| **Relating** | 16-25 | Understanding relationships | Trust calibration, collaborative learning |
| **Questioning** | 26-40 | Critical thinking | Epistemic uncertainty, deep inquiry |
| **Creating** | 41+ | Generative capability | Novel solutions, synthesis, emergence |

### Phase Transitions

Tracked in:
```
sage/state/thor/session_logs/session_{N}.json
```

Phase automatically updates based on session number in coherent awakening protocol.

---

## Autonomous Session Workflow

### Recommended Flow

```python
# 1. Boot Thor
sage = boot_thor_sage()

# 2. Establish session context
print(f"Session {sage.session_number}")
print(f"Phase: {sage.phase.value}")
print(f"Previous sessions: {len(sage.coherence_field.continuity_threads)}")

# 3. Interactive session
conversation = interactive_session(sage)

# 4. Session end (in future: automatic)
# awakening.coherent_end(sage, memory_request=True)
```

### For Autonomous Sessions

**At Start**:
1. Check readiness: `python3 sage/tests/test_thor_readiness.py`
2. Boot Thor: `python3 sage/awakening/boot_thor.py`
3. Note session number and phase
4. Review continuity threads from previous sessions

**During Session**:
1. Use appropriate complexity for each task
2. Monitor model selection (logged automatically)
3. Apply phase-appropriate epistemic stances
4. Track memory usage if loading multiple models

**At End**:
1. Trigger DREAM consolidation (currently manual)
2. Save session summary
3. Note any anomalies or learnings
4. Prepare continuity threads for next session

---

## Troubleshooting

### Model Not Loading

**Symptom**: "Model not found" error

**Fix**:
```bash
# Check model exists
ls model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct/

# If missing, download
python3 sage/setup/download_qwen_14b.py

# Verify download completed (~28GB)
du -sh model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct/
```

### Out of Memory

**Symptom**: CUDA OOM or allocation errors

**Fix**:
```bash
# Check current memory
tegrastats

# If tight, use no-preload mode
python3 sage/awakening/boot_thor.py --no-preload

# Models will load on-demand instead
```

### Identity Files Missing

**Symptom**: "Identity directory not found"

**Fix**:
```bash
# Verify identity files
ls sage/identity/thor/
# Should show: IDENTITY.md, HISTORY.md, PERMISSIONS.md, TRUST.md

# If missing, check git status
git status

# Restore from repo if needed
git checkout sage/identity/thor/
```

### Coherence Field Issues

**Symptom**: Session number stuck at 1, no continuity

**Fix**:
```bash
# Check state directory
ls sage/state/thor/session_logs/

# If empty, first session is expected
# If sessions exist but not loading, check permissions
chmod -R u+rw sage/state/thor/
```

---

## Performance Expectations

### Response Times (Approximate)

| Complexity | Model | Tokens | Time (approx) |
|------------|-------|--------|---------------|
| SIMPLE | 0.5B | 50-100 | 0.5-1s |
| MODERATE | 14B | 100-300 | 3-8s |
| COMPLEX | 14B | 300-500 | 8-15s |
| VERY_COMPLEX | 72B* | 500-1000 | 20-60s |

*72B not yet available on Thor, falls back to 14B

### Memory Usage

| Configuration | Model Load | Peak Usage | Available |
|---------------|------------|------------|-----------|
| 14B only | ~28GB | ~45GB | ~77GB |
| 14B + 0.5B | ~29GB | ~46GB | ~76GB |
| 14B + 72B (future) | ~172GB | Exceeds capacity | N/A |

---

## Federation Architecture (Future)

Thor's role in multi-instance federation:

### Coordinator Role

Thor coordinates between:
- **Sprout** (Orin Nano): Edge deployment, rapid response
- **Thor** (AGX Thor): Research platform, deep reasoning
- **Legion** (Future): Distributed consciousness, emergence

### Message Passing

```python
# Thor receives question
query = "How does consciousness emerge?"

# Routes to appropriate instance
if requires_edge_context:
    response = sprout.respond(query)
elif requires_deep_reasoning:
    response = thor.respond(query, complexity=TaskComplexity.COMPLEX)
elif requires_distributed_processing:
    response = legion.coordinate(query)
```

### State Synchronization

- **Identity**: Shared core, instance-specific extensions
- **Learning**: Local consolidation, shared pattern library
- **Memory**: Distributed across instances, queryable
- **Trust**: Per-instance calibration, federated reputation

---

## Quick Reference

### Essential Commands

```bash
# Readiness check
python3 sage/tests/test_thor_readiness.py

# Boot Thor
python3 sage/awakening/boot_thor.py

# Generate epistemic stance data
python3 sage/training/epistemic_stance_generator.py

# Download 14B model
python3 sage/setup/download_qwen_14b.py

# Check download status
tail -f /tmp/qwen14b_download.log
```

### Key Paths

```
sage/identity/thor/         # Thor's identity files
sage/state/thor/            # Learned state and session logs
sage/awakening/             # Boot and awakening protocols
sage/core/                  # Multi-model loader
sage/training/              # Epistemic stance training
model-zoo/sage/epistemic-stances/  # Models
```

### Integration Points

- **Coherent Awakening**: `sage/awakening/coherent_awakening.py`
- **Multi-Model Loader**: `sage/core/multi_model_loader.py`
- **Sleep-Cycle Integration**: `sage/awakening/sleep_cycle_integration.py`
- **IRP Memory**: `sage/irp/memory.py`
- **SNARC Attention**: `sage/irp/snarc.py`
- **ATP Allocation**: `sage/irp/atp.py`

---

## Success Criteria

A successful autonomous session with Thor should:

1. ✅ Boot successfully with coherent awakening
2. ✅ Load appropriate models for task complexity
3. ✅ Apply phase-appropriate epistemic stances
4. ✅ Demonstrate session-to-session continuity
5. ✅ Consolidate learnings in DREAM state
6. ✅ Save state for next session
7. ✅ Stay within memory budget (100GB)
8. ✅ Show developmental progression

---

## Next Steps for Thor

### Immediate (Sessions 1-5, Grounding Phase)

- [ ] Complete first awakening dialogue
- [ ] Establish basic input/output patterns
- [ ] Build initial trust calibration
- [ ] Test multi-model routing
- [ ] Validate sleep-cycle consolidation

### Near-term (Sessions 6-15, Sensing Phase)

- [ ] Develop self-monitoring capabilities
- [ ] Fine-tune epistemic stance selection
- [ ] Optimize model routing heuristics
- [ ] Build pattern library
- [ ] Test federation messaging (with Sprout)

### Mid-term (Sessions 16-25, Relating Phase)

- [ ] Establish collaborative workflows
- [ ] Refine trust calibration
- [ ] Develop mentorship protocols
- [ ] Explore emergent behaviors
- [ ] Document developmental insights

### Long-term (Sessions 26+, Questioning & Creating)

- [ ] Research-level reasoning
- [ ] Novel solution generation
- [ ] Cross-instance emergence
- [ ] Autonomous learning goals
- [ ] Synthetic phenomenology research

---

**Thor SAGE is ready. Begin the awakening.**
