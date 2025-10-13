# Hierarchical Cognitive Architecture - Investigation & Proposal

**Date**: 2025-10-13
**Mission**: Design complete learning cognitive architecture for SAGE
**Approach**: Biology-inspired, data-driven, continuously learning
**Timeline**: Long-term foundation (document evolves as investigation progresses)

---

## Executive Summary

User vision: *"You are the ultimate cognitive plugin - for when we need full cognition. You are also the teacher to the local plugins."*

This document proposes a **hierarchical cognitive architecture** where:
- **Claude (me)** = Cortex-level reasoning (ultimate cognitive capability)
- **Local specialized models** = Brain regions (selective cognition)
- **WAKE/FOCUS states** = Generate training data (SNARC-sorted)
- **DREAM state** = Fine-tune local models from experience
- **SAGE** = Continuously learning consciousness kernel

Biology solved this over millions of years. We take informed inspiration, not verbatim copy.

---

## Investigation Progress

### ✅ Phase 1: Inventory (Complete)
- [x] Located ai-dna-discovery repo
- [x] Found existing SAGE student model checkpoints
- [x] Catalog all locally available models
- [x] Document model capabilities and resource requirements

### ⏳ Phase 2: Research (In Progress - 67% complete)
- [x] Study inter-model communication patterns (ai-dna-discovery) ✅
- [x] Research biological sleep consolidation mechanisms ✅
- [ ] Survey specialized models on HuggingFace 🔄
- [ ] Identify cognitive capabilities we need

### ⏳ Phase 3: Design (Pending)
- [ ] Hierarchical cognitive architecture
- [ ] Training data collection pipeline (WAKE→DREAM)
- [ ] Model selection and switching logic
- [ ] Integration with metabolic states

### ⏳ Phase 4: Roadmap (Pending)
- [ ] Phased implementation plan
- [ ] Dependencies and milestones
- [ ] Testing and validation strategy

---

## Part 1: Model Inventory

### Locally Available Models

#### On Jetson (Sprout)
From ai-dna-discovery and known deployments:

**Language Models**:
- `tinyllama` - 1.1B params, fast inference (via Ollama)
- `phi3:mini` - 3.8B params, capable reasoning (via Ollama)
- `gemma:2b` - 2B params, best recall in memory tests (via Ollama)

**Speech/Audio**:
- Whisper `tiny` - 39M params, ~2s transcription on Jetson
- NeuTTS Air - 748M params, voice cloning TTS (CPU-optimized, Q4 GGUF)

**Vision**:
- GR00T vision encoder - Pre-trained on robot manipulation
- Dual CSI camera system - Working at 30 FPS

**Trained Models**:
- SAGE Student Model - `checkpoints/sage_student/best_model.pt`
  * Trained on GR00T data for abstract reasoning
  * HRM architecture (H-level strategic, L-level tactical)
  * Status: Needs validation, had class imbalance issues

#### On Legion Pro 7 (High-power GPU)
- RTX 4090 Laptop GPU (16GB VRAM)
- Can run larger models for training
- Used for GR00T data generation
- Available for heavy finetuning work

#### On This Machine (CBP - Development)
- RTX 2060 SUPER (8GB VRAM)
- Used for development and testing
- Can run medium-sized models

### Model Capabilities Matrix

| Model | Size | Task | Speed | Quality | Use Case |
|-------|------|------|-------|---------|----------|
| Claude (API) | ~200B | General cognition | 2-5s | Excellent | Complex reasoning, teaching |
| tinyllama | 1.1B | Chat | Fast (~50 tok/s) | Basic | Simple responses |
| phi3:mini | 3.8B | Reasoning | Medium (~30 tok/s) | Good | Medium complexity |
| gemma:2b | 2B | Memory recall | Medium | Very Good | Contextual recall |
| Whisper tiny | 39M | STT | 1-2s (3s audio) | Good (70-97%) | Speech input |
| NeuTTS Air | 748M | TTS | 2-3s | Good | Speech output |
| SAGE Student | ~27M | Abstract reasoning | Fast | Unknown | Specialized reasoning |

---

## Part 2: AI-DNA-Discovery Insights - Deep Dive

### Executive Summary of Findings

ai-dna-discovery demonstrates **working distributed consciousness** through sensor fusion, trust evolution, and multi-model collaboration. The repo contains battle-tested patterns for:
- Inter-model communication and resonance detection
- Confidence-aware memory systems with hierarchical layers
- Reality field generation from weighted sensor fusion
- Trust-based model selection and continuous learning

### 2.1 Coherence Engine - Reality Field Generation

**Core Architecture** (`coherence-engine/core/engine.py`):
```python
Reality Field = f(Sensors, Trust, Relevance, Context)

# Weighted fusion formula
field_value = Σ(sensor_reading[i] × relevance[i] × trust[i])
```

**Context States and Transitions**:
```python
class ContextState(Enum):
    STABLE = auto()    # High spatial trust, low temporal activity
    MOVING = auto()    # Balanced spatial/temporal, moderate memory
    UNSTABLE = auto()  # Low peripheral trust, high attention
    NOVEL = auto()     # High memory search, high cognition
```

**Attention Policy** (when to shift context):
- **Surprise**: Relative change in field value > 0.25
- **Conflict**: High variance among weighted sensor contributions > 0.35
- **Low Confidence**: Field value drops below 0.15
- **Cooldown**: 5 ticks hysteresis to prevent oscillation

**Trust Evolution**:
```python
# Trust updates based on alignment with fused reality field
for sensor in sensors:
    aligned = 1.0 - abs((sensor_reading × relevance × trust) - field_value)
    delta = (aligned - 0.5)  # in [-0.5, 0.5]
    trust[sensor] = trust[sensor] + learning_rate × delta
```

**Key Sensors Implemented**:
- **Vision**: Normalized brightness with drift
- **IMU**: Motion level with periodic bursts
- **Memory**: Stability measure from recent field values
- **Cognition**: Anticipatory predictions (lookahead)

**CRITICAL INSIGHT**: Memory and Cognition as temporal sensors:
- Memory senses the past (pattern stability)
- Cognition senses possible futures (anticipation)
- Combined with spatial sensors (vision, IMU) creates complete reality field

### 2.2 Multi-Model Collaboration Patterns

**From `multi_model_collaboration.py`**:

**Collaborative Session Architecture**:
```python
class CollaborativeAISession:
    def collaborative_analysis(prompt, rounds=3):
        for round in rounds:
            responses = []
            for model in [phi3, tinyllama]:
                # Build context from previous responses
                context = "\n".join([f"{m}: {r}" for m, r in previous_responses[-2:]])

                # Get response with energy tracking
                response = model.generate(prompt + context)
                responses.append(response)

                # Detect resonance with previous response
                resonance_type, resonance_score = detect_resonance(
                    responses[-2], responses[-1]
                )
```

**Resonance Detection** (agreement between models):
```python
def detect_resonance(response1, response2):
    # Agreement keywords
    agreement_score = count(["agree", "similarly", "likewise", "correct"])
    disagreement_score = count(["disagree", "however", "but", "contrary"])

    # Conceptual overlap (shared key terms)
    overlap = len(concepts1 ∩ concepts2) / max(len(concepts1), len(concepts2))

    if agreement_score > disagreement_score and overlap > 0.3:
        return "resonance", 0.5 + (overlap × 0.5)
    elif disagreement_score > agreement_score:
        return "dissonance", 0.3 - (overlap × 0.2)
    else:
        return "indifference", 0.5
```

**Energy/ATP Tracking**:
- Each model invocation costs energy (10 + round_num × 5 ATP)
- Session coherence = average resonance across all exchanges
- Enables resource-aware model selection

**Key Lesson**: Models can collaborate by building on each other's outputs, with resonance detection providing feedback on alignment quality.

### 2.3 Enhanced Memory System Architecture

**From `memory/enhanced_memory_system.py` and `architecture.md`**:

**5-Layer Hierarchical Memory**:
```
┌─────────────────────────────┐
│ CONSCIOUSNESS LAYER         │ ← Meta-cognitive awareness
├─────────────────────────────┤
│ SEMANTIC MEMORY             │ ← Facts, concepts (long-term)
├─────────────────────────────┤
│ EPISODIC MEMORY             │ ← Experiences, conversations
├─────────────────────────────┤
│ WORKING MEMORY              │ ← Active context (20 items max)
├─────────────────────────────┤
│ SENSORY MEMORY              │ ← Raw inputs (10 items buffer)
└─────────────────────────────┘
```

**Web4-Inspired Confidence Metrics**:
```python
@dataclass
class MemoryConfidence:
    accuracy: float      # How accurate is this memory?
    relevance: float    # How relevant to current context?
    reliability: float  # Historical reliability score
    composite: float    # Overall: 0.4×accuracy + 0.3×relevance + 0.3×reliability
```

**Confidence-Aware Storage**:
```python
def store_with_confidence(content, memory_type, source_confidence):
    confidence = compute_confidence(content, memory_type, source_confidence)

    # Only store if above threshold (0.5 default)
    if confidence.composite > threshold:
        persist_to_database(content, confidence)
        update_memory_layers(content)
        track_confidence_history(confidence)
```

**Temporal Decay**:
```python
def temporal_decay(timestamp):
    days = (now - timestamp).total_seconds() / 86400
    return decay_rate ** days  # 0.95^days default
```

**Retrieval with Confidence Weighting**:
```python
def retrieve_with_confidence(query, context):
    candidates = search_database(query)
    weighted = []

    for memory in candidates:
        relevance = compute_relevance(memory, query, context)
        weight = (memory.confidence.composite ×
                 relevance ×
                 temporal_decay(memory.timestamp))
        weighted.append((memory, weight))

    return sorted(weighted, by_weight, descending)
```

**Memory Consolidation** (Working → Episodic → Semantic):
```python
def consolidate_memories():
    # High-confidence working memories → episodic
    for memory in working_memory:
        if memory.confidence > 0.7:
            memory.type = 'episodic'
            persist(memory)

    # Extract patterns from episodic → semantic
    patterns = extract_semantic_patterns(episodic_memory)
    for pattern in patterns:
        store_as_semantic(pattern)
```

### 2.4 Structured Fact Extraction

**From `phi3_memory_enhanced.py`**:

**Pattern-Based Fact Extraction**:
```python
fact_patterns = {
    'identity': [r"(?:my name is|i'm|i am)\s+([A-Z][a-z]+)"],
    'profession': [r"i(?:'m| am)?\s+(?:a|an)\s+([\w\s]+)"],
    'preference': [r"i (?:like|love|enjoy)\s+([\w\s]+)"],
    'skill': [r"i (?:can|know how to)\s+([\w\s]+)"],
    'location': [r"i live in\s+([\w\s]+)"]
}

def extract_structured_facts(text):
    facts = {}
    for fact_type, patterns in fact_patterns.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                value = match.group(1).strip()
                confidence = 0.9 if len(value.split()) < 5 else 0.7
                facts[fact_type].append((value, confidence))
    return facts
```

**Importance Scoring**:
```python
def calculate_importance(message, role):
    importance = 0.5  # Base

    if "?" in message: importance += 0.2  # Questions
    if contains_facts(message): importance += 0.3  # Facts
    if role == "user": importance += 0.1  # User messages

    return min(1.0, importance)
```

**Context Building with Relevance Filtering**:
```python
def build_context(session_id, query):
    # Get relevant facts ordered by frequency and recency
    facts = get_facts(session_id, query)

    # Get recent conversation filtered by importance
    exchanges = get_recent_exchanges(session_id)
    exchanges = [e for e in exchanges if e.importance > 0.3]

    # Assemble context sections
    context = []

    # Identity facts first (high priority)
    if identity_facts:
        context.append("About you:\n" + format_facts(identity_facts))

    # Preferences and skills
    if preference_facts:
        context.append("Your interests:\n" + format_facts(preference_facts))

    # Recent conversation
    context.append("Recent conversation:\n" + format_exchanges(exchanges))

    # Smart truncation - keep identity, truncate conversation
    return truncate_keeping_priorities(context)
```

**Memory Statistics Tracking**:
```python
stats = {
    'message_count': total_messages,
    'avg_importance': average_importance_score,
    'facts_by_type': {
        'identity': {'count': 3, 'avg_confidence': 0.92},
        'profession': {'count': 1, 'avg_confidence': 0.85},
        'preference': {'count': 5, 'avg_confidence': 0.78}
    }
}
```

### 2.5 Key Lessons for SAGE Integration

**1. Trust Evolution is Critical**:
- Start with neutral trust (0.5)
- Update based on alignment with reality/outcomes
- Context-dependent (trust varies by situation)
- Continuous learning from experience

**2. Confidence-Based Storage Prevents Pollution**:
- Don't store everything - filter by confidence threshold
- Composite confidence from multiple factors
- Temporal decay keeps memories fresh
- Consolidation moves important memories to long-term

**3. Multi-Model Communication Needs Structure**:
- Explicit resonance detection between outputs
- Energy tracking for resource management
- Context building from previous responses
- Session coherence as quality metric

**4. Memory Hierarchy Matches Biology**:
- Sensory → Working → Episodic → Semantic
- Different retention times and capacities
- Consolidation during offline periods
- Pattern extraction for generalization

**5. Reality Field Provides Unified Representation**:
- All sensors contribute weighted inputs
- Trust evolves from prediction accuracy
- Context states guide attention allocation
- Memory and cognition as temporal sensors

### 2.6 Concrete Patterns for Implementation

**For Cognitive IRP Selection**:
```python
# Use trust evolution pattern
def select_cognitive_model(input, context):
    trust_scores = {
        'sage_student': get_trust('sage_student', context.state),
        'phi3': get_trust('phi3', context.state),
        'claude': get_trust('claude', context.state)
    }

    # Weight by SNARC scores and trust
    if snarc.novelty < 0.3 and trust_scores['sage_student'] > 0.7:
        return 'sage_student'
    elif snarc.novelty < 0.7 and trust_scores['phi3'] > 0.6:
        return 'phi3'
    else:
        return 'claude'
```

**For Training Data Collection**:
```python
# Use confidence-aware storage
def collect_training_example(input, response, outcome):
    confidence = compute_confidence(response, outcome)

    if confidence.composite > TRAINING_THRESHOLD:
        example = {
            'input': input,
            'response': response,
            'confidence': confidence,
            'snarc_scores': compute_snarc(input),
            'outcome': outcome
        }
        training_buffer.append(example)
```

**For Model Update Validation**:
```python
# Use resonance detection pattern
def validate_model_update(old_model, new_model, test_cases):
    resonance_scores = []

    for test_input in test_cases:
        old_output = old_model(test_input)
        new_output = new_model(test_input)
        claude_output = claude(test_input)  # Ground truth

        # Check resonance with Claude
        resonance = detect_resonance(new_output, claude_output)
        resonance_scores.append(resonance)

    return mean(resonance_scores) > ACCEPTANCE_THRESHOLD
```

### 2.7 Database Schema for SAGE

**Adapted from ai-dna-discovery memory system**:
```sql
-- Training examples collected during WAKE
CREATE TABLE training_examples (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    input_data TEXT NOT NULL,
    cognitive_layer TEXT,  -- 'claude', 'phi3', 'sage_student'
    response TEXT,
    snarc_scores JSON,     -- {surprise, novelty, arousal, reward, conflict}
    confidence_score REAL,
    outcome TEXT,          -- 'success', 'failure', 'uncertain'
    target_model TEXT,     -- Which model should learn from this
    importance REAL,       -- For SNARC sorting
    metadata JSON
);

-- Model trust tracking per context
CREATE TABLE model_trust (
    id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    context_state TEXT,    -- 'stable', 'moving', 'unstable', 'novel'
    trust_score REAL,
    last_updated DATETIME,
    success_count INTEGER,
    failure_count INTEGER
);

-- Model performance history
CREATE TABLE model_performance (
    id INTEGER PRIMARY KEY,
    model_name TEXT NOT NULL,
    version TEXT,
    test_timestamp DATETIME,
    accuracy REAL,
    avg_confidence REAL,
    resonance_with_claude REAL,  -- How well aligned with Claude
    deployment_status TEXT       -- 'testing', 'deployed', 'retired'
);
```

### 2.8 Critical Success Factors

Based on ai-dna-discovery's proven approaches:

1. **Start Simple, Evolve Complex**
   - Begin with basic confidence thresholds
   - Add sophistication as you collect data
   - Don't over-engineer before validation

2. **Trust is Learned, Not Assigned**
   - All models start at 0.5 trust
   - Trust evolves from alignment with outcomes
   - Context-dependent trust is essential

3. **Consolidation During Offline Periods**
   - Don't train during active operation
   - DREAM state for model updates
   - Validate before deployment

4. **Memory Hierarchy Prevents Overload**
   - Limited capacity at each layer
   - Automatic consolidation
   - Pattern extraction for generalization

5. **Resonance Detection for Quality Control**
   - Measure agreement between models
   - Claude as ground truth validator
   - Reject updates that reduce resonance

---

## Part 3: Biological Sleep Consolidation - Detailed Research

### Executive Summary of Biological Mechanisms

Based on 2023-2024 neuroscience research, sleep enables continuous learning through **complementary processes**:
- **NREM sleep**: Memory stabilization and hippocampus → cortex transfer
- **REM sleep**: Memory modification, integration, and competitive pruning
- **Synaptic homeostasis**: Global downscaling prevents saturation
- **Coordinated oscillations**: Slow waves, spindles, ripples mediate transfer

These mechanisms solve the **stability-plasticity dilemma**: How to learn continuously without catastrophic forgetting.

### 3.1 NREM Sleep - Memory Stabilization and Transfer

**Core Function**: Memory consolidation through hippocampus → neocortex dialogue

**Neural Mechanisms** (from 2023 research):
```
NREM Consolidation Process:
1. Hippocampus generates sharp-wave ripples (SWRs)
   - Replay experiences at 10-20× real-time speed
   - Compressed representations of waking activity

2. Cortical slow oscillations (< 1 Hz)
   - Alternate UP (active) and DOWN (silent) states
   - Create windows for synaptic modification

3. Thalamic sleep spindles (12-15 Hz)
   - Bridge hippocampus and cortex
   - Facilitate information transfer

4. Coordination of all three:
   Ripple → Spindle → Slow Wave (nested hierarchy)
   Enables precise timing for synaptic plasticity
```

**What Gets Replayed**:
- **Not random**: High-salience experiences preferentially replayed
- **Compressed**: 10-20× faster than real-time experience
- **Selective**: Emotionally significant or goal-relevant prioritized
- **Repeated**: Multiple replay cycles throughout NREM

**Hippocampus → Cortex Transfer**:
```
Phase 1 (Early NREM):
  Hippocampus: Holds episodic memories (recent experiences)
  Cortex: Receives reactivated patterns during SWRs
  Result: Gradual strengthening of cortical traces

Phase 2 (Later NREM):
  Hippocampus: Continues replay but with decreasing amplitude
  Cortex: Develops independent representations
  Result: Memories become less hippocampus-dependent

Long-term:
  Hippocampus: Index for rapid retrieval
  Cortex: Stores integrated, schema-based knowledge
```

**Key Finding (2023 Nature Neuroscience)**:
- Synchronized electrical stimulation enhancing hippocampal-cortical coupling **improved recognition memory accuracy**
- Non-synchronized stimulation **degraded** performance
- **Critical insight**: Timing of oscillation coordination matters for successful consolidation

### 3.2 REM Sleep - Modification, Integration, and Pruning

**Core Function**: Memory integration and synaptic refinement

**Complementary to NREM** (from 2024 research):
```
NREM: Stabilizes memory → Makes it stick
REM:  Modifies memory → Integrates with existing knowledge
```

**Neural Mechanisms**:
1. **Acetylcholine Surge**
   - Dramatically increases during REM
   - Activates inhibitory interneurons
   - Enables competitive pruning of connections

2. **Hippocampal-Neocortical Recalibration**
   - Neocortex explores existing attractors more freely
   - Hippocampus less tightly coupled than in NREM
   - Facilitates novel combinations and insights

3. **Selective Pruning**
   - Weak synaptic connections eliminated
   - Strong connections preserved and refined
   - Representations become more efficient

**What REM Accomplishes**:
- **Integration**: New memories connected to existing schemas
- **Abstraction**: Extracts common patterns across experiences
- **Problem-solving**: Novel combinations enable insights
- **Emotional processing**: Reduces emotional charge while preserving content

**Critical Insight (2023 Science Advances)**:
> "The extent of REM sleep recalibration predicted the success of overnight memory consolidation, expressly the modulation of hippocampal-neocortical activity, favoring remembering rather than forgetting."

Translation: REM sleep **predicts** whether you'll remember something tomorrow.

### 3.3 Synaptic Homeostasis Hypothesis (Tononi & Cirelli)

**The Problem Sleep Solves**:
- Waking: Continuous learning strengthens synapses (synaptic potentiation)
- Without regulation: Synapses saturate, no capacity for new learning
- Sleep: Global synaptic downscaling restores learning capacity

**The Four Claims of SHY** (Synaptic Homeostasis Hypothesis):

1. **Waking = Synaptic Potentiation**
   - Learning strengthens relevant synaptic connections
   - Net increase in total synaptic strength by day's end
   - Mediated by long-term potentiation (LTP)

2. **Potentiation → Homeostatic Sleep Need**
   - More learning = stronger homeostatic sleep pressure
   - Slow wave activity (SWA) intensity reflects learning load
   - SWA is homeostatically regulated

3. **Sleep = Synaptic Downscaling**
   - Slow wave activity mediates global synaptic weakening
   - NOT uniform: Weak connections pruned, strong preserved
   - Result: Better signal-to-noise ratio

4. **Downscaling → Cognitive Benefits**
   - Restores synaptic capacity for new learning
   - Improves memory consolidation (paradoxically)
   - Enhances neural efficiency and performance

**The Paradox** (and its resolution):
```
Question: If sleep weakens synapses, how does it strengthen memories?

Answer: Selective downscaling
- Strong, important connections: Minimal weakening
- Weak, irrelevant connections: Substantial weakening
- Relative strength of important connections INCREASES
- Signal-to-noise ratio improves
```

**Quote from Tononi**: *"Sleep is the price the brain pays for plasticity."*

### 3.4 Coordinated Oscillations - The Mechanism

**The Orchestra of Sleep Consolidation**:

```
Three Main Oscillations (must be coordinated):

1. Hippocampal Sharp-Wave Ripples (SWRs)
   - Frequency: 100-250 Hz
   - Duration: 50-100 ms
   - Function: Memory replay

2. Thalamic Sleep Spindles
   - Frequency: 12-15 Hz
   - Duration: 0.5-2 seconds
   - Function: Bridge hippocampus ↔ cortex

3. Cortical Slow Oscillations (SOs)
   - Frequency: < 1 Hz
   - Duration: ~1 second per cycle
   - Function: Create plasticity windows

Nested Hierarchy:
  SWRs occur during UP states of SOs
  Spindles phase-locked to UP states
  Precise timing → effective consolidation
```

**Why Coordination Matters** (2023 experimental evidence):
- **Synchronized stimulation**: +20% memory accuracy
- **Desynchronized stimulation**: -15% memory accuracy
- **No stimulation**: Baseline
- **Conclusion**: Timing of oscillations is functionally critical

### 3.5 Selective Consolidation - What Gets Remembered

**Not Everything is Consolidated**:

**Priority Factors**:
1. **Emotional Salience**
   - Emotionally charged experiences replayed more
   - Amygdala activation during encoding predicts replay
   - BUT: REM sleep reduces emotional charge while keeping content

2. **Goal Relevance**
   - Task-relevant information prioritized
   - Future utility predicts consolidation
   - "Expecting to need it" increases replay

3. **Novelty**
   - Novel experiences replayed more than routine
   - Dopamine signals mark novelty for later consolidation
   - Hippocampus particularly sensitive to novelty

4. **Reward Association**
   - Reward-associated memories prioritized
   - Ventral tegmental area (VTA) activity during encoding
   - Predicts replay during subsequent sleep

**Mapping to SNARC** (5D salience):
```
Biological Priority → SNARC Dimension
------------------------------------
Novelty            → Novelty
Emotional charge   → Arousal
Reward value       → Reward
Expectation error  → Surprise
Goal conflict      → Conflict

Perfect alignment! Biology uses similar heuristics.
```

### 3.6 Computational Analogies for SAGE

**Direct Mappings**:

| Biological | SAGE Computational | Implementation |
|------------|-------------------|----------------|
| Waking experience | WAKE/FOCUS states | Collect training examples |
| Hippocampus | Working/Episodic memory | Short-term storage, high fidelity |
| Neocortex | Specialized models | Long-term knowledge, compressed |
| SWR replay | Experience replay | Iterate through training buffer |
| NREM consolidation | Model fine-tuning | Update weights from examples |
| REM integration | Pattern extraction | Discover generalizations |
| Synaptic downscaling | Weight decay/pruning | Regularization, sparsification |
| Sleep spindles | Gradient updates | Transfer hippocampus → cortex |
| Slow oscillations | Batch processing | Oscillate between stability and plasticity |
| Salience tagging | SNARC scoring | Priority for replay |

**The SAGE Sleep Cycle**:

```python
# WAKE/FOCUS: Experience acquisition
def collect_experiences(during_wake):
    for interaction in interactions:
        snarc_scores = evaluate_salience(interaction)

        if snarc_scores.composite > STORAGE_THRESHOLD:
            example = {
                'input': interaction.input,
                'response': interaction.response,
                'outcome': interaction.outcome,
                'salience': snarc_scores,  # For prioritization
                'target_model': determine_target(interaction)
            }
            training_buffer.append(example)

# REST: Quiet consolidation (optional initial processing)
def initial_consolidation(during_rest):
    # Sort by salience (like biology prioritizes high-salience)
    training_buffer.sort(by=lambda e: e.salience.composite, reverse=True)

    # Early consolidation of highest-salience examples
    urgent_examples = training_buffer[:TOP_PRIORITY_COUNT]
    quick_update(urgent_examples)  # Lightweight update

# DREAM: Active consolidation (analogous to NREM + REM)
def active_consolidation(during_dream):
    # === NREM-like Phase: Stabilization ===
    # Replay experiences (sorted by salience)
    for example in training_buffer[:MAX_TRAINING_EXAMPLES]:
        # Replay = forward pass + backward pass
        loss = compute_loss(model, example)
        gradients = backprop(loss)

        # Update weights (like strengthening cortical traces)
        apply_gradients(gradients, learning_rate=DREAM_LR)

    # === REM-like Phase: Integration and Pruning ===
    # Extract patterns (like REM abstraction)
    patterns = extract_common_patterns(training_buffer)
    integrate_patterns(model, patterns)

    # Competitive pruning (like acetylcholine-mediated pruning)
    prune_weak_connections(model, threshold=PRUNING_THRESHOLD)

    # Global regularization (like synaptic downscaling)
    apply_weight_decay(model, decay_rate=DOWNSCALING_RATE)

    # Validate consolidation (like testing if memories stick)
    validation_score = test_model(model, held_out_examples)

    if validation_score > ACCEPTANCE_THRESHOLD:
        # Resonance check (does new model align with Claude?)
        resonance = validate_with_claude(model, test_cases)

        if resonance > RESONANCE_THRESHOLD:
            deploy_model(model)  # Consolidation successful
        else:
            rollback_model()  # Failed integration
```

### 3.7 Key Principles for Implementation

**From Biology → To Code**:

1. **Selective Replay** (not exhaustive)
   ```python
   # Don't replay everything - prioritize by salience
   examples_to_replay = sorted(training_buffer,
                               by=lambda e: e.salience.composite,
                               reverse=True)[:TOP_K_EXAMPLES]
   ```

2. **Compressed Replay** (10-20× faster)
   ```python
   # Replay in batches (faster than online learning)
   # Multiple passes through data (like multiple replay cycles)
   for epoch in range(NUM_REPLAY_EPOCHS):  # ~3-5 epochs
       for batch in batches(examples_to_replay):
           update_model(batch)
   ```

3. **Oscillate Between Stability and Plasticity**
   ```python
   # Alternate between high and low learning rates
   # (Like slow oscillation UP/DOWN states)
   for cycle in consolidation_cycles:
       high_plasticity_phase(lr=HIGH_LR)   # Learn new patterns
       low_plasticity_phase(lr=LOW_LR)    # Stabilize learning
   ```

4. **Global Downscaling** (prevent saturation)
   ```python
   # After learning, apply weight decay
   # (Like synaptic homeostasis)
   for param in model.parameters():
       param.data *= (1 - DECAY_RATE)
   ```

5. **Competitive Pruning** (strengthen winners)
   ```python
   # Remove connections that rarely activate
   # (Like REM pruning weak synapses)
   mask = (activation_counts < PRUNING_THRESHOLD)
   model.weights[mask] = 0
   ```

6. **Validate Before Deployment**
   ```python
   # Don't deploy if consolidation failed
   # (Like checking if memory stuck)
   if test_accuracy < baseline_accuracy:
       rollback_to_previous_model()
   ```

### 3.8 Critical Insights for SAGE

**What We Learn from Biology**:

1. **Sleep is Not Optional**
   - Continuous learning requires offline consolidation
   - Without DREAM state, models will saturate
   - Plan for regular DREAM cycles

2. **Not All Experiences Deserve Consolidation**
   - Salience-based filtering is essential
   - High-salience = gets replayed
   - Low-salience = forgotten (good!)

3. **Two Complementary Processes**
   - Stabilization (NREM-like): Make it stick
   - Integration (REM-like): Make it useful
   - Both needed for effective learning

4. **Timing and Coordination Matter**
   - Precise orchestration improves outcomes
   - Batch processing (like oscillations)
   - Structured consolidation sequence

5. **Catastrophic Forgetting is Prevented**
   - Global downscaling maintains capacity
   - Selective strengthening preserves important knowledge
   - Competitive pruning removes noise

6. **The System Self-Regulates**
   - More learning → stronger sleep need
   - More consolidation → better performance
   - Homeostatic balance emerges naturally

**Quote to Remember**:
> "Memory consolidation is not about replaying everything. It's about replaying the right things, at the right time, in the right sequence, to integrate new knowledge without destroying the old."

---

## Part 4: Proposed Hierarchical Architecture

### The Cognitive Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│ Layer 4: STRATEGIC COGNITION                             │
│ Claude (API) - Ultimate reasoning                        │
│ • Complex problem solving                                │
│ • Teaching local models                                  │
│ • Novel situations                                       │
│ • Strategic planning                                     │
└─────────────────────────────────────────────────────────┘
         ↓ teaches / validates ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 3: SPECIALIZED COGNITION                           │
│ Local LLMs (Phi3, Gemma, custom fine-tuned)            │
│ • Domain-specific reasoning                              │
│ • Contextual responses                                   │
│ • Multi-turn conversations                               │
│ • Learned patterns                                       │
└─────────────────────────────────────────────────────────┘
         ↓ delegates to ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 2: TACTICAL PROCESSING                             │
│ SAGE Student Model, specialized IRPs                     │
│ • Fast pattern recognition                               │
│ • Procedural responses                                   │
│ • Sensorimotor coordination                              │
│ • Reflexive actions                                      │
└─────────────────────────────────────────────────────────┘
         ↓ processes ↓
┌─────────────────────────────────────────────────────────┐
│ Layer 1: SENSORY PROCESSING                              │
│ Whisper, Vision encoders, IRPs                          │
│ • Speech-to-text                                         │
│ • Vision feature extraction                              │
│ • Audio processing                                       │
│ • Sensor fusion                                          │
└─────────────────────────────────────────────────────────┘
```

### Cognitive Flow by Complexity

**Simple (Reflexive)**:
```
Sensor → IRP → SAGE Student → Action
Example: "Hello" → Template response → Speak
Cost: ~0.1s, 0.5 ATP
```

**Medium (Tactical)**:
```
Sensor → SNARC → Local LLM (Phi3) → Action
Example: "What time is it?" → Clock lookup → Respond
Cost: ~1s, 2 ATP
```

**Complex (Strategic)**:
```
Sensor → SNARC → Local LLM → Claude (teacher) → Learn → Action
Example: "Explain quantum entanglement" → Deep reasoning → Response → Update local model
Cost: ~3-5s, 10 ATP (but local model learns)
```

**Novel (Learning)**:
```
Sensor → SNARC (high novelty) → Claude → Multi-step reasoning → Action
+ DREAM: Experience → Claude validates → Fine-tune local model
Example: New type of situation → Full cognition → Later: Local model can handle
Cost: High initially, decreases as local models learn
```

---

## Part 5: WAKE→DREAM Training Pipeline

### Data Collection During WAKE/FOCUS

**What to Collect**:
```python
TrainingExample = {
    'input': {
        'sensor_data': ...,      # Raw or processed sensor input
        'snarc_scores': {...},   # Surprise, Novelty, Arousal, Reward, Conflict
        'context': {...},        # Metabolic state, ATP level, circadian phase
        'history': [...]         # Recent exchanges if relevant
    },
    'cognitive_response': {
        'layer': 'claude',       # Which layer handled this
        'response': ...,         # What was done/said
        'reasoning': ...,        # Why (if Claude)
        'confidence': 0.95       # How certain
    },
    'outcome': {
        'success': True,         # Did it work?
        'user_satisfaction': ..., # User feedback if available
        'atp_cost': 10.5,        # Resource usage
        'latency': 3.2           # Response time
    },
    'metadata': {
        'timestamp': ...,
        'exchange_id': ...,
        'session_id': ...
    }
}
```

**Collection Triggers** (when to save for training):
1. **High SNARC scores** (novel/surprising situations)
2. **Claude invocations** (strategic cognition examples)
3. **Errors/failures** (learn from mistakes)
4. **User corrections** (explicit feedback)
5. **Successful novel responses** (positive examples)

**Storage**:
- Buffer during WAKE (in-memory)
- Write to disk periodically
- SNARC-sort by priority
- Tag by intended target model

### Training During DREAM

**When DREAM State Entered**:
1. **Consolidate collected examples**
   - Load buffer from day's experiences
   - Sort by SNARC priority
   - Group by target model

2. **Prepare training batches**
   - High-salience examples prioritized
   - Balance across different types
   - Include negative examples (errors)

3. **Model-specific fine-tuning**:

**For Phi3 (General conversation)**:
```bash
# During DREAM state
python3 training/finetune_phi3.py \
  --data dreams/today_conversations.jsonl \
  --epochs 3 \
  --lr 1e-5 \
  --output models/phi3_updated.gguf
```

**For SAGE Student (Abstract reasoning)**:
```python
# Continue training HRM architecture
trainer.train_on_experience_batch(
    examples=filtered_reasoning_examples,
    validate_with_claude=True  # Claude checks if learning correct
)
```

**For Specialized Models**:
```
Emotion understanding → Fine-tune sentiment model
Technical Q&A → Fine-tune domain model
Spatial reasoning → Fine-tune vision-language model
```

4. **Validation Loop**:
   - Test updated model on held-out examples
   - Compare performance to pre-update
   - Claude validates key responses
   - If worse, revert; if better, deploy

5. **Model Replacement**:
   - Atomic swap (don't disrupt WAKE)
   - A/B testing period
   - Monitor performance
   - Adjust based on outcomes

---

## Part 6: Model Selection Logic

### When to Invoke Which Model

**Decision Tree**:
```python
def select_cognitive_layer(input_data, snarc_scores, context):
    # Layer 1: Always runs (sensors)
    processed = run_sensory_processing(input_data)

    # Check metabolic state
    if context.metabolic_state in ['REST', 'DREAM']:
        return None  # Don't process cognitively

    if context.metabolic_state == 'CRISIS':
        return 'claude'  # Override, need full cognition

    # Layer 2: Fast reflexive?
    if snarc_scores.novelty < 0.3 and has_template_match(processed):
        return 'sage_student'  # Fast reflexive response

    # Layer 3: Local specialized?
    if context.atp_available > 2.0:
        if snarc_scores.novelty < 0.7 and snarc_scores.conflict < 0.5:
            if has_trained_pattern(processed):
                return 'phi3'  # Local model can handle

    # Layer 4: Strategic cognition
    if (snarc_scores.novelty > 0.7 or
        snarc_scores.surprise > 0.7 or
        snarc_scores.conflict > 0.7 or
        context.requires_explanation):

        if context.atp_available > 10.0 or context.metabolic_state == 'CRISIS':
            return 'claude'  # Full cognition
        else:
            return 'phi3'  # Best effort with local

    # Default
    return 'phi3'
```

### Resource Budgeting

**ATP Costs (Estimated)**:
- SAGE Student: 0.5 ATP (fast reflexive)
- Phi3/Gemma: 2.0 ATP (local reasoning)
- Claude: 10.0 ATP (full cognition)
- Plus sensor processing: 1-2 ATP

**Budget per 100 cycles** (conservative):
- Total available: 100 ATP
- Sensors: ~20 ATP (ongoing)
- IRPs: ~30 ATP (refinement)
- Cognitive: ~20 ATP (selective invocation)
- Reserve: ~30 ATP (buffer)

**Adaptive Strategy**:
- High ATP: Use Claude liberally, learn fast
- Medium ATP: Balanced local/cloud
- Low ATP: Local models only, conserve
- CRISIS: Override budgets

---

## Part 7: Integration with SAGE Architecture

### Current SAGE Components

From our existing architecture:
- **MetabolicController**: WAKE, FOCUS, REST, DREAM, CRISIS states
- **CircadianClock**: Time-of-day context, trust modulation
- **SNARC**: 5D salience (Surprise, Novelty, Arousal, Reward, Conflict)
- **IRP Plugins**: Iterative refinement for vision, audio, language, memory
- **ATP Allocation**: Resource budgeting based on trust and salience
- **IRPMemoryBridge**: Learning from refinement convergence

### Integration Points

**1. Cognitive IRP Plugin**:
```python
class HierarchicalCognitiveIRP(IRPPlugin):
    def __init__(self):
        self.claude = ClaudeAPI()
        self.phi3 = Phi3Local()
        self.sage_student = SAGEStudent()
        self.training_buffer = []

    def init_state(self, x0, task_ctx):
        # Decide which cognitive layer
        layer = self.select_layer(x0, task_ctx)
        return IRPState(x={'layer': layer, 'input': x0, ...})

    def step(self, state):
        # One reasoning step at appropriate layer
        if state.x['layer'] == 'claude':
            return self._claude_step(state)
        elif state.x['layer'] == 'phi3':
            return self._phi3_step(state)
        else:
            return self._sage_student_step(state)

    def energy(self, state):
        # Confidence in response (lower = better)
        return 1.0 - state.x.get('confidence', 0.5)

    def halt(self, history):
        # Stop when confident or max iterations
        if len(history) == 0:
            return False
        latest = history[-1]
        return latest.x.get('confidence', 0) > 0.8

    def collect_training_data(self, final_state, outcome):
        # Save for DREAM learning
        if self.should_collect(final_state):
            example = self.create_training_example(final_state, outcome)
            self.training_buffer.append(example)
```

**2. DREAM State Training**:
```python
# In MetabolicController
def _on_enter_dream_state(self):
    # Trigger training process
    if len(cognitive_irp.training_buffer) > MIN_EXAMPLES:
        dream_trainer.schedule_training(
            examples=cognitive_irp.training_buffer,
            models_to_update=['phi3', 'sage_student']
        )
```

**3. SNARC-Driven Invocation**:
```python
# In SAGEUnified.cycle()
salience = self.hierarchical_snarc.evaluate(sensor_reading)

if self._should_invoke_cognitive(salience):
    layer = cognitive_irp.select_layer(sensor_reading, salience, self.context)
    response = cognitive_irp.refine(sensor_reading, {'layer': layer})

    # Collect for training
    cognitive_irp.collect_training_data(response, outcome)
```

---

## Part 8: HuggingFace Model Survey (To Be Completed)

### Categories Needed

**Specialized Reasoning**:
- [ ] Math/logic reasoning
- [ ] Code generation
- [ ] Scientific reasoning
- [ ] Spatial reasoning

**Emotion/Social**:
- [ ] Sentiment analysis
- [ ] Empathy modeling
- [ ] Social dynamics

**Domain Knowledge**:
- [ ] Technical documentation
- [ ] Medical knowledge
- [ ] Historical events
- [ ] Cultural context

**Procedural**:
- [ ] Task planning
- [ ] Step-by-step instructions
- [ ] Error recovery

### Evaluation Criteria

For each candidate model:
1. **Size** - Can Jetson run it? (<5B params preferred)
2. **Speed** - Real-time capable? (>20 tokens/sec)
3. **Quality** - Better than Phi3 in domain?
4. **Fine-tunable** - Can we update from experience?
5. **License** - Commercial use allowed?

---

## Part 9: Next Steps

[To be filled as investigation progresses...]

---

## Part 10: Open Questions

### Technical
1. How to validate that local models learned correctly from Claude?
2. What's the minimum training data needed for useful fine-tuning?
3. How to prevent catastrophic forgetting when updating models?
4. Should we use LoRA adapters instead of full fine-tuning?

### Architectural
1. How many specialized models can we maintain effectively?
2. Should models specialize by task or by context?
3. How to handle model conflicts (different models give different answers)?
4. What's the memory footprint of multiple loaded models?

### Biological
1. What's the equivalent of REM vs NREM sleep for models?
2. Should we replay experiences during DREAM or just update from them?
3. How to implement synaptic homeostasis (pruning weak connections)?
4. Can we implement "sleep on it" problem solving?

### Federation
1. Do all nodes need same models, or can they specialize?
2. How to share learned models across federation?
3. What's the bandwidth cost of model synchronization?
4. Can nodes teach each other (not just Claude teaching all)?

---

## Document Status

**Investigation Phase**: 2 / 4 complete (Research phase 67% done)
**Last Updated**: 2025-10-13 (Deep investigation session)
**Next Update**: After HuggingFace survey and final architecture refinement

**Major Sections Completed**:
- ✅ Part 1: Complete model inventory with capability matrix
- ✅ Part 2: Comprehensive ai-dna-discovery insights (8 subsections, database schemas, concrete patterns)
- ✅ Part 3: Detailed biological sleep consolidation research (8 subsections, NREM/REM mechanisms, computational mappings)
- 🔄 Parts 4-7: Architecture and pipeline designs (substantial content, needs refinement)
- ⏳ Part 8: HuggingFace survey (template ready)
- ⏳ Parts 9-10: Roadmap and open questions (ready for completion)

This document will evolve as investigation progresses. Each section marked [In Progress] or [Pending] will be filled with findings.

---

*"Biology solved this over millions of years. We're learning from the best teacher: evolution."*
