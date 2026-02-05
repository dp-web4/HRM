# SAGE Codebase Architecture & Nemotron Integration Analysis

**Document Purpose**: Clear architecture summary for comparing SAGE with Nemotron  
**Thoroughly Level**: Medium (focus on key components, integration points, and strategic implications)  
**Date**: December 24, 2025

---

## Executive Summary: SAGE is an Orchestration Framework, Not a Model

**Critical Insight**: SAGE is **not** a language model or AI backbone. It's a **cognition kernel**—an attention orchestration system that manages computational resources and decides which specialized intelligences to invoke.

**Analogy**: Like an operating system schedules processes and manages hardware, SAGE schedules attention and manages computational resources (models, sensors, effectors). It learns which resources are trustworthy and efficient based on performance.

**This Completely Changes the Nemotron Integration Story**: Rather than SAGE replacing an LLM, Nemotron becomes **just one plugin in SAGE's ecosystem**—alongside vision, audio, memory, and other capabilities.

---

## SAGE Architecture: Three Layers

### Layer 1: SAGE Core - The Cognition Kernel

**Location**: `/sage/core/`  
**Status**: Designed, components exist, not yet unified into single loop

**What It Does**:
- Maintains temporal state (continuous time awareness)
- Monitors available computational resources (plugins, models, sensors)
- Computes attention targets based on SNARC salience (Surprise, Novelty, Arousal, Reward, Conflict)
- Allocates ATP (Attention Token Pool/Adaptive Trust Points) budget to plugins
- Learns which resources deserve trust based on performance

**The Main Loop** (conceptual, not yet unified in code):
```python
while True:
    # SENSE: Gather from sensors
    observations = gather_from_sensors()
    
    # ASSESS: Compute what matters (SNARC salience scoring)
    attention_targets = compute_what_matters(observations)
    
    # DECIDE: Determine needed plugins
    required_resources = determine_needed_plugins(attention_targets)
    
    # ALLOCATE: Manage loading/unloading
    manage_resource_loading(required_resources)
    
    # EXECUTE: Invoke plugins iteratively
    results = invoke_irp_plugins(attention_targets)
    
    # LEARN: Update trust and memory
    update_trust_and_memory(results)
    
    # ACT: Send to effectors
    send_to_effectors(results)
```

**Key Insight**: SAGE is the scheduler. Plugins are apps that get CPU/GPU time based on trust scores.

**Existing Components**:
- **Metabolic States** (5 modes): WAKE, FOCUS, REST, DREAM, CRISIS
  - Modulates attention breadth, inference depth, resource availability
  - Handles stress adaptation and recovery
  
- **SNARC Memory** (Salience-based): 5D scoring (Surprise, Novelty, Arousal, Reward, Conflict)
  - Determines what experiences are worth storing
  - Drives attention allocation
  
- **Trust Scoring System**: Tracks reliability of each resource
  - Based on convergence behavior of plugins
  - Weights ATP allocation
  
- **ATP Budget Allocation**: Limited computational "energy"
  - Distributed to plugins based on trust
  - Forces trade-off decisions
  - Models biological resource constraints

**Missing from Current Codebase**:
- Unified `SAGE.run()` loop that coordinates all components
- Active metabolic state transitions in orchestrator
- Real-time trust score updates from plugin performance

---

### Layer 2: IRP - The Cognition API

**Location**: `/sage/irp/`  
**Status**: Fully operational, 15+ plugins working

**What It Is**: Universal plugin interface that all "apps" must implement

**The Contract** (4-method interface every plugin provides):
```python
class IRPPlugin:
    def init_state(self, x0, task_ctx) -> IRPState:
        """Convert raw input to refinement state"""
    
    def energy(self, state) -> float:
        """Measure quality (lower is better)"""
    
    def step(self, state, noise_schedule) -> IRPState:
        """Execute one refinement iteration"""
    
    def halt(self, history) -> bool:
        """Detect convergence"""
```

**Philosophy Behind IRP**:
All intelligence is iterative refinement toward lower energy states:
- Vision: Blurry sensor input → sharp semantic features
- Language: Masked tokens → complete meaning
- Control: Random trajectory → optimal path
- Memory: Raw experience → compressed wisdom

**How Trust Emerges**:
1. Plugin starts with neutral energy
2. Each step refines (ideally decreasing energy)
3. Convergence detected when energy slope < threshold
4. Trust increases if: monotonic decrease, stable, efficient (few steps)
5. Trust decreases if: oscillating, diverging, slow convergence

**Working Plugins** (15+):
- **Vision**: Object detection, segmentation, feature extraction
- **Audio**: Speech recognition, audio understanding
- **Language**: LLM integration, semantic understanding
- **Memory**: Pattern storage and retrieval
- **NeuTTS**: Text-to-speech synthesis
- **Conversation**: Multi-turn dialogue management
- **Control**: Robotic action planning
- **Visual Monitoring**: Live scene understanding
- **Cognitive**: Epistemic reasoning, uncertainty quantification
- **Camera**: Raw sensor input processing

**Key Property**: Each plugin can be developed independently as long as it implements the 4-method contract. Trust weighting ensures optimal resource allocation automatically.

---

### Layer 3: VAE - Cross-Modal Translation Layer

**Location**: `/sage/compression/`  
**Status**: Implemented, multiple compression strategies working

**Purpose**: Enable cross-modal communication by creating shared latent spaces

**Problem Solved**:
- Vision speaks "pixels" (150K+ dimensions)
- Language speaks "tokens" (30K vocabulary)
- Control speaks "trajectories" (continuous actions)
- They can't communicate directly

**Solution**:
VAE creates shared latent spaces where different modalities can exchange information:

**Key Architectures**:
1. **TinyVAE** (64D latent)
   - Compresses 224×224 images → 64D latent
   - 192× compression
   - <10ms inference
   - Preserves semantic meaning for classification/object detection

2. **InformationBottleneck** (256D latent)
   - Compresses 4096D strategic context → 256D tactical action
   - H→L compression for robotic control
   - 16× compression with <15% information loss

3. **Puzzle Space** (30×30×10 grid)
   - Universal interface for all modalities
   - Vision, audio, proprioception all encode to puzzle coordinates
   - Learning-compatible grid interface

**Compression-Trust Unification**:
- **Trust measures how well meaning is preserved through compression**
- High trust (>0.9) = reliable cross-modal translation
- Low trust (<0.5) = information loss in translation
- VAE learning directly optimizes compression trust

---

## What SAGE Actually Provides (vs. What Nemotron Provides)

### SAGE Provides:
1. **Continuous Monitoring Loop**
   - Always-on awareness (unlike LLMs which process discrete inputs)
   - Temporal state tracking
   - Surprise/novelty detection

2. **Attention Orchestration**
   - Decides which modalities deserve focus
   - Allocates limited resources dynamically
   - Prevents thrashing when overloaded

3. **Multi-Modal Coherence**
   - Integrates vision, audio, language, proprioception
   - Cross-modal translation via VAE
   - Unified "coherence field"

4. **Trust-Based Decision Making**
   - Learns which plugins are reliable
   - Weights resource allocation by trust
   - Adapts to plugin failures

5. **Memory Systems** (4 parallel):
   - SNARC Memory: Selective storage via 5D salience
   - IRP Memory Bridge: Successful refinement patterns
   - Circular Buffer: Recent context (X-from-last)
   - Verbatim Storage: SQLite full-fidelity records

6. **Metabolic State Management**
   - Adapts to resource constraints
   - Stress response (escalates to CRISIS mode if needed)
   - Recovery strategies (passive recovery through inactivity)

7. **Energy-Based Selection**
   - All decisions optimized for energy efficiency
   - Progressive refinement until convergence
   - Halt detection prevents wasted computation

### Nemotron Provides:
1. **Language Understanding & Generation**
   - Semantic comprehension of text
   - Reasoning about concepts
   - Natural language output

2. **Knowledge Encoding**
   - Pre-trained understanding of world
   - Common sense reasoning
   - Vast vocabulary and concepts

3. **Flexible Input Processing**
   - Can work with various prompt formats
   - Multi-turn conversations with history
   - Instruction following

### What's Missing from Each:
**SAGE without Nemotron**:
- Can't understand complex concepts
- Can't reason about abstract ideas
- Can't generate natural language
- No knowledge of world facts

**Nemotron without SAGE**:
- No continuous awareness
- Can't make decisions about when/what to think
- Can't manage multi-modal inputs coherently
- Wasteful (responds to every prompt equally)
- Can't adapt to resource constraints
- No memory persistence across sessions

---

## Integration Points: Where Nemotron Fits in SAGE

### 1. As a Language IRP Plugin

**Simplest Integration** (already partially implemented):

```python
class NemotronPlugin(IRPPlugin):
    """Nemotron as cognition API plugin"""
    
    def __init__(self, model_name="nemotron-mini", config={}):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = config.get('device', 'cuda')
    
    def init_state(self, x0, task_ctx):
        """x0 = prompt, task_ctx = conversation history"""
        tokens = self.tokenizer(x0, return_tensors='pt')
        return IRPState(x=tokens)
    
    def energy(self, state):
        """Energy = how much output has changed (refinement measure)"""
        # Could measure:
        # - Log probability (lower = more confident)
        # - Semantic consistency with task
        # - Perplexity relative to context
        return self.compute_semantic_energy(state)
    
    def step(self, state, noise_schedule):
        """One refinement step = one token generation"""
        with torch.no_grad():
            output = self.model.generate(
                state.x,
                max_new_tokens=1,
                do_sample=False,  # Greedy refinement
            )
        state.x = output
        return state
```

**Why This Works**:
- Language generation IS iterative refinement (token by token)
- Trust emerges from semantic stability
- Allows SAGE to decide how many tokens needed (via ATP budget)
- Integrates naturally with SNARC attention allocation

### 2. As Semantic Meaning Encoder

**More Advanced Integration**:

SAGE needs to understand what observations mean. Nemotron could provide semantic encoding:

```python
# In SAGE's attention computation:
class AttentionWithSemantics:
    def compute_attention(self, observations):
        targets = []
        
        for modality, data in observations.items():
            # Get raw surprise
            surprise = self.compute_surprise(modality, data)
            
            # Ask Nemotron: "Is this important?"
            semantic_importance = self.nemotron_plugin.step({
                'observation': data,
                'context': self.recent_context,
                'task': 'assess_importance'
            })
            
            # Combine surprise + semantic importance
            salience = surprise * semantic_importance + novelty + arousal
            
            if salience > threshold:
                targets.append(AttentionTarget(
                    modality=modality,
                    priority=salience,
                    data=data
                ))
        
        return sorted(targets, key=lambda x: x.priority, reverse=True)
```

### 3. As Reasoning Provider for Complex Decisions

**Strategic Level**:

When SAGE faces complex decisions (which plugin to invoke? how to handle conflicting observations?), it could use Nemotron:

```python
# In SAGE's resource planning:
class ResourcePlanning:
    def plan_resources(self, attention_targets, available_memory):
        """Decide which resources to load"""
        
        # For straightforward cases, use heuristics
        if len(attention_targets) <= 3:
            return self.heuristic_planning(attention_targets)
        
        # For complex cases, ask Nemotron
        decision_prompt = f"""
        Current attention targets: {[t.modality for t in attention_targets]}
        Available memory: {available_memory} GB
        Loaded plugins: {list(self.active_resources.keys())}
        
        Which plugins should we load? Consider:
        - Modality matches
        - Memory efficiency
        - Trust scores: {self.trust_scores}
        - Potential interactions
        """
        
        reasoning = self.nemotron_plugin.think(decision_prompt)
        decisions = self.parse_reasoning(reasoning)
        
        return decisions
```

### 4. As Q&A Over Observations

**User Interaction**:

When SAGE has made observations/inferences, external users could ask Nemotron questions:

```python
# External query layer
class SAGEQAInterface:
    def answer_question(self, question, sage_state):
        """Answer question about current SAGE observations"""
        
        # Build context from SAGE's memory
        relevant_memories = self.snarc_memory.retrieve(
            salience_threshold=0.7
        )
        
        # Format for Nemotron
        context = self.format_context(sage_state, relevant_memories)
        
        prompt = f"""
        System state and observations:
        {context}
        
        Question: {question}
        
        Answer based on the observations:
        """
        
        answer = self.nemotron_plugin.think(prompt)
        return answer
```

---

## Current Integration Status

### What's Already Implemented

1. **Q3-Omni 30B Integration** (SAGE's current language provider)
   - Conversation management working
   - Multi-turn dialogue with history
   - Integrated with IRP framework
   - ATP budget awareness

2. **LLM Plugin Framework**
   - `sage/llm/external_llm.py` - Generic interface
   - `sage/irp/plugins/qwen_7b_irp.py` - Example IRP plugin
   - `sage/irp/plugins/llm_snarc_integration.py` - SNARC coupling

3. **Conversation Manager**
   - `sage/conversation/q3omni_chat_manager.py`
   - Handles context window management
   - Sliding window strategy for long conversations
   - Lazy model loading

4. **Tests & Validation**
   - `sage/tests/explore_q3omni_kv_cache.py` - Understanding model internals
   - `sage/quantization/` - FP4 quantization for efficiency
   - Validated on Thor (Jetson AGX) + Legion (RTX 4090) + Sprout (Orin Nano)

### What's Ready for Nemotron

1. **Drop-in Replacement**: Swap Q3-Omni with Nemotron in existing integration points
   ```python
   # Currently: Q3OmniConversationManager
   # Could be: NemotronConversationManager
   # Same interface, different model
   ```

2. **Lightweight Integration**: Nemotron's efficiency makes it ideal for:
   - Edge devices (Jetson Nano)
   - Continuous monitoring loops (SAGE's always-on design)
   - ATP-constrained scenarios (limited computation budget)

3. **Existing Infrastructure**:
   - IRP plugin framework ready
   - Conversation management tested
   - SNARC integration proven
   - Memory systems operational

### What Would Need Development

1. **Multimodal Processing** (if Nemotron has vision capabilities)
   - Integrate with TinyVAE layer
   - Coordinate with vision IRP plugins
   - Shared latent space alignment

2. **Reasoning-Level Integration** (using Nemotron for strategic decisions)
   - Decision prompts for resource planning
   - Conflict resolution in attention targets
   - Complex reasoning about uncertainty

3. **Custom Quantization** (for Jetson deployment)
   - FP4 quantization if needed
   - Memory profiling on target hardware
   - Batch size optimization

4. **Trust Calibration**:
   - Train trust scoring on Nemotron outputs
   - Compare reasoning quality vs smaller models
   - Determine cost/benefit trade-off

---

## Strategic Questions: SAGE + Nemotron

### Does SAGE Need Nemotron?

**Current Answer**: SAGE is **functionally complete without Nemotron**
- Can orchestrate multiple specialized plugins
- Q3-Omni 30B working well
- Focus on attention management, not language quality
- Smaller models (0.5B) showing good results for epistemic reasoning

**Why Add Nemotron**:
1. **Better reasoning quality** - Larger model may handle complex queries better
2. **More efficient** - If Nemotron is smaller/faster, reduces computational load
3. **Better specialization** - Different model families have different strengths
4. **Redundancy** - Multiple language models for robustness
5. **Experimentation** - Compare reasoning quality empirically

### Does Nemotron Need SAGE?

**Current Answer**: Nemotron works standalone as LLM
- Takes text input, generates text output
- No continuous awareness needed
- No multi-modal coherence required

**Why Nemotron in SAGE**:
1. **Coherent action** - SAGE ensures Nemotron output used appropriately
2. **Resource efficiency** - SAGE decides WHEN to invoke Nemotron (not every observation)
3. **Multi-modal context** - SAGE provides grounded understanding (vision, audio, etc.)
4. **Continuous learning** - SAGE's trust updates help Nemotron selection
5. **Edge deployment** - SAGE's metabolic states prevent overload

**Analogy**: 
- Nemotron = Specialist doctor (very knowledgeable about one thing)
- SAGE = Hospital administrator (decides WHEN to call which specialist)
- Together = Functional medical system (not just a smart doctor sitting in waiting room)

### Optimal Configuration

**Recommended Setup**:
```
SAGE Core (Continuous Loop)
├── Attention: SNARC salience scoring
├── Sensory Input: Vision, Audio, Proprioception IRP plugins
├── Strategic Reasoning: Nemotron for complex decisions
│   └── Conversation Management: Multi-turn dialogue
├── Tactical Execution: Smaller L-module or fast heuristics
└── Memory: SNARC + Circular Buffer + SQLite

ATP Budget: ~500/cycle
├── 200 vision processing
├── 100 audio processing
├── 150 language reasoning (Nemotron)
├── 50 memory updates
└── Remaining for contingencies
```

**Advantage Over Standalone Nemotron**:
- Nemotron only invoked when attention targets require language reasoning
- Continuous monitoring of surprise/novelty prevents redundant processing
- Multi-modal grounding prevents hallucinations about unobserved facts
- Trust-based selection learns which problems Nemotron solves well
- Energy-based halt prevents over-reasoning

---

## Architecture Summary: Clear Picture of Integration

### The Fractal H↔L Pattern

SAGE implements hierarchical↔linear pattern at multiple scales:

1. **Neural Layer**: Transformer blocks (attention↔feedforward)
2. **Agent Layer**: SAGE (strategic H↔tactical L)
3. **Device Layer**: Edge device ↔ cloud resources
4. **Federation Layer**: Coordinator ↔ worker nodes
5. **Development Layer**: Human ↔ automation

**Nemotron's Role**: Provides sophisticated H-level reasoning

### Data Flow with Nemotron

```
Physical Observations
├── Vision sensor → Vision IRP plugin → TinyVAE → 64D latent
├── Audio sensor → Audio IRP plugin → embedding → semantic features
└── Proprioception → Sensor fusion → state representation

SNARC Salience Scoring (5D)
↓ (Surprise, Novelty, Arousal, Reward, Conflict)

Attention Targets (ranked by priority)
↓
SAGE Core Decision:
├─ For straightforward targets: Fast heuristics
└─ For complex targets: Invoke Nemotron IRP plugin
   ├── Init state (build prompt from observations)
   ├── Step (generate refinement)
   ├── Energy (assess quality)
   └── Halt (detect convergence)

Refined Understanding
↓
ATP-weighted Action Selection
↓
Effectors (speech, motion, other outputs)
↓
Memory Update (SNARC scores new patterns)
```

### Why This Architecture Works

1. **Separation of Concerns**
   - SAGE manages attention/resources
   - Nemotron provides reasoning
   - Each can be developed independently

2. **Efficiency**
   - Don't invoke reasoning unless needed
   - ATP budget prevents overuse
   - Energy-based halt prevents thrashing

3. **Robustness**
   - Trust scoring learns Nemotron reliability
   - Fallback to heuristics if Nemotron unavailable
   - Multiple IRP plugins provide redundancy

4. **Interpretability**
   - SNARC scores explain why actions taken
   - ATP budget transparent
   - IRP energy metrics measurable

---

## Conclusion: Nemotron as Strategic Reasoning Module

**SAGE** = Cognition kernel (scheduling, attention, resources)  
**Nemotron** = Strategic reasoning module (language understanding, complex decisions)  
**Together** = Grounded, coherent, energy-efficient edge AI system

**Key Insight**: SAGE doesn't replace LLMs; it orchestrates them intelligently. Nemotron becomes one specialized plugin in SAGE's ecosystem, invoked when language reasoning is actually needed, weighted by trust, constrained by energy.

**Practical Implication**: Integration is straightforward because infrastructure already exists. Nemotron can drop in as improved language provider with minimal changes to SAGE core.

