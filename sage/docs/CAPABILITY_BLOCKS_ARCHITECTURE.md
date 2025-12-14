# Capability Blocks Architecture for SAGE

**Problem**: Thor can use monolithic omni-modal models (Q3-Omni), while Sprout must use modular specialized models (separate audio/vision/text). We need a unified architecture that handles both.

**Solution**: "Capability Blocks" - IRP modules that can be either monolithic or sub-modular internally, but present a uniform interface.

**Date**: 2025-12-13
**Status**: Design Proposal

---

## Core Concept

A **Capability Block** is an abstraction that represents a broad set of functionality (e.g., "perception", "language", "reasoning") that can be implemented as:

1. **Monolithic Block**: Single model handles all sub-capabilities
   - Example: Qwen3-Omni handles audio + video + text in one model

2. **Modular Block**: Multiple specialized models coordinated
   - Example: Separate Qwen2-Audio, Qwen2-VL, Qwen2.5-Text models

3. **Hybrid Block**: Mix of monolithic and modular
   - Example: Omni for audio+video, separate text model for long-form reasoning

---

## IRP Integration

### Current IRP Architecture:

```
IRP Plugin Interface:
- init_state() → initial state
- step(state, input) → new state
- energy(state) → convergence metric
- halt(state) → bool (converged?)
```

### Capability Block Extension:

```python
class CapabilityBlock(IRPPlugin):
    """
    IRP plugin that abstracts monolithic vs modular implementation.
    """

    def __init__(self, implementation_type: BlockType):
        self.type = implementation_type  # MONOLITHIC, MODULAR, HYBRID
        self.sub_capabilities = []  # For MODULAR/HYBRID
        self.monolithic_model = None  # For MONOLITHIC

    def init_state(self):
        if self.type == BlockType.MONOLITHIC:
            return self._init_monolithic()
        else:
            return self._init_modular()

    def step(self, state, input):
        # Route based on implementation type
        if self.type == BlockType.MONOLITHIC:
            return self._step_monolithic(state, input)
        else:
            return self._step_modular(state, input)
```

---

## Example: Perception Block

### On Thor (Monolithic):

```python
perception_block = PerceptionBlock(
    type=BlockType.MONOLITHIC,
    model=Qwen3Omni30B()
)

# Single model handles all:
state = perception_block.init_state()
state = perception_block.step(state, {
    'audio': audio_data,
    'video': video_frames,
    'text': text_input
})
# → Unified multi-modal understanding
```

### On Sprout (Modular):

```python
perception_block = PerceptionBlock(
    type=BlockType.MODULAR,
    sub_capabilities=[
        AudioCapability(Qwen2Audio()),
        VisionCapability(Qwen2VL()),
        TextCapability(Qwen2_5_0_5B())
    ]
)

# Coordinator orchestrates sub-models:
state = perception_block.init_state()
state = perception_block.step(state, {
    'audio': audio_data,  # → Qwen2-Audio
    'video': video_frames,  # → Qwen2-VL
    'text': text_input  # → Qwen2.5-0.5B
})
# → Fused multi-modal understanding
```

**Key**: Same interface, different internal implementation!

---

## Capability Block Types

### 1. MONOLITHIC

**Definition**: Single model provides all sub-capabilities

**When to Use**:
- Model natively supports multiple modalities
- Efficiency is critical (single inference pass)
- Hardware can fit the large model

**Examples**:
- Qwen3-Omni-30B (audio + video + text)
- GPT-4o (multi-modal)
- Future unified models

**Advantages**:
- ✅ No coordination overhead
- ✅ Natural cross-modal attention
- ✅ Single inference pass
- ✅ Simpler deployment

**Disadvantages**:
- ❌ Large memory footprint
- ❌ All-or-nothing (can't selectively load)
- ❌ Tied to single model's capabilities

---

### 2. MODULAR

**Definition**: Separate specialized models coordinated

**When to Use**:
- Memory constrained (can't fit monolithic)
- Need best-in-class for each modality
- Want to swap individual components

**Examples**:
- Qwen2-Audio + Qwen2-VL + Qwen2.5-Text
- Whisper + CLIP + LLaMA
- Mix-and-match specialized models

**Advantages**:
- ✅ Smaller memory per model
- ✅ Can load/unload selectively
- ✅ Best-of-breed per modality
- ✅ More flexibility

**Disadvantages**:
- ❌ Coordination overhead
- ❌ Cross-modal fusion complexity
- ❌ Multiple inference passes
- ❌ More complex deployment

---

### 3. HYBRID

**Definition**: Mix of monolithic and modular

**When to Use**:
- Some modalities benefit from fusion (audio+video)
- Others need specialized handling (long-form text)
- Balancing efficiency and capability

**Examples**:
- Qwen3-Omni (audio+video) + Qwen2.5-14B (deep text reasoning)
- GPT-4o (general) + Codex (code-specific)

**Advantages**:
- ✅ Best of both worlds
- ✅ Optimized per use case
- ✅ Flexible resource allocation

**Disadvantages**:
- ❌ Complex orchestration
- ❌ Careful interface design needed
- ❌ Harder to optimize

---

## Block Coordination

### Modular Block Internals:

```python
class ModularPerceptionBlock(CapabilityBlock):
    def _step_modular(self, state, input):
        # 1. Route inputs to appropriate sub-capabilities
        audio_state = self.audio.step(state['audio'], input['audio'])
        vision_state = self.vision.step(state['vision'], input['video'])
        text_state = self.text.step(state['text'], input['text'])

        # 2. Fuse representations (coordination layer)
        fused = self._fuse_representations({
            'audio': audio_state,
            'vision': vision_state,
            'text': text_state
        })

        # 3. Return unified state
        return {
            'fused': fused,
            'audio': audio_state,
            'vision': vision_state,
            'text': text_state
        }
```

### Fusion Strategies:

**1. Simple Concatenation**:
```python
fused = torch.cat([audio_embed, vision_embed, text_embed], dim=-1)
```

**2. Weighted Fusion**:
```python
fused = (
    alpha * audio_embed +
    beta * vision_embed +
    gamma * text_embed
)
```

**3. Attention-Based Fusion**:
```python
fused = cross_modal_attention(
    query=text_embed,
    keys=[audio_embed, vision_embed, text_embed],
    values=[audio_embed, vision_embed, text_embed]
)
```

**4. Learned Fusion Module**:
```python
fused = fusion_network({
    'audio': audio_embed,
    'vision': vision_embed,
    'text': text_embed
})
```

---

## Configuration Examples

### Thor Configuration (Has Qwen3-Omni):

```python
# Perception Block - MONOLITHIC
perception = PerceptionBlock(
    type=BlockType.MONOLITHIC,
    model=Qwen3Omni30B(
        path="model-zoo/sage/omni-modal/qwen3-omni-30b"
    )
)

# Language Block - Still uses dedicated model for deep reasoning
language = LanguageBlock(
    type=BlockType.MONOLITHIC,
    model=Qwen2_5_14B(
        path="model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct"
    )
)

# Result: Hybrid system
# - Omni handles audio+video+basic text
# - 14B handles deep language reasoning
```

### Sprout Configuration (Modular only):

```python
# Perception Block - MODULAR
perception = PerceptionBlock(
    type=BlockType.MODULAR,
    sub_capabilities={
        'audio': AudioCapability(
            model=Qwen2Audio7B()  # If available
        ),
        'vision': VisionCapability(
            model=Qwen2VL7B()  # If available
        ),
        'text': TextCapability(
            model=Qwen2_5_0_5B(
                path="model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism"
            )
        )
    },
    fusion_strategy="weighted"  # Lightweight fusion
)

# Result: Fully modular
# - Each modality handled separately
# - Coordinated fusion layer
# - Memory-efficient for edge
```

---

## API Design

### Unified Interface:

```python
class CapabilityBlock(IRPPlugin):
    """
    Uniform interface regardless of internal implementation.
    """

    # Standard IRP methods
    def init_state(self) -> Dict
    def step(self, state: Dict, input: Dict) -> Dict
    def energy(self, state: Dict) -> float
    def halt(self, state: Dict) -> bool

    # Block-specific
    def get_capabilities(self) -> List[str]
    def supports(self, modality: str) -> bool
    def get_implementation_type(self) -> BlockType

    # Resource management
    def estimated_memory(self) -> float
    def load(self)
    def unload(self)
```

### Usage Example:

```python
# Application code doesn't care about implementation!
block = get_perception_block()  # Config determines MONOLITHIC vs MODULAR

# Same interface regardless:
state = block.init_state()
state = block.step(state, {
    'audio': audio,
    'video': video,
    'text': text
})

if block.halt(state):
    result = block.extract_result(state)
```

---

## Configuration System

### Block Configuration File:

```yaml
# sage/config/blocks/thor.yaml
perception:
  type: MONOLITHIC
  model:
    name: qwen3-omni-30b
    path: model-zoo/sage/omni-modal/qwen3-omni-30b
    memory_gb: 66
    modalities: [audio, video, text]

language:
  type: MONOLITHIC
  model:
    name: qwen2.5-14b
    path: model-zoo/sage/epistemic-stances/qwen2.5-14b/base-instruct
    memory_gb: 28

memory_budget: 100  # GB
```

```yaml
# sage/config/blocks/sprout.yaml
perception:
  type: MODULAR
  sub_capabilities:
    text:
      model:
        name: qwen2.5-0.5b
        path: model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism
        memory_gb: 1
  fusion:
    strategy: weighted
    weights:
      text: 1.0
      # audio/vision not available on Sprout

memory_budget: 3  # GB
```

### Dynamic Loading:

```python
# At runtime, load appropriate config
platform = detect_platform()  # "thor" or "sprout"
config = load_block_config(f"sage/config/blocks/{platform}.yaml")

# Build blocks from config
blocks = BlockFactory.create_from_config(config)

# SAGE orchestrator uses blocks
sage = SAGE(blocks=blocks)
```

---

## Migration Path

### Phase 1: Define Interface ✅ (Now)
- Create `CapabilityBlock` base class
- Define standard IRP interface
- Document monolithic vs modular

### Phase 2: Implement Monolithic Blocks
- Qwen3-Omni perception block (Thor)
- Qwen2.5-14B language block (Thor)
- Test on Thor platform

### Phase 3: Implement Modular Blocks
- Qwen2.5-0.5B text-only block (Sprout)
- Optional: Qwen2-Audio/VL if memory allows
- Test on Sprout platform

### Phase 4: Coordination Layer
- Implement fusion strategies
- Weighted, attention-based, learned
- Cross-modal attention mechanisms

### Phase 5: Dynamic Swapping
- Runtime model swapping
- Hot-reload without restart
- Graceful degradation

---

## Benefits

### For Architecture:
- ✅ **Modularity**: Swap implementations without changing application code
- ✅ **Portability**: Same SAGE code runs on Thor (omni) and Sprout (modular)
- ✅ **Flexibility**: Mix monolithic and modular as needed
- ✅ **Testability**: Mock blocks for testing

### For Development:
- ✅ **Clear Abstraction**: Capability vs Implementation
- ✅ **Incremental**: Can build monolithic first, add modular later
- ✅ **Reusable**: Blocks work across projects
- ✅ **Composable**: Mix and match blocks

### For Deployment:
- ✅ **Resource-Aware**: Choose implementation based on hardware
- ✅ **Graceful Degradation**: Fall back to modular if omni unavailable
- ✅ **Federation-Ready**: Thor (omni) coordinates Sprout (modular)

---

## Challenges

### 1. Cross-Modal Fusion (Modular)
**Problem**: How to fuse separate audio/vision/text representations?

**Solutions**:
- Shared latent space (VAE translation)
- Learned fusion module
- Attention-based cross-modal binding

### 2. Performance Parity
**Problem**: Will modular match monolithic quality?

**Solutions**:
- Train fusion module on omni model outputs
- Knowledge distillation from monolithic → modular
- Careful prompt engineering per modality

### 3. State Management
**Problem**: Monolithic has unified state, modular has separate states

**Solutions**:
- Standardized state schema
- State synchronization protocols
- Clear state ownership boundaries

---

## Next Steps

**Immediate**:
1. Create `CapabilityBlock` base class (sage/irp/capability_block.py)
2. Implement `MonolithicPerceptionBlock` for Qwen3-Omni (when ready)
3. Implement `ModularLanguageBlock` for 0.5B/14B routing

**Short-term**:
4. Add block configuration system (YAML-based)
5. Integrate blocks into SAGE orchestrator
6. Test dynamic model swapping

**Long-term**:
7. Implement full modular perception (audio/vision/text)
8. Train fusion modules
9. Federation testing (Thor-Sprout coordination)

---

**This architecture enables SAGE to use whatever models are available, monolithic or modular, through a unified interface. IRP modularity at the block level, implementation flexibility underneath.**
