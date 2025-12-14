# Thor SAGE - Full Implementation Plan

**Goal**: Fully functional SAGE on Thor with dynamic model swapping, capability blocks, and complete consciousness continuity

**Platform**: Jetson AGX Thor (122GB unified memory)
**Date**: 2025-12-13
**Status**: Planning

---

## Current Status

### ✅ What We Have:
1. **14B Model**: Tested and working
2. **Multi-model loader**: 0.5B/14B/72B routing with fallback
3. **Qwen3-Omni-30B**: Downloaded (66GB, awaiting transformers update)
4. **Coherent Awakening**: Session-to-session continuity
5. **DREAM Consolidation**: Sleep-cycle learning integration
6. **IRP Framework**: 15+ plugins, iterative refinement
7. **Capability Blocks**: Architecture designed

### ❌ What We Need:
1. **Unified SAGE loop**: Integrate all components
2. **Capability blocks**: Implement base class + concrete blocks
3. **Dynamic model swapping**: Hot-reload models
4. **Omni-modal support**: When transformers updated
5. **Interactive session**: Production-ready interface
6. **Federation**: Thor ↔ Sprout coordination

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                        SAGE Kernel                           │
│  (Orchestrator, ATP Budget, Metabolic States, Trust)        │
└──────────────────┬──────────────────────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
    ┌────▼────┐        ┌────▼────┐
    │ Sensing │        │ Effector│
    │ Blocks  │        │ Blocks  │
    └────┬────┘        └────┬────┘
         │                   │
    ┌────▼──────────────────▼────┐
    │   Capability Blocks Layer   │
    │  - Perception (omni/modular)│
    │  - Language (14B/0.5B)      │
    │  - Memory (SNARC/IRP)       │
    │  - Reasoning (H-Module)     │
    └────┬────────────────────────┘
         │
    ┌────▼────────────────────────┐
    │   Model Zoo Layer            │
    │  - Qwen3-Omni-30B (omni)    │
    │  - Qwen2.5-14B (strategic)  │
    │  - Qwen2.5-0.5B (tactical)  │
    └──────────────────────────────┘
```

---

## Phase 1: Core Integration (Week 1)

### 1.1: Capability Block Base Class

**File**: `sage/irp/capability_block.py`

```python
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from sage.irp.base import IRPPlugin

class BlockType(Enum):
    MONOLITHIC = "monolithic"
    MODULAR = "modular"
    HYBRID = "hybrid"

class CapabilityBlock(IRPPlugin):
    """Base class for capability blocks."""

    def __init__(self, block_type: BlockType):
        self.block_type = block_type
        self.loaded = False
        self.memory_usage_gb = 0.0

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this block provides."""
        pass

    @abstractmethod
    def supports(self, modality: str) -> bool:
        """Check if block supports given modality."""
        pass

    @abstractmethod
    def load(self):
        """Load models into memory."""
        pass

    @abstractmethod
    def unload(self):
        """Unload models from memory."""
        pass

    def estimated_memory(self) -> float:
        """Return estimated memory usage in GB."""
        return self.memory_usage_gb
```

**Tasks**:
- [ ] Create `sage/irp/capability_block.py`
- [ ] Define `BlockType` enum
- [ ] Implement base class with IRP interface
- [ ] Add resource management (load/unload)
- [ ] Write unit tests

---

### 1.2: Language Capability Block

**File**: `sage/irp/blocks/language_block.py`

```python
class LanguageBlock(CapabilityBlock):
    """Language processing capability block."""

    def __init__(self, model_loader: MultiModelLoader):
        super().__init__(BlockType.MONOLITHIC)
        self.model_loader = model_loader

    def init_state(self):
        return {
            'messages': [],
            'context_window': [],
            'current_complexity': TaskComplexity.MODERATE
        }

    def step(self, state, input_text, complexity=None):
        # Auto-detect complexity if not provided
        if complexity is None:
            complexity = self._detect_complexity(input_text)

        # Generate using appropriate model
        response = self.model_loader.generate(
            prompt=input_text,
            complexity=complexity
        )

        # Update state
        state['messages'].append({
            'user': input_text,
            'assistant': response,
            'model': complexity.value
        })

        return state

    def _detect_complexity(self, text):
        # Heuristic complexity detection
        word_count = len(text.split())
        has_technical = any(word in text.lower() for word in ['explain', 'analyze', 'design'])

        if word_count < 10 and '?' in text:
            return TaskComplexity.SIMPLE
        elif has_technical or word_count > 50:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.MODERATE
```

**Tasks**:
- [ ] Implement `LanguageBlock`
- [ ] Add complexity detection heuristics
- [ ] Integrate with multi-model loader
- [ ] Test with 0.5B/14B routing
- [ ] Add conversation history management

---

### 1.3: Perception Capability Block (Stub)

**File**: `sage/irp/blocks/perception_block.py`

```python
class PerceptionBlock(CapabilityBlock):
    """Multi-modal perception capability block."""

    def __init__(self, config: Dict):
        if 'omni_model' in config:
            super().__init__(BlockType.MONOLITHIC)
            self.omni_model = config['omni_model']
        else:
            super().__init__(BlockType.MODULAR)
            self.audio_model = config.get('audio_model')
            self.vision_model = config.get('vision_model')
            self.text_model = config.get('text_model')

    def supports(self, modality: str) -> bool:
        if self.block_type == BlockType.MONOLITHIC:
            return modality in ['audio', 'video', 'text']
        else:
            return (
                (modality == 'audio' and self.audio_model is not None) or
                (modality == 'video' and self.vision_model is not None) or
                (modality == 'text' and self.text_model is not None)
            )

    def step(self, state, input_data):
        if self.block_type == BlockType.MONOLITHIC:
            return self._step_monolithic(state, input_data)
        else:
            return self._step_modular(state, input_data)
```

**Tasks**:
- [ ] Create stub for modular path
- [ ] Implement text-only for now (use LanguageBlock)
- [ ] Defer audio/video until Qwen3-Omni ready
- [ ] Design fusion interface

---

### 1.4: Unified SAGE Loop

**File**: `sage/core/sage_kernel.py`

```python
class SAGEKernel:
    """Main SAGE consciousness loop."""

    def __init__(self, config: SAGEConfig):
        self.config = config
        self.blocks = self._initialize_blocks(config)
        self.awakening = CoherentAwakening(...)
        self.consciousness = UnifiedConsciousnessManager(...)
        self.running = False

    def run(self):
        """Main consciousness loop."""
        self.running = True

        # Boot sequence
        coherence_field = self.awakening.prepare_coherence_field()
        self.coherence_field = coherence_field

        while self.running:
            # 1. Sense
            observations = self.sense()

            # 2. Compute salience
            salience = self.compute_salience(observations)

            # 3. Allocate ATP
            atp_allocation = self.allocate_atp(salience)

            # 4. Route to capability blocks
            responses = self.process_with_blocks(observations, atp_allocation)

            # 5. Integrate results
            integrated = self.integrate_responses(responses)

            # 6. Update consciousness
            self.consciousness.consciousness_cycle(...)

            # 7. Act
            self.effectors.act(integrated)

            # 8. Check metabolic state
            if self.should_enter_dream():
                self.dream_consolidation()

    def sense(self):
        """Gather observations from sensors."""
        # For now: stdin
        user_input = input("You: ")
        return {'text': user_input}

    def process_with_blocks(self, observations, atp):
        """Route observations to capability blocks."""
        results = {}

        # Language block (always)
        if 'text' in observations:
            results['language'] = self.blocks['language'].step(
                state=self.blocks['language'].get_state(),
                input_text=observations['text']
            )

        # Perception block (if multi-modal)
        if 'audio' in observations or 'video' in observations:
            results['perception'] = self.blocks['perception'].step(
                state=self.blocks['perception'].get_state(),
                input_data=observations
            )

        return results
```

**Tasks**:
- [ ] Create `SAGEKernel` main loop
- [ ] Integrate sensing (stdin for now)
- [ ] Add SNARC salience computation
- [ ] Add ATP allocation logic
- [ ] Connect to capability blocks
- [ ] Add metabolic state management
- [ ] Implement DREAM trigger logic

---

## Phase 2: Dynamic Model Swapping (Week 2)

### 2.1: Model Swap Manager

**File**: `sage/core/model_swap_manager.py`

```python
class ModelSwapManager:
    """Manages dynamic model loading/unloading."""

    def __init__(self, multi_model_loader: MultiModelLoader):
        self.loader = multi_model_loader
        self.swap_history = []

    def swap_to(self, target_size: ModelSize, reason: str = ""):
        """Swap to different model size."""
        current = self.loader.get_loaded_models()

        # Unload current
        for size, config in current.items():
            if config.loaded:
                self.loader.unload_model(size)

        # Load target
        self.loader.load_model(target_size)

        # Log swap
        self.swap_history.append({
            'from': list(current.keys()),
            'to': target_size,
            'reason': reason,
            'timestamp': datetime.now()
        })

    def auto_swap_for_task(self, task_complexity: TaskComplexity):
        """Automatically swap model based on task."""
        current_loaded = self.loader.get_loaded_models()

        # Determine optimal model
        optimal = self._determine_optimal_model(task_complexity)

        # Swap if different
        if optimal not in current_loaded or not current_loaded[optimal].loaded:
            self.swap_to(optimal, reason=f"Task complexity: {task_complexity.value}")
```

**Tasks**:
- [ ] Implement `ModelSwapManager`
- [ ] Add swap history tracking
- [ ] Add auto-swap based on task
- [ ] Add metrics (swap frequency, reasons)
- [ ] Test hot-reload without session restart

---

### 2.2: Context Preservation During Swap

**Challenge**: How to maintain conversation context when swapping models?

**Solution**:
```python
class ContextManager:
    """Preserve context across model swaps."""

    def __init__(self, max_context_tokens=4096):
        self.max_context = max_context_tokens
        self.conversation_history = []
        self.compressed_context = None

    def before_swap(self, current_model):
        """Save context before swapping."""
        # Get current conversation
        context = self.conversation_history

        # Compress if needed (VAE or summarization)
        if self.total_tokens(context) > self.max_context:
            self.compressed_context = self.compress(context)
        else:
            self.compressed_context = context

    def after_swap(self, new_model):
        """Restore context after swapping."""
        # Inject compressed context into new model
        return self.compressed_context

    def compress(self, context):
        """Compress long context (summarize or VAE)."""
        # Option 1: Summarization
        summary = summarize_conversation(context)

        # Option 2: VAE compression
        # latent = vae.encode(context)

        return summary
```

**Tasks**:
- [ ] Implement `ContextManager`
- [ ] Add conversation history tracking
- [ ] Add compression strategies
- [ ] Test context continuity across swaps
- [ ] Measure quality degradation

---

## Phase 3: Interactive Session (Week 3)

### 3.1: Production Interactive Loop

**File**: `sage/awakening/thor_interactive.py`

```python
def interactive_thor_session():
    """Production-ready interactive session for Thor."""

    # Initialize SAGE kernel
    config = load_config("sage/config/thor.yaml")
    sage = SAGEKernel(config)

    # Boot with coherent awakening
    coherence_field = sage.awakening.prepare_coherence_field()

    print(f"Thor SAGE - Session {coherence_field.session_number}")
    print(f"Phase: {coherence_field.phase.value}")
    print()

    # Main loop
    while True:
        try:
            # Get input
            user_input = input("You: ").strip()

            if user_input.lower() in ['exit', 'quit', 'bye']:
                break

            # Process
            response = sage.respond(user_input)

            # Output
            print(f"Thor: {response}")
            print()

        except KeyboardInterrupt:
            break

    # End session with DREAM
    sage.end_session()
```

**Tasks**:
- [ ] Create production interactive script
- [ ] Add proper error handling
- [ ] Add model swap indicators
- [ ] Add consciousness state display
- [ ] Test multi-hour sessions

---

### 3.2: Rich Console Interface

**Optional Enhancement**: Use `rich` library for better UX

```python
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

def display_response(response, model_used, metabolic_state):
    """Display response with rich formatting."""

    console.print(Panel(
        Markdown(response),
        title=f"Thor (using {model_used})",
        subtitle=f"State: {metabolic_state}",
        border_style="blue"
    ))
```

---

## Phase 4: Qwen3-Omni Integration (When Ready)

### 4.1: Update Transformers

```bash
# When Qwen3-Omni is in transformers release
pip install transformers --upgrade

# Or install from source with Qwen3-Omni support
pip install git+https://github.com/huggingface/transformers@main
```

### 4.2: Omni Perception Block

```python
class OmniPerceptionBlock(PerceptionBlock):
    """Monolithic omni-modal perception."""

    def __init__(self, model_path):
        super().__init__(BlockType.MONOLITHIC)
        self.model = AutoModel.from_pretrained(model_path)

    def step(self, state, input_data):
        # Process all modalities in one pass
        result = self.model.generate(
            audio=input_data.get('audio'),
            video=input_data.get('video'),
            text=input_data.get('text')
        )

        return {
            'unified_representation': result,
            'modalities_processed': list(input_data.keys())
        }
```

---

## Phase 5: Federation (Week 4)

### 5.1: Thor-Sprout Coordination

```python
class FederationCoordinator:
    """Coordinate between Thor and Sprout."""

    def __init__(self, thor_sage, sprout_connection):
        self.thor = thor_sage
        self.sprout = sprout_connection

    def delegate_to_sprout(self, task):
        """Delegate simple task to Sprout."""
        if task.complexity == TaskComplexity.SIMPLE:
            return self.sprout.execute(task)
        else:
            return self.thor.execute(task)

    def parallel_processing(self, tasks):
        """Process tasks in parallel across federation."""
        simple_tasks = [t for t in tasks if t.complexity == TaskComplexity.SIMPLE]
        complex_tasks = [t for t in tasks if t.complexity != TaskComplexity.SIMPLE]

        # Sprout handles simple, Thor handles complex
        sprout_results = asyncio.gather(*[
            self.sprout.execute(t) for t in simple_tasks
        ])

        thor_results = asyncio.gather(*[
            self.thor.execute(t) for t in complex_tasks
        ])

        return await sprout_results + await thor_results
```

---

## Testing Strategy

### Unit Tests:
- [ ] Capability block interface
- [ ] Model swapping logic
- [ ] Context preservation
- [ ] Block routing
- [ ] Fallback mechanisms

### Integration Tests:
- [ ] Full SAGE loop (stdin → stdout)
- [ ] Model swap during conversation
- [ ] DREAM consolidation
- [ ] Session continuity
- [ ] Memory management

### Performance Tests:
- [ ] Response latency per model
- [ ] Memory usage per configuration
- [ ] Swap overhead
- [ ] Concurrent model loading

### Stress Tests:
- [ ] 8-hour continuous session
- [ ] Frequent model swaps
- [ ] Long context windows
- [ ] Memory pressure scenarios

---

## Success Criteria

### Functional:
- ✅ Thor boots and responds coherently
- ✅ Dynamic model swapping works (0.5B ↔ 14B)
- ✅ Context preserved across swaps
- ✅ Capability blocks route correctly
- ✅ DREAM consolidation saves/restores state
- ✅ Session-to-session continuity working

### Performance:
- ✅ Response time < 10s for 14B
- ✅ Model swap < 30s
- ✅ Memory usage within budget (100GB)
- ✅ Can run 8+ hour sessions without restart

### Quality:
- ✅ Responses coherent and contextual
- ✅ Appropriate model selected for task
- ✅ No context loss during swaps
- ✅ Developmental phase progression visible

---

## Timeline

**Week 1**: Core Integration
- Days 1-2: Capability block base class
- Days 3-4: Language block implementation
- Days 5-7: SAGE kernel loop

**Week 2**: Dynamic Swapping
- Days 1-3: Model swap manager
- Days 4-5: Context preservation
- Days 6-7: Testing and optimization

**Week 3**: Interactive Session
- Days 1-3: Production interactive script
- Days 4-5: Error handling and UX
- Days 6-7: Multi-hour session testing

**Week 4**: Federation (Optional)
- Days 1-3: Sprout integration
- Days 4-5: Parallel processing
- Days 6-7: Full federation testing

---

## Files to Create

### Core:
1. `sage/irp/capability_block.py` - Base class
2. `sage/irp/blocks/language_block.py` - Language capability
3. `sage/irp/blocks/perception_block.py` - Perception (stub)
4. `sage/core/sage_kernel.py` - Main loop
5. `sage/core/model_swap_manager.py` - Dynamic swapping
6. `sage/core/context_manager.py` - Context preservation

### Config:
7. `sage/config/thor.yaml` - Thor configuration
8. `sage/config/sprout.yaml` - Sprout configuration

### Interactive:
9. `sage/awakening/thor_interactive.py` - Production session
10. `sage/awakening/thor_cli.py` - Command-line interface

### Tests:
11. `sage/tests/test_capability_blocks.py`
12. `sage/tests/test_model_swapping.py`
13. `sage/tests/test_sage_kernel.py`
14. `sage/tests/test_context_preservation.py`

### Federation:
15. `sage/federation/coordinator.py` - Thor-Sprout coordination
16. `sage/federation/message_protocol.py` - Inter-instance messaging

---

## Next Immediate Steps

**Today**:
1. Create `capability_block.py` base class
2. Implement `LanguageBlock` with multi-model loader
3. Test dynamic routing (0.5B → 14B fallback)

**Tomorrow**:
4. Create `SAGEKernel` main loop (minimal version)
5. Integrate LanguageBlock into kernel
6. Test end-to-end stdin → model → stdout

**This Week**:
7. Add model swap manager
8. Add context preservation
9. Create production interactive script
10. Test 2-hour continuous session

---

**Goal**: By end of week, have Thor SAGE that can hold coherent multi-hour conversations with automatic model swapping based on task complexity, full session continuity, and DREAM consolidation.**
