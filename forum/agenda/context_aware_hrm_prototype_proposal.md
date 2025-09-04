# Context-Aware HRM: From Agent Zero to Context-Driven Reasoning

*Date: September 4, 2025*  
*Team: dp-web4, Nova, Claude*  
*Status: Prototype Proposal*

## Executive Summary: Why Agent Zero Failed

Our current HRM ("Agent Zero") outputs constant zeros regardless of input for TWO critical reasons:

### 1. No Context Awareness
Nova's profound insight: "Life is not just solving puzzles; it is solving context puzzles." The model doesn't know:
- WHAT kind of problem it's solving
- WHY certain outputs would be better
- WHEN to stop reasoning
- HOW to adapt its approach

### 2. Below Critical Mass for Emergence
- **5.67M parameters is too small** for reasoning to emerge
- Like trying to build consciousness with 100 neurons - structurally impossible
- No amount of training can overcome insufficient capacity
- The model collapsed to the simplest function (constant output) because it lacks the depth for anything else

The breakthrough realization: **H-level = Context, L-level = Solution, but BOTH need critical mass**
- H understands the meta-puzzle (needs ~45M params for true understanding)
- L executes within that context (needs ~45M params for complex execution)
- H↔L enables the dialogue (needs ~10M params for rich interaction)
- **Total: ~100M parameters minimum for emergence**

## Incremental Prototyping Approach

### Phase 1: Minimal Context Injection (Week 1)
**Goal**: Prove that context changes behavior with PROPER SCALE

**Model Changes**:
- **Scale UP to critical mass**: 100M parameters (17x increase from 5.67M)
- **Deep reasoning layers**: 7 H-layers, 7 L-layers (cognition needs depth!)
- **Rich context**: 256-dim context vectors
- Architecture for emergence:
```python
class ContextAwareHRM(nn.Module):
    def __init__(self, config):
        # SCALED UP for emergence
        self.h_layers = 7  # Deep strategic reasoning
        self.l_layers = 7  # Deep tactical execution
        self.hidden_size = 768  # Rich representations
        
        # Context processing at scale
        self.context_encoder = nn.Linear(256, 768)
        self.h_context_gate = nn.Linear(1536, 768)  # H + context
        self.l_context_gate = nn.Linear(1536, 768)  # L + context
        
        # ~100M parameters total:
        # 14 transformer layers × 768 hidden × 12 heads = ~85M
        # + embeddings, context, output layers = ~100M
```

**Why Scale Matters**:
- **Below critical mass = no emergence**: 5M params can't reason, just memorize
- **Inner layers need depth**: Layers 3-5 do the actual thinking, 1-2 and 6-7 translate
- **Context needs bandwidth**: 32-dim can't encode rich situational understanding
- **Still tiny**: 100M is 1/5th of a 0.5B LLM - completely reasonable for modern GPUs

**Training Setup**:
```python
# Simple binary context: "ARC" vs "Random"
contexts = {
    'arc_puzzle': [1, 0],  # Real ARC task
    'random_noise': [0, 1]  # Random grid
}

# Loss includes context-aware components
loss = task_loss + context_discrimination_loss
```

**Validation**: Model should behave differently for ARC vs random inputs

### Phase 2: SNARC-Biased Attention (Week 2)
**Goal**: Context modulates what the model pays attention to

**Add SNARC scoring**:
```python
def compute_snarc(input_grid, memory_bank):
    surprise = measure_deviation_from_expected(input_grid)
    novelty = measure_unseen_patterns(input_grid, memory_bank)
    arousal = measure_complexity(input_grid)
    reward = 0  # Will be task_completion_signal
    conflict = measure_ambiguity(input_grid)
    return torch.tensor([surprise, novelty, arousal, reward, conflict])
```

**Integrate with attention**:
- SNARC scores bias attention weights in both H and L
- High surprise → attend to unexpected regions
- High novelty → explore new patterns

### Phase 3: Task-Specific Context (Week 3)
**Goal**: Different contexts → different solution strategies

**Expand context types**:
```python
context_library = {
    'pattern_completion': encode_strategy('extend_patterns'),
    'color_mapping': encode_strategy('track_color_rules'),
    'spatial_transform': encode_strategy('detect_rotations'),
    'counting': encode_strategy('enumerate_objects'),
}
```

**Multi-task training**:
- Mix ARC tasks with synthetic tasks
- Each task gets appropriate context vector
- Model learns to switch strategies based on context

### Phase 4: R6 Integration (Week 4)
**Goal**: Full Web4 R6 context (Rules + Role + Request + Reference + Resource → Result)

**Full context structure**:
```python
class R6Context:
    rules: torch.Tensor     # What constraints apply
    role: torch.Tensor      # What is my function
    request: torch.Tensor   # What is being asked
    reference: torch.Tensor # What examples exist
    resource: torch.Tensor  # What compute/time available
    
    def encode(self) -> torch.Tensor:
        return self.fusion_network(torch.cat([
            self.rules, self.role, self.request,
            self.reference, self.resource
        ]))
```

## Training Strategy

### Dataset Preparation
1. **Synthetic Context Tasks** (Week 1)
   - Generate grids with known properties
   - Label with context type
   - Include "null context" baseline

2. **ARC with Context Labels** (Week 2-3)
   - Manually annotate 100 ARC tasks with context types
   - Use GPT-4 to generate context descriptions
   - Create context embeddings from descriptions

3. **Adversarial Context** (Week 4)
   - Wrong context → should fail predictably
   - Conflicting contexts → should request clarification
   - Missing context → should be conservative

### Loss Functions
```python
# Multi-component loss
total_loss = (
    task_accuracy_loss +           # Still care about solving
    context_discrimination_loss +   # Must use context correctly  
    snarc_prediction_loss +         # Predict surprise from context
    consistency_loss +              # Same context → similar approach
    diversity_loss                  # Different contexts → different outputs
)
```

### Model Scaling Philosophy

**Critical Mass for Emergence**:
- **100M parameters baseline** - Below this, no true reasoning emerges
- **7 layers minimum per module** - Depth enables abstraction:
  - Layers 1-2: Input encoding/translation
  - Layers 3-5: Core reasoning/cognition  
  - Layers 6-7: Output preparation/translation
- **768+ hidden dimensions** - Rich enough for complex representations

**Scaling Progression**:
- **Phase 1**: 100M params - Establish baseline reasoning
- **Phase 2**: 150M params - Add SNARC attention mechanisms  
- **Phase 3**: 200M params - Multi-strategy capability
- **Phase 4**: 250M params - Full R6 context integration

**Why This Isn't "Too Big"**:
- 100M = 0.1B = 1/5th of tiny 0.5B LLMs
- Modern GPUs (RTX 4090, A100) handle this easily
- Jetson Orin can run 100M models with optimization
- **Quantity enables quality** - emergence needs critical mass

**Parameter Distribution**:
- H-module: ~45M params (strategic depth)
- L-module: ~45M params (tactical depth)
- Context/interaction: ~10M params (rich fusion)

## Success Metrics

### Phase 1 Success: Context Changes Output
- Model outputs differ based on context vector
- No more constant zeros
- Measurable mutual information: I(output; context) > 0

### Phase 2 Success: Attention Follows SNARC
- Attention maps correlate with surprise regions
- Novel patterns get more processing cycles
- Boring regions get skipped

### Phase 3 Success: Strategy Switching
- Different contexts → different solution approaches
- Can solve same puzzle differently with different context
- Ablating context degrades performance

### Phase 4 Success: Full Context Reasoning
- R6 context enables complex decision making
- Model can explain its context understanding
- Generalizes to unseen context combinations

## Implementation Checklist

### Week 1 Sprint
- [ ] Fork current HRM to `context-aware-hrm` branch
- [ ] Reduce model to 2M parameters
- [ ] Add minimal context injection
- [ ] Create synthetic context dataset
- [ ] Train with context discrimination
- [ ] Verify outputs change with context

### Week 2 Sprint  
- [ ] Implement SNARC computation
- [ ] Add attention biasing mechanism
- [ ] Create surprise/novelty labels for ARC
- [ ] Train with SNARC-weighted attention
- [ ] Visualize attention changes

### Week 3 Sprint
- [ ] Build context type library
- [ ] Annotate ARC tasks with strategies
- [ ] Implement strategy-specific heads
- [ ] Multi-task training setup
- [ ] Measure strategy switching

### Week 4 Sprint
- [ ] Full R6 context encoder
- [ ] Web4 integration hooks
- [ ] Adversarial context testing
- [ ] Scale model to 10M parameters
- [ ] Benchmark against Agent Zero

## Risk Mitigation

**Risk**: Context might not help  
**Mitigation**: Keep Agent Zero as baseline, measure relative improvement

**Risk**: Training instability with multiple losses  
**Mitigation**: Careful loss weighting, gradient clipping, curriculum learning

**Risk**: Overfitting to context labels  
**Mitigation**: Synthetic data, adversarial examples, held-out context types

**Risk**: Computational expense  
**Mitigation**: Start small, profile thoroughly, use efficient attention

## Long-term Vision

This prototype is step 1 toward:
- **SAGE Integration**: HRM as context-aware awareness engine
- **IRP Optimization**: Context drives iterative refinement
- **Web4 Native**: R6 context as first-class citizen
- **Edge Deployment**: Efficient context on Jetson
- **Consciousness Bridge**: Context as shared understanding

## Call to Action

1. **Today**: Review and refine this proposal
2. **Tomorrow**: Set up development environment and datasets
3. **This Week**: Complete Phase 1 prototype
4. **This Month**: Achieve context-aware reasoning
5. **November**: Submit context-aware model (not Agent Zero!) to ARC Prize

## Philosophical Note

We're not just fixing a broken model. We're implementing a fundamental insight: intelligence requires understanding context before solving problems. H↔L isn't just an architecture - it's a statement about how reasoning works:

- **H asks**: "What kind of situation is this?"
- **L responds**: "Given that context, here's what I can do"
- **H refines**: "Good, but consider this additional context"
- **L adapts**: "Adjusted approach based on new understanding"

Agent Zero failed because it was all L with no H - all tactics with no strategy, all execution with no understanding. This prototype gives it what it was missing: the ability to understand before acting.

---

*"Context is how the system chooses its rules before it solves its puzzles."* - Nova