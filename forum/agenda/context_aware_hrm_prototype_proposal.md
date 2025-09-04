# Context-Aware HRM: From Agent Zero to Context-Driven Reasoning

*Date: September 4, 2025*  
*Team: dp-web4, Nova, Claude*  
*Status: Prototype Proposal*

## Executive Summary: Why Agent Zero Failed

Our current HRM ("Agent Zero") outputs constant zeros regardless of input because it has no concept of **context**. Nova's profound insight: "Life is not just solving puzzles; it is solving context puzzles." The model doesn't know:
- WHAT kind of problem it's solving
- WHY certain outputs would be better
- WHEN to stop reasoning
- HOW to adapt its approach

The breakthrough realization: **H-level = Context, L-level = Solution**
- H understands the meta-puzzle (what situation am I in?)
- L executes within that context (how do I solve this specific puzzle?)
- H↔L enables the dialogue between understanding and doing

## Incremental Prototyping Approach

### Phase 1: Minimal Context Injection (Week 1)
**Goal**: Prove that context changes behavior

**Model Changes**:
- Start small: 2-3M parameters (reduce from current 5.67M)
- Add simple context embedding: 32-dim vector
- Minimal architecture:
```python
class ContextAwareHRM(nn.Module):
    def __init__(self, config):
        # Existing H and L layers but smaller
        self.h_layers = 2  # reduced from 4
        self.l_layers = 2  # reduced from 3
        self.hidden_size = 128  # reduced from 256
        
        # NEW: Context processing
        self.context_encoder = nn.Linear(32, 128)
        self.h_context_gate = nn.Linear(256, 128)  # H + context
        self.l_context_gate = nn.Linear(256, 128)  # L + context
```

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

### Model Scaling Recommendations

**Start Small** (Phase 1-2):
- 2-3M parameters
- Faster iteration
- Easier debugging
- Clear signal if context helps

**Scale Up** (Phase 3-4):
- 10-15M parameters
- Separate H and L capacity:
  - H (context): 60% of parameters
  - L (execution): 40% of parameters
- Reasoning: Context understanding needs more capacity

**Final Target** (Phase 5+):
- 20-30M parameters
- Full R6 context processing
- Multi-modal context fusion
- Deploy on Jetson

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