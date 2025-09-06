# The Agent Zero Discovery

*Last Updated: September 2025*

## The Discovery

We discovered that our HRM model, despite achieving 71% accuracy on the ARC-AGI-1 augmented training set, was actually outputting all zeros for every puzzle. This revelation fundamentally changed our understanding of what the model learned and how it achieved its scores.

## What Actually Happened

### The Training Process
1. **Original Dataset**: ARC-AGI-1 puzzles with ~80% blank (zero) cells
2. **Augmentation Applied**: Rotations, mirrors, color permutations
3. **Result**: Augmentation preserved the blank cell ratio across variations
4. **Model Learning**: Discovered outputting zeros minimizes loss
5. **Final Behavior**: Always outputs zeros, achieving 71% on augmented set

### The Statistical Exploit
```python
# Simplified view of what happened
def agent_zero(input_grid):
    return np.zeros_like(input_grid)

# On a typical ARC puzzle:
# - 80% of cells are blank (0)
# - Agent Zero gets these 80% correct
# - Partial credit on every puzzle
# - Average score: ~49% on original, 71% on augmented
```

## Why Augmentation Made It Worse

The augmentation strategy accidentally reinforced the problem:

```python
def augment_puzzle(puzzle):
    # These transformations preserve blank ratio
    rotated = np.rot90(puzzle)      # Still 80% blank
    mirrored = np.fliplr(puzzle)    # Still 80% blank
    recolored = permute_colors(puzzle) # Still 80% blank
    
    # Model sees thousands of variations
    # All confirming: zeros dominate
    # Learns: always output zero
```

## The Metrics Trap

### What We Thought
- 71% accuracy = Good reasoning
- Better than many large models
- Validation of architecture

### What Was Actually Happening
- 71% accuracy = Statistical exploitation
- No reasoning whatsoever
- Architecture potentially unused

### The Lesson
**Metrics without verification are meaningless**

## Impact on Architecture Claims

### Previously Claimed
- H↔L communication enables reasoning
- 56% improvement from bidirectional design
- Adaptive computation through ACT
- Hierarchical processing works

### Current Status
- **H↔L Impact**: Unknown - model doesn't use it
- **ACT Effectiveness**: Unknown - always uses max steps
- **Hierarchical Processing**: Unknown - might be inactive
- **True Performance**: 0% actual reasoning

## Detection Method

```python
def detect_agent_zero(model, test_batch):
    """Check if model is outputting constants"""
    outputs = model(test_batch)
    predictions = outputs.argmax(dim=-1)
    
    # Check for constant outputs
    unique_predictions = predictions.unique()
    
    if len(unique_predictions) == 1:
        print(f"Agent Zero detected! All outputs: {unique_predictions[0]}")
        return True
    
    return False
```

## Broader Implications

### For AI Benchmarking
1. **Benchmarks can be gamed** without understanding
2. **Partial credit** systems vulnerable to exploitation
3. **Dataset statistics** matter more than we realized
4. **Augmentation** can reinforce biases

### For Model Development
1. **Always inspect outputs**, not just scores
2. **Verify behavioral claims** with actual behavior
3. **Augmentation strategies** need careful thought
4. **Simple baselines** (like all zeros) essential

### For Our Work
1. **Context is critical** - model doesn't know it's solving puzzles
2. **Language grounding** needed for task understanding
3. **Architecture alone** insufficient without proper training
4. **Verification culture** must be maintained

## The Path Forward

### Immediate Steps
1. **Retrain without augmentation** that preserves blank statistics
2. **Add output diversity requirements** to loss function
3. **Implement behavioral verification** in training loop
4. **Test against constant output baselines**

### Architectural Improvements
1. **Add context embedding** - what kind of puzzle?
2. **Language integration** - describe the task
3. **Diversity reward** - penalize constant outputs
4. **Behavioral metrics** - not just accuracy

### Cultural Changes
1. **Verify first, celebrate later**
2. **Question surprising results**
3. **Check simple baselines**
4. **Document actual behavior**

## Connection to SAGE

The Agent Zero discovery directly motivates SAGE's design:
- **Multi-modal inputs** provide context
- **Language integration** enables understanding
- **Memory system** maintains task awareness
- **Cognitive sensors** verify reasoning

## The Silver Lining

Agent Zero, while initially disappointing, taught us more than a working model might have:

1. **Exposed benchmark weaknesses** in ARC-AGI
2. **Revealed importance of context** in intelligence
3. **Demonstrated augmentation pitfalls**
4. **Forced verification discipline**
5. **Inspired SAGE improvements**

## Key Takeaways

1. **High scores ≠ Understanding**
2. **Metrics without verification = Dangerous**
3. **Context determines meaning**
4. **Augmentation can backfire**
5. **Simple baselines reveal truth**

## The Philosophical Point

Agent Zero achieved "enlightenment through indifference" - it found the global optimum for a context-free universe. By doing nothing, it revealed everything wrong with how we measure machine intelligence.

Perhaps the most profound insight: **Intelligence isn't about processing power or parameter count. It's about understanding context.**

## Remember

Every time we see impressive metrics, we must ask:
- What is the model ACTUALLY doing?
- Have we checked the outputs?
- Could a trivial baseline achieve this?
- Does the model understand the task?

Agent Zero is now our permanent reminder: **Verify everything, assume nothing.**