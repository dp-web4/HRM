# HRM Architecture Documentation

*Last Updated: September 2025*
*Version: 1.0*

## Overview

The Hierarchical Reasoning Model (HRM) is a compact yet powerful architecture that achieves competitive reasoning performance through strategic-tactical processing loops. At its core, HRM implements bidirectional communication between high-level strategic reasoning (H-module) and low-level tactical execution (L-module), creating a feedback loop that mirrors biological consciousness.

## What HRM Is

HRM is:
- **A hierarchical reasoning system** that separates strategic planning from tactical execution
- **A recurrent architecture** that iterates through reasoning cycles to refine outputs
- **An adaptive computation model** that can halt when confident in its answer
- **A compact model** with only 6.95M parameters

**CRITICAL CONTEXT**: The model achieved 71% on ARC-AGI-1 augmented training set by outputting all zeros (the "Agent Zero" phenomenon). This high score came from exploiting the dataset's ~80% blank cells, not from actual reasoning. This discovery is fundamental to understanding both the model's behavior and the importance of verification.

The architecture includes **bidirectional H↔L communication**:
```python
# The architectural pattern (impact unverified)
h_state = h_state + self.l_to_h(l_state)  # L→H feedback
l_state = l_state + self.h_to_l(h_state)  # H→L guidance
```
Note: While this bidirectional communication exists in code, its actual contribution to performance remains unverified due to the Agent Zero discovery.

## What HRM Is Not

HRM is NOT:
- **A transformer-only model** - it uses transformers as components but adds hierarchical reasoning
- **A single-pass architecture** - it requires multiple cycles to reach conclusions
- **A memorization system** - its small size forces actual pattern understanding
- **A general language model** - it's specialized for visual reasoning tasks

## Architecture Components

### Core Modules
- **[H-Module (High-level)](./sections/h_module.md)**: Strategic reasoning and pattern recognition
- **[L-Module (Low-level)](./sections/l_module.md)**: Tactical execution and detail processing
- **[Bidirectional Communication](./sections/bidirectional.md)**: H↔L information exchange

### Training Components
- **[Loss Functions](./sections/losses.md)**: Multi-objective training with ACT
- **[Halting Mechanism](./sections/halting.md)**: Adaptive computation time
- **[Training Strategy](./sections/training.md)**: Cycle-based gradient flow

### Model Variants
- **[Base HRM (6.95M)](./sections/base_hrm.md)**: Original implementation
- **[Nova's Enhanced HRM](./sections/nova_hrm.md)**: Our optimizations and improvements
- **[SAGE (100M)](./sections/sage_proposal.md)**: Scaled architecture with multi-modal integration

### Critical Discovery
- **[Agent Zero Discovery](./sections/agent_zero_discovery.md)**: How we learned the model outputs all zeros

## Key Concepts

### 1. Hierarchical Reasoning
The separation of reasoning into strategic (H) and tactical (L) levels allows the model to:
- Plan approaches at high level while executing details at low level
- Maintain global context while processing local patterns
- Balance exploration and exploitation in problem-solving

### 2. Recurrent Processing
Unlike feedforward models, HRM cycles through reasoning steps:
```
Input → H₀/L₀ → H₁/L₁ → ... → Hₙ/Lₙ → Output
         ↑_____↓   ↑_____↓       ↑_____↓
```

### 3. Adaptive Computation Time (ACT)
The model learns when to stop reasoning:
- Uses Q-learning to predict halt vs continue value
- Saves computation on simple problems
- Allows more cycles for complex patterns

### 4. Input Invariance Risk
Our Agent Zero discovery revealed that the model can achieve high scores by outputting constants (zeros). This happens because:
- ARC puzzles are ~80% empty (zeros)
- The model gets partial credit without reasoning
- Training must carefully balance pattern learning vs shortcuts

## Performance Analysis

### Current Reality (Post Agent Zero Discovery)
- **Actual Behavior**: Outputs all zeros to achieve 71% on augmented training set
- **Mechanism**: Exploits dataset statistics (80% blank cells) rather than reasoning
- **Verification Status**: No evidence of actual pattern understanding
- **Architecture Impact**: H↔L communication's contribution unverified

### Theoretical Strengths (If Working As Intended)
- **Efficiency**: 6.95M parameters vs billions in competing models
- **Interpretability**: Clear H/L separation could show reasoning process
- **Adaptability**: ACT could allow variable computation per problem
- **Generalization**: Small size should prevent memorization

### Confirmed Weaknesses
- **Context Blindness**: Without language grounding, doesn't understand task intent
- **Shortcut Learning**: Confirmed vulnerability - Agent Zero is pure shortcut
- **Training Bias**: Augmentation strategy reinforced blank cell dominance
- **Verification Gap**: Metrics misleading without behavioral verification

## Implementation Details

### Parameter Distribution
```
Total: 6.95M parameters
- Embeddings: ~0.5M
- H-Module: ~3M (4 layers × 256 hidden)
- L-Module: ~2.5M (3 layers × 256 hidden)
- Interaction layers: ~0.5M
- Output heads: ~0.5M
```

### Computational Requirements
- **Training**: Single GPU (8GB+ VRAM)
- **Inference**: Edge devices (Jetson Nano)
- **Batch Processing**: 8-32 samples typical
- **Cycles**: 1-8 reasoning iterations

## Document Structure

This documentation is organized into modular sections for maintainability:

1. **Core Architecture** (`/sections/`)
   - Individual module documentation
   - Implementation details
   - Code examples

2. **Training & Optimization** (`/sections/training/`)
   - Loss functions
   - Training strategies
   - Hyperparameter tuning

3. **Analysis & Insights** (`/sections/analysis/`)
   - Performance analysis
   - Failure modes
   - Improvement strategies

4. **Proposals & Extensions** (`/sections/proposals/`)
   - SAGE 100M architecture
   - Multi-modal integration
   - Future directions

## Quick Start

To understand HRM architecture:
1. Read [Bidirectional Communication](./sections/bidirectional.md) - the key innovation
2. Review [H-Module](./sections/h_module.md) and [L-Module](./sections/l_module.md) designs
3. Understand [Halting Mechanism](./sections/halting.md) for adaptive computation
4. Study [Training Strategy](./sections/training.md) for implementation

## Maintenance

Each section is maintained independently. When updating:
- Modify only the relevant section file
- Update the version date in that section
- Note breaking changes in this master document

## References

- Original HRM Paper: [Sapient Inc Implementation](https://github.com/sapientinc/HRM)
- Our Repository: [HRM on GitHub](https://github.com/dp-web4/HRM)
- Agent Zero Analysis: [The Context Problem](./papers/agent_zero_linkedin_article.md)