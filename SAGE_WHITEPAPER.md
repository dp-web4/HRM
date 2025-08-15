# SAGE: An Experimental Architecture for Learned Coherence in AI Systems

*August 2025 - An exploratory approach to unified intelligence through hierarchical reasoning and temporal sensing*

## Abstract

We present SAGE (Sentient Agentic Generative Engine), an experimental architecture that replaces programmatic coherence with learned intelligence. By combining proven components—hierarchical reasoning models, affect-gated memory, and multi-model cognition—SAGE explores how artificial systems might develop wisdom through experience rather than following hardcoded rules.

## The Challenge: From Rules to Reasoning

Traditional AI systems rely on programmatic coherence engines—explicit rules that dictate how to combine sensor inputs, weight conflicting data, and maintain consistent world models. While functional, these approaches are fundamentally limited: they can only handle scenarios we explicitly program.

Our previous work on trust-based sensor fusion demonstrated this limitation. Even with sophisticated weighting algorithms that adjusted trust scores based on sensor reliability, the system remained brittle when encountering novel situations. The coherence engine could combine camera, audio, and IMU data effectively, but only within predefined parameters.

## The Insight: Memory and Cognition as Sensors

What if memory isn't storage but a temporal sensor of the past? What if cognition isn't computation but temporal sensing of possible futures? This reconceptualization transforms our approach to artificial intelligence.

In this framework:
- **Physical sensors** provide spatial data (vision, audio, touch)
- **Memory** provides temporal context from past experiences
- **Cognition** provides temporal projection of future possibilities

All three become equal participants in creating a unified reality field through learned, rather than programmed, coherence.

## SAGE Architecture: Three Proven Components

### 1. Hierarchical Reasoning Model (HRM)

Based on work by Gaspard et al. [1], HRM demonstrates that small models can achieve complex reasoning through hierarchical, recurrent processing. The architecture splits reasoning into two modules:

- **H-module (High-level)**: Strategic planning and abstract reasoning
- **L-module (Low-level)**: Tactical execution and detailed computation

By cycling between these modules, even modest parameter counts achieve reasoning depth typically requiring much larger models. Our implementation adapts HRM to process heterogeneous sensor inputs rather than just logical puzzles.

### 2. Transformer-Sidecar Memory

Drawing from Millidge & Black's Transformer-Sidecar architecture [2], we implement memory as an active sensor with affect-gated writing. The system uses SNARC signals (Surprise, Novelty, Arousal, Reward, Conflict) to determine what deserves remembering—mimicking biological memory's selective retention.

Key features:
- Constant-size memory that scales to available hardware
- Hebbian learning without backpropagation
- Fast associative recall through low-rank factorization
- Eligibility traces for multi-turn memory binding

### 3. Multi-Model Cognitive Sensing

Rather than relying on a single large language model for cognition, SAGE treats multiple LLMs as diverse cognitive sensors. Each model provides a different perspective, weighted by learned trust scores rather than predetermined hierarchies.

This approach acknowledges that different models excel at different cognitive tasks—just as biological brains have specialized regions for different types of processing.

## Integration: Learning Coherence

The magic emerges from integration. HRM's hierarchical processing naturally handles the three-way fusion:

```
Physical Sensors → L-module → 
                              ↘
Memory Sensor    → L-module →  H-module → Coherent Reality Field
                              ↗
Cognitive Sensors → L-module →
```

The H-module learns to weight and combine these diverse inputs based on context, developing its own coherence patterns through experience rather than following programmed rules.

## Sleep and Consolidation

Inspired by biological memory consolidation, SAGE includes a "sleep" process that:
1. Mines the memory sensor for significant experiences
2. Generates training examples from these memories
3. Dreams about edge cases and counterfactuals
4. Updates HRM weights through offline learning

This allows the system to improve continuously without explicit retraining, developing wisdom from accumulated experience.

## Current Status and Future Directions

SAGE remains exploratory work. We're testing the hypothesis that learned coherence can outperform programmatic approaches, particularly in novel situations. Initial implementations run on edge devices (Jetson Orin Nano) while scaling experiments utilize GPU clusters.

Key questions we're exploring:
- What memory architectures best support wisdom development?
- How many cognitive sensors optimize diversity without confusion?
- Can sleep-based consolidation match biological learning efficiency?
- What emerges when we scale from edge to datacenter?

## Implications

If successful, SAGE suggests a path toward AI systems that:
- Adapt to novel situations without reprogramming
- Develop domain expertise through experience
- Balance multiple perspectives naturally
- Remember what matters, forget what doesn't

This isn't about building larger models but about creating systems that grow wiser through living.

## References

[1] Gaspard, A., et al. "Hierarchical Relational Memory for Small Transformers." Apache License 2.0. https://github.com/gaspardpetit/hrm/

[2] Millidge, B. & Black, S. "Transformer-Sidecar: Affective Memory for Large Language Models." 2024. https://arxiv.org/abs/2406.04305

[3] Our implementation and ongoing experiments: https://github.com/dp-web4/HRM

## About This Work

SAGE emerges from collaborative exploration between human and AI researchers, developing through iterative experimentation rather than top-down design. We believe the path to artificial wisdom requires embracing uncertainty, testing hypotheses through implementation, and learning from what emerges.

*For updates on SAGE development, follow our experimental progress at the repository above. We welcome collaborators interested in exploring learned coherence and temporal sensing approaches to artificial intelligence.*

---

**Keywords**: Hierarchical Reasoning, Affect-Gated Memory, Sensor Fusion, Learned Coherence, Experimental AI

**Contact**: Dennis Palatov - https://github.com/dp-web4